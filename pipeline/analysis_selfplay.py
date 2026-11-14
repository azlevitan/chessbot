"""Metadata-only self-play diagnostics retained outside prunable chunk directories."""

from __future__ import annotations

import json
from pathlib import Path
import re

from run_layout import write_json_atomic

FORMAT = "chessbot-selfplay-summary-v1"
COLUMNS = tuple(("gen chunk games samples samples_per_game model seed sims max_plies "
                 "temp_plies white_wins black_wins draws draw_rate checkmate resign "
                 "rule_draw repetition insufficient_material ply_cap resign_threshold "
                 "playthrough_games false_positives false_positive_rate source").split())


def _read(path, warnings):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("expected an object")
        return value
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as error:
        warnings.append(f"{path}: {error}")
        return {}


def _count(value):
    return type(value) is int and value >= 0


def _chunks(directory: Path, warnings):
    rows = []
    for sidecar in sorted(directory.glob("chunk_*.json")):
        match = re.fullmatch(r"chunk_([0-9]{6})\.json", sidecar.name)
        if match is None or sidecar.is_symlink():
            continue
        metadata = _read(sidecar, warnings)
        samples, games = metadata.get("samples"), metadata.get("games")
        binary = sidecar.with_suffix(".bin")
        valid = (metadata.get("format") == "chessbot-chunk-v1"
                 and metadata.get("kind") == "selfplay"
                 and _count(samples) and _count(games) and games > 0)
        try:
            valid = valid and binary.is_file() and not binary.is_symlink() and binary.stat().st_size == samples * 492
        except OSError:
            valid = False
        if not valid:
            warnings.append(f"{sidecar}: ignoring incomplete or invalid selfplay chunk")
            continue
        results, terms, audit = (metadata.get(key, {}) for key in ("results", "terminations", "resign"))
        results, terms, audit = (value if isinstance(value, dict) else {} for value in (results, terms, audit))
        groups = ((results, ("white", "black", "draw")), (terms, (
            "checkmate", "resign", "rule_draw", "repetition", "insufficient_material", "ply_cap")))
        for counters, names in groups:
            if not all(_count(counters.get(key)) for key in names):
                warnings.append(f"{sidecar}: some optional selfplay counters are unavailable")
            elif sum(counters[key] for key in names) != games:
                warnings.append(f"{sidecar}: inconsistent selfplay counters; leaving them unavailable")
                counters.clear()
        value = lambda counters, key: counters.get(key) if _count(counters.get(key)) else None
        row = {key: metadata.get(key) for key in ("model", "seed", "sims", "max_plies", "temp_plies")}
        row.update(chunk=int(match[1]), samples=samples, games=games,
                   samples_per_game=samples / games, white_wins=value(results, "white"),
                   black_wins=value(results, "black"), draws=value(results, "draw"),
                   draw_rate=results["draw"] / games if _count(results.get("draw")) else None,
                   **{key: value(terms, key) for key in (
                       "checkmate", "resign", "rule_draw", "repetition", "insufficient_material", "ply_cap")})
        played, false = audit.get("playthrough_games"), audit.get("false_positives")
        valid_audit = _count(played) and _count(false) and false <= played <= games
        row.update(resign_threshold=audit.get("threshold"),
                   playthrough_games=played if valid_audit else None,
                   false_positives=false if valid_audit else None,
                   false_positive_rate=false / played if valid_audit and played else None)
        rows.append(row)
    return rows


def _saved(directory, warnings):
    saved = _read(directory / "selfplay_summary.json", warnings)
    rows = saved.get("chunks")
    if saved.get("format") == FORMAT and isinstance(rows, list) and all(
            isinstance(row, dict) and _count(row.get("chunk"))
            and all(_count(row.get(key)) for key in ("samples", "games"))
            for row in rows):
        return rows
    return []


def selfplay_rows(directory: Path, gen: int, warnings: list[str]) -> list[dict]:
    """Prefer surviving complete chunks; a full snapshot survives partial pruning."""
    actual = _chunks(directory / "selfplay", warnings)
    archived = _saved(directory, warnings)
    rows, source = (archived, "snapshot") if len(archived) > len(actual) else (actual, "chunks")
    return [dict(row, gen=gen, source=source) for row in rows]


def archive_selfplay(generation_dir, expected_games: int | None = None) -> None:
    """Publish compact diagnostics after selfplay, or preserve them before pruning.

    An existing snapshot is retained during pre-prune retries, including after a
    partially completed recursive deletion. A successful selfplay stage supplies
    expected_games and replaces the snapshot from its complete authoritative data.
    """
    directory, warnings = Path(generation_dir), []
    if expected_games is None and _saved(directory, warnings):
        return
    rows = _chunks(directory / "selfplay", warnings)
    if expected_games is not None and sum(row["games"] for row in rows) != expected_games:
        raise ValueError(f"{directory}: cannot archive incomplete selfplay diagnostics")
    if not rows:
        return
    target = directory / "selfplay_summary.json"
    summary = {"format": FORMAT, "chunks": rows}
    if _read(target, []) != summary:
        write_json_atomic(target, summary)


def snapshot_run_selfplay(run_dir) -> None:
    for directory in sorted((Path(run_dir) / "gens").glob("gen_*")):
        if directory.is_dir() and not directory.is_symlink():
            archive_selfplay(directory)
