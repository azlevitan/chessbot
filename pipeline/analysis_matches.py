"""Per-block match evidence and per-game outcomes, without loading engines."""

from __future__ import annotations

import json
from pathlib import Path
import re

import elo
from analysis_selfplay import _read

COLUMNS = tuple(("gen block a b block_games a_wins draws b_wins games cumulative_wins "
                 "cumulative_draws cumulative_losses score elo elo_lo elo_hi llr "
                 "sprt_lower sprt_upper settings_source sims seed opening_offset max_plies").split())
GAME_COLUMNS = tuple("gen block game opening white result termination plies".split())


def resolve_settings(directory, fallback, fallback_source, stages, warnings):
    """Use captured match-stage settings only when all attempts agree."""
    configs = []
    for row in stages:
        if row.get("stage") == "match" and row.get("config_json"):
            config = json.loads(row["config_json"])
            if isinstance(config, dict) and isinstance(config.get("match"), dict):
                configs.append(config["match"])
    if configs:
        if all(config == configs[0] for config in configs):
            if fallback_source == "generation.json" and configs[0] != fallback:
                warnings.append(f"{directory}: using effective match-stage config, which differs from generation.json")
            return configs[0], "telemetry.match.config"
        warnings.append(f"{directory}: match-stage configs differ; historical LLR is unavailable")
        return None, "ambiguous"
    if fallback_source == "generation.json":
        return fallback, fallback_source
    if any(directory.glob("match/block_*.json")):
        warnings.append(f"{directory}: historical SPRT settings were not captured; LLR is unavailable")
    return None, "unavailable"


def match_rows(directory: Path, gen: int, settings: dict | None, settings_source: str, warnings):
    rows, games, totals = [], [], [0, 0, 0]
    paths = sorted(directory.glob("match/block_*.json"))
    indices = []
    for path in paths:
        match = re.fullmatch(r"block_([0-9]{2,})\.json", path.name)
        if match:
            indices.append((int(match[1]), path))
    for block, path in sorted(indices):
        if block != len(rows):
            warnings.append(f"{path}: non-contiguous match blocks; cumulative series stops here")
            break
        report = _read(path, warnings)
        counts = [report.get(key) for key in ("a_wins", "draws", "b_wins")]
        if (report.get("format") != "chessbot-match-v1"
                or any(type(value) is not int or value < 0 for value in counts)
                or sum(counts) <= 0 or report.get("games") != sum(counts)):
            warnings.append(f"{path}: invalid published match totals; cumulative series stops here")
            break
        totals = [a + b for a, b in zip(totals, counts)]
        lo, hi = elo.elo_ci95(*totals)
        try:
            if settings is None:
                raise ValueError("historical settings unavailable")
            lower, upper = elo.sprt_bounds(settings.get("alpha", 0.05), settings.get("beta", 0.05))
            likelihood = elo.llr(*totals, settings.get("elo0", 0), settings.get("elo1", 40))
        except ValueError:
            lower = upper = likelihood = None
            if settings is not None:
                warnings.append(f"{directory}: invalid SPRT settings; leaving LLR unavailable")
        row = {key: report.get(key) for key in (
            "a", "b", "sims", "seed", "opening_offset", "max_plies")}
        row.update(gen=gen, block=block, block_games=sum(counts),
                   a_wins=counts[0], draws=counts[1], b_wins=counts[2], games=sum(totals),
                   cumulative_wins=totals[0], cumulative_draws=totals[1], cumulative_losses=totals[2],
                   score=elo.score(*totals), elo=elo.elo(*totals), elo_lo=lo, elo_hi=hi,
                   llr=likelihood, sprt_lower=lower, sprt_upper=upper, settings_source=settings_source)
        rows.append(row)
        results = report.get("results", [])
        if not isinstance(results, list):
            warnings.append(f"{path}: missing or malformed per-game results")
            results = []
        for game in results:
            if not isinstance(game, dict):
                warnings.append(f"{path}: ignoring malformed game result")
                continue
            games.append(dict({key: game.get(key) for key in GAME_COLUMNS}, gen=gen, block=block))
    return rows, games
