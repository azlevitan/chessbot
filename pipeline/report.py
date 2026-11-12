"""Rebuild the SPEC §10 promotion ladder and Task 9.5 TSV from saved metadata.

Ladder intervals shift each match's bounds by the incumbent's point Elo; they
describe conditional match uncertainty, not accumulated ladder uncertainty.
Missing metrics are null in JSON and NA in TSV. Measured stage spans include
interrupted attempts, and archived selfplay diagnostics survive chunk pruning.
No model or sample contents are read, and no training libraries are imported.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
from pathlib import Path
import re
import sys

from run_layout import read_json
from telemetry import stages_summary

COLUMNS = tuple(("gen promoted reason games score elo elo_lo elo_hi ladder_elo "
                 "vs_random vs_material train_loss val_top1 selfplay_samples "
                 "selfplay_secs train_secs match_secs").split())


def _number(value, lower=None, upper=None):
    try:
        if type(value) not in (int, float) or not math.isfinite(value):
            return None
    except OverflowError:
        return None
    if (lower is not None and value < lower) or (upper is not None and value > upper):
        return None
    return value


def _optional(path: Path) -> dict:
    try:
        return read_json(path)
    except (OSError, ValueError):
        return {}


def _generations(run: Path):
    root = run / "gens"
    if not root.is_dir():
        return []
    generations = []
    for directory in root.iterdir():
        match = re.fullmatch(r"gen_([0-9]{4,})", directory.name)
        if match and directory.is_dir() and not directory.is_symlink():
            gen = int(match[1])
            if directory.name == f"gen_{gen:04d}":
                generations.append((gen, directory))
    return sorted(generations)


def _decision(directory: Path, gen: int) -> dict | None:
    path = directory / "decision.json"
    try:
        decision = read_json(path)
    except FileNotFoundError:
        return None
    except ValueError as error:
        raise ValueError(f"{path}: invalid published decision") from error
    valid = type(decision.get("promoted")) is bool
    if gen == 0:
        valid = valid and decision["promoted"] and decision.get("reason") == "bootstrap"
    else:
        interval = decision.get("elo_ci95")
        valid = (valid and decision.get("reason") in ("sprt_accept", "sprt_reject", "max_games")
                 and type(decision.get("games")) is int and decision["games"] > 0
                 and _number(decision.get("score"), 0, 1) is not None
                 and _number(decision.get("elo")) is not None
                 and isinstance(interval, list) and len(interval) == 2
                 and all(_number(value) is not None for value in interval)
                 and interval[0] <= interval[1])
    if not valid:
        raise ValueError(f"{path}: invalid published decision")
    return decision


def _samples(directory: Path) -> int | None:
    total, found = 0, False
    for binary in (directory / "selfplay").glob("chunk_*.bin"):
        if not re.fullmatch(r"chunk_[0-9]{6}\.bin", binary.name):
            continue
        metadata = _optional(binary.with_suffix(".json"))
        count = metadata.get("samples")
        if (metadata.get("format") != "chessbot-chunk-v1"
                or metadata.get("kind") != "selfplay"
                or type(count) is not int or count < 0):
            continue
        try:
            if not binary.is_file() or binary.stat().st_size != count * 492:
                continue
        except OSError:
            continue
        total, found = total + count, True
    saved = _optional(directory / "selfplay_summary.json")
    chunks = saved.get("chunks")
    if (saved.get("format") == "chessbot-selfplay-summary-v1" and isinstance(chunks, list)
            and chunks and all(isinstance(row, dict) and type(row.get("samples")) is int
                               and row["samples"] >= 0 for row in chunks)):
        return max(total, sum(row["samples"] for row in chunks))
    return total if found else None


def _collect(run_dir):
    run = Path(run_dir).resolve(strict=True)
    if not run.is_dir():
        raise ValueError(f"{run}: expected a run directory")
    ladder, rows, best = [], [], 0.0
    for gen, directory in _generations(run):
        decision = _decision(directory, gen)
        if decision is None:
            continue
        row = dict.fromkeys(COLUMNS)
        row.update(gen=gen, promoted=decision["promoted"], reason=decision["reason"])
        for field in ("games", "score", "elo"):
            row[field] = _number(decision.get(field))
        interval = decision.get("elo_ci95")
        if isinstance(interval, list) and len(interval) == 2:
            row["elo_lo"], row["elo_hi"] = map(_number, interval)
        for opponent in ("random", "material"):
            anchor = _optional(directory / "anchors" / f"vs_{opponent}.json")
            row[f"vs_{opponent}"] = _number(anchor.get("a_score"), 0, 1)
        if decision["promoted"]:
            elo = 0.0 if gen == 0 else best + decision["elo"]
            ci = [0, 0] if gen == 0 else [best + value for value in interval]
            if any(_number(value) is None for value in (elo, *ci)):
                raise ValueError(f"{directory}: ladder Elo overflow")
            row["ladder_elo"] = best = elo
            ladder.append({"gen": gen, "elo": elo, "ci95": ci,
                           "vs_random": row["vs_random"], "vs_material": row["vs_material"]})
        train = _optional(directory / "train.json")
        validation = train.get("val")
        row.update(train_loss=_number(train.get("final_loss"), 0),
                   val_top1=_number(validation.get("policy_top1"), 0, 1)
                   if isinstance(validation, dict) else None,
                   selfplay_samples=_samples(directory),
                   train_secs=_number(train.get("seconds"), 0))
        measured = stages_summary(directory)
        for stage, column in (("selfplay", "selfplay_secs"), ("train", "train_secs"),
                              ("match", "match_secs")):
            if stage in measured:
                row[column] = measured[stage]["seconds"]
        rows.append(row)
    return run, ladder, rows


def build_ladder(run_dir) -> list[dict]:
    """Return promoted generations in numeric order, anchored at bootstrap Elo 0."""
    return _collect(run_dir)[1]


def _publish(contents) -> None:
    """Stage both outputs before replacement; each file is published atomically.

    A crash between replacements is repaired by a deterministic rebuild. Avoid
    rewriting unchanged output, including the loop's normal done boundary.
    """
    staged = []
    try:
        for path, text in contents:
            pending = path.with_name(path.name + ".tmp")
            try:
                unchanged = path.read_text(encoding="utf-8") == text
            except (FileNotFoundError, UnicodeError):
                unchanged = False
            if unchanged:
                continue
            with pending.open("w", encoding="utf-8", newline="") as stream:
                stream.write(text)
                stream.flush()
                os.fsync(stream.fileno())
            staged.append((pending, path))
        for pending, path in staged:
            pending.replace(path)
    finally:
        for path, _ in contents:
            path.with_name(path.name + ".tmp").unlink(missing_ok=True)


def write_report(run_dir) -> None:
    """Rebuild ladder.json and the exact 17-column report.tsv from one snapshot."""
    run, ladder, rows = _collect(run_dir)
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
    writer.writerow(COLUMNS)
    writer.writerows(["NA" if row[column] is None else row[column] for column in COLUMNS]
                     for row in rows)
    _publish([(run / "ladder.json", json.dumps(ladder, allow_nan=False, sort_keys=True) + "\n"),
              (run / "report.tsv", stream.getvalue())])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        write_report(args.run)
        sys.stdout.write((args.run / "report.tsv").read_text(encoding="utf-8"))
        return 0
    except (OSError, ValueError) as error:
        print(f"report: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
