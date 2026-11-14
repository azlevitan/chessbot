#!/usr/bin/env python3
"""Resumable block matches and SPEC §10 promotion decisions (stdlib only).

Resume with the same settings: match reports record players, sims, seed, opening
offset and ply cap, but do not record batch size, provider or thread count.
Completed blocks are the source of truth; decisions can always be rebuilt.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import subprocess
import sys

import elo
from run_layout import read_json, write_json_atomic
from process_metrics import measured_run


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("engine", "a", "b"):
        parser.add_argument(f"--{flag}", required=True)
    for flag in ("out-dir", "decision"):
        parser.add_argument(f"--{flag}", type=Path, required=True)
    parser.add_argument("--sims", type=int, required=True)
    parser.add_argument("--seed-base", type=int, required=True)
    for flag, default in (("block-games", 40), ("max-games", 400), ("batch", 8),
                          ("max-plies", 300), ("threads", 1)):
        parser.add_argument(f"--{flag}", type=int, default=default)
    for flag, default in (("elo0", 0.0), ("elo1", 40.0), ("alpha", 0.05), ("beta", 0.05)):
        parser.add_argument(f"--{flag}", type=float, default=default)
    parser.add_argument("--ep", choices=("auto", "cpu", "coreml"), default="auto")
    parser.add_argument("--infer-batch", type=int)
    args = parser.parse_args(argv)
    try:
        _validate(args)
    except ValueError as error:
        parser.error(str(error))
    return args


def _validate(args) -> None:
    infer_batch = getattr(args, "infer_batch", None)
    if infer_batch is not None and (type(infer_batch) is not int or not 1 <= infer_batch <= 64):
        raise ValueError("--infer-batch must be an integer in 1..64")
    for field in ("block_games", "max_games", "batch", "max_plies", "threads"):
        value = getattr(args, field)
        if type(value) is not int or value <= 0:
            raise ValueError(f"--{field.replace('_', '-')} must be positive")
    if args.block_games % 2 or args.max_games % 2:
        raise ValueError("--block-games and --max-games must be even")
    if not 2 <= args.sims <= 2**32 - 1 or args.max_plies > 2**32 - 1:
        raise ValueError("--sims must be 2..2^32-1 and --max-plies at most 2^32-1")
    if min(args.block_games, args.max_games) > 2**32 - 1:
        raise ValueError("a match block must fit the engine's u32 game count")
    if not 0 <= args.seed_base <= 2**64 - 1:
        raise ValueError("--seed-base must fit a nonnegative u64")
    last = (args.max_games - 1) // args.block_games
    # Check the largest game seed in the final full block and the tail block.
    for block in {max(0, last - 1), last}:
        games = min(args.block_games, args.max_games - block * args.block_games)
        if args.seed_base + block + games - 1 > 2**64 - 1:
            raise ValueError("--seed-base + block + game index overflows u64")
    if not all(getattr(args, field) for field in ("engine", "a", "b")):
        raise ValueError("engine and player specs must not be empty")
    elo.llr(1, 0, 1, args.elo0, args.elo1)
    elo.sprt_bounds(args.alpha, args.beta)


def _check_paths(args) -> None:
    directory = args.out_dir.resolve()
    targets = (args.decision, args.decision.with_name(args.decision.name + ".tmp"))
    existing = list(args.out_dir.glob("block_*.json*"))
    for path in targets:
        # Preserve the basename too: a dangling symlink may hide a reserved name.
        for candidate in (path.parent.resolve() / path.name, path.resolve()):
            reserved = (directory in candidate.parents and re.fullmatch(
                r"block_\d{2,}\.json(?:\..*)?", candidate.relative_to(directory).parts[0]))
            if candidate == directory or candidate in directory.parents or reserved:
                raise ValueError("decision and block output paths must not overlap")
        if any(path.exists() and other.exists() and path.samefile(other) for other in existing):
            raise ValueError("decision and block output paths must not overlap")


def _read_block(path: Path, args, block: int, games: int) -> tuple[int, int, int]:
    report = read_json(path)
    expected = {"format": "chessbot-match-v1", "a": args.a, "b": args.b,
                "games": games, "sims": args.sims, "seed": args.seed_base + block,
                "opening_offset": 20 * block, "max_plies": args.max_plies}
    for key, value in expected.items():
        if type(report.get(key)) is not type(value) or report[key] != value:
            raise ValueError(f"{path}: incompatible match field {key}")
    counts = tuple(report.get(key) for key in ("a_wins", "draws", "b_wins"))
    if any(type(count) is not int or count < 0 for count in counts) or sum(counts) != games:
        raise ValueError(f"{path}: invalid match W/D/L totals")
    score = report.get("a_score")
    if (type(score) not in (int, float) or not 0 <= score <= 1
            or not math.isclose(score, elo.score(*counts), rel_tol=0, abs_tol=1e-12)):
        raise ValueError(f"{path}: inconsistent match score")
    results = report.get("results")
    if not isinstance(results, list) or len(results) != games:
        raise ValueError(f"{path}: missing game results")
    observed = [0, 0, 0]
    for index, result in enumerate(results):
        if (not isinstance(result, dict)
                or type(result.get("game")) is not int or result["game"] != index
                or type(result.get("opening")) is not int
                or result["opening"] != (20 * block + index // 2) % 50
                or result.get("white") != ("a" if index % 2 == 0 else "b")
                or result.get("result") not in ("1-0", "0-1", "1/2-1/2")
                or type(result.get("plies")) is not int
                or not 0 <= result["plies"] <= args.max_plies):
            raise ValueError(f"{path}: invalid game result {index}")
        draw = result["result"] == "1/2-1/2"
        a_win = (result["result"] == "1-0") == (result["white"] == "a")
        observed[1 if draw else (0 if a_win else 2)] += 1
    if tuple(observed) != counts:
        raise ValueError(f"{path}: game results disagree with W/D/L totals")
    return counts


def _cleanup(temporary: Path) -> None:
    temporary.unlink(missing_ok=True)
    # engine match may itself be interrupted while staging its requested output.
    for path in temporary.parent.glob(temporary.name + ".*.tmp"):
        if re.fullmatch(re.escape(temporary.name) + r"\.[0-9a-f]{16}\.tmp", path.name):
            path.unlink(missing_ok=True)


def _block(args, block: int, games: int) -> tuple[int, int, int]:
    path = args.out_dir / f"block_{block:02d}.json"
    temporary = path.with_name(path.name + ".tmp")
    _cleanup(temporary)
    if path.exists():
        return _read_block(path, args, block, games)
    command = [args.engine, "match", "--a", args.a, "--b", args.b,
               "--games", str(games), "--sims", str(args.sims),
               "--opening-offset", str(20 * block), "--seed", str(args.seed_base + block),
               "--out", str(temporary), "--batch", str(args.batch),
               "--max-plies", str(args.max_plies), "--ep", args.ep,
               "--threads", str(args.threads)]
    if getattr(args, "infer_batch", None) is not None:
        command.extend(["--infer-batch", str(args.infer_batch)])
    try:
        measured_run(command, unit=f"block_{block:02d}", check=True)
        counts = _read_block(temporary, args, block, games)
        temporary.replace(path)
        return counts
    finally:
        _cleanup(temporary)


def evaluate(args) -> dict:
    """Replay saved blocks, run missing ones, and return the first decision."""
    _validate(args)
    _check_paths(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    decision_tmp = args.decision.with_name(args.decision.name + ".tmp")
    decision_tmp.unlink(missing_ok=True)
    w = d = l = 0
    for block in range((args.max_games + args.block_games - 1) // args.block_games):
        counts = _block(args, block, min(args.block_games, args.max_games - w - d - l))
        w, d, l = (total + count for total, count in zip((w, d, l), counts))
        games = w + d + l
        reason, promoted = elo.decide(w, d, l, games, args.max_games,
                                      args.elo0, args.elo1, args.alpha, args.beta)
        if reason is None:
            continue
        decision = {"promoted": promoted, "reason": reason, "games": games,
                    "score": elo.score(w, d, l), "elo": elo.elo(w, d, l),
                    "elo_ci95": list(elo.elo_ci95(w, d, l)),
                    "llr": elo.llr(w, d, l, args.elo0, args.elo1)}
        canonical = json.dumps(decision, allow_nan=False, sort_keys=True) + "\n"
        try:
            unchanged = args.decision.read_text(encoding="utf-8") == canonical
        except (FileNotFoundError, ValueError):
            unchanged = False
        if not unchanged:
            try:
                write_json_atomic(args.decision, decision)
            finally:
                decision_tmp.unlink(missing_ok=True)
        return decision
    raise RuntimeError("evaluation reached the game cap without a decision")


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        decision = evaluate(args)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"evaluation failed: {error}", file=sys.stderr)
        return 1
    print(f"{decision['reason']}: promoted={decision['promoted']}, games={decision['games']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
