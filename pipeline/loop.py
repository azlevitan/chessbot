"""Checkpointed, sequential RL orchestration (SPEC §10, Phase 9.3).

--generations includes the resumed current generation; --until-gen is inclusive.
When both are set the earlier bound wins. With neither, run until interrupted.
The prune boundary is a no-op until Phase 9.4; reporting belongs to Phase 9.5.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import subprocess
import sys

from config import normalize_config
from loop_lock import LoopLockError, run_lock
from loop_worker import run_stage
from run_layout import read_run_json, write_run_json

STAGES = ("selfplay", "train", "export", "match", "decide", "anchors", "prune", "done")


def _positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--generations", type=_positive)
    parser.add_argument("--until-gen", type=_positive)
    args = parser.parse_args(argv)
    args.run, args.engine = args.run.resolve(), args.engine.resolve()
    return args


def validate_state(state: dict) -> None:
    """Reject invalid checkpoints before launching work or changing artifacts."""
    gen, best = state.get("current_gen"), state.get("best_gen")
    if type(gen) is not int or gen < 1:
        raise ValueError("current_gen must be a positive integer")
    if type(best) is not int or not 0 <= best <= gen:
        raise ValueError("best_gen must be an integer between 0 and current_gen")
    if state.get("stage") not in STAGES:
        raise ValueError("run.json has an unsupported stage")
    threshold = state.get("resign_threshold")
    if (type(threshold) not in (int, float) or not -1 <= threshold <= 0
            or not math.isfinite(threshold)):
        raise ValueError("resign_threshold must be finite and in [-1, 0]")
    normalize_config(state.get("config"))


def run_loop(args) -> None:
    with run_lock(args.run):
        state = read_run_json(args.run)
        validate_state(state)
        pending = args.run / "run.json.tmp"
        pending.unlink(missing_ok=True)
        bounds = []
        if args.generations is not None:
            bounds.append(state["current_gen"] + args.generations - 1)
        if args.until_gen is not None:
            bounds.append(args.until_gen)
        last = min(bounds) if bounds else None
        try:
            while last is None or state["current_gen"] <= last:
                gen, stage = state["current_gen"], state["stage"]
                if stage in ("prune", "done"):
                    proposed = dict(state)
                else:
                    proposed = run_stage(args.run, args.engine, state)
                    validate_state(proposed)
                # Simulate work published successfully but its checkpoint lost.
                if os.environ.get("CHESSBOT_CRASH_AT") == f"{gen}:{stage}":
                    os._exit(17)
                if stage == "done":
                    proposed.update(current_gen=gen + 1, stage="selfplay")
                else:
                    proposed["stage"] = STAGES[STAGES.index(stage) + 1]
                write_run_json(args.run, proposed)
                state = proposed
        finally:
            pending.unlink(missing_ok=True)


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        run_loop(args)
        return 0
    except KeyboardInterrupt:
        return 130
    except (OSError, ValueError, LoopLockError, subprocess.SubprocessError) as error:
        print(f"loop: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
