"""Deterministic, resumable disk pruning (SPEC §10); sizes use decimal GB.

Only metadata and regular-file sizes are inspected. Symbolic links are never
followed, and best/current generations are protected from every deletion rule.
The transient root loop.lock is excluded so ownership cannot change the plan.
The caller owns the run lock; the standalone CLI acquires it for live pruning.
"""

from __future__ import annotations

import argparse
from decimal import Decimal
import os
from pathlib import Path
import re
import shutil
import stat
import sys

from config import normalize_config
from loop_lock import LoopLockError, run_lock
from run_layout import read_json, read_run_json


class DiskBudgetError(OSError):
    """All eligible cleanup is exhausted but the run remains over budget."""


def _mode(path: Path) -> int:
    try:
        return path.lstat().st_mode
    except FileNotFoundError:
        return 0


def _bytes(path: Path) -> int:
    mode = _mode(path)
    if stat.S_ISREG(mode):
        return path.stat().st_size
    if not stat.S_ISDIR(mode):
        return 0
    total = 0

    def failed(error):
        raise error

    for directory, _, files in os.walk(path, followlinks=False, onerror=failed):
        for name in files:
            entry = Path(directory) / name
            info = entry.lstat()
            if stat.S_ISREG(info.st_mode):
                total += info.st_size
    return total


def _generations(run: Path, current: int, best: int) -> list[tuple[int, Path]]:
    root = run / "gens"
    if not stat.S_ISDIR(_mode(root)):
        return []
    generations = []
    for directory in root.iterdir():
        match = re.fullmatch(r"gen_([0-9]{4,})", directory.name)
        if match is None or not stat.S_ISDIR(_mode(directory)):
            continue
        generation = int(match[1])
        if (directory.name == f"gen_{generation:04d}"
                and generation < current and generation != best):
            generations.append((generation, directory))
    return sorted(generations)


def _usage(run: Path) -> int:
    lock = run / "loop.lock"
    transient = _bytes(lock) if stat.S_ISREG(_mode(lock)) else 0
    return _bytes(run) - transient


def _promotion(directory: Path) -> bool | None:
    path = directory / "decision.json"
    if not stat.S_ISREG(_mode(path)):
        return None
    try:
        value = read_json(path).get("promoted")
    except (OSError, ValueError):
        return None
    return value if type(value) is bool else None


def _plan(run_dir, state, cfg):
    run = Path(run_dir).resolve(strict=True)
    if not run.is_dir():
        raise ValueError(f"{run}: expected a run directory")
    current, best = state.get("current_gen"), state.get("best_gen")
    if type(current) is not int or current < 1:
        raise ValueError("current_gen must be a positive integer")
    if type(best) is not int or not 0 <= best <= current:
        raise ValueError("best_gen must be an integer between 0 and current_gen")
    cfg = normalize_config(cfg)
    budget = int(Decimal(str(cfg["disk_budget_gb"])) * 1_000_000_000)
    generations = _generations(run, current, best)
    targets, checkpoints = [], []
    cutoff = current - cfg["window"]["generations"] + 1
    for generation, directory in generations:
        path = directory / "selfplay"
        if generation < cutoff and stat.S_ISDIR(_mode(path)):
            targets.append(path)
    for _, directory in generations:
        promoted = _promotion(directory)
        if promoted is False:
            targets.extend(path for name in ("model.pt", "model.onnx")
                           if stat.S_ISREG(_mode(path := directory / name)))
        elif promoted is True:
            path = directory / "model.pt"
            if stat.S_ISREG(_mode(path)):
                checkpoints.append(path)
    before = _usage(run)
    remaining = before - sum(_bytes(path) for path in targets)
    for path in checkpoints:
        if remaining <= budget:
            break
        targets.append(path)
        remaining -= _bytes(path)
    return run, targets, before, remaining, budget


def _check_budget(remaining: int, budget: int) -> None:
    if remaining > budget:
        raise DiskBudgetError(f"disk budget unreachable: {remaining} bytes remain; "
                              f"budget {budget} bytes; no further eligible cleanup")


def _execute(run: Path, targets: list[Path], budget: int) -> list[Path]:
    removed = []
    for path in targets:
        # Do not traverse a parent replaced by a symlink after inspection.
        if path.parent.resolve() != path.parent:
            raise ValueError(f"{path}: pruning parent changed to a symbolic link")
        mode = _mode(path)
        if stat.S_ISDIR(mode):
            shutil.rmtree(path)
        elif stat.S_ISREG(mode):
            path.unlink()
        else:
            continue
        removed.append(path)
    _check_budget(_usage(run), budget)
    return removed


def prune(run_dir, state, cfg) -> list[Path]:
    """Remove eligible directories/files in policy order; return removed paths.

    Always clean expired selfplay and rejected past candidates, even under
    budget. Only promoted checkpoints are budget-dependent. Repeating after a
    partial deletion needs no marker; missing outputs are already complete.
    """
    run, targets, _, _, budget = _plan(run_dir, state, cfg)
    return _execute(run, targets, budget)


def _cli_plan(run):
    state = read_run_json(run)
    return _plan(run, state, state["config"])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        run = args.run.resolve(strict=True)
        if args.dry_run:
            _, targets, before, remaining, budget = _cli_plan(run)
            for path in targets:
                print(f"would remove {path}")
            print(f"bytes: {before} -> {remaining}; budget: {budget}")
            _check_budget(remaining, budget)
        else:
            with run_lock(run):
                _, targets, before, _, budget = _cli_plan(run)
                removed = _execute(run, targets, budget)
                for path in removed:
                    print(f"removed {path}")
                print(f"bytes: {before} -> {_usage(run)}; budget: {budget}")
        return 0
    except KeyboardInterrupt:
        return 130
    except (OSError, ValueError, KeyError, LoopLockError) as error:
        print(f"prune: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
