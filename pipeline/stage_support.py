"""Metadata and publication helpers for the SPEC §10 stages (stdlib only)."""

from __future__ import annotations

import math
from pathlib import Path
import shutil
import subprocess
import sys

from process_metrics import measured_run

from run_layout import REPO_ROOT, generation_dir, read_json, write_json_atomic


def context(run_dir, gen):
    if type(gen) is not int or gen < 1:
        raise ValueError("RL stages require a positive generation")
    run = Path(run_dir).resolve()
    return run, generation_dir(run, gen)


def incumbent(run: Path, gen: int, state: dict) -> Path:
    """Recover the best before G even when decide has already promoted G."""
    best = state["best_gen"]
    if type(best) is not int or best < 0:
        raise ValueError("best_gen must be a nonnegative integer")
    if best < gen:
        return generation_dir(run, best)
    for previous in range(gen - 1, -1, -1):
        decision = object_or_none(generation_dir(run, previous) / "decision.json")
        if decision is not None and decision.get("promoted") is True:
            return generation_dir(run, previous)
    raise ValueError(f"{run}: cannot recover the best before generation {gen}")


def object_or_none(path: Path) -> dict | None:
    try:
        return read_json(path)
    except (OSError, ValueError):
        return None


def temporary(path: Path) -> Path:
    return path.with_name(path.name + ".tmp")


def remove_outputs(*paths: Path) -> None:
    for path in paths:
        path.unlink(missing_ok=True)
        temporary(path).unlink(missing_ok=True)


def invalidate(gen_dir: Path, *, model: bool = False) -> None:
    """Invalidate dependents before replacing the candidate's upstream output."""
    remove_outputs(gen_dir / "decision.json")
    if model:
        remove_outputs(gen_dir / "model.onnx")
    for name in ("match", "anchors"):
        directory = gen_dir / name
        if directory.exists():
            shutil.rmtree(directory)


def training_ready(gen_dir: Path) -> bool:
    checkpoint, summary = gen_dir / "model.pt", gen_dir / "train.json"
    return (checkpoint.is_file() and checkpoint.stat().st_size > 0
            and object_or_none(summary) is not None
            and not any(temporary(path).exists() for path in (checkpoint, summary)))


def flags(values: dict, keys) -> list[str]:
    return [part for key in keys if key in values
            for part in ("--" + key.replace("_", "-"), str(values[key]))]


def training_process(script: str, *arguments) -> None:
    measured_run([sys.executable, str(REPO_ROOT / "training" / script),
                  *map(str, arguments)], unit=script, check=True)


def valid_onnx(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        measured_run(
            [sys.executable, "-c", "import sys; from export_onnx import check_contract; "
             "check_contract(sys.argv[1])", str(path)], cwd=REPO_ROOT / "training",
            unit="check_contract", check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        return False
    return True


def selfplay_info(binary: Path, expected: dict) -> dict | None:
    info = object_or_none(binary.with_suffix(".json"))
    if info is None:
        return None
    if any(type(info.get(key)) is not type(value) or info[key] != value
           for key, value in expected.items()):
        return None
    samples, audit = info.get("samples"), info.get("resign")
    if type(samples) is not int or samples < 0 or not isinstance(audit, dict):
        return None
    try:
        if not binary.is_file() or binary.stat().st_size != samples * 492:
            return None
    except OSError:
        return None
    threshold = audit.get("threshold")
    if type(threshold) not in (int, float) or not math.isfinite(threshold):
        return None
    played, false = audit.get("playthrough_games"), audit.get("false_positives")
    if (type(played) is not int or type(false) is not int
            or not 0 <= false <= played <= info["games"]):
        return None
    return info


def input_threshold(infos: list[dict], state: dict) -> float:
    thresholds = [info["resign"]["threshold"] for info in infos]
    # Rust writes a binary32 threshold; recover the intended decimal schedule.
    threshold = float(format(thresholds[0], ".7g")) if thresholds else state["resign_threshold"]
    if (type(threshold) not in (int, float) or not math.isfinite(threshold)
            or not -1 <= threshold <= 0):
        raise ValueError("resign_threshold must be finite and in [-1, 0]")
    if any(not math.isclose(value, threshold, rel_tol=0, abs_tol=1e-7)
           for value in thresholds):
        raise ValueError("selfplay chunks disagree on the generation's input threshold")
    return threshold


def generation_threshold(directory: Path, model: Path, infos: list[dict], state: dict) -> float:
    """Freeze inputs before work, including recovery with every sidecar missing."""
    path = directory / ".inputs.json"
    temporary(path).unlink(missing_ok=True)
    saved = object_or_none(path)
    if saved is None:
        if path.exists():
            raise ValueError(f"{path}: invalid selfplay input metadata")
        threshold = input_threshold(infos, state)
        write_json_atomic(path, {"model": str(model), "resign_threshold": threshold})
        return threshold
    if saved.get("model") != str(model) or "resign_threshold" not in saved:
        raise ValueError(f"{path}: incompatible selfplay inputs")
    threshold = input_threshold([], saved)
    if not math.isclose(input_threshold(infos, saved), threshold, rel_tol=0, abs_tol=1e-7):
        raise ValueError(f"{path}: chunks disagree with the recorded input threshold")
    return threshold


def update_threshold(infos: list[dict], threshold: float, state: dict) -> None:
    played = sum(info["resign"]["playthrough_games"] for info in infos)
    false = sum(info["resign"]["false_positives"] for info in infos)
    # Integer comparison keeps the strict 5% boundary exact.
    state["resign_threshold"] = (round(max(threshold - 0.02, -0.99), 7)
                                 if played and false * 20 > played else threshold)


def decision_ready(path: Path) -> dict | None:
    decision = object_or_none(path)
    if decision is None or type(decision.get("promoted")) is not bool:
        return None
    if decision.get("reason") not in ("sprt_accept", "sprt_reject", "max_games"):
        return None
    games, score = decision.get("games"), decision.get("score")
    if type(games) is not int or games <= 0 or type(score) not in (int, float) or not 0 <= score <= 1:
        return None
    interval = decision.get("elo_ci95")
    values = [decision.get("elo"), decision.get("llr")]
    if not isinstance(interval, list) or len(interval) != 2:
        return None
    if any(type(value) not in (int, float) or not math.isfinite(value)
           for value in [*values, *interval]):
        return None
    return decision
