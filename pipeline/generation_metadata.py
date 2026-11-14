"""Generation provenance and explicit coverage of pre-telemetry work (stdlib)."""

from __future__ import annotations

from pathlib import Path
import subprocess

from config import normalize_config
from run_layout import REPO_ROOT, read_json, write_json_atomic
from telemetry import read_attempts


def ensure_manifest(directory: Path, engine: Path, state: dict) -> None:
    # Called under the root lock after prior workers have exited. Keep a lost
    # final publication's started record, but remove its abandoned temp file.
    for pending in (directory / "telemetry").glob("*.json.tmp"):
        pending.unlink()
    path = directory / "generation.json"
    if path.exists():
        if read_json(path).get("format") != "chessbot-generation-v1":
            raise ValueError(f"{path}: invalid generation provenance")
        return
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                            capture_output=True, text=True, check=False)
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT,
                           capture_output=True, text=True, check=False)
    write_json_atomic(path, {
        "format": "chessbot-generation-v1", "gen": state["current_gen"],
        "best_gen_at_capture": state["best_gen"],
        "config": normalize_config(state["config"]), "engine": str(engine),
        "source_commit": result.stdout.strip() if result.returncode == 0 else None,
        "source_dirty": bool(dirty.stdout.strip()) if dirty.returncode == 0 else None,
    })


def coverage_complete(directory: Path, stage: str) -> bool:
    previous = [record for record in read_attempts(directory)
                if record.get("kind") == "stage" and record.get("stage") == stage]
    if previous:
        return all(record.get("coverage_complete", True) for record in previous)
    # Existing work has an unknown historical cost. Checking cached outputs is
    # useful, but cannot retroactively measure the work that produced them.
    patterns = {
        "selfplay": ("selfplay/*", "selfplay/.inputs.json"),
        "train": ("model.pt*", "train.json*", "train_history.jsonl"),
        "export": ("model.onnx*",),
        "match": ("match/block_*.json*", "decision.json"),
        "anchors": ("anchors/*.json*",),
    }
    return not any(any(directory.glob(pattern)) for pattern in patterns.get(stage, ()))
