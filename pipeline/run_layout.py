"""SPEC §10 paths and atomic run metadata; no training imports or sample decoding."""

from __future__ import annotations

import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ENGINE_DIR = REPO_ROOT / "engine"
ENGINE_BIN = ENGINE_DIR / "target" / "release" / "engine"
RUN_FORMAT = "chessbot-run-v1"


def generation_dir(run_dir: str | Path, generation: int) -> Path:
    if generation < 0:
        raise ValueError("generation must be nonnegative")
    return Path(run_dir) / "gens" / f"gen_{generation:04d}"


def read_json(path: str | Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def write_json_atomic(path: str | Path, value: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, allow_nan=False, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def read_run_json(run_dir: str | Path) -> dict:
    state = read_json(Path(run_dir) / "run.json")
    if state.get("format") != RUN_FORMAT:
        raise ValueError(f"{run_dir}: expected {RUN_FORMAT}")
    return state


def write_run_json(run_dir: str | Path, state: dict) -> None:
    write_json_atomic(Path(run_dir) / "run.json", {**state, "format": RUN_FORMAT})


def chunk_manifest(directory: Path) -> list[dict]:
    """Inspect sidecars and sizes only; the Rust sample bytes stay on disk."""
    chunks = []
    for index, binary in enumerate(sorted(directory.glob("chunk_*.bin"))):
        if binary.name != f"chunk_{index:06d}.bin":
            raise ValueError(f"{directory}: chunk indices must be dense")
        metadata = read_json(binary.with_suffix(".json"))
        samples = metadata.get("samples")
        if (metadata.get("format") != "chessbot-chunk-v1"
                or metadata.get("kind") != "pgn"
                or type(samples) is not int or samples < 0
                or binary.stat().st_size != samples * 492):
            raise ValueError(f"{binary}: incomplete PGN chunk")
        chunks.append({"name": binary.name, "samples": samples})
    if not chunks or not sum(chunk["samples"] for chunk in chunks):
        raise ValueError(f"{directory}: conversion produced no samples")
    return chunks
