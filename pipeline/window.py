"""Metadata-only replay selection and trainer source flags (SPEC §§6, 10).

All selected self-play chunks form one sampling source, so every selected row
has equal probability within self-play regardless of its generation or chunk.
The supervised source is separate and never includes held-out validation data.
"""

from __future__ import annotations

from collections.abc import Iterator
import math
from pathlib import Path
import re

from run_layout import generation_dir, read_json


def _chunks(directory: Path, *, newest_first: bool = False) -> Iterator[tuple[Path, int]]:
    """Inspect sidecars and sizes only; ignore partial writes and empty chunks."""
    for binary in sorted(directory.glob("chunk_*.bin"), reverse=newest_first):
        if not re.fullmatch(r"chunk_[0-9]{6}\.bin", binary.name):
            continue
        try:
            metadata = read_json(binary.with_suffix(".json"))
            samples = metadata.get("samples")
            if (metadata.get("format") != "chessbot-chunk-v1"
                    or type(samples) is not int or samples <= 0
                    or not binary.is_file() or binary.stat().st_size != samples * 492):
                continue
        except (OSError, ValueError):
            continue
        yield binary, samples


def select_window(
    run_dir: str | Path, gen: int, cfg: dict,
) -> list[tuple[list[Path], float]]:
    """Return self-play then supervised sources for a validated RL config.

    Visit generations and chunk indices in descending order. Take whole chunks
    until the self-play sample total first reaches window.max_samples: the last
    chunk may cross that threshold. The supervised source is outside this cap.
    Empty and zero-weight sources are omitted, as in WeightedSourceSampler;
    remaining weights are relative if one source has no usable data.
    """
    if type(gen) is not int or gen < 1:
        raise ValueError("gen must be a positive integer (RL starts at generation 1)")
    run_dir = Path(run_dir)
    window = cfg["window"]
    schedule = window["supervised_weight"]
    supervised_weight = float(schedule[min(gen - 1, len(schedule) - 1)])
    sources = []
    if supervised_weight < 1:
        paths, samples = [], 0
        oldest = max(1, gen - window["generations"] + 1)
        for generation in range(gen, oldest - 1, -1):
            directory = generation_dir(run_dir, generation) / "selfplay"
            for binary, count in _chunks(directory, newest_first=True):
                paths.append(binary)
                samples += count
                if samples >= window["max_samples"]:
                    break
            if samples >= window["max_samples"]:
                break
        if paths:
            sources.append((paths, 1.0 - supervised_weight))
    if supervised_weight > 0:
        paths = [binary for binary, _ in _chunks(run_dir / "supervised")]
        if paths:
            sources.append((paths, supervised_weight))
    if not sources:
        raise ValueError(f"{run_dir}: replay window has no nonempty, positive-weight source")
    return sources


def train_args(sources: list[tuple[list[Path], float]]) -> list[str]:
    """Serialize selected sources for training/train.py, without shell quoting.

    Pass this list directly to subprocess. Spaces in paths are preserved, while
    commas cannot be represented by the trainer's comma-separated source format.
    """
    arguments = []
    for paths, weight in sources:
        try:
            finite = type(weight) in (int, float) and math.isfinite(weight)
        except OverflowError:
            finite = False
        if not finite or weight < 0:
            raise ValueError("source weights must be finite and nonnegative")
        if not paths or weight == 0:
            continue
        names = [str(path) for path in paths]
        if any(not name or "," in name for name in names):
            raise ValueError("source paths must be nonempty and must not contain commas")
        arguments.extend(("--source", ",".join(names) + f":{float(weight)}"))
    if not arguments:
        raise ValueError("training requires a nonempty, positive-weight source")
    return arguments
