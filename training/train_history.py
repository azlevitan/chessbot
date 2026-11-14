"""Durable attempt-separated training curves and fixed, bounded probe selection."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import uuid

import torch
from torch.utils.data import Subset

from data.replay_buffer import make_loader
from data.sampler import WeightedSourceSampler

FORMAT = "chessbot-train-history-v1"


def fixed_loader(dataset, indices, batch_size: int, seed: int):
    """Use only bounded row indices, and keep probe iteration off the training RNG."""
    loader = make_loader(Subset(dataset, indices), batch_size, shuffle=False)
    loader.generator = torch.Generator().manual_seed(seed)
    return loader


def train_probe(dataset, ranges, sources, count: int, batch_size: int, seed: int):
    """Freeze source-weighted draws; repeated rows are intentional sampling draws."""
    indices = list(WeightedSourceSampler(ranges, [s["weight"] for s in sources],
                                         min(count, len(dataset)), seed))
    return fixed_loader(dataset, indices, batch_size, seed), len(indices)


def validation_indices(length: int, count: int) -> list[int]:
    """Spread a stable subset across the entire held-out source without loading it."""
    count = min(length, count)
    return [i * (length - 1) // max(count - 1, 1) for i in range(count)]


def _repair_tail(path: Path) -> None:
    """An abruptly killed append can leave one partial line; keep complete rows."""
    if not path.exists():
        return
    with path.open("r+b") as stream:
        stream.seek(0, os.SEEK_END)
        end = stream.tell()
        while end:
            start = max(0, end - 4096)
            stream.seek(start)
            block = stream.read(end - start)
            newline = block.rfind(b"\n")
            if newline >= 0:
                stream.truncate(start + newline + 1)
                return
            end = start
        stream.truncate(0)


class TrainingHistory:
    """Append/fsync each event so interrupted attempts remain independently usable."""

    def __init__(self, path: Path, metadata: dict):
        self.path = path
        self.attempt_id = uuid.uuid4().hex
        self.path.parent.mkdir(parents=True, exist_ok=True)
        _repair_tail(self.path)
        self.write("start", metadata=metadata)

    def write(self, event: str, **record) -> None:
        row = {"format": FORMAT, "attempt_id": self.attempt_id, "event": event,
               "timestamp": datetime.now(timezone.utc).isoformat(), **record}
        text = json.dumps(row, allow_nan=False, sort_keys=True) + "\n"
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())

    def record(self, record: dict) -> None:
        self.write("metrics", **record)

    def failed(self, error: BaseException) -> None:
        self.write("interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                   error_type=type(error).__name__, error=str(error))
