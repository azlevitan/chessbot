"""Lazy multi-chunk access to the Rust sample format (SPEC §§3, 6)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .replay_buffer import (
    NUM_MOVES, NUM_PLANES, SAMPLE_BYTES, SAMPLE_DTYPE,
    collate_samples, open_samples, sample_to_tensors,
)


def read_sidecar(bin_path: str | Path) -> dict:
    """Read and validate a complete chunk; reject missing or invalid metadata."""
    path = Path(bin_path)
    if path.suffix != ".bin":
        raise ValueError(f"{path}: expected a .bin chunk")
    sidecar = path.with_suffix(".json")
    try:
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise ValueError(f"{sidecar}: invalid JSON") from exc
    if not isinstance(metadata, dict) or metadata.get("format") != "chessbot-chunk-v1":
        raise ValueError(f"{sidecar}: expected chessbot-chunk-v1 metadata")
    samples = metadata.get("samples")
    if type(samples) is not int or samples < 0:
        raise ValueError(f"{sidecar}: samples must be a nonnegative integer")
    if path.stat().st_size != samples * SAMPLE_BYTES:
        raise ValueError(f"{path}: binary size does not match sidecar samples")
    return metadata


def list_complete_chunks(directory: str | Path) -> list[Path]:
    """List complete .bin chunks in filename order, ignoring partial writes."""
    complete = []
    for path in sorted(Path(directory).glob("*.bin")):
        try:
            read_sidecar(path)
        except (OSError, ValueError):
            continue
        complete.append(path)
    return complete


class ChunkDataset(Dataset):
    """Concatenate read-only memmaps in caller-supplied order, without preloading."""

    def __init__(self, paths: list[Path]):
        self.paths = [Path(path) for path in paths]
        lengths = [read_sidecar(path)["samples"] for path in self.paths]
        # Empty chunks are complete but cannot be memory-mapped by NumPy.
        self._samples = [open_samples(path) if n else None for path, n in zip(self.paths, lengths)]
        self._offsets = np.concatenate(([0], np.cumsum(lengths, dtype=np.int64)))

    def __len__(self) -> int:
        return int(self._offsets[-1])

    def _locate(self, indices: list[int]) -> tuple[np.ndarray, np.ndarray]:
        indices = np.asarray(indices)
        if indices.ndim != 1 or (indices.size and indices.dtype.kind not in "iu"):
            raise TypeError("sample indices must be a one-dimensional sequence of integers")
        if np.any(indices >= len(self)) or (
            indices.dtype.kind == "i" and np.any(indices < -len(self))
        ):
            raise IndexError("sample index out of range")
        indices = indices.astype(np.int64)
        indices[indices < 0] += len(self)
        chunks = np.searchsorted(self._offsets[1:], indices, side="right")
        return chunks, indices - self._offsets[chunks]

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        chunks, local = self._locate([i])
        return sample_to_tensors(self._samples[chunks[0]][local[0]])

    def __getitems__(self, indices: list[int]) -> dict[str, torch.Tensor]:
        """Gather only requested rows, then decode and collate in request order."""
        chunks, local = self._locate(indices)
        if not len(chunks):
            q_value = torch.empty(0, dtype=torch.float32)
            return {
                "planes": torch.empty((0, NUM_PLANES, 8, 8), dtype=torch.float32),
                "policy": torch.empty((0, NUM_MOVES), dtype=torch.float32),
                "legal_mask": torch.empty((0, NUM_MOVES), dtype=torch.bool),
                "wdl": torch.empty(0, dtype=torch.int64),
                "q": q_value,
                "q_value": q_value,
            }
        records = np.empty(len(chunks), dtype=SAMPLE_DTYPE)
        for chunk in np.unique(chunks):
            positions = np.flatnonzero(chunks == chunk)
            records[positions] = self._samples[chunk][local[positions]]
        return collate_samples([sample_to_tensors(rec) for rec in records])
