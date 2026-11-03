"""Zero-copy ``np.memmap`` reader for Rust-written ``TrainingSample`` records.

Law 1 (CLAUDE.md): this module only *decodes* the bytes defined in SPEC.md §3.
It never encodes boards or moves; all of that lives in the Rust engine.
Law 2: records are sliced lazily from the memmap; loaders use
``num_workers=0`` and ``pin_memory=False``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

SAMPLE_BYTES = 492
NUM_MOVES = 4168
POLICY_SLOTS = 64
UNUSED_MOVE = 0xFFFF
NUM_PLANES = 18

POLICY_ENTRY_DTYPE = np.dtype([("move_idx", "<u2"), ("prob", "<f4")])

# Mirrors `#[repr(C, packed)] struct TrainingSample` (no padding).
SAMPLE_DTYPE = np.dtype(
    [
        ("bitboards", "<u8", (12,)),
        ("castling_and_ep", "u1"),
        ("ep_square", "u1"),
        ("halfmove_clock", "u1"),
        ("outcome_wdl", "u1"),
        ("q_value", "<f4"),
        ("policy", POLICY_ENTRY_DTYPE, (POLICY_SLOTS,)),
        ("num_legal_moves", "<u4"),
    ]
)

FIELD_OFFSETS = {
    "bitboards": 0,
    "castling_and_ep": 96,
    "ep_square": 97,
    "halfmove_clock": 98,
    "outcome_wdl": 99,
    "q_value": 100,
    "policy": 104,
    "num_legal_moves": 488,
}

assert POLICY_ENTRY_DTYPE.itemsize == 6
assert SAMPLE_DTYPE.itemsize == SAMPLE_BYTES
assert {k: SAMPLE_DTYPE.fields[k][1] for k in FIELD_OFFSETS} == FIELD_OFFSETS

_SHIFTS = np.arange(64, dtype=np.uint64)


def open_samples(path: str | Path) -> np.memmap:
    """Memory-map a ``.bin`` file as an array of ``SAMPLE_DTYPE`` records (read-only)."""
    path = Path(path)
    size = path.stat().st_size
    if size == 0 or size % SAMPLE_BYTES != 0:
        raise ValueError(
            f"{path}: size {size} is not a positive multiple of {SAMPLE_BYTES} bytes"
        )
    return np.memmap(path, dtype=SAMPLE_DTYPE, mode="r")


def unpack_planes(rec: np.void) -> np.ndarray:
    """Expand one stored record into ``float32[18, 8, 8]`` planes.

    Index layout is ``[plane, square // 8, square % 8]``, identical to the Rust
    flat ``plane * 64 + square`` tensor.
    """
    planes = np.zeros((NUM_PLANES, 64), dtype=np.float32)
    bitboards = np.asarray(rec["bitboards"], dtype=np.uint64)
    planes[:12] = (bitboards[:, None] >> _SHIFTS) & np.uint64(1)
    flags = int(rec["castling_and_ep"])
    for bit in range(4):
        if flags & (1 << bit):
            planes[12 + bit] = 1.0
    if flags & 0x10:
        planes[16, int(rec["ep_square"])] = 1.0
    planes[17] = int(rec["halfmove_clock"]) / 100.0
    return planes.reshape(NUM_PLANES, 8, 8)


def unpack_policy(rec: np.void) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(dense float32[4168] target, bool[4168] legal mask)``."""
    n = int(rec["num_legal_moves"])
    if n > POLICY_SLOTS:
        raise ValueError(f"num_legal_moves {n} exceeds {POLICY_SLOTS} slots")
    entries = rec["policy"][:n]
    idx = entries["move_idx"].astype(np.int64)
    if np.any(idx >= NUM_MOVES):
        raise ValueError("active policy entry has an out-of-range move index")
    dense = np.zeros(NUM_MOVES, dtype=np.float32)
    mask = np.zeros(NUM_MOVES, dtype=bool)
    dense[idx] = entries["prob"]
    mask[idx] = True
    return dense, mask


class ReplayBuffer(Dataset):
    """Torch ``Dataset`` over one ``.bin`` file; each item is sliced lazily."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._samples = open_samples(self.path)

    def __len__(self) -> int:
        return len(self._samples)

    def record(self, idx: int) -> np.void:
        """The raw structured record at ``idx`` (a view into the memmap)."""
        return self._samples[idx]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        rec = self._samples[idx]
        policy, legal_mask = unpack_policy(rec)
        return {
            "planes": torch.from_numpy(unpack_planes(rec)),
            "policy": torch.from_numpy(policy),
            "legal_mask": torch.from_numpy(legal_mask),
            "wdl": torch.tensor(int(rec["outcome_wdl"]), dtype=torch.int64),
            "q": torch.tensor(float(rec["q_value"]), dtype=torch.float32),
        }


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool = True) -> DataLoader:
    """DataLoader with the 8GB-M1 guardrails baked in (CLAUDE.md Law 2)."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False,
    )
