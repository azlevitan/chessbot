"""Device selection and memory hygiene for the 8GB M1 (CLAUDE.md Law 2)."""

from __future__ import annotations

import torch


def default_device() -> torch.device:
    """Apple-GPU `mps` when available, else `cpu`."""
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def empty_cache(device: torch.device) -> None:
    """Release cached MPS memory. Call at the end of every training epoch."""
    if device.type == "mps":
        torch.mps.empty_cache()
