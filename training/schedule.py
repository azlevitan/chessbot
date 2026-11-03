"""Learning-rate multiplier: linear warmup, then cosine decay to 10%."""

from __future__ import annotations

import math


def lr_multiplier(step: int, steps: int, warmup_steps: int) -> float:
    """Multiplier for zero-based optimizer update `step`.

    Short runs cap warmup to leave a final update at the 10% floor.
    A single-update run uses that floor; zero warmup starts at full LR.
    """
    if steps < 1 or not 0 <= step < steps or warmup_steps < 0:
        raise ValueError("require steps > 0, 0 <= step < steps, warmup_steps >= 0")
    warmup = min(warmup_steps, steps - 1)
    if step < warmup:
        return (step + 1) / warmup
    decay_steps = steps - warmup
    progress = (step - warmup) / (decay_steps - 1) if decay_steps > 1 else 1.0
    return 0.1 + 0.45 * (1.0 + math.cos(math.pi * progress))
