"""Elementwise logit and probability checks for PyTorch/ONNX outputs."""

from __future__ import annotations

from numbers import Real

import numpy as np

PARITY_ATOL = 1e-4
PARITY_RTOL = 1e-6
PROBABILITY_ATOL = 1e-5
_HEADS = (("policy_logits", 4168), ("wdl_logits", 3))


def _heads(outputs, label: str) -> list[np.ndarray]:
    if not isinstance(outputs, (list, tuple)) or len(outputs) != len(_HEADS):
        raise ValueError(f"{label} must contain exactly two output heads")
    arrays = []
    for (name, width), value in zip(_HEADS, outputs):
        if not isinstance(value, np.ndarray) or not np.issubdtype(value.dtype, np.floating):
            raise ValueError(f"{label} {name} must be a real floating NumPy array")
        if value.ndim != 2 or value.shape[0] == 0 or value.shape[1] != width:
            raise ValueError(f"{label} {name} shape must be (B, {width}) with B > 0; got {value.shape}")
        with np.errstate(over="ignore", invalid="ignore"):
            value = value.astype(np.float64, copy=False)
        if not np.isfinite(value).all():
            raise ValueError(f"{label} {name} contains nonfinite values")
        arrays.append(value)
    if arrays[0].shape[0] != arrays[1].shape[0]:
        raise ValueError(f"{label} policy_logits and wdl_logits batch sizes differ")
    return arrays


def _softmax(logits: np.ndarray, legal_mask: np.ndarray | None) -> np.ndarray:
    values = logits if legal_mask is None else np.where(legal_mask, logits, -np.inf)
    with np.errstate(over="ignore", under="ignore"):
        probabilities = np.exp(values - values.max(axis=1, keepdims=True))
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def compare_outputs(ref, actual, legal_mask=None, *, rtol=PARITY_RTOL) -> float:
    """Check both heads, returning their largest absolute raw-logit difference.

    Each logit uses ``ATOL + rtol * abs(reference_logit)``. Probabilities have
    an independent absolute bound; policy masking never exempts raw logits.
    """
    if (isinstance(rtol, (bool, np.bool_)) or not isinstance(rtol, Real)
            or not np.isfinite(rtol) or rtol < 0):
        raise ValueError("rtol must be a finite nonnegative real number")
    reference, observed = _heads(ref, "reference"), _heads(actual, "actual")
    for (name, _), expected, result in zip(_HEADS, reference, observed):
        if expected.shape != result.shape:
            raise ValueError(f"{name} shape mismatch: reference {expected.shape}, actual {result.shape}")
    if legal_mask is not None:
        if (not isinstance(legal_mask, np.ndarray) or legal_mask.dtype != np.bool_
                or legal_mask.shape != reference[0].shape):
            raise ValueError("legal_mask must be a boolean NumPy array matching policy_logits shape")
        if not legal_mask.any(axis=1).all():
            raise ValueError("legal_mask must contain at least one legal action per row")

    worst = 0.0
    for index, ((name, _), expected, result) in enumerate(zip(_HEADS, reference, observed)):
        with np.errstate(over="ignore", invalid="ignore"):
            difference = np.abs(expected - result)
            bound = PARITY_ATOL + float(rtol) * np.abs(expected)
            ratio = difference / bound
        if not np.isfinite(difference).all():
            raise ValueError(f"{name} raw parity failed: nonfinite absolute difference")
        maximum = float(difference.max())
        worst = max(worst, maximum)
        if np.any(difference > bound):
            location = np.unravel_index(np.argmax(ratio), ratio.shape)
            raise ValueError(
                f"{name} raw parity failed: max_abs_error={maximum:.9g}; at {location}, "
                f"abs_error={difference[location]:.9g}, bound={bound[location]:.9g}, "
                f"scaled_error={ratio[location]:.9g} > 1"
            )
        mask = legal_mask if index == 0 else None
        probability_error = np.abs(_softmax(expected, mask) - _softmax(result, mask))
        if np.any(probability_error > PROBABILITY_ATOL):
            maximum = float(probability_error.max())
            raise ValueError(
                f"{name} probability parity failed: max_abs_error={maximum:.9g} "
                f"> {PROBABILITY_ATOL:g}"
            )
    return worst
