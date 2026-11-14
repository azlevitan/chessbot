"""Full numerical evidence for export probes; acceptance stays with SPEC §4."""

from __future__ import annotations

import math
from numbers import Real

import numpy as np

from onnx_parity import PARITY_ATOL, PARITY_RTOL, PROBABILITY_ATOL, _heads, _softmax, compare_outputs

HEADS = (("policy_logits", 4168), ("wdl_logits", 3))


def _number(value):
    value = float(value)
    return value if math.isfinite(value) else None


def _pair(expected, actual, width):
    arrays = []
    for label, value in (("reference", expected), ("observed", actual)):
        if not isinstance(value, np.ndarray) or not np.issubdtype(value.dtype, np.floating):
            raise ValueError(f"{label} must be a real floating NumPy array")
        if value.ndim != 2 or value.shape[0] == 0 or value.shape[1] != width:
            raise ValueError(f"{label} shape must be (B, {width}) with B > 0")
        with np.errstate(over="ignore", invalid="ignore"):
            value = value.astype(np.float64, copy=False)
        if not np.isfinite(value).all():
            raise ValueError(f"{label} contains nonfinite values")
        arrays.append(value)
    if arrays[0].shape != arrays[1].shape:
        raise ValueError("reference and observed shapes differ")
    return arrays


def _gap(row):
    top = np.partition(row, -2)[-2:]
    return _number(top.max() - top.min())


def _flips(expected, observed, ref_prob, got_prob, mask):
    reference = expected if mask is None else np.where(mask, expected, -np.inf)
    actual = observed if mask is None else np.where(mask, observed, -np.inf)
    ref_indices, got_indices = reference.argmax(axis=1), actual.argmax(axis=1)
    details = []
    with np.errstate(over="ignore", invalid="ignore"):
        for row in np.flatnonzero(ref_indices != got_indices):
            details.append({
                "row": int(row), "reference_index": int(ref_indices[row]),
                "observed_index": int(got_indices[row]),
                "reference_top2_logit_margin": _gap(reference[row]),
                "observed_top2_logit_margin": _gap(actual[row]),
                "reference_top2_probability_margin": _gap(ref_prob[row]),
                "observed_top2_probability_margin": _gap(got_prob[row]),
            })
    return {"argmax_flips": len(details), "flip_details": details}


def _metrics(expected, observed, mask, rtol, *, policy):
    with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
        difference = np.abs(expected - observed)
        bound = PARITY_ATOL + rtol * np.abs(expected)
        ratio = difference / bound
        location = np.unravel_index(np.argmax(ratio), ratio.shape)
        ref_prob, got_prob = _softmax(expected, mask), _softmax(observed, mask)
        prob_difference = np.abs(ref_prob - got_prob)
    metrics = {
        "raw_max_abs_error": _number(difference.max()),
        "raw_violation_count": int(np.count_nonzero(difference > bound)),
        "raw_worst_scaled_error": _number(ratio[location]),
        "raw_arithmetic_overflow": not bool(np.isfinite(difference).all()),
        "raw_worst": {"row": int(location[0]), "column": int(location[1]),
                      "reference": _number(expected[location]), "observed": _number(observed[location]),
                      "abs_error": _number(difference[location]), "bound": _number(bound[location]),
                      "scaled_error": _number(ratio[location])},
        "probability_max_abs_error": _number(prob_difference.max()),
        "probability_violation_count": int(np.count_nonzero(prob_difference > PROBABILITY_ATOL)),
        "probability_scope": "legal" if mask is not None else "unmasked",
    }
    if mask is not None or not policy:
        metrics.update(_flips(expected, observed, ref_prob, got_prob, mask))
        metrics["argmax_scope"] = "legal_policy" if policy else "wdl"
    if not policy:
        reference_value, observed_value = ref_prob[:, 2] - ref_prob[:, 0], got_prob[:, 2] - got_prob[:, 0]
        metrics["value_max_abs_error"] = _number(np.abs(reference_value - observed_value).max())
    return metrics


def compare_case(reference, observed, legal_mask=None, *, rtol=PARITY_RTOL):
    """Report every valid head even when the production comparator rejects early.

    Nonfinite/invalid output contracts are explicit failures. Infinite arithmetic
    diagnostics are represented by null plus an overflow marker, never JSON NaN.
    Argmax changes are diagnostic and do not add a new acceptance requirement.
    """
    valid_rtol = (not isinstance(rtol, (bool, np.bool_)) and isinstance(rtol, Real)
                  and np.isfinite(rtol) and rtol >= 0)
    result = {"gate_pass": False, "failure_kind": None, "error": None,
              "rtol": float(rtol) if valid_rtol else None, "atol": PARITY_ATOL,
              "probability_atol": PROBABILITY_ATOL, "heads": {}}
    contract_error = not valid_rtol
    for index, (name, width) in enumerate(HEADS):
        try:
            if not valid_rtol:
                raise ValueError("rtol must be finite and nonnegative")
            if (not isinstance(reference, (list, tuple)) or not isinstance(observed, (list, tuple))
                    or len(reference) != 2 or len(observed) != 2):
                raise ValueError("reference and observed must each contain two heads")
            expected, actual = _pair(reference[index], observed[index], width)
            mask = legal_mask if index == 0 else None
            if mask is not None and (not isinstance(mask, np.ndarray) or mask.dtype != np.bool_
                                     or mask.shape != expected.shape or not mask.any(axis=1).all()):
                raise ValueError("legal_mask must match policy shape and contain a legal action per row")
            result["heads"][name] = _metrics(expected, actual, mask, float(rtol), policy=index == 0)
        except (TypeError, ValueError, OverflowError) as error:
            contract_error = True
            result["heads"][name] = {"error": str(error), "failure_kind": "output_contract"}
    # Validate the combined output contract after retaining each valid head's
    # evidence; individually valid heads may still disagree on batch size.
    for label, outputs in (("reference", reference), ("observed", observed)):
        try:
            _heads(outputs, label)
        except (TypeError, ValueError, OverflowError):
            contract_error = True
    try:
        compare_outputs(reference, observed, legal_mask, rtol=rtol)
        result["gate_pass"] = True
    except (TypeError, ValueError, AssertionError, OverflowError) as error:
        result.update(error=str(error), failure_kind="output_contract" if contract_error else "numerical")
    return result
