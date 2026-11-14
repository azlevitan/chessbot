"""Diagnostic evidence remains complete without weakening the export gate."""

import json

import numpy as np
import pytest

from parity_diagnostics import compare_case


def outputs(batch=1):
    return [np.zeros((batch, 4168)), np.zeros((batch, 3))]


def json_safe(value):
    return json.loads(json.dumps(value, allow_nan=False))


def test_policy_raw_failure_still_reports_policy_probabilities_and_wdl():
    reference, actual = outputs(), outputs()
    reference[0][0, 0] = actual[0][0, 0] = 96
    actual[0][0, 0] += 0.0004
    actual[1][0, 2] = 0.002
    result = compare_case(reference, actual)
    assert not result["gate_pass"] and result["failure_kind"] == "numerical"
    policy, wdl = result["heads"]["policy_logits"], result["heads"]["wdl_logits"]
    assert policy["raw_violation_count"] == 1
    assert policy["raw_worst"]["reference"] == 96
    assert policy["raw_worst"]["observed"] == actual[0][0, 0]
    assert policy["probability_violation_count"] == 0
    assert wdl["raw_violation_count"] == 1
    assert wdl["probability_violation_count"] > 0
    assert wdl["value_max_abs_error"] > 0
    assert "argmax_flips" not in policy
    json_safe(result)


def test_near_tie_legal_argmax_flip_is_evidence_not_an_extra_failure():
    reference, actual = outputs(), outputs()
    mask = np.zeros((1, 4168), dtype=bool)
    mask[0, :2] = True
    reference[0][0, 0], actual[0][0, 1] = 1e-7, 1e-7
    result = compare_case(reference, actual, mask)
    assert result["gate_pass"]
    policy = result["heads"]["policy_logits"]
    assert policy["argmax_scope"] == "legal_policy"
    assert policy["argmax_flips"] == 1
    detail = policy["flip_details"][0]
    assert (detail["row"], detail["reference_index"], detail["observed_index"]) == (0, 0, 1)
    assert detail["reference_top2_logit_margin"] == pytest.approx(1e-7)
    assert detail["reference_top2_probability_margin"] > 0
    json_safe(result)


def test_legal_mask_never_exempts_an_illegal_raw_logit():
    reference, actual = outputs(), outputs()
    mask = np.zeros((1, 4168), dtype=bool)
    mask[0, 1] = True
    actual[0][0, 0] = 0.001
    result = compare_case(reference, actual, mask)
    assert not result["gate_pass"]
    policy = result["heads"]["policy_logits"]
    assert policy["raw_violation_count"] == 1
    assert policy["probability_max_abs_error"] == 0
    assert policy["argmax_flips"] == 0


def test_real_position_bound_remains_stricter_than_random_stress_bound():
    reference, actual = outputs(), outputs()
    reference[0][0, 0] = actual[0][0, 0] = 256
    actual[0][0, 0] += 0.00018
    assert compare_case(reference, actual)["gate_pass"]
    assert not compare_case(reference, actual, rtol=0)["gate_pass"]


@pytest.mark.parametrize("failure", ["shape", "nan", "infinity", "mask", "count"])
def test_contract_failures_are_explicit_and_do_not_hide_other_valid_head(failure):
    reference, actual = outputs(), outputs()
    mask = None
    if failure == "shape":
        actual[0] = np.zeros((1, 4167))
    elif failure == "nan":
        actual[0][0, 1] = np.nan
    elif failure == "infinity":
        actual[0][0, 1] = np.inf
    elif failure == "mask":
        mask = np.zeros((1, 4168), dtype=bool)
    else:
        actual.pop()
    result = compare_case(reference, actual, mask)
    assert not result["gate_pass"] and result["failure_kind"] == "output_contract"
    assert result["error"]
    if failure != "count":
        assert result["heads"]["wdl_logits"]["raw_violation_count"] == 0
    json_safe(result)


def test_finite_extreme_values_cannot_turn_overflow_into_a_false_pass_or_invalid_json():
    reference, actual = outputs(), outputs()
    reference[0][:], actual[0][:] = 1.7e308, -1.7e308
    result = compare_case(reference, actual)
    assert not result["gate_pass"]
    assert result["heads"]["policy_logits"]["raw_arithmetic_overflow"]
    assert result["heads"]["policy_logits"]["raw_max_abs_error"] is None
    json_safe(result)


def test_cross_head_batch_mismatch_is_a_contract_failure_with_both_heads_reported():
    reference, actual = outputs(), outputs()
    reference[1] = np.zeros((2, 3))
    actual[1] = np.zeros((2, 3))
    result = compare_case(reference, actual)
    assert not result["gate_pass"] and result["failure_kind"] == "output_contract"
    assert "batch sizes differ" in result["error"]
    assert result["heads"]["policy_logits"]["raw_violation_count"] == 0
    assert result["heads"]["wdl_logits"]["raw_violation_count"] == 0
    json_safe(result)


@pytest.mark.parametrize("rtol", [True, -1, np.nan, "1e-6"])
def test_invalid_relative_limits_are_contract_errors(rtol):
    result = compare_case(outputs(), outputs(), rtol=rtol)
    assert not result["gate_pass"] and result["failure_kind"] == "output_contract"
    json_safe(result)
