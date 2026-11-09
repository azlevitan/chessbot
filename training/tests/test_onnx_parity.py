"""Numerical parity rejects local errors even when unrelated logits are large."""

import numpy as np
import pytest

from onnx_parity import PARITY_ATOL, compare_outputs


def outputs(batch=1, dtype=np.float64):
    return [np.zeros((batch, 4168), dtype=dtype), np.zeros((batch, 3), dtype=dtype)]


def test_large_logits_allow_elementwise_relative_rounding_but_strict_fails():
    ref, actual = outputs(), outputs()
    ref[0][0, 0] = 256
    actual[0][0, 0] = 256 + 0.00018310546875
    assert compare_outputs(ref, actual) == pytest.approx(0.00018310546875)
    with pytest.raises(ValueError, match="policy_logits raw parity.*scaled_error"):
        compare_outputs(ref, actual, rtol=0)


def test_large_unrelated_logit_does_not_relax_a_near_zero_element():
    ref, actual = outputs(), outputs()
    ref[0][0, 0] = actual[0][0, 0] = 1e9
    actual[0][0, 1] = 2e-4
    with pytest.raises(ValueError, match="policy_logits raw parity"):
        compare_outputs(ref, actual)


def test_relative_bound_scales_reference_not_actual():
    ref, actual = outputs(), outputs()
    actual[0][0, 0] = PARITY_ATOL + 5e-11
    with pytest.raises(ValueError, match="policy_logits raw parity"):
        compare_outputs(ref, actual)


def test_strict_boundary_is_inclusive_and_does_not_mutate_inputs():
    ref, actual = outputs(dtype=np.float32), outputs()
    for head in actual:
        head[:] = PARITY_ATOL
    before = [head.copy() for head in actual]
    assert compare_outputs(ref, actual, rtol=0) == PARITY_ATOL
    for head, saved in zip(actual, before):
        np.testing.assert_array_equal(head, saved)


@pytest.mark.parametrize("head", [0, 1])
def test_softmax_probability_changes_have_an_independent_bound(head):
    ref, actual = outputs(), outputs()
    ref[head][0, :2] = actual[head][0, :2] = 256
    actual[head][0, 0] += 2e-4
    name = "policy_logits" if head == 0 else "wdl_logits"
    with pytest.raises(ValueError, match=f"{name} probability parity"):
        compare_outputs(ref, actual)


def test_legal_mask_excludes_illegal_probabilities_but_not_raw_logits():
    ref, actual = outputs(), outputs()
    ref[0][0, :2] = actual[0][0, :2] = 256
    actual[0][0, 0] += 2e-4
    legal = np.zeros_like(ref[0], dtype=bool)
    legal[0, 1] = True
    assert compare_outputs(ref, actual, legal) == pytest.approx(2e-4)
    actual[0][0, 2] = 2e-4
    with pytest.raises(ValueError, match="policy_logits raw parity"):
        compare_outputs(ref, actual, legal)


def test_near_tie_argmax_flip_is_allowed():
    ref, actual = outputs(), outputs()
    ref[0][:] = actual[0][:] = -30
    ref[0][0, :2], actual[0][0, :2] = [1e-7, 0], [0, 1e-7]
    assert ref[0].argmax() != actual[0].argmax()
    assert compare_outputs(ref, actual) == pytest.approx(1e-7)


def test_softmax_is_stable_for_large_offsets():
    ref, actual = outputs(), outputs()
    for expected, result in zip(ref, actual):
        expected[:] = result[:] = 1e300
    assert compare_outputs(ref, actual) == 0


def test_finite_values_cannot_overflow_into_a_false_raw_pass():
    ref, actual = outputs(), outputs()
    ref[0][:], actual[0][:] = 1.7e308, -1.7e308
    with pytest.raises(ValueError, match="policy_logits raw parity.*nonfinite"):
        compare_outputs(ref, actual, rtol=10)


@pytest.mark.parametrize("side", [0, 1])
@pytest.mark.parametrize("head", [0, 1])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_logits_are_rejected_in_either_output(side, head, value):
    pair = [outputs(), outputs()]
    pair[side][head][0, 0] = value
    with pytest.raises(ValueError, match="nonfinite"):
        compare_outputs(*pair)


@pytest.mark.parametrize("rtol", [-1, np.nan, np.inf, -np.inf, None, "1e-6", True, [1e-6]])
def test_invalid_rtol_is_rejected(rtol):
    with pytest.raises(ValueError, match="rtol"):
        compare_outputs(outputs(), outputs(), rtol=rtol)


@pytest.mark.parametrize("invalid", [None, [], [np.zeros((1, 4168))], [0, 0, 0]])
def test_wrong_output_count_is_rejected(invalid):
    with pytest.raises(ValueError, match="exactly two"):
        compare_outputs(outputs(), invalid)


@pytest.mark.parametrize("head, shape", [(0, (1, 4167)), (1, (1, 4)), (0, (4168,)), (1, (0, 3))])
def test_invalid_head_shapes_are_rejected(head, shape):
    actual = outputs()
    actual[head] = np.zeros(shape)
    with pytest.raises(ValueError, match="shape"):
        compare_outputs(outputs(), actual)


def test_empty_or_inconsistent_batches_are_rejected():
    with pytest.raises(ValueError, match="B > 0"):
        compare_outputs(outputs(0), outputs(0))
    actual = outputs()
    actual[1] = np.zeros((2, 3))
    with pytest.raises(ValueError, match="batch sizes differ"):
        compare_outputs(outputs(), actual)
    with pytest.raises(ValueError, match="shape mismatch"):
        compare_outputs(outputs(), outputs(2))


@pytest.mark.parametrize("dtype", [int, bool, complex, object])
def test_nonfloating_logits_are_rejected(dtype):
    actual = outputs()
    actual[0] = actual[0].astype(dtype)
    with pytest.raises(ValueError, match="real floating"):
        compare_outputs(outputs(), actual)


@pytest.mark.parametrize("mask", [np.ones((1, 4167), dtype=bool), np.ones((1, 4168), dtype=int),
                                 np.zeros((1, 4168), dtype=bool), [[True] * 4168]])
def test_invalid_legal_masks_are_rejected(mask):
    with pytest.raises(ValueError, match="legal_mask"):
        compare_outputs(outputs(), outputs(), mask)


def test_each_mask_row_must_have_a_legal_action():
    legal = np.ones((2, 4168), dtype=bool)
    legal[1] = False
    with pytest.raises(ValueError, match="per row"):
        compare_outputs(outputs(2), outputs(2), legal)
