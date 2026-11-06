"""SPEC §9/§10 reference statistics, finite-sample rules and SPRT decisions."""

import math

import pytest

import elo


@pytest.mark.parametrize("counts, expected", [
    ((60, 20, 20), 0.7), ((0, 10, 0), 0.5),
    ((1, 0, 0), 1.0), ((0, 0, 1), 0.0),
])
def test_score_is_the_raw_empirical_fraction(counts, expected):
    assert elo.score(*counts) == expected


@pytest.mark.parametrize("counts, expected", [
    ((60, 0, 40), 70.43650362227247), ((50, 0, 50), 0.0),
    ((10, 0, 0), 511.50144038113183), ((0, 0, 10), -511.50144038113154),
    ((1, 0, 0), 0.0), ((0, 0, 1), 0.0),
])
def test_elo_reference_values_and_finite_sample_clamping(counts, expected):
    value = elo.elo(*counts)
    assert math.isfinite(value)
    assert value == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize("counts, expected", [
    ((60, 0, 40), (2.7656471210969977, 143.91068168752204)),
    ((30, 40, 30), (-53.1589719045787, 53.15897190457869)),
    ((10, 0, 0), (421.9547858882145, 511.50144038113183)),
    ((0, 0, 10), (-511.50144038113154, -421.95478588821453)),
    ((1, 0, 0), (0.0, 0.0)), ((0, 0, 1), (0.0, 0.0)),
    ((0, 10, 0), (0.0, 0.0)),
])
def test_confidence_interval_uses_clamped_variance_and_endpoints(counts, expected):
    interval = elo.elo_ci95(*counts)
    assert isinstance(interval, tuple)
    assert interval == pytest.approx(expected, abs=1e-9)
    assert all(math.isfinite(value) for value in interval)
    assert interval[0] <= elo.elo(*counts) <= interval[1]


def test_sprt_bounds_reference_and_asymmetric_error_rates():
    assert elo.sprt_bounds(0.05, 0.05) == pytest.approx(
        (-2.9444389791664403, 2.9444389791664403))
    assert elo.sprt_bounds(0.1, 0.2) == pytest.approx(
        (-1.5040773967762742, 2.0794415416798357))


@pytest.mark.parametrize("counts, expected", [
    ((100, 0, 100), -1.31384934580127),
    ((130, 0, 70), 6.113787588652595),
    ((10, 0, 0), 96.59169404312043),
    ((0, 0, 10), -109.73018750113333),
    ((1, 0, 0), -0.00656924672900635),
    ((0, 0, 1), -0.00656924672900635),
    ((0, 1, 0), -1.6423116822515875),
])
def test_llr_reference_values_use_literal_clamped_x_and_variance_floor(counts, expected):
    assert elo.llr(*counts, 0, 40) == pytest.approx(expected, abs=1e-9)


def test_extreme_finite_hypotheses_and_small_error_rates_remain_finite():
    assert elo.llr(50, 0, 50, -1e308, 1e308) == 0.0
    lower, upper = elo.sprt_bounds(1e-300, 1e-300)
    assert math.isfinite(lower) and math.isfinite(upper)
    assert lower < 0 < upper


@pytest.mark.parametrize("counts, max_games, expected", [
    ((90, 20, 10), 400, ("sprt_accept", True)),
    ((20, 20, 80), 400, ("sprt_reject", False)),
    ((60, 0, 40), 400, (None, False)),
    ((56, 0, 44), 100, ("max_games", True)),
    ((50, 0, 50), 100, ("max_games", False)),
    ((56, 0, 44), 90, ("max_games", True)),
    ((90, 20, 10), 120, ("sprt_accept", True)),
    ((20, 20, 80), 120, ("sprt_reject", False)),
])
def test_decision_continuation_sprt_and_cap_precedence(counts, max_games, expected):
    assert elo.decide(*counts, sum(counts), max_games, 0, 40, 0.05, 0.05) == expected


@pytest.mark.parametrize("bound, expected", [
    (0, ("sprt_reject", False)), (1, ("sprt_accept", True)),
])
def test_sprt_threshold_comparisons_are_inclusive(monkeypatch, bound, expected):
    threshold = elo.sprt_bounds(0.05, 0.05)[bound]
    monkeypatch.setattr(elo, "llr", lambda *args: threshold)
    assert elo.decide(50, 0, 50, 100, 100, 0, 40, 0.05, 0.05) == expected


def test_zero_llr_at_cap_is_not_promoted():
    assert elo.llr(50, 0, 50, -40, 40) == 0.0
    assert elo.decide(50, 0, 50, 100, 100, -40, 40, 0.05, 0.05) == (
        "max_games", False)


@pytest.mark.parametrize("statistic", ["score", "elo", "elo_ci95", "llr", "decide"])
@pytest.mark.parametrize("counts", [
    (0, 0, 0), (-1, 0, 11), (1, -1, 10), (1, 10, -1),
    (True, 0, 9), (1.0, 0, 9), (1, "0", 9), (1, None, 9),
])
def test_invalid_result_counts_raise_value_error(statistic, counts):
    extra = {"llr": (0, 40), "decide": (10, 400, 0, 40, 0.05, 0.05)}
    with pytest.raises(ValueError):
        getattr(elo, statistic)(*counts, *extra.get(statistic, ()))


@pytest.mark.parametrize("hypotheses", [
    (0, 0), (40, 0), (float("nan"), 40), (0, float("nan")),
    (float("-inf"), 40), (0, float("inf")), ("0", 40), (0, "40"),
    (False, 40),
])
@pytest.mark.parametrize("statistic", ["llr", "decide"])
def test_invalid_hypotheses_raise_value_error(statistic, hypotheses):
    args = (50, 0, 50)
    if statistic == "decide":
        args += (100, 400)
    args += hypotheses
    if statistic == "decide":
        args += (0.05, 0.05)
    with pytest.raises(ValueError):
        getattr(elo, statistic)(*args)


@pytest.mark.parametrize("probabilities", [
    (0, 0.05), (1, 0.05), (0.05, 0), (0.05, 1),
    (-0.1, 0.05), (0.05, 1.1), (float("nan"), 0.05),
    (0.05, float("inf")), ("0.05", 0.05),
])
@pytest.mark.parametrize("statistic", ["sprt_bounds", "decide"])
def test_invalid_error_probabilities_raise_value_error(statistic, probabilities):
    args = () if statistic == "sprt_bounds" else (50, 0, 50, 100, 400, 0, 40)
    with pytest.raises(ValueError):
        getattr(elo, statistic)(*args, *probabilities)


@pytest.mark.parametrize("games_played", [True, 100.0, "100", -1, 0, 99, 101])
def test_decision_requires_integer_consistent_game_count(games_played):
    with pytest.raises(ValueError):
        elo.decide(50, 0, 50, games_played, 400, 0, 40, 0.05, 0.05)


@pytest.mark.parametrize("max_games", [True, 400.0, "400", 0, -1])
def test_decision_requires_positive_integer_game_cap(max_games):
    with pytest.raises(ValueError):
        elo.decide(50, 0, 50, 100, max_games, 0, 40, 0.05, 0.05)
