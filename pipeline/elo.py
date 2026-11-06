"""Pure Elo/GSPRT math for SPEC §9 and promotion decisions for SPEC §10.

``score`` reports the raw empirical score. As defined in §9, the other
statistics use x clamped to [1/(2n), 1 - 1/(2n)], including the variance and
LLR. Confidence endpoints are clamped to that interval before Elo conversion.
Invalid inputs, including empty results, raise ValueError.
"""

from __future__ import annotations

import math


def _total(w: int, d: int, l: int) -> int:
    if any(type(count) is not int or count < 0 for count in (w, d, l)):
        raise ValueError("win, draw and loss counts must be nonnegative integers")
    n = w + d + l
    if n == 0:
        raise ValueError("at least one game is required")
    return n


def _finite(value: float, name: str) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{name} must be a finite number")
    try:
        value = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} must be a finite number") from error
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return value


def _clamp(x: float, n: int) -> float:
    epsilon = 0.5 / n
    return min(max(x, epsilon), 1.0 - epsilon)


def _statistics(w: int, d: int, l: int) -> tuple[int, float, float]:
    n = _total(w, d, l)
    x = _clamp((w + 0.5 * d) / n, n)
    variance = (w * (1.0 - x) ** 2 + d * (0.5 - x) ** 2 + l * x ** 2) / n
    return n, x, variance


def _to_elo(x: float, n: int) -> float:
    x = _clamp(x, n)
    return 400.0 * (math.log10(x) - math.log10(1.0 - x))


def _expected_score(rating: float) -> float:
    # This form of the SPEC logistic avoids overflow for finite large ratings.
    exponent = rating * (math.log(10.0) / 400.0)
    tail = math.exp(-abs(exponent))
    return 1.0 / (1.0 + tail) if exponent >= 0 else tail / (1.0 + tail)


def score(w: int, d: int, l: int) -> float:
    """Empirical score, with a draw worth half a win (no clamping)."""
    n = _total(w, d, l)
    return (w + 0.5 * d) / n


def elo(w: int, d: int, l: int) -> float:
    """Relative Elo from the finite-sample clamped score."""
    n, x, _ = _statistics(w, d, l)
    return _to_elo(x, n)


def elo_ci95(w: int, d: int, l: int) -> tuple[float, float]:
    """Normal 95% score interval mapped to Elo; no SPRT variance floor."""
    n, x, variance = _statistics(w, d, l)
    margin = 1.96 * math.sqrt(variance / n)
    return _to_elo(x - margin, n), _to_elo(x + margin, n)


def llr(w: int, d: int, l: int, elo0: float, elo1: float) -> float:
    """SPEC's GSPRT approximation, positive in favor of elo1."""
    elo0, elo1 = _finite(elo0, "elo0"), _finite(elo1, "elo1")
    if elo0 >= elo1:
        raise ValueError("elo0 must be less than elo1")
    n, x, variance = _statistics(w, d, l)
    s0, s1 = _expected_score(elo0), _expected_score(elo1)
    return n * (s1 - s0) * (2.0 * x - s0 - s1) / (2.0 * max(variance, 1e-3))


def sprt_bounds(alpha: float, beta: float) -> tuple[float, float]:
    """Return the lower rejection and upper acceptance thresholds."""
    alpha, beta = _finite(alpha, "alpha"), _finite(beta, "beta")
    if not 0.0 < alpha < 1.0 or not 0.0 < beta < 1.0:
        raise ValueError("alpha and beta must be strictly between 0 and 1")
    return math.log(beta) - math.log1p(-alpha), math.log1p(-beta) - math.log(alpha)


def decide(w: int, d: int, l: int, games_played: int, max_games: int,
           elo0: float, elo1: float, alpha: float, beta: float) -> tuple[str | None, bool]:
    """Check inclusive SPRT bounds before the game cap; None means continue."""
    n = _total(w, d, l)
    if type(games_played) is not int or games_played != n:
        raise ValueError("games_played must equal the win, draw and loss total")
    if type(max_games) is not int or max_games <= 0:
        raise ValueError("max_games must be a positive integer")
    lower, upper = sprt_bounds(alpha, beta)
    likelihood = llr(w, d, l, elo0, elo1)
    if likelihood >= upper:
        return "sprt_accept", True
    if likelihood <= lower:
        return "sprt_reject", False
    if games_played >= max_games:
        return "max_games", likelihood > 0.0
    return None, False
