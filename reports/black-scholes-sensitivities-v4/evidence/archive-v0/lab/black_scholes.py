"""Small, auditable Black--Scholes kernels.

This module is a research baseline, not a production trading library.  The
`stable_prices` function evaluates the same mathematical BSM formula as
`naive_prices`; it changes only the floating-point representation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


_SQRT_2 = math.sqrt(2.0)
_SQRT_PI_OVER_2 = math.sqrt(math.pi / 2.0)
_LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)


@dataclass(frozen=True)
class PricePair:
    call: float
    put: float
    parity: float


@dataclass(frozen=True)
class DetailedPrice:
    call: float
    put: float
    parity: float
    otm_option: str
    otm_value: float
    log_otm_value: float
    call_intrinsic: float
    put_intrinsic: float
    time_value: float
    log_time_value: float
    underflow: bool


def _validate(s: float, k: float, t: float, r: float, q: float, sigma: float) -> None:
    values = (s, k, t, r, q, sigma)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("all inputs must be finite")
    if s <= 0.0 or k <= 0.0:
        raise ValueError("spot and strike must be positive")
    if t < 0.0 or sigma < 0.0:
        raise ValueError("time and volatility must be non-negative")


def _discounted_legs(
    s: float, k: float, t: float, r: float, q: float
) -> tuple[float, float, float]:
    log_stock_leg = math.log(s) - q * t
    log_strike_leg = math.log(k) - r * t
    try:
        stock_leg = math.exp(log_stock_leg)
        strike_leg = math.exp(log_strike_leg)
    except OverflowError as exc:
        raise OverflowError("a discounted cash leg is not representable in float64") from exc
    return stock_leg, strike_leg, log_stock_leg - log_strike_leg


def _normal_cdf(x: float) -> float:
    return 0.5 * math.erfc(-x / _SQRT_2)


def _normal_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x - _LOG_SQRT_2PI)


def _erfcx_positive(x: float) -> float:
    """Scaled complementary error function for x >= 0.

    The direct branch is safe below 26.  Above it, an alternating asymptotic
    expansion avoids `exp(x*x)` overflow.  This is intentionally compact; a
    production kernel must compare against Cody/Jaeckel implementations.
    """

    if x < 0.0:
        raise ValueError("_erfcx_positive expects x >= 0")
    if x < 26.0:
        return math.exp(x * x) * math.erfc(x)

    inv_2x2 = 0.5 / (x * x)
    term = 1.0
    series = 1.0
    for n in range(1, 9):
        term *= -(2 * n - 1) * inv_2x2
        series += term
    return series / (math.sqrt(math.pi) * x)


def _left_mills_ratio(z: float) -> float:
    """Return Phi(z) / phi(z) for z <= 0 without tail underflow."""

    if z > 0.0:
        return _normal_cdf(z) / _normal_pdf(z)
    return _SQRT_PI_OVER_2 * _erfcx_positive(-z / _SQRT_2)


def _mills_difference(a: float, gap: float) -> float:
    """Return Mills(a)-Mills(a+gap) for positive inputs.

    Here Mills(a)=Phi(-a)/phi(a).  For large, nearby arguments the direct
    subtraction loses all digits.  The asymptotic series is differenced term
    by term using `expm1`, so the small distance b-a remains visible.
    """

    if not a > 0.0 or not gap > 0.0:
        raise ValueError("expected positive a and gap")
    if a < 8.0:
        b = a + gap
        return _left_mills_ratio(-a) - _left_mills_ratio(-b)

    # `a + gap` may round back to `a`; retaining gap explicitly is the point
    # of this branch.
    log_ratio = math.log1p(gap / a)
    coefficient = 1.0
    total = 0.0
    inverse_a = 1.0 / a
    for n in range(8):
        power = 2 * n + 1
        power_difference = (inverse_a**power) * (
            -math.expm1(-power * log_ratio)
        )
        total += coefficient * power_difference
        coefficient *= -(2 * n + 1)
    return total


def _stable_parity(stock_leg: float, strike_leg: float, m: float) -> float:
    """Compute stock_leg - strike_leg while preserving near-ATM digits."""

    if m >= 0.0:
        return stock_leg * (-math.expm1(-m))
    return strike_leg * math.expm1(m)


def _tail_common(log_leg: float, d: float) -> float:
    log_value = log_leg - 0.5 * d * d - _LOG_SQRT_2PI
    if log_value < math.log(math.ulp(0.0)):
        return 0.0
    return math.exp(log_value)


def naive_prices(
    s: float,
    k: float,
    t: float,
    r: float,
    q: float,
    sigma: float,
) -> PricePair:
    """Direct textbook evaluation used as a deliberately simple baseline."""

    _validate(s, k, t, r, q, sigma)
    stock_leg, strike_leg, m = _discounted_legs(s, k, t, r, q)
    parity = stock_leg - strike_leg
    if t == 0.0 or sigma == 0.0:
        return PricePair(max(parity, 0.0), max(-parity, 0.0), parity)

    total_vol = sigma * math.sqrt(t)
    d1 = m / total_vol + 0.5 * total_vol
    d2 = d1 - total_vol
    call = stock_leg * _normal_cdf(d1) - strike_leg * _normal_cdf(d2)
    put = strike_leg * _normal_cdf(-d2) - stock_leg * _normal_cdf(-d1)
    return PricePair(call, put, parity)


def stable_prices(
    s: float,
    k: float,
    t: float,
    r: float,
    q: float,
    sigma: float,
) -> PricePair:
    """Regime-aware float64 evaluation of a European call and put.

    The cheaper OTM leg is evaluated first.  A central branch uses `erf` so
    the small ATM time value is not obtained by subtracting two numbers near
    one half.  A tail branch uses scaled Mills ratios.  The ITM leg is then
    reconstructed with a parity difference computed through `expm1`.
    """

    _validate(s, k, t, r, q, sigma)
    stock_leg, strike_leg, m = _discounted_legs(s, k, t, r, q)
    parity = _stable_parity(stock_leg, strike_leg, m)
    if t == 0.0 or sigma == 0.0:
        return PricePair(max(parity, 0.0), max(-parity, 0.0), parity)

    total_vol = sigma * math.sqrt(t)
    if total_vol == 0.0:
        return PricePair(max(parity, 0.0), max(-parity, 0.0), parity)

    d1 = m / total_vol + 0.5 * total_vol
    d2 = d1 - total_vol

    if m <= 0.0:
        # The call is OTM.  In a far left tail both CDFs are tiny and share
        # the same density scale; factoring it avoids two underflows.
        direct_left = stock_leg * _normal_cdf(d1)
        direct_right = strike_leg * _normal_cdf(d2)
        direct = direct_left - direct_right
        direct_score = abs(direct_left) + abs(direct_right)
        if d1 < 0.0:
            common = _tail_common(math.log(stock_leg), d1)
            mills_1 = _left_mills_ratio(d1)
            mills_2 = _left_mills_ratio(d2)
            alternative = common * (mills_1 - mills_2)
            alternative_score = common * (abs(mills_1) + abs(mills_2))
        else:
            erf_left = stock_leg * math.erf(d1 / _SQRT_2)
            erf_right = strike_leg * math.erf(d2 / _SQRT_2)
            alternative = 0.5 * parity + 0.5 * (erf_left - erf_right)
            alternative_score = 0.5 * (
                abs(parity) + abs(erf_left) + abs(erf_right)
            )
        # This is a condition-based choice, not a fitted threshold.  Each
        # score bounds the scale of round-off in the corresponding linear
        # combination; special-function approximation error still needs an
        # external oracle.
        call = (
            alternative
            if direct < 0.0 or alternative_score < direct_score
            else direct
        )
        call = max(call, 0.0)
        put = call - parity
    else:
        # The put is OTM.  The symmetric right-tail construction avoids
        # subtracting two probabilities rounded to one.
        direct_left = strike_leg * _normal_cdf(-d2)
        direct_right = stock_leg * _normal_cdf(-d1)
        direct = direct_left - direct_right
        direct_score = abs(direct_left) + abs(direct_right)
        if d2 > 0.0:
            common = _tail_common(math.log(strike_leg), d2)
            mills_1 = _left_mills_ratio(-d2)
            mills_2 = _left_mills_ratio(-d1)
            alternative = common * (mills_1 - mills_2)
            alternative_score = common * (abs(mills_1) + abs(mills_2))
        else:
            erf_left = stock_leg * math.erf(d1 / _SQRT_2)
            erf_right = strike_leg * math.erf(d2 / _SQRT_2)
            alternative = -0.5 * parity + 0.5 * (erf_left - erf_right)
            alternative_score = 0.5 * (
                abs(parity) + abs(erf_left) + abs(erf_right)
            )
        put = (
            alternative
            if direct < 0.0 or alternative_score < direct_score
            else direct
        )
        put = max(put, 0.0)
        call = put + parity

    return PricePair(call, put, parity)


def stable_detailed(
    s: float,
    k: float,
    t: float,
    r: float,
    q: float,
    sigma: float,
) -> DetailedPrice:
    """Return prices plus a log-domain representation of the OTM leg.

    `underflow=True` means the mathematical OTM price is positive but below
    the least positive float64 returned by the ordinary API.  The log value is
    retained for comparison and implied-volatility diagnostics.
    """

    pair = stable_prices(s, k, t, r, q, sigma)
    stock_leg, strike_leg, m = _discounted_legs(s, k, t, r, q)
    option = "call" if m <= 0.0 else "put"
    value = pair.call if option == "call" else pair.put
    call_intrinsic = max(pair.parity, 0.0)
    put_intrinsic = max(-pair.parity, 0.0)

    if t == 0.0 or sigma == 0.0:
        log_value = math.log(value) if value > 0.0 else -math.inf
        return DetailedPrice(
            call=pair.call,
            put=pair.put,
            parity=pair.parity,
            otm_option=option,
            otm_value=value,
            log_otm_value=log_value,
            call_intrinsic=call_intrinsic,
            put_intrinsic=put_intrinsic,
            time_value=value,
            log_time_value=log_value,
            underflow=False,
        )

    total_vol = sigma * math.sqrt(t)
    d1 = m / total_vol + 0.5 * total_vol
    d2 = d1 - total_vol
    if m <= 0.0 and d1 < 0.0:
        difference = _mills_difference(-d1, total_vol)
        log_value = (
            math.log(stock_leg)
            - 0.5 * d1 * d1
            - _LOG_SQRT_2PI
            + math.log(difference)
        )
    elif m > 0.0 and d2 > 0.0:
        difference = _mills_difference(d2, total_vol)
        log_value = (
            math.log(strike_leg)
            - 0.5 * d2 * d2
            - _LOG_SQRT_2PI
            + math.log(difference)
        )
    else:
        log_value = math.log(value) if value > 0.0 else -math.inf

    return DetailedPrice(
        call=pair.call,
        put=pair.put,
        parity=pair.parity,
        otm_option=option,
        otm_value=value,
        log_otm_value=log_value,
        call_intrinsic=call_intrinsic,
        put_intrinsic=put_intrinsic,
        time_value=value,
        log_time_value=log_value,
        underflow=value == 0.0 and math.isfinite(log_value),
    )
