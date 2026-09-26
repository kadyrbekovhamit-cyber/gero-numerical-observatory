"""High-precision oracle with an explicit exact-binary64 input contract.

The two precision levels test stability, not a rigorous interval error bound.
The separate integral is an analytic cross-check sharing mpmath arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import mpmath as mp


@dataclass(frozen=True)
class Reference:
    call: object
    put: object
    parity: object
    otm: object
    log_otm: object
    option: str
    stock_leg: object
    strike_leg: object
    m: object
    total_vol: object
    d1: object
    d2: object
    dps: int


def evaluate(s, k, t, r, q, sigma, dps=180):
    raw = (s, k, t, r, q, sigma)
    if not all(isinstance(x, (float, int)) and math.isfinite(x) for x in raw):
        raise ValueError("oracle expects finite binary64 numbers")
    if s <= 0 or k <= 0 or t < 0 or sigma < 0:
        raise ValueError("invalid BSM domain")
    with mp.workdps(dps):
        # mp.mpf(float) preserves the exact binary value at this precision.
        ss, kk, tt, rr, qq, vv = (mp.mpf(x) for x in raw)
        stock = ss * mp.exp(-qq * tt)
        strike = kk * mp.exp(-rr * tt)
        parity = stock - strike
        m = mp.log(ss / kk) + (rr - qq) * tt
        vol = vv * mp.sqrt(tt)
        option = "call" if m <= 0 else "put"
        if vol == 0:
            call, put = max(parity, mp.mpf(0)), max(-parity, mp.mpf(0))
            return Reference(call, put, parity, mp.mpf(0), mp.ninf,
                             option, stock, strike, m, vol, None, None, dps)
        d1, d2 = m / vol + vol / 2, m / vol - vol / 2
        cdf = lambda z: mp.erfc(-z / mp.sqrt(2)) / 2
        call = stock * cdf(d1) - strike * cdf(d2)
        put = strike * cdf(-d2) - stock * cdf(-d1)
        otm = call if option == "call" else put
        if otm <= 0:
            raise ArithmeticError("insufficient oracle precision: non-positive OTM")
        return Reference(call, put, parity, otm, mp.log(otm), option,
                         stock, strike, m, vol, d1, d2, dps)


def checked(s, k, t, r, q, sigma, low_dps=100, high_dps=180):
    low = evaluate(s, k, t, r, q, sigma, dps=low_dps)
    attempts = []
    for precision in (high_dps, high_dps + 80):
        high = evaluate(s, k, t, r, q, sigma, dps=precision)
        with mp.workdps(precision):
            if high.otm == 0:
                relative_change = abs(low.otm)
                log_change = mp.mpf(0) if low.otm == 0 else mp.inf
            else:
                relative_change = abs(low.otm - high.otm) / high.otm
                log_change = abs(low.log_otm - high.log_otm)
            attempt = {
                "low_dps": low.dps, "high_dps": precision,
                "otm_relative_change": mp.nstr(relative_change, 30),
                "log_absolute_change": mp.nstr(log_change, 30),
                "passed": bool(relative_change <= mp.mpf("1e-60")
                               and log_change <= mp.mpf("1e-55")),
            }
        attempts.append(attempt)
        if attempt["passed"]:
            break
        low = high
    return high, {**attempts[-1], "precision_increased": len(attempts) > 1,
                  "attempts": attempts}


def integral_log_otm(s, k, t, r, q, sigma, dps=90):
    """Positive payoff integral, scaled to avoid cancellation and underflow.

    For an OTM call: C = X*phi(d1) * integral_0^inf
    exp(d1*u-u^2/2) * (1-exp(-v*u)) du. Put is symmetric.
    Use u=w/L, b=v/L and divide the integrand by b so tiny tail values
    cannot pass an absolute quadrature tolerance merely by being tiny.
    """
    with mp.workdps(dps):
        ss, kk, tt, rr, qq, vv = (mp.mpf(x) for x in (s, k, t, r, q, sigma))
        v = vv * mp.sqrt(tt)
        if v == 0:
            return mp.ninf
        m = mp.log(ss / kk) + (rr - qq) * tt
        d1, d2 = m / v + v / 2, m / v - v / 2
        z = d1 if m <= 0 else -d2
        log_leg = mp.log(ss) - qq * tt if m <= 0 else mp.log(kk) - rr * tt
        scale = max(mp.mpf(1), -z)
        b = v / scale
        def integrand(w):
            return mp.exp(z*w/scale - w*w/(2*scale*scale)) * (-mp.expm1(-b*w) / b)
        area = mp.quad(integrand, [0, 1, 4, 12, mp.inf])
        return log_leg - z*z/2 - mp.log(2*mp.pi)/2 + mp.log(v) - 2*mp.log(scale) + mp.log(area)


def prices(
    s: float,
    k: float,
    t: float,
    r: float,
    q: float,
    sigma: float,
    dps: int = 100,
) -> tuple[mp.mpf, mp.mpf]:
    ref = evaluate(s, k, t, r, q, sigma, dps=dps)
    return ref.call, ref.put
