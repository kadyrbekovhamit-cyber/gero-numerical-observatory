"""Independent high-precision reference based on mpmath."""

from __future__ import annotations

import mpmath as mp


def prices(
    s: float,
    k: float,
    t: float,
    r: float,
    q: float,
    sigma: float,
    dps: int = 100,
) -> tuple[mp.mpf, mp.mpf]:
    with mp.workdps(dps):
        ss = mp.mpf(str(s))
        kk = mp.mpf(str(k))
        tt = mp.mpf(str(t))
        rr = mp.mpf(str(r))
        qq = mp.mpf(str(q))
        vv = mp.mpf(str(sigma))
        stock_leg = ss * mp.exp(-qq * tt)
        strike_leg = kk * mp.exp(-rr * tt)
        parity = stock_leg - strike_leg
        if tt == 0 or vv == 0:
            return max(parity, 0), max(-parity, 0)
        total_vol = vv * mp.sqrt(tt)
        m = mp.log(stock_leg / strike_leg)
        d1 = m / total_vol + total_vol / 2
        d2 = d1 - total_vol
        cdf = lambda x: mp.erfc(-x / mp.sqrt(2)) / 2
        call = stock_leg * cdf(d1) - strike_leg * cdf(d2)
        put = strike_leg * cdf(-d2) - stock_leg * cdf(-d1)
        return call, put
