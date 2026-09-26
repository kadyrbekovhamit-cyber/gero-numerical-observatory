"""Research hybrid BSM evaluator; fast analytic regions plus frozen v2.

The analytic remainder estimates below are component estimates in floating
arithmetic, not certified bounds for all input/libm/reconstruction errors.
No multiprecision or reference-oracle calls occur in this implementation.
"""
from dataclasses import dataclass
import math
import sys

from lab.black_scholes import _validate
from lab.black_scholes_v2 import (
    INV_SQRT_2PI, LOG_SQRT_2PI, log_ratio, scaled_product, prices as integral_prices,
)

EPS = sys.float_info.epsilon
SQRT2 = math.sqrt(2.0)
SMALL_V_MAX = 0.125
SMALL_Z_MIN = -1.0
ASYMPTOTIC_A_MIN = 12.0
DIRECT_ERROR_BUDGET = 1e-13
SERIES_RELATIVE_BUDGET = EPS


@dataclass(frozen=True)
class Result:
    call: float
    put: float
    parity: float
    otm_option: str
    otm_value: float
    log_otm_value: float
    log_moneyness: float
    method: str
    work_evaluations: int
    component_relative_estimate: float
    component_estimate_kind: str
    underflow: bool


def _small_v_series(z, v):
    """Integral / v = sum (-v)^(n-1) I_n(z)/n!, n>=1.

    I_n(z)=int_0^infinity u^n exp(z*u-u*u/2)du.
    Integration by parts gives I_1=1+z*I_0 and
    I_n=(n-1)*I_(n-2)+z*I_(n-1). Restrict z to [-1,1/16]
    to avoid the severe large-negative-z cancellation of this recurrence.
    Taylor's theorem for exp(-v*u) bounds the integral remainder by the
    first omitted term in exact arithmetic, irrespective of monotonicity.
    """
    if not SMALL_Z_MIN <= z <= SMALL_V_MAX * 0.5 or not 0 < v <= SMALL_V_MAX:
        return None
    i0 = math.sqrt(math.pi / 2.0) * math.exp(0.5*z*z) * math.erfc(-z/SQRT2)
    i1 = math.fsum((1.0, z*i0))
    terms = [i1]
    previous, current = i0, i1
    coefficient = 1.0
    for n in range(2, 19):
        moment = math.fsum(((n-1)*previous, z*current))
        coefficient *= -v/n
        term = coefficient * moment
        total = math.fsum(terms)
        if not moment > 0.0 or not total > 0.0:
            return None
        relative = abs(term)/total
        if relative <= SERIES_RELATIVE_BUDGET:
            return total, relative, len(terms)
        terms.append(term)
        previous, current = current, moment
    return None


def _tail_series(a, v):
    """Scaled [Mills(a)-Mills(a+v)]/v with no subtraction of Mills values.

    Divide the positive-payoff integral by v/a**2 and expand exp(-u*u/2).
    For r=v/a and p=2n+1, H_p(r)=[1-(1+r)**(-p)]/r. The n-th term is
    (-1)^n (2n-1)!! a**(-2n) H_(2n+1)(r). The first omitted term bounds
    the exact-arithmetic truncation remainder by Taylor's theorem under
    the nonnegative weight exp(-a*u)*(1-exp(-v*u)).
    """
    if not a >= ASYMPTOTIC_A_MIN or not v > 0.0:
        return None
    ratio = v/a
    if not math.isfinite(ratio):
        return None
    inverse_square = (1.0/a)*(1.0/a)
    log_ratio_value = math.log1p(ratio)
    terms = [1.0/(1.0+ratio)]
    coefficient = 1.0
    for n in range(1, 65):
        coefficient *= -(2*n-1)*inverse_square
        p = 2*n+1
        h = p if ratio == 0.0 else -math.expm1(-p*log_ratio_value)/ratio
        term = coefficient*h
        total = math.fsum(terms)
        if not total > 0.0 or not math.isfinite(term):
            return None
        relative = abs(term)/total
        if relative <= SERIES_RELATIVE_BUDGET:
            return total, relative, len(terms)
        if abs(term) >= abs(terms[-1]):
            return None
        terms.append(term)
    return None


def _direct(z, v, absolute_m):
    """Conditional direct CDF evaluation with a conservative heuristic gate.

    The score includes cancellation and squared normal arguments. It is an
    engineering screen, not a libm error certificate. Underflow-prone CDF
    regions are excluded before evaluation, independently of monetary scale.
    """
    w = z-v
    if not (-26.0 <= w <= z <= 26.0) or absolute_m > 600.0:
        return None
    left = 0.5*math.erfc(-z/SQRT2)
    right = math.exp(absolute_m)*(0.5*math.erfc(-w/SQRT2))
    value = left-right
    if not value > 0.0 or not math.isfinite(value):
        return None
    score = 16.0*EPS*(1.0+z*z+w*w+absolute_m)*(left+right)/value
    if score > DIRECT_ERROR_BUDGET:
        return None
    return value, score, 2


def _fallback(args):
    old = integral_prices(*args)
    return Result(old.call, old.put, old.parity, old.otm_option, old.otm_value,
                  old.log_otm_value, old.log_moneyness, 'v2_fallback',
                  old.quadrature_evaluations, old.quadrature_relative_estimate,
                  'v2 quadrature agreement; not total error', old.underflow)


def prices(s, k, t, r, q, sigma):
    args = (s, k, t, r, q, sigma)
    _validate(*args)
    carry = (r-q)*t
    m = math.fsum((log_ratio(s,k),carry))
    v = sigma*math.sqrt(t)
    if not math.isfinite(m) or not math.isfinite(v):
        raise OverflowError('nonfinite moneyness/total volatility')
    if v == 0.0:
        return _fallback(args)
    option = 'call' if m <= 0.0 else 'put'
    z = -abs(m)/v+v*0.5
    if not math.isfinite(z) or not math.isfinite(z*z):
        raise OverflowError('log price beyond binary64 exponent arithmetic')
    base, discount = (s,-q*t) if option == 'call' else (k,-r*t)

    # Only exact input ATM is eligible; rounded m==0 with unequal carry is
    # not evidence of an exact forward-ATM contract.
    if s == k and r == q:
        if v < 1e-4:
            # Factoring v preserves prices when erf(v/(2sqrt(2))) underflows.
            v2 = v*v
            correction = 1.0-v2/24.0+v2*v2/640.0
            factors = (v, INV_SQRT_2PI, correction)
            log_relative = math.fsum((math.log(v), -LOG_SQRT_2PI, math.log(correction)))
            estimate = v2*v2*v2/21504.0
        else:
            relative = math.erf(v/(2.0*SQRT2))
            factors = (relative,)
            log_relative = math.log(relative)
            estimate = 0.0
        method, kind, work, density = 'exact_atm', 'ATM identity or short-series truncation only', 1, 0.0
    else:
        small = _small_v_series(z,v)
        tail = _tail_series(-z,v) if small is None else None
        direct = _direct(z,v,abs(m)) if small is None and tail is None else None
        if small is not None:
            integral, estimate, work = small
            factors = (v, integral, INV_SQRT_2PI)
            density = -0.5*z*z
            log_relative = math.fsum((density,math.log(v),math.log(integral),-LOG_SQRT_2PI))
            method, kind = 'small_v_moments', 'first omitted Taylor term; truncation only'
        elif tail is not None:
            integral, estimate, work = tail
            a = -z
            factors = (v,1.0/a,1.0/a,integral,INV_SQRT_2PI)
            density = -0.5*z*z
            log_relative = math.fsum((density,math.log(v),-2*math.log(a),math.log(integral),-LOG_SQRT_2PI))
            method, kind = 'tail_difference_series', 'first omitted Taylor term; truncation only'
        elif direct is not None:
            relative, estimate, work = direct
            factors, density, log_relative = (relative,), 0.0, math.log(relative)
            method, kind = 'conditioned_direct', 'heuristic rounding/cancellation screen; not certificate'
        else:
            return _fallback(args)

    if r == q or t == 0.0:
        parity = math.copysign(scaled_product(abs(s-k),-r*t),s-k)
    else:
        pbase, exponent = (s,-q*t) if m >= 0 else (k,-r*t)
        parity = math.copysign(scaled_product(pbase,exponent,(-math.expm1(-abs(m)),)),m)
    value = scaled_product(base,math.fsum((discount,density)),factors)
    log_value = math.fsum((math.log(base),discount,log_relative))
    call,put = (value,math.fsum((value,-parity))) if option == 'call' else (math.fsum((value,parity)),value)
    if not all(math.isfinite(x) and x >= 0.0 for x in (call,put)):
        raise ArithmeticError('invalid reconstructed prices')
    return Result(call,put,parity,option,value,log_value,m,method,work,estimate,kind,value==0.0)
