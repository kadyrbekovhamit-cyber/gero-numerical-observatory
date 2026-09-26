"""Analytic BSM sensitivities, separate from the frozen price branch graph.

Research API, positive T and sigma only. These are derivatives of the smooth
BSM model, not derivatives of every rounded arithmetic instruction. No oracle,
finite-difference bump, autodiff framework, or multiprecision fallback is used.
Vega is per unit volatility (divide by 100 for a one percentage-point change).
"""
from dataclasses import dataclass
import math
import sys

from lab.black_scholes import _validate
from lab.black_scholes_v2 import log_ratio, LN2_HI, LN2_LO, INV_SQRT_2PI, LOG_SQRT_2PI

SQRT2 = math.sqrt(2.)
TAIL_START = 12.


@dataclass(frozen=True)
class Sensitivities:
    call_delta: float
    put_delta: float
    gamma: float
    vega: float
    log_call_delta: float
    log_abs_put_delta: float
    log_gamma: float
    log_vega: float
    inverse_vega: float
    log_inverse_vega: float
    d1: float
    log_moneyness: float
    ranges: dict


def _scaled_ratio(exponential, numerator=(), denominator=()):
    """exp(e)*prod(n)/prod(d), postponing all range decisions until the end."""
    mantissa, power = 1., 0
    for factors, divide in ((numerator, False), (denominator, True)):
        for factor in factors:
            if not math.isfinite(factor) or factor <= 0.:
                raise ArithmeticError('positive finite scaling factors required')
            part, exponent = math.frexp(factor)
            mantissa = mantissa/part if divide else mantissa*part
            power += -exponent if divide else exponent
            mantissa, adjustment = math.frexp(mantissa)
            power += adjustment
    if not math.isfinite(exponential):
        raise OverflowError('nonfinite exponent outside research API domain')
    shift = int(round(exponential/math.log(2.)))
    if power+shift < -1076:
        return 0.
    if power+shift > 1026:
        return math.inf
    residual = math.fsum((exponential, -shift*LN2_HI, -shift*LN2_LO))
    try:
        return math.ldexp(mantissa*math.exp(residual), power+shift)
    except OverflowError:
        return math.inf


def _tail_mills_factor(a):
    # Phi(-a)=phi(a)/a * (1-1/a^2+3/a^4-...). DLMF 7.12.1.
    # First omitted term is a truncation bound in exact arithmetic only.
    inverse_square=(1./a)*(1./a)
    term, terms = 1., [1.]
    for n in range(1, 96):
        term *= -(2*n-1)*inverse_square
        total=math.fsum(terms)
        if abs(term) <= sys.float_info.epsilon*total:
            return total
        if abs(term) >= abs(terms[-1]):
            raise ArithmeticError('tail series did not reach its truncation target')
        terms.append(term)
    raise ArithmeticError('tail series iteration budget exhausted')


def _discounted_cdf(x, discount):
    if x <= -TAIL_START:
        a=-x
        factor=_tail_mills_factor(a)
        density=-.5*x*x
        log_value=math.fsum((discount,density,-LOG_SQRT_2PI,-math.log(a),math.log(factor)))
        value=_scaled_ratio(math.fsum((discount,density)),(INV_SQRT_2PI,factor),(a,))
        return value,log_value
    cdf=.5*math.erfc(-x/SQRT2)
    return _scaled_ratio(discount,(cdf,)),math.fsum((discount,math.log(cdf)))


def _range(x):
    if math.isinf(x):
        return 'overflow'
    if x == 0.:
        return 'rounded_zero'
    return 'subnormal' if abs(x) < sys.float_info.min else 'normal'


def greeks(s, k, t, r, q, sigma):
    _validate(s,k,t,r,q,sigma)
    if t == 0. or sigma == 0.:
        raise ValueError('positive time and volatility required; boundary Greeks need a separate contract')
    root_t=math.sqrt(t)
    v=sigma*root_t
    if v == 0.:
        raise ArithmeticError('positive total volatility underflows; log-volatility interface required')
    m=math.fsum((log_ratio(s,k),(r-q)*t))
    d1=math.fsum((m/v,.5*v))
    if not all(math.isfinite(x) for x in (m,v,d1,d1*d1,q*t)):
        raise OverflowError('logarithmic arithmetic outside research API domain')
    discount=-q*t
    call_delta,log_call=_discounted_cdf(d1,discount)
    abs_put_delta,log_put=_discounted_cdf(-d1,discount)
    exponent=math.fsum((discount,-.5*d1*d1))
    gamma=_scaled_ratio(exponent,(INV_SQRT_2PI,),(s,sigma,root_t))
    vega=_scaled_ratio(exponent,(s,root_t,INV_SQRT_2PI))
    log_gamma=math.fsum((discount,-.5*d1*d1,-LOG_SQRT_2PI,-math.log(s),-math.log(sigma),-.5*math.log(t)))
    log_vega=math.fsum((discount,-.5*d1*d1,-LOG_SQRT_2PI,math.log(s),.5*math.log(t)))
    inverse_vega=_scaled_ratio(-exponent,(),(s,root_t,INV_SQRT_2PI))
    values={'call_delta':call_delta,'put_delta':-abs_put_delta,'gamma':gamma,'vega':vega,'inverse_vega':inverse_vega}
    return Sensitivities(call_delta,-abs_put_delta,gamma,vega,log_call,log_put,
                         log_gamma,log_vega,inverse_vega,-log_vega,d1,m,
                         {name:_range(value) for name,value in values.items()})
