"""Research BSM evaluator using positive binary64 quadrature, not a new model.

No multiprecision, external libraries or reference-oracle fallback. A bounded
quadrature estimator checks integration agreement, not total floating error.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import sys

from lab.black_scholes import _validate

LN2_HI = 0.6931471803691238
LN2_LO = 1.9082149292705877e-10
LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)
INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)
QUAD_TOL = 8.0 * sys.float_info.epsilon


@dataclass(frozen=True)
class Result:
    call: float
    put: float
    parity: float
    otm_option: str
    otm_value: float
    log_otm_value: float
    log_moneyness: float
    quadrature_evaluations: int
    quadrature_relative_estimate: float
    branch: str
    underflow: bool


def _gauss_rule(order):
    """Legendre roots by recurrence/Newton; normalize the positive weights."""
    nodes, weights = [], []
    for index in range(1, order//2+1):
        root = math.cos(math.pi*(index-0.25)/(order+0.5))
        for _ in range(20):
            previous, current = 1.0, root
            for degree in range(2, order+1):
                previous, current = current, ((2*degree-1)*root*current-(degree-1)*previous)/degree
            derivative = order*(root*current-previous)/(root*root-1.0)
            step = current/derivative
            revised = root-step
            if revised == root or abs(step) < 0.25*math.ulp(root):
                break
            root = revised
        previous, current = 1.0, root
        for degree in range(2, order+1):
            previous, current = current, ((2*degree-1)*root*current-(degree-1)*previous)/degree
        derivative = order*(root*current-previous)/(root*root-1.0)
        weight = 2.0/((1-root*root)*derivative*derivative)
        nodes.extend((-root,root))
        weights.extend((weight,weight))
    scale = 2.0/math.fsum(weights)
    return tuple(sorted((x,w*scale) for x,w in zip(nodes,weights)))


RULE16, RULE32 = _gauss_rule(16), _gauss_rule(32)


def _rule(function, left, right, rule):
    midpoint, halfwidth = (left+right)*0.5, (right-left)*0.5
    return halfwidth*math.fsum(weight*function(midpoint+halfwidth*x) for x,weight in rule)


def _integrate(function, cuts):
    evaluations = 0
    def interval(left, right, depth):
        nonlocal evaluations
        coarse = _rule(function,left,right,RULE16)
        fine = _rule(function,left,right,RULE32)
        evaluations += 48
        error = abs(fine-coarse)
        if error <= QUAD_TOL*abs(fine) or fine == coarse:
            return fine,error
        if depth == 12:
            raise ArithmeticError('binary64 quadrature did not converge')
        middle = (left+right)*0.5
        a,ea = interval(left,middle,depth+1)
        b,eb = interval(middle,right,depth+1)
        return math.fsum((a,b)),ea+eb
    parts = [interval(a,b,0) for a,b in zip(cuts,cuts[1:]) if a < b]
    value = math.fsum(x for x,_ in parts)
    error = math.fsum(e for _,e in parts)
    if not value > 0.0 or not math.isfinite(value):
        raise ArithmeticError('invalid positive integral')
    return value, error/value, evaluations


def log_ratio(s, k):
    if 0.5*k <= s <= 2.0*k:
        # Sterbenz subtraction keeps near-ATM input differences visible.
        return math.log1p((s-k)/k)
    sm,se = math.frexp(s)
    km,ke = math.frexp(k)
    difference = se-ke
    return math.fsum((math.log(sm/km),difference*LN2_HI,difference*LN2_LO))


def scaled_product(base, exponential, factors=()):
    """Positive product with one exponential and no early monetary underflow."""
    if base == 0.0 or any(x == 0.0 for x in factors):
        return 0.0
    mantissa, power = math.frexp(base)
    for factor in factors:
        part, exponent = math.frexp(factor)
        mantissa *= part
        power += exponent
        mantissa, adjustment = math.frexp(mantissa)
        power += adjustment
    if not math.isfinite(exponential):
        raise OverflowError('nonfinite logarithmic scale outside supported domain')
    if exponential == 0.0:
        return math.ldexp(mantissa,power)
    shift = int(round(exponential/math.log(2.0)))
    if power+shift < -1076:
        return 0.0
    if power+shift > 1026:
        raise OverflowError('monetary result overflows binary64')
    residual = math.fsum((exponential,-shift*LN2_HI,-shift*LN2_LO))
    return math.ldexp(mantissa*math.exp(residual),power+shift)


def prices(s, k, t, r, q, sigma):
    _validate(s,k,t,r,q,sigma)
    carry = (r-q)*t
    m = math.fsum((log_ratio(s,k),carry))
    v = sigma*math.sqrt(t)
    if not math.isfinite(m) or not math.isfinite(v):
        raise OverflowError('nonfinite moneyness/total volatility')
    option = 'call' if m <= 0.0 else 'put'
    if r == q or t == 0.0:
        parity = math.copysign(scaled_product(abs(s-k),-r*t),s-k)
    else:
        base, exponent = (s,-q*t) if m >= 0 else (k,-r*t)
        parity = math.copysign(scaled_product(base,exponent,(-math.expm1(-abs(m)),)),m)
    if v == 0.0:
        return Result(max(parity,0.),max(-parity,0.),parity,option,0.,-math.inf,m,0,0.,'deterministic',False)
    base, discount = (s,-q*t) if option == 'call' else (k,-r*t)
    z = -abs(m)/v+v*0.5
    if not math.isfinite(z) or not math.isfinite(z*z):
        raise OverflowError('log price beyond binary64 exponent arithmetic')
    if z <= 0.0:
        scale = max(1.0,-z)
        b = v/scale
        def function(w):
            payoff = w if b == 0.0 else -math.expm1(-b*w)/b
            return math.exp((z/scale)*w-0.5*(w/scale)**2)*payoff
        integral,estimate,evaluations = _integrate(function,(0.,1.,2.,4.,8.,16.,32.,64.))
        log_density = -0.5*z*z
        factors = (v,1.0/scale,1.0/scale,integral,INV_SQRT_2PI)
        log_relative = math.fsum((log_density,math.log(v),-2.0*math.log(scale),math.log(integral),-LOG_SQRT_2PI))
        branch = 'scaled_negative_z_integral'
    else:
        # Completing the square avoids exp(z*z/2) overflow at large v.
        left,right = max(0.0,z-12.0),z+12.0
        cuts = sorted({left,right,*[z+delta for delta in (-8.,-4.,-2.,-1.,0.,1.,2.,4.,8.) if left<z+delta<right]})
        def function(u):
            return math.exp(-0.5*(u-z)**2)*(-math.expm1(-v*u)/v)
        integral,estimate,evaluations = _integrate(function,cuts)
        log_density = 0.0
        factors = (v,integral,INV_SQRT_2PI)
        log_relative = math.fsum((math.log(v),math.log(integral),-LOG_SQRT_2PI))
        branch = 'centered_positive_z_integral'
    value = scaled_product(base,math.fsum((discount,log_density)),factors)
    log_value = math.fsum((math.log(base),discount,log_relative))
    if option == 'call':
        call,put = value,math.fsum((value,-parity))
    else:
        put,call = value,math.fsum((value,parity))
    if not all(math.isfinite(x) and x >= 0.0 for x in (call,put)):
        raise ArithmeticError('invalid reconstructed prices')
    return Result(call,put,parity,option,value,log_value,m,evaluations,estimate,branch,value==0.0)
