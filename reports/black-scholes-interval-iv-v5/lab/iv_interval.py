"""BSM quote-set inversion using exact quotes and Arb-certified comparisons.

Research utility, not the frozen binary64 price engine. Certification is
conditional on the documented Arb enclosure contract and this implementation.
The process must use one thread: python-flint precision context is global.
"""
from dataclasses import dataclass
from fractions import Fraction
import math
import struct
from flint import arb, fmpq, ctx


MAX_RATIONAL_BITS = 4096


def rational(x):
    if isinstance(x, bool):
        raise TypeError('Boolean is not a numeric quote')
    if not isinstance(x, (int, float, str, Fraction)):
        raise TypeError('Use int, finite binary64 float, exact decimal string or Fraction')
    if isinstance(x, str) and len(x) > 1300:
        raise ValueError('Numeric string exceeds the input size budget')
    if isinstance(x, str) and 'e' in x.lower():
        # Check before Fraction allocates a power of ten.
        exponent = x.lower().split('e')[-1]
        if len(exponent) > 6 or abs(int(exponent)) > 1233:
            raise ValueError('Decimal exponent exceeds the input size budget')
    f = Fraction(x)
    if max(abs(f.numerator).bit_length(), f.denominator.bit_length()) > MAX_RATIONAL_BITS:
        raise ValueError('Rational input exceeds the declared size budget')
    return f


def ball(x):
    return arb(fmpq(x.numerator, x.denominator))


def encode(x):
    return None if x is None else str(x)


@dataclass(frozen=True)
class QuoteInterval:
    lower: Fraction
    upper: Fraction
    lower_closed: bool = True
    upper_closed: bool = True
    origin: str = 'explicit exact price interval'

    def __post_init__(self):
        object.__setattr__(self, 'lower', rational(self.lower))
        object.__setattr__(self, 'upper', rational(self.upper))
        if self.lower > self.upper:
            raise ValueError('Lower price must not exceed upper price')
        if type(self.lower_closed) is not bool or type(self.upper_closed) is not bool:
            raise TypeError('Endpoint closure must be bool')

    def as_dict(self):
        return {'lower': str(self.lower), 'upper': str(self.upper),
                'lower_closed': self.lower_closed, 'upper_closed': self.upper_closed,
                'origin': self.origin}


def rounding_cell(price):
    """Nonnegative real-price cell for binary64 round-to-nearest, ties-to-even."""
    if type(price) is not float or not math.isfinite(price) or price < 0:
        raise ValueError('Expected a finite nonnegative binary64 float')
    p = Fraction(price)
    if price == 0:
        return QuoteInterval(0, Fraction(1, 2**1075), True, True, 'binary64 nearest-even cell')
    previous = Fraction(math.nextafter(price, -math.inf))
    following = math.nextafter(price, math.inf)
    successor = Fraction(2**1024) if math.isinf(following) else Fraction(following)
    even = struct.unpack('>Q', struct.pack('>d', price))[0] % 2 == 0
    return QuoteInterval((previous+p)/2, (p+successor)/2, even, even,
                         'binary64 nearest-even cell')


@dataclass(frozen=True)
class Contract:
    s: Fraction
    k: Fraction
    t: Fraction
    r: Fraction = Fraction(0)
    q: Fraction = Fraction(0)
    option: str = 'call'

    def __post_init__(self):
        for name in ('s', 'k', 't', 'r', 'q'):
            object.__setattr__(self, name, rational(getattr(self, name)))
        if self.s <= 0 or self.k <= 0 or self.t < 0 or self.option not in ('call', 'put'):
            raise ValueError('Expected S,K>0, T>=0 and call/put')
        if abs(self.r*self.t) > 1024 or abs(self.q*self.t) > 1024:
            raise ValueError('Discount exponents outside the declared [-1024,1024] budget')

    def as_dict(self):
        return {**{n: str(getattr(self,n)) for n in ('s','k','t','r','q')}, 'option':self.option}


def legs(c):
    s,k,t,r,q = [ball(getattr(c,n)) for n in ('s','k','t','r','q')]
    return s*(-q*t).exp(), k*(-r*t).exp()


def intrinsic(c):
    if c.t == 0:
        return ball(max((c.s-c.k) if c.option=='call' else (c.k-c.s), Fraction(0)))
    if c.r == c.q:
        difference = (c.s-c.k) if c.option=='call' else (c.k-c.s)
        return ball(max(difference,Fraction(0)))*(-ball(c.r)*ball(c.t)).exp()
    a,b=legs(c)
    difference = a-b if c.option=='call' else b-a
    return difference.max(arb(0))


def ceiling(c):
    a,b=legs(c)
    return a if c.option=='call' else b


def price_ball(c, sigma):
    """Enclose the unspecialized model price at exact rational sigma."""
    sigma = rational(sigma)
    if sigma < 0:
        raise ValueError('Negative volatility')
    if sigma == 0 or c.t == 0:
        return intrinsic(c)
    a,b=legs(c)
    t=ball(c.t)
    v=ball(sigma)*t.sqrt()
    m=(ball(c.s)/ball(c.k)).log()+(ball(c.r)-ball(c.q))*t
    d1=m/v+v/2
    d2=d1-v
    root2=arb(2).sqrt()
    if c.option=='call':
        return (a*(-d1/root2).erfc()-b*(-d2/root2).erfc())/2
    return (b*(d2/root2).erfc()-a*(d1/root2).erfc())/2


def exact_ball_bounds(x):
    """Outward bounds as mantissa * 2**exponent, without expanding huge powers."""
    if not x.is_finite():
        return None, None
    def dyadic(y):
        m,e=y.man_exp();m,e=int(m),int(e)
        return {'mantissa':str(m),'exponent':str(e)}
    return dyadic(x.lower()),dyadic(x.upper())


class Engine:
    def __init__(self, contract, precisions, max_sigma, max_steps, tolerance):
        self.c=contract;self.precisions=precisions;self.max_sigma=max_sigma
        self.max_steps=max_steps;self.tolerance=tolerance;self.comparisons=0
        self.evaluations=0;self.highest_precision=0

    def compare(self, kind, target, sigma=None, retain=False):
        self.comparisons+=1
        witness=None
        for bits in self.precisions:
            self.evaluations+=1;self.highest_precision=max(self.highest_precision,bits)
            with ctx.workprec(bits):
                value=(intrinsic(self.c) if kind=='intrinsic' else ceiling(self.c)
                       if kind=='ceiling' else price_ball(self.c,sigma))
                difference=value-ball(target)
                sign=-1 if difference < 0 else 1 if difference > 0 else 0 if difference==0 else None
                if retain or sign is None:
                    low,high=exact_ball_bounds(difference)
                    witness={'kind':kind,'target':str(target),'sigma':encode(sigma),'precision_bits':bits,
                             'difference_lower':low,'difference_upper':high,'sign':sign}
                if sign is not None:
                    return sign,witness
        return None,witness

    def root(self, target):
        # Caller has certified intrinsic < target < unattained ceiling.
        low,high=Fraction(0),min(Fraction(1),self.max_sigma)
        expansions=0
        while True:
            sign,_=self.compare('price',target,high)
            if sign==0:
                low=high;break
            if sign==1:break
            if sign is None:
                return self.finish_root(target,low,high,'unresolved_bracketing',False)
            low=high
            if high==self.max_sigma:
                return self.finish_root(target,low,None,'sigma_budget_exhausted',False)
            high=min(2*high,self.max_sigma);expansions+=1
        state='converged';iterations=0
        while high-low > self.tolerance and iterations < self.max_steps:
            middle=(low+high)/2
            sign,_=self.compare('price',target,middle)
            iterations+=1
            if sign is None:
                state='precision_budget_exhausted';break
            if sign==0:
                low=high=middle;break
            if sign<0:low=middle
            else:high=middle
        if high-low>self.tolerance and state=='converged':state='iteration_budget_exhausted'
        result=self.finish_root(target,low,high,state,True)
        result.update(iterations=iterations,bracket_expansions=expansions)
        return result

    def finish_root(self,target,low,high,state,bracketed):
        # If a sign cannot be proved during bracketing, the finite high is
        # NOT an upper bound. Preserve [low,+inf), never invent completeness.
        if not bracketed:high=None
        lower_sign,lower_witness=self.compare('price',target,low,True)
        upper_witness=None
        if high is not None:
            upper_sign,upper_witness=self.compare('price',target,high,True)
            assert lower_sign in (-1,0) and upper_sign in (0,1)
        else:
            assert lower_sign in (-1,0)
        return {'kind':'root','lower':str(low),'upper':encode(high),'state':state,
                'target':str(target),'certified_bracket':bracketed,
                'lower_witness':lower_witness,'upper_witness':upper_witness}


def invert_quote(contract, quote, *, tolerance=Fraction(1,10**12),
                 precisions=(128,256,512,1024,2048), max_sigma=Fraction(2**20), max_steps=256):
    """Return certified outer enclosure and endpoint root brackets.

    `resolved` concerns root-location tolerance, not identifiability. Closure
    flags describe exact mathematical endpoints, not approximate bracket ends.
    None at outer_upper means +infinity. Unknown comparisons are never false.
    """
    if not isinstance(contract,Contract) or not isinstance(quote,QuoteInterval):
        raise TypeError('Expected Contract and QuoteInterval')
    tolerance,max_sigma=rational(tolerance),rational(max_sigma)
    if tolerance<=0 or max_sigma<=0 or max_sigma>2**40:
        raise ValueError('Invalid tolerance or sigma search budget')
    if max(abs(max_sigma.numerator).bit_length(), max_sigma.denominator.bit_length())>1024:
        raise ValueError('Sigma search budget exceeds 1024 rational bits')
    if not precisions or any(type(n) is not int or not 64<=n<=4096 for n in precisions):
        raise ValueError('Precision budget must be integers from 64 to 4096')
    if tuple(sorted(set(precisions)))!=tuple(precisions) or type(max_steps) is not int or not 1<=max_steps<=2048:
        raise ValueError('Invalid precision schedule or iteration budget')
    if ctx.threads!=1:
        raise RuntimeError('Set flint.ctx.threads=1 before sequential research use')
    engine=Engine(contract,tuple(precisions),max_sigma,max_steps,tolerance)
    result={'contract':contract.as_dict(),'quote':quote.as_dict(),
            'certification':'conditional on Arb enclosure contract; not an independent formal verification',
            'tolerance':str(tolerance),'boundary_comparisons':[]}
    def done(status,**extra):
        return {**result,'status':status,**extra,'comparisons':engine.comparisons,
                'arb_evaluations':engine.evaluations,'highest_precision_bits':engine.highest_precision}
    def cmp(kind,target):
        sign,witness=engine.compare(kind,target,retain=True)
        result['boundary_comparisons'].append(witness)
        return sign
    if quote.lower==quote.upper and not(quote.lower_closed and quote.upper_closed):
        return done('empty',reason='empty open price interval',resolved=True)
    lower0,upper0=cmp('intrinsic',quote.lower),cmp('intrinsic',quote.upper)
    if lower0 is None or upper0 is None:
        return done('unresolved',reason='boundary precision budget',resolved=False,outer_lower='0',outer_upper=None)
    if contract.t==0:
        inside=(lower0>0 or lower0==0 and quote.lower_closed) and (upper0<0 or upper0==0 and quote.upper_closed)
        return done('all_volatilities' if inside else 'empty',reason='zero-maturity payoff independent of sigma',
                    resolved=True,outer_lower='0' if inside else None,outer_upper=None)
    lowercap,uppercap=cmp('ceiling',quote.lower),cmp('ceiling',quote.upper)
    if lowercap is None or uppercap is None:
        return done('unresolved',reason='boundary precision budget',resolved=False,outer_lower='0',outer_upper=None)
    if upper0>0 or upper0==0 and not quote.upper_closed:
        return done('empty',reason='quote below or excludes intrinsic',resolved=True)
    if lowercap<=0:
        return done('empty',reason='ceiling is unattained for finite sigma',resolved=True)
    if upper0==0:
        return done('singleton_zero',resolved=True,outer_lower='0',outer_upper='0',zero_included=True)
    lower={'kind':'zero','lower':'0','upper':'0','state':'exact'} if lower0>=0 else engine.root(quote.lower)
    upper={'kind':'unbounded','lower':None,'upper':None,'state':'exact'} if uppercap<=0 else engine.root(quote.upper)
    zero_included=lower0>0 or lower0==0 and quote.lower_closed
    resolved=all(x['state'] in ('exact','converged') for x in (lower,upper))
    return done('interval',resolved=resolved,lower_endpoint=lower,upper_endpoint=upper,
                lower_endpoint_closed=zero_included if lower['kind']=='zero' else quote.lower_closed,
                upper_endpoint_closed=False if upper['kind']=='unbounded' else quote.upper_closed,
                outer_lower=lower['lower'],outer_upper=upper['upper'],zero_included=zero_included,
                unbounded=upper['kind']=='unbounded')
