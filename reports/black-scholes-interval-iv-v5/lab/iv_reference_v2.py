"""Independent mpmath control; no Arb or candidate imports, not an interval proof."""
from fractions import Fraction
import mpmath as mp


def number(x):
    q=Fraction(x)
    return mp.mpf(q.numerator)/q.denominator


def value(contract, sigma, kind='price'):
    s,k,t,r,q=[number(contract[n]) for n in ('s','k','t','r','q')]
    a,b=s*mp.exp(-q*t),k*mp.exp(-r*t)
    call=contract['option']=='call'
    log_forward=mp.log(s/k)+(r-q)*t
    difference=a*(-mp.expm1(-log_forward)) if log_forward>=0 else b*mp.expm1(log_forward)
    intrinsic=max(difference if call else -difference,mp.mpf(0))
    if kind=='intrinsic' or t==0 or sigma is not None and Fraction(sigma)==0:
        return intrinsic
    if kind=='ceiling':return a if call else b
    v=number(sigma)*mp.sqrt(t)
    z=log_forward/v
    d1=z+v/2;d2=z-v/2;root2=mp.sqrt(2)
    # Evaluate OTM first and add the exact model parity term when ITM.
    if a<=b:
        otm=(a*mp.erfc(-d1/root2)-b*mp.erfc(-d2/root2))/2
        return otm if call else otm-difference
    otm=(b*mp.erfc(d2/root2)-a*mp.erfc(d1/root2))/2
    return otm+difference if call else otm


def dyadic(d):
    return mp.ldexp(mp.mpf(d['mantissa']),int(d['exponent']))


def check_witness(contract,witness):
    if witness is None or witness['sign'] is None:
        return {'checked':False,'reason':'uncertain comparison'}
    vals=[]
    for digits in (180,260):
        with mp.workdps(digits):
            x=value(contract,witness['sigma'],witness['kind'])-number(witness['target'])
            vals.append(x)
    with mp.workdps(280):
        lo,hi=dyadic(witness['difference_lower']),dyadic(witness['difference_upper'])
        sign=witness['sign']
        contained=lo<=vals[1]<=hi
        change=abs(vals[1]-vals[0]);width=hi-lo
        scale=max(abs(number(witness['target'])),abs(vals[1]),abs(lo),abs(hi),mp.mpf('1e-400'))
        convergence=(change<=mp.mpf('1e-160')*scale and
                     (change==0 if width==0 else change<=mp.mpf('1e-8')*width))
        proven=(sign==-1 and hi<0) or (sign==1 and lo>0) or (sign==0 and lo==hi==0)
        return {'checked':True,'contained':bool(contained),'reference_converged':bool(convergence),
                'witness_sign_valid':bool(proven),'pass':bool(contained and convergence and proven)}


def witnesses(result):
    out=list(result['boundary_comparisons'])
    for key in ('lower_endpoint','upper_endpoint'):
        endpoint=result.get(key,{})
        out += [endpoint[n] for n in ('lower_witness','upper_witness') if endpoint.get(n) is not None]
    return out
