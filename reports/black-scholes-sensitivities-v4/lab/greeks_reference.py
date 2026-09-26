"""Exact binary64-input multiprecision analytic reference for sensitivity tests."""
import mpmath as mp

NAMES=('call_delta','put_delta','gamma','vega','inverse_vega')
LOG_NAMES=('log_call_delta','log_abs_put_delta','log_gamma','log_vega','log_inverse_vega')


def evaluate(args, dps=180):
    with mp.workdps(dps):
        s,k,t,r,q,sigma=map(mp.mpf,args)
        root=mp.sqrt(t)
        v=sigma*root
        d1=(mp.log(s/k)+(r-q)*t)/v+v/2
        discount=mp.exp(-q*t)
        pdf=mp.exp(-d1*d1/2)/mp.sqrt(2*mp.pi)
        dc=discount*mp.erfc(-d1/mp.sqrt(2))/2
        dp=-discount*mp.erfc(d1/mp.sqrt(2))/2
        gamma=discount*pdf/(s*v)
        vega=s*discount*pdf*root
        values=(dc,dp,gamma,vega,1/vega)
        return {**dict(zip(NAMES,values)), **dict(zip(LOG_NAMES,(mp.log(abs(x)) for x in values)))}


def checked(args):
    low,high=evaluate(args,100),evaluate(args,180)
    with mp.workdps(180):
        relative=max(abs(low[n]/high[n]-1) for n in NAMES)
        log_diff=max(abs(low[n]-high[n]) for n in LOG_NAMES)
        # The log comparison scales for deliberately enormous exponents.
        log_scaled=max(abs(low[n]-high[n])/max(1,abs(high[n])) for n in LOG_NAMES)
        good=relative<mp.mpf('1e-60') and log_scaled<mp.mpf('1e-60')
        return high,{'passed':bool(good),'dps':[100,180],
                     'max_relative_change':mp.nstr(relative,30),
                     'max_absolute_log_change':mp.nstr(log_diff,30)}
