"""Exploratory derivative/IV diagnostics of the frozen price kernels.

This is not a new blind confirmation or a vendor defect report. All stencils,
including bad step sizes, are retained. A rounded high-precision-price control
separates finite-difference conditioning from kernel error.
"""
from datetime import datetime, timezone
import argparse
import hashlib
import json
import math
from pathlib import Path

import mpmath as mp

from lab.black_scholes_v2 import prices as v2
from lab.black_scholes_v3 import prices as v3
from lab.corpus_v3 import verify
from lab.reference import evaluate

ROOT = Path(__file__).resolve().parents[1]
STEPS = (1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12)


def stencil(fm, f0, fp, a, b):
    left, right = (f0-fm)/a, (fp-f0)/b
    return (b*left+a*right)/(a+b), 2*(right-left)/(a+b)


def text(x):
    return mp.nstr(x, 40)


def analytic(args):
    s, k, t, r, q, sig = map(mp.mpf, args)
    v = sig*mp.sqrt(t)
    d1 = (mp.log(s/k)+(r-q)*t)/v+v/2
    discount = mp.exp(-q*t)
    pdf = mp.exp(-d1*d1/2)/mp.sqrt(2*mp.pi)
    return (discount*mp.erfc(-d1/mp.sqrt(2))/2,
            discount*pdf/(s*v), s*discount*pdf*mp.sqrt(t))


def bases():
    cases = [('ATM', (100., 100., 1., 0., 0., .3)),
             ('regular', (107., 100., 1.3, .035, .012, .27))]
    for kind, z, vol in (('small_z', -1., .07),
                         ('small_v', -.4, .125),
                         ('tail', -12., .4),
                         ('fallback', -3., .3)):
        for side in (-1., 1.):
            m = side*vol*(vol/2-z)
            cases.append((kind+('_call' if side < 0 else '_put'),
                          (100.*math.exp(m), 100., 1., 0., 0., vol)))
    return cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT/'evidence/sensitivity-audit-v3.json')
    target = parser.parse_args().output
    if target.exists():
        raise FileExistsError('Preserve previous exploratory evidence')
    freeze = verify()
    rows = []
    with mp.workdps(100):
        for name, args in bases():
            s = args[0]
            delta, gamma, vega = analytic(args)
            for step in STEPS:
                sm, sp = float(s*(1-step)), float(s*(1+step))
                if not sm < s < sp:
                    raise ArithmeticError('Collapsed stencil')
                aa, bb = mp.mpf(s)-mp.mpf(sm), mp.mpf(sp)-mp.mpf(s)
                triplet = [(x,)+args[1:] for x in (sm,s,sp)]
                refs = [evaluate(*a,dps=180).call for a in triplet]
                exact = stencil(*refs,aa,bb)
                control = stencil(*(float(x) for x in refs),float(aa),float(bb))
                samples = {}
                for version, fn in (('v2',v2),('v3',v3)):
                    points = [fn(*a) for a in triplet]
                    got = stencil(*(x.call for x in points),float(aa),float(bb))
                    samples[version] = {
                        'prices_hex':[x.call.hex() for x in points],
                        'branches':[getattr(x,'method',getattr(x,'branch','')) for x in points],
                        'delta':got[0], 'gamma':got[1],
                        'delta_error':text(mp.mpf(got[0])-delta),
                        'gamma_relative_error':text(abs(mp.mpf(got[1])/gamma-1)),
                        'gamma_error_vs_exact_stencil':text(mp.mpf(got[1])-exact[1]),
                        'gamma_difference_from_rounded_oracle_control':text(mp.mpf(got[1])-control[1]),
                    }
                rows.append({'base':name,'args_hex':[x.hex() for x in args],
                             'relative_step':step,'spot_stencil_hex':[x.hex() for x in (sm,s,sp)],
                             'reference_delta':text(delta),'reference_gamma':text(gamma),
                             'reference_vega':text(vega),
                             'exact_stencil_delta':text(exact[0]),'exact_stencil_gamma':text(exact[1]),
                             'truncation_gamma_relative_error':text(abs(exact[1]/gamma-1)),
                             'rounded_oracle_delta':control[0], 'rounded_oracle_gamma':control[1],
                             'rounded_oracle_gamma_relative_error':text(abs(mp.mpf(control[1])/gamma-1)),
                             'samples':samples})

        args=(100.,100.,1.,0.,0.,.3)
        delta,gamma,vega=analytic(args)
        # This is analytic replay of the selected ATM leaf, NOT an actual AD
        # execution of v3 (which has no AD interface). On the S=K submanifold,
        # call=S*erf(v/(2sqrt(2))). It is not an identity in S with K fixed.
        leaf_delta=mp.erf(mp.mpf(.3)/(2*mp.sqrt(2)))
        atm={'args_hex':[x.hex() for x in args], 'reference_delta':text(delta),
             'fixed_leaf_delta':text(leaf_delta),'fixed_leaf_gamma':'0',
             'reference_gamma':text(gamma),'reference_vega':text(vega),
             'status':'analytic counterexample to blind branch differentiation; no native AD API claimed'}

        iv_rows=[]
        for vol in (.03,.04,.05,.06,.07,.08):
            args=(200.,100.,1.,0.,0.,vol)
            ref=evaluate(*args,dps=180)
            got=v3(*args)
            iv_rows.append({'sigma_hex':vol.hex(),'sigma':vol,'rounded_reference_call_hex':float(ref.call).hex(),
                            'v3_call_hex':got.call.hex(),'v3_put':got.put,
                            'reference_time_value':text(ref.put),'reference_log_time_value':text(ref.log_otm)})
        half_ulp=mp.mpf(math.ulp(100.))/2
        lo,hi=mp.mpf(0),mp.mpf(1)
        def put(sig):
            d1=mp.log(2)/sig+sig/2
            d2=d1-sig
            return 100*mp.erfc(d2/mp.sqrt(2))/2-200*mp.erfc(d1/mp.sqrt(2))/2
        for _ in range(180):
            mid=(lo+hi)/2
            if put(mid)<half_ulp:
                lo=mid
            else:
                hi=mid
        iv={'contract':'S=200,K=100,T=1,r=q=0, quoted call rounded to nearest binary64',
            'price':100.,'half_ulp':text(half_ulp),'sigma_threshold_bracket':[text(lo),text(hi)],
            'meaning':'All continuous volatilities strictly between 0 and the lower endpoint round to the same 100 call. Upper endpoint is a high-precision numerical estimate, not an interval proof.',
            'examples':iv_rows}
        summary={'stencils':len(rows),'bases':len(bases()),'steps':list(STEPS),
                 'negative_gamma':{v:sum(r['samples'][v]['gamma']<0 for r in rows) for v in ('v2','v3')},
                 'negative_gamma_rounded_oracle_control':sum(r['rounded_oracle_gamma']<0 for r in rows),
                 'branch_crossing_stencils_v3':sum(len(set(r['samples']['v3']['branches']))>1 for r in rows),
                 'iv_distinct_sigmas_same_call':len({r['sigma_hex'] for r in iv_rows}),
                 'all_iv_examples_round_to_100':all(float.fromhex(r['rounded_reference_call_hex'])==100. and float.fromhex(r['v3_call_hex'])==100. for r in iv_rows)}
    verify()
    report={'status':'exploratory audit; retain all cases; no confirmation superiority claim',
            'generated_utc':datetime.now(timezone.utc).isoformat(),
            'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'candidate_sha256':freeze['source_sha256']['lab/black_scholes_v3.py'],
            'summary':summary,'atm_branch_derivative':atm,'iv_information_loss':iv,'stencils':rows}
    target.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'summary':summary,'atm':atm,'iv_threshold':iv['sigma_threshold_bracket']},indent=2))


if __name__=='__main__':
    main()
