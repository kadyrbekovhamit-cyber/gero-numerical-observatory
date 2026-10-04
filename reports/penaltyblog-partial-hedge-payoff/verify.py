"""Replay unchanged pinned arbitrage.py; synthetic back-bet accounting, no wagers.

Python 3.9+ and scipy required. Run with numerical thread limits = 1.
The full-hedge control adds only HiGHS options={'threads': 1} to linprog.
"""
from pathlib import Path
from fractions import Fraction as F
from datetime import datetime, timezone
import dataclasses, hashlib, importlib.util, json, math, os, platform, sys, time, warnings

P=Path(__file__).resolve().parent
source=P/'vendor/arbitrage.py'
receipt=json.loads((P/'evidence/SOURCE_REVIEW.json').read_text())
assert hashlib.sha256(source.read_bytes()).hexdigest()==receipt['files']['vendor/arbitrage.py']['sha256']
start=time.process_time()
spec=importlib.util.spec_from_file_location('pinned_penaltyblog_arbitrage',source)
arb=importlib.util.module_from_spec(spec);sys.modules[spec.name]=arb;spec.loader.exec_module(arb)
import scipy
real_linprog=arb.linprog
def one_thread_lp(*args,**kwargs):
    kwargs['options']={**kwargs.get('options',{}),'threads':1}
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',message='Unrecognized options detected.*')
        return real_linprog(*args,**kwargs)
arb.linprog=one_thread_lp

def payoff(s,o,q,h):
    # Independent cash ledger: payout of the winning position minus all paid stakes.
    s,o,q,h=([F(str(v)) for v in a] for a in (s,o,q,h))
    paid=sum(s)+sum(h)
    return [s[i]*o[i]+h[i]*q[i]-paid for i in range(len(s))]

cases=[
 ('primary',[100,0],[3,1.5],[2,1.9]),
 ('scaled',[1,0],[3,1.5],[2,1.9]),
 ('three_way',[10,0,0],[5,3.4,2],[3,3.5,2.6]),
 ('multiple_positions',[100,0,50],[2,3,4],[1.9,2.8,3.8]),
 ('docstring_inputs',[100,0],[3,2.5],[3,2.5]),
 ('zero_exposure',[0,0],[3,1.5],[2,1.9]),
]
out=[]
for name,s,o,q in cases:
    r=arb.arbitrage_hedge(s,o,q,hedge_all=False)
    net=payoff(s,o,q,r.practical_hedge_stakes)
    out.append({'name':name,'existing_stakes':s,'existing_odds':o,'hedge_odds':q,'result':dataclasses.asdict(r),'net_payoffs_exact':[str(v) for v in net],'actual_worst':float(min(net)),'reported_matches_cash_ledger':math.isclose(r.guaranteed_profit,float(min(net)),abs_tol=1e-9,rel_tol=1e-12)})
assert out[0]['result']['practical_hedge_stakes']==[100.0,0.0]
assert out[0]['result']['guaranteed_profit']==0
assert out[0]['net_payoffs_exact']==['300','-200']
control=arb.arbitrage_hedge([100,0],[3,1.5],[2,1.9],hedge_all=True)
control_net=payoff([100,0],[3,1.5],[2,1.9],control.practical_hedge_stakes)
assert control.lp_success
assert math.isclose(control.guaranteed_profit,800/19,abs_tol=1e-8)
assert all(math.isclose(float(v),800/19,abs_tol=1e-8) for v in control_net)
exact_hedge=F(3000,19)
exact_net=payoff([100,0],[3,1.5],[2,1.9],[0,exact_hedge])
rounded_net=payoff([100,0],[3,1.5],[2,1.9],[0,'157.89'])
assert exact_net==[F(800,19),F(800,19)]
result={'checked_at':datetime.now(timezone.utc).isoformat(),'upstream_commit':receipt['source_commit'],'python':platform.python_version(),'scipy':scipy.__version__,'execution':'Unchanged upstream module loaded with importlib, bypassing unrelated package __init__; HiGHS threads=1 for one full-hedge control. No installed whole-package claim.','partial_cases':out,'partial_mismatches':sum(not r['reported_matches_cash_ledger'] for r in out),'full_hedge_control':dataclasses.asdict(control),'full_hedge_actual_net':[str(v) for v in control_net],'exact_unrestricted_comparison':{'hedge_on_B':str(exact_hedge),'net_payoffs':[str(v) for v in exact_net],'cent_rounded_B_stake':'157.89','net_with_cent_rounded_stake':[str(v) for v in rounded_net],'limitation':'Unrestricted comparison, not the mandated fix for a mode that forbids new outcomes. Assumes exhaustive mutually exclusive outcomes, no commissions or limits.'},'cpu_seconds':time.process_time()-start,'OMP_NUM_THREADS':os.environ.get('OMP_NUM_THREADS'),'OPENBLAS_NUM_THREADS':os.environ.get('OPENBLAS_NUM_THREADS'),'gpu_used':False}
(P/'evidence/REPLAY.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'partial_cases':len(out),'partial_mismatches':result['partial_mismatches'],'primary_payoffs':out[0]['net_payoffs_exact'],'full_control_profit':control.guaranteed_profit,'exact_comparison':result['exact_unrestricted_comparison'],'cpu_seconds':result['cpu_seconds']}))
