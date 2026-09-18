"""Execute pinned real PyLoan variants against an exact Decimal amortization oracle."""
from pathlib import Path
from decimal import Decimal, ROUND_HALF_EVEN
from dataclasses import asdict
from itertools import product
import importlib.util, sys, json, hashlib, platform
B=Path(__file__).resolve().parent
D=Decimal
Q=D('0.01')
def q(x): return x.quantize(Q,rounding=ROUND_HALF_EVEN)
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path/'__init__.py',submodule_search_locations=[str(path)])
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m)
    return m.Loan
paths={'current':B/'source/src/pyloan','release':B/'release-0.7.2/pyloan','prior_release':B/'release-0.7.0/pyloan','candidate':B/'candidate/pyloan','restored':B/'restored/pyloan'}
def output(schedule): return [{k:str(v) for k,v in asdict(p).items()} for p in schedule]
def ref(P,rate,freq,n,pay):
    bal=q(P); result=[]
    for i in range(n):
        if bal<=0: break
        interest=q(bal*rate/D(100)/D(freq))
        payment=min(pay,bal+interest)
        principal=q(payment-interest);bal=q(bal-principal)
        result.append({'payment_amount':str(q(payment)),'interest_amount':str(interest),'principal_amount':str(principal),'loan_balance_amount':str(bal)})
    return result
cases=[]
for P,rate,freq,n,factor,method in product(map(D,['1000','12345.67','100000']),map(D,['1.2','6','12']),[1,2,4,12],[2,6,12],map(D,['0.25','0.75','1.5','2']),['30E/360 ISDA','30E/360']):
    pay=q(P*rate/D(100)/D(freq)+P/D(n)*factor)
    kw=dict(loan_amount=float(P),interest_rate=float(rate),loan_term=n*(12//freq),loan_term_period='M',start_date='2026-01-01',payment_end_of_month=False,annual_payments=freq,payment_amount=float(pay),compounding_method=method)
    cases.append((kw,ref(P,rate,freq,n,pay)))
controls=[]
for typ,method,grace,freq in product(['annuity','linear','interest-only'],['30E/360 ISDA','30E/360','A/360','A/365','A/A'],[0,1],[4,12]):
    controls.append(dict(loan_amount=12000,interest_rate=12,loan_term=2,start_date='2026-01-01',payment_end_of_month=False,annual_payments=freq,loan_type=typ,compounding_method=method,interest_only_period=grace))
summary={}
for mode,path in paths.items():
    Loan=load('pyloan_'+mode,path)
    rows=[]
    for i,(kw,expected) in enumerate(cases):
        actual=output(Loan(**kw).get_payment_schedule())[1:]
        capfail=any(D(x['payment_amount'])>D(str(kw['payment_amount'])) for x in actual)
        mismatch=len(actual)!=len(expected) or any(any(D(x[k])!=D(y[k]) for k in y) for x,y in zip(actual,expected))
        maxdiff=max([abs(D(x[k])-D(y[k])) for x,y in zip(actual,expected) for k in y] or [D(0)])
        rows.append(dict(id=i,inputs=kw,reference=expected,observed=actual,payment_cap_failure=capfail,exact_oracle_mismatch=mismatch,max_cent_difference=str(maxdiff)))
    cp=[]
    for i,kw in enumerate(controls):
        cp.append(dict(id=i,inputs=kw,observed=output(Loan(**kw).get_payment_schedule())))
    (B/'evidence'/f'grid-{mode}.json').write_text(json.dumps(rows,indent=2)+'\n')
    (B/'evidence'/f'controls-{mode}.json').write_text(json.dumps(cp,indent=2)+'\n')
    summary[mode]={'cases':len(rows),'payment_cap_failures':sum(x['payment_cap_failure'] for x in rows),'exact_oracle_mismatches':sum(x['exact_oracle_mismatch'] for x in rows),'controls':len(cp),'grid_sha256':hashlib.sha256((B/'evidence'/f'grid-{mode}.json').read_bytes()).hexdigest()}
summary['environment']={'python':sys.version,'platform':platform.platform(),'precision':__import__('decimal').getcontext().prec,'rounding':str(__import__('decimal').getcontext().rounding)}
for a,b in [('current','release'),('current','restored')]:
    summary[a+'_'+b+'_byte_identical']=(B/'evidence'/f'grid-{a}.json').read_bytes()==(B/'evidence'/f'grid-{b}.json').read_bytes()
summary['candidate_controls_identical']=(B/'evidence/controls-current.json').read_bytes()==(B/'evidence/controls-candidate.json').read_bytes()
(B/'evidence/GRID_RECEIPT.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
