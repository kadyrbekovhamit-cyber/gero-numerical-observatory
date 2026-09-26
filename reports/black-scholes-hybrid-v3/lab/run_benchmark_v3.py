"""Confirmation of a fully locally frozen v3 pipeline; preserve all failures."""
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import argparse
import json
import math
from pathlib import Path
import mpmath as mp
from lab.black_scholes import stable_prices
from lab.black_scholes_v2 import prices as v2_prices
from lab.black_scholes_v3 import prices as v3_prices
from lab.corpus import arguments
from lab.corpus_v3 import ROOT, verify
from lab.metrics import log_error, price_error, round_binary64, rounding_class, text
from lab.reference import checked, integral_log_otm
from lab.run_benchmark_v1 import comparison, native_batch, safe

INTEGRAL_INDICES=(0,47,95,135,175,215,255,303,351,391,431,479)
METHODS=('v0','v2','v3','jackel_native')
GATES=('price','log','finite_nonnegative_both_legs','correct_otm_side','no_unexpected_zero')

def summarize(rows):
    good=[r for r in rows if 'oracle_failure' not in r]
    nonzero=[r for r in good if r['rounding_class'] in ('normal','subnormal')]
    result={'cases':len(rows),'oracle_failures':len(rows)-len(good),
            'rounding_classes':dict(Counter(r['rounding_class'] for r in good)),
            'oracle_precision_increased':sum(r['convergence']['precision_increased'] for r in good),
            'integral_checks':sum('integral' in r for r in good),
            'integral_failures':sum(not r['integral']['passed'] for r in good if 'integral' in r),
            'branches':dict(Counter(r['details']['v3'].get('method','exception') for r in good)),
            'gates':{},'methods':{},
            'comparisons_nonzero':{name:dict(Counter(r['comparisons'][name] for r in nonzero)) for name in ('v0','v2','jackel_native')}}
    for version in ('v2','v3'):
        result['gates'][version]={gate:{'passed':sum(r['gates'][version][gate] for r in good),
                                      'failed_ids':[r['id'] for r in good if not r['gates'][version][gate]]} for gate in GATES}
    with mp.workdps(180):
        for name in METHODS:
            finite=[r for r in nonzero if r['errors'][name]['finite']]
            normal=[r for r in finite if r['rounding_class']=='normal']
            worst=max(finite,key=lambda r:mp.mpf(r['errors'][name]['relative']),default=None)
            worst_normal=max(normal,key=lambda r:mp.mpf(r['errors'][name]['relative']),default=None)
            result['methods'][name]={
                'nonfinite':sum(not r['errors'][name]['finite'] for r in good),
                'negative':sum(r['errors'][name]['finite'] and r['errors'][name]['value']<0 for r in good),
                'unexpected_zero':sum(r['errors'][name]['value']==0 for r in nonzero),
                'rounding_matches_nonzero':sum(r['errors'][name]['rounded_ulp_distance']==0 for r in finite),
                'max_relative_nonzero':worst['errors'][name]['relative'] if worst else None,
                'max_relative_case':worst['id'] if worst else None,
                'max_relative_normal':worst_normal['errors'][name]['relative'] if worst_normal else None,
                'max_relative_normal_case':worst_normal['id'] if worst_normal else None}
        logs=[r for r in good if r['logs']['v3']['finite'] and r['gates']['v3']['correct_otm_side']]
        worst_abs=max(logs,key=lambda r:mp.mpf(r['logs']['v3']['absolute']),default=None)
        worst_ulp=max(logs,key=lambda r:mp.mpf(r['logs']['v3']['real_ulp_error']),default=None)
        result['v3_log']={'max_absolute':worst_abs['logs']['v3']['absolute'] if worst_abs else None,
                          'max_absolute_case':worst_abs['id'] if worst_abs else None,
                          'max_real_ulp':worst_ulp['logs']['v3']['real_ulp_error'] if worst_ulp else None,
                          'max_real_ulp_case':worst_ulp['id'] if worst_ulp else None}
    result['all_v3_practical_gates_passed']=not(result['oracle_failures'] or result['integral_failures']) and all(not g['failed_ids'] for g in result['gates']['v3'].values())
    return result

def run():
    freeze=verify()
    cases=json.loads((ROOT/'evidence/corpus-v3.json').read_text())['cases']
    mappings,native=native_batch(cases)
    rows=[]
    for index,case in enumerate(cases):
        args=arguments(case)
        try:
            ref,convergence=checked(*args)
            if not convergence['passed']:
                raise ArithmeticError('reference precision did not converge')
        except Exception as exc:
            rows.append({**case,'oracle_failure':f'{type(exc).__name__}: {exc}'})
            continue
        category=rounding_class(ref.otm)
        with mp.workdps(ref.dps):
            price_limit=max(mp.mpf('1e-10')*ref.otm,mp.mpf(math.ulp(round_binary64(ref.otm))))
            log_limit=max(mp.mpf('2e-11'),16*mp.mpf(math.ulp(float(ref.log_otm))))
        details,logs,gates,values={},{},{},{}
        for name,fn in (('v2',v2_prices),('v3',v3_prices)):
            try:
                got=fn(*args)
                details[name]={k:safe(v) if isinstance(v,float) else v for k,v in asdict(got).items()}
                values[name]=getattr(got,ref.option)
                logs[name]=log_error(got.log_otm_value,ref.log_otm,dps=ref.dps)
                side=got.otm_option==ref.option
                finite_both=all(math.isfinite(v) and v>=0 for v in (got.call,got.put))
            except Exception as exc:
                details[name]={'exception':f'{type(exc).__name__}: {exc}'}
                values[name]=math.nan
                logs[name]=log_error(math.nan,ref.log_otm)
                side,finite_both=False,False
            with mp.workdps(ref.dps):
                gates[name]={
                    'price':bool(math.isfinite(values[name]) and abs(mp.mpf(values[name])-ref.otm)<=price_limit),
                    'log':bool(side and logs[name]['finite'] and mp.mpf(logs[name]['absolute'])<=log_limit),
                    'finite_nonnegative_both_legs':finite_both,'correct_otm_side':side,
                    'no_unexpected_zero':math.isfinite(values[name]) and not(values[name]==0 and category in ('normal','subnormal'))}
            logs[name]['comparable_same_leg']=side
        try:
            values['v0']=getattr(stable_prices(*args),ref.option)
        except Exception:
            values['v0']=math.nan
        forward,discount=mappings[index]
        values['jackel_native']=native[index][0 if ref.option=='call' else 1]*discount
        row={**case,'reference_otm':text(ref.otm),'reference_log_otm':text(ref.log_otm),
             'reference_round_binary64_hex':round_binary64(ref.otm).hex(),
             'otm_option':ref.option,'reference_actual_m':text(ref.m),'rounding_class':category,
             'convergence':convergence,'details':details,'logs':logs,'gates':gates,
             'gate_limits':{'price_absolute':text(price_limit),'log_absolute':text(log_limit)},
             'errors':{name:price_error(value,ref.otm,dps=ref.dps) for name,value in values.items()},
             'comparisons':{name:comparison(values['v3'],values[name],ref.otm) for name in ('v0','v2','jackel_native')},
             'jackel_adapter':{'forward_hex':forward.hex(),'discount_hex':discount.hex()}}
        if args[3]!=0 or args[4]!=0:
            mapped,mcheck=checked(forward,args[1],args[2],0.,0.,args[5])
            with mp.workdps(max(ref.dps,mapped.dps)):
                mvalue=getattr(mapped,ref.option)*mp.mpf(discount)
                row['jackel_adapter'].update({'mapped_oracle_passed':mcheck['passed'],
                    'mapping_absolute_error':text(abs(mvalue-ref.otm)),
                    'mapping_relative_error':text(abs(mvalue-ref.otm)/ref.otm),
                    'evaluation_after_mapping':price_error(values['jackel_native'],mvalue,dps=max(ref.dps,mapped.dps))})
        if index in INTEGRAL_INDICES:
            try:
                val=integral_log_otm(*args,dps=100)
                with mp.workdps(ref.dps):
                    delta=abs(val-ref.log_otm)
                    row['integral']={'dps':100,'log_absolute_difference':text(delta),'passed':bool(delta<=mp.mpf('1e-50'))}
            except Exception as exc:
                row['integral']={'passed':False,'exception':str(exc)}
        rows.append(row)
        if (index+1)%80==0:
            print(f'Completed {index+1}/{len(cases)}',flush=True)
    verify()
    return {'schema':3,'generated_utc':datetime.now(timezone.utc).isoformat(),
            'status':'frozen confirmation; all cases retained; no post-result tuning',
            'freeze':freeze,'integral_indices':INTEGRAL_INDICES,'summary':summarize(rows),
            'by_family':{f:summarize([r for r in rows if r['family']==f]) for f in sorted({r['family'] for r in rows})},
            'rows':rows}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT/'evidence/benchmark-v3.json')
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Choose a fresh output path')
    report=run()
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report['summary'],indent=2))

if __name__=='__main__':
    main()
