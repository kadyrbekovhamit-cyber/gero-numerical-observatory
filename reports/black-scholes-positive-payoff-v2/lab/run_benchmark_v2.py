"""Evaluate frozen v2 once, retain failures, and report predeclared gates."""
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import sys

import mpmath as mp

from lab.black_scholes import stable_prices
from lab.black_scholes_v2 import prices
from lab.corpus import arguments
from lab.metrics import log_error, price_error, round_binary64, rounding_class, text
from lab.reference import checked, integral_log_otm
from lab.run_benchmark_v1 import comparison, native_batch, safe

ROOT=Path(__file__).resolve().parents[1]
INTEGRAL_INDICES=(0,31,63,95,127,159,191,223,255,287)
METHODS=('candidate_v0','candidate_v2','jackel_native')


def summarize(rows):
    valid=[r for r in rows if 'oracle_failure' not in r]
    nonzero=[r for r in valid if r['rounding_class'] in ('normal','subnormal')]
    result={'cases':len(rows),'oracle_failures':len(rows)-len(valid),
            'rounding_classes':dict(Counter(r['rounding_class'] for r in valid)),
            'oracle_precision_increased':sum(r['convergence']['precision_increased'] for r in valid),
            'integral_checks':sum('integral' in r for r in valid),
            'integral_failures':sum(not r['integral']['passed'] for r in valid if 'integral' in r),
            'v2_exceptions':sum('exception' in r['v2'] for r in valid),
            'v2_orientation_mismatches':sum(not r['gates']['correct_otm_side'] for r in valid),
            'gates':{gate:{'passed':sum(r['gates'][gate] for r in valid),
                           'failed_ids':[r['id'] for r in valid if not r['gates'][gate]]}
                     for gate in ('price','log','finite_nonnegative_both_legs','correct_otm_side','no_unexpected_zero')},
            'comparisons_nonzero':{},'methods':{}}
    for other in ('candidate_v0','jackel_native'):
        result['comparisons_nonzero'][other]=dict(Counter(r['comparisons'][other] for r in nonzero))
    with mp.workdps(180):
        for method in METHODS:
            finite=[r for r in nonzero if r['methods'][method]['finite']]
            normal=[r for r in finite if r['rounding_class']=='normal']
            worst=max(finite,key=lambda r:mp.mpf(r['methods'][method]['relative']),default=None)
            worst_normal=max(normal,key=lambda r:mp.mpf(r['methods'][method]['relative']),default=None)
            result['methods'][method]={
                'nonfinite':sum(not r['methods'][method]['finite'] for r in valid),
                'negative':sum(r['methods'][method]['finite'] and r['methods'][method]['value']<0 for r in valid),
                'unexpected_zero':sum(r['methods'][method]['value']==0 for r in nonzero),
                'max_relative_nonzero':worst['methods'][method]['relative'] if worst else None,
                'max_relative_case':worst['id'] if worst else None,
                'max_relative_normal':worst_normal['methods'][method]['relative'] if worst_normal else None,
                'max_relative_normal_case':worst_normal['id'] if worst_normal else None,
                'rounding_matches_nonzero':sum(r['methods'][method]['rounded_ulp_distance']==0 for r in nonzero)}
        logs=[r for r in valid if r['v2_log']['finite'] and r['gates']['correct_otm_side']]
        worst_log=max(logs,key=lambda r:mp.mpf(r['v2_log']['real_ulp_error']),default=None)
        result['v2_log']={'comparable':len(logs),
                          'max_real_ulp':worst_log['v2_log']['real_ulp_error'] if worst_log else None,
                          'max_real_ulp_case':worst_log['id'] if worst_log else None}
    result['max_quadrature_evaluations']=max((r['v2'].get('quadrature_evaluations',0) for r in valid),default=0)
    result['all_practical_gates_passed']=(not result['oracle_failures'] and not result['integral_failures']
                                        and not result['v2_exceptions']
                                        and all(not v['failed_ids'] for v in result['gates'].values()))
    return result


def run():
    freeze=json.loads((ROOT/'evidence/freeze-v2.json').read_text())
    corpus=ROOT/'evidence/corpus-v2.json'
    if hashlib.sha256(corpus.read_bytes()).hexdigest()!=freeze['corpus_sha256']:
        raise RuntimeError('Corpus changed after freeze')
    for name,digest in freeze['source_sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:
            raise RuntimeError('Source changed after freeze: '+name)
    native_manifest=json.loads((ROOT/'evidence/jackel-build-v1.json').read_text())
    if hashlib.sha256((ROOT/'build/jackel_cli').read_bytes()).hexdigest()!=native_manifest['binary_sha256']:
        raise RuntimeError('Native binary changed')
    for name,digest in native_manifest['source_sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:
            raise RuntimeError('Native source changed')
    cases=json.loads(corpus.read_text())['cases']
    mappings,native=native_batch(cases)
    rows=[]
    for index,case in enumerate(cases):
        args=arguments(case)
        try:
            ref,convergence=checked(*args)
            if not convergence['passed']:
                raise ArithmeticError('reference precision did not converge')
        except Exception as exc:
            rows.append({**case,'oracle_failure':{'type':type(exc).__name__,'message':str(exc)}})
            continue
        try:
            candidate=prices(*args)
            v2={key:safe(value) if isinstance(value,float) else value for key,value in asdict(candidate).items()}
            val_v2=getattr(candidate,ref.option)
            log=log_error(candidate.log_otm_value,ref.log_otm,dps=ref.dps)
            correct_side=candidate.otm_option==ref.option
            finite_both=all(math.isfinite(x) and x>=0 for x in (candidate.call,candidate.put))
        except Exception as exc:
            v2={'exception':{'type':type(exc).__name__,'message':str(exc)}}
            val_v2=math.nan
            log=log_error(math.nan,ref.log_otm)
            correct_side,finite_both=False,False
        try:
            v0=stable_prices(*args)
            val_v0=getattr(v0,ref.option)
        except Exception:
            val_v0=math.nan
        fwd,discount=mappings[index]
        val_native=native[index][0 if ref.option=='call' else 1]*discount
        vals={'candidate_v0':val_v0,'candidate_v2':val_v2,'jackel_native':val_native}
        errs={method:price_error(value,ref.otm,dps=ref.dps) for method,value in vals.items()}
        with mp.workdps(ref.dps):
            price_limit=max(mp.mpf('1e-10')*ref.otm,mp.mpf(math.ulp(round_binary64(ref.otm))))
            log_limit=max(mp.mpf('2e-11'),16*mp.mpf(math.ulp(float(ref.log_otm))))
            pg=bool(math.isfinite(val_v2) and abs(mp.mpf(val_v2)-ref.otm)<=price_limit)
            lg=bool(correct_side and log['finite'] and mp.mpf(log['absolute'])<=log_limit)
        category=rounding_class(ref.otm)
        row={**case,'reference_otm':text(ref.otm),'reference_log_otm':text(ref.log_otm),
             'reference_round_binary64_hex':round_binary64(ref.otm).hex(),'otm_option':ref.option,
             'reference_actual_m':text(ref.m),'rounding_class':category,'convergence':convergence,
             'methods':errs,'v2':v2,'v2_log':log,
             'comparisons':{other:comparison(val_v2,vals[other],ref.otm) for other in ('candidate_v0','jackel_native')},
             'gates':{'price':pg,'log':lg,'finite_nonnegative_both_legs':finite_both,
                      'correct_otm_side':correct_side,
                      'no_unexpected_zero':math.isfinite(val_v2) and not(val_v2==0 and category in ('normal','subnormal'))},
             'gate_limits':{'price_absolute':text(price_limit),'log_absolute':text(log_limit)},
             'jackel_adapter':{'forward_hex':fwd.hex(),'discount_hex':discount.hex()}}
        if not correct_side:
            row['v2_log']['comparable_same_leg']=False
            row['v2_log']['cross_leg_contract_discrepancy']={key:log[key] for key in ('absolute','real_ulp_error')}
            row['v2_log']['absolute']=None
            row['v2_log']['real_ulp_error']=None
        else:
            row['v2_log']['comparable_same_leg']=True
        if args[3]!=0 or args[4]!=0:
            mapped,mapped_check=checked(fwd,args[1],args[2],0.,0.,args[5])
            with mp.workdps(max(ref.dps,mapped.dps)):
                mapped_value=getattr(mapped,ref.option)*mp.mpf(discount)
                row['jackel_adapter'].update({'mapped_oracle_passed':mapped_check['passed'],
                                              'mapping_absolute_error':text(abs(mapped_value-ref.otm)),
                                              'mapping_relative_error':text(abs(mapped_value-ref.otm)/ref.otm),
                                              'evaluation_after_mapping':price_error(val_native,mapped_value)})
        if index in INTEGRAL_INDICES:
            check=integral_log_otm(*args,dps=100)
            with mp.workdps(ref.dps):
                delta=abs(check-ref.log_otm)
                row['integral']={'dps':100,'log_absolute_difference':text(delta),
                                 'passed':bool(delta<=mp.mpf('1e-50'))}
        rows.append(row)
    return {'schema':2,'generated_utc':datetime.now(timezone.utc).isoformat(),
            'status':'frozen candidate confirmation; no post-result tuning',
            'freeze':freeze,'environment':{'python':sys.version,'mpmath':mp.__version__,'platform':platform.platform()},
            'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'native_build':native_manifest,'integral_indices':INTEGRAL_INDICES,
            'limitations':['Empirical converged oracle, not interval-certified',
                           'Quadrature is also used by a supplementary oracle; primary reference is erfc formula',
                           'Old v1 corpus is development data; no novelty or market-accuracy claim',
                           'Historical pinned Jaeckel; end-to-end, not isolated-kernel comparison',
                           'No equal-accuracy performance comparison'],
            'summary':summarize(rows),
            'by_family':{family:summarize([r for r in rows if r['family']==family]) for family in sorted({r['family'] for r in rows})},
            'rows':rows}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT/'evidence/benchmark-v2.json')
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Choose a fresh output path')
    report=run()
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report['summary'],indent=2))


if __name__=='__main__':
    main()
