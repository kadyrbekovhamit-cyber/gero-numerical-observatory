"""Development and confirmation measurement for the separate analytic Greek API."""
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import mpmath as mp
from lab.black_scholes_greeks import greeks
from lab.greeks_reference import checked, NAMES, LOG_NAMES
from lab.corpus import arguments

ROOT=Path(__file__).resolve().parents[1]


def number(x):
    return x if math.isfinite(x) else str(x)


def category(x):
    if math.isinf(x):return 'overflow'
    if x==0.:return 'rounded_zero'
    return 'subnormal' if abs(x)<sys.float_info.min else 'normal'


def evaluate_case(case):
    args=arguments(case)
    ref,convergence=checked(args)
    result={**case,'reference_convergence':convergence}
    try:
        got=greeks(*args)
    except Exception as exc:
        return {**result,'exception':f'{type(exc).__name__}: {exc}','all_gates_passed':False}
    raw=asdict(got)
    result['candidate']={k:number(v) if isinstance(v,float) else v for k,v in raw.items()}
    metrics={}
    with mp.workdps(180):
        for name in NAMES:
            exact=ref[name]
            rounded=float(exact)
            value=getattr(got,name)
            expected=category(rounded)
            if math.isinf(rounded):
                passed=value==rounded
                err,rel,limit=None,None,None
            else:
                err=abs(mp.mpf(value)-exact)
                rel=err/abs(exact)
                limit=max(mp.mpf('2e-11')*abs(exact),mp.mpf(math.ulp(rounded)))
                passed=bool(math.isfinite(value) and err<=limit)
            no_spurious_zero=not(value==0 and rounded!=0)
            # Near representability boundaries one-ULP acceptance alone must
            # not silently approve a zero in place of a representable Greek.
            good_range=(got.ranges[name]==expected)
            metrics[name]={'value':number(value),'reference':mp.nstr(exact,45),
                           'rounded_reference':number(rounded),'reference_range':expected,
                           'absolute_error':mp.nstr(err,30) if err is not None else None,
                           'relative_error':mp.nstr(rel,30) if rel is not None else None,
                           'absolute_limit':mp.nstr(limit,30) if limit is not None else None,
                           'value_passed':passed,'range_passed':good_range,
                           'no_unexpected_zero':no_spurious_zero,
                           'passed':passed and good_range and no_spurious_zero}
        for name in LOG_NAMES:
            exact=ref[name]
            value=getattr(got,name)
            limit=max(mp.mpf('2e-11'),32*mp.mpf(math.ulp(float(exact))))
            err=abs(mp.mpf(value)-exact)
            metrics[name]={'value':number(value),'reference':mp.nstr(exact,45),
                           'absolute_error':mp.nstr(err,30),'absolute_limit':mp.nstr(limit,30),
                           'passed':bool(math.isfinite(value) and err<=limit)}
    result['metrics']=metrics
    result['all_gates_passed']=convergence['passed'] and all(x['passed'] for x in metrics.values())
    return result


def summarize(rows):
    result={'cases':len(rows),'passed_cases':sum(r['all_gates_passed'] for r in rows),
            'failed_ids':[r['id'] for r in rows if not r['all_gates_passed']],
            'oracle_convergence_failures':[r['id'] for r in rows if not r['reference_convergence']['passed']],
            'fields':{}}
    with mp.workdps(100):
        for name in NAMES+LOG_NAMES:
            present=[r for r in rows if name in r.get('metrics',{})]
            entry={'passed':sum(r['metrics'][name]['passed'] for r in present),
                   'failed_ids':[r['id'] for r in present if not r['metrics'][name]['passed']]}
            if name in NAMES:
                normal=[r for r in present if r['metrics'][name]['reference_range']=='normal']
                worst=max(normal,key=lambda r:mp.mpf(r['metrics'][name]['relative_error']),default=None)
                entry['reference_ranges']=dict(Counter(r['metrics'][name]['reference_range'] for r in present))
                entry['max_relative_normal']=worst['metrics'][name]['relative_error'] if worst else None
                entry['max_relative_normal_case']=worst['id'] if worst else None
            else:
                worst=max(present,key=lambda r:mp.mpf(r['metrics'][name]['absolute_error']),default=None)
                entry['max_absolute']=worst['metrics'][name]['absolute_error'] if worst else None
                entry['max_absolute_case']=worst['id'] if worst else None
            result['fields'][name]=entry
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--development',action='store_true')
    parser.add_argument('--output',type=Path,required=True)
    opts=parser.parse_args()
    if opts.output.exists():raise FileExistsError('Do not overwrite evidence')
    if opts.development:
        source=[]
        for version in (1,2,3):
            source.extend(json.loads((ROOT/f'evidence/corpus-v{version}.json').read_text())['cases'])
        excluded=[x['id'] for x in source if arguments(x)[2]<=0 or arguments(x)[5]<=0]
        cases=[x for x in source if x['id'] not in excluded]
        freeze=None
    else:
        from lab.corpus_greeks_v1 import verify
        freeze=verify()
        cases=json.loads((ROOT/'evidence/corpus-greeks-v1.json').read_text())['cases']
        excluded=[]
    rows=[evaluate_case(case) for case in cases]
    if not opts.development:verify()
    report={'generated_utc':datetime.now(timezone.utc).isoformat(),
            'status':'development on disclosed price inputs' if opts.development else 'locally frozen new Greek confirmation; all rows retained',
            'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'candidate_sha256':hashlib.sha256((ROOT/'lab/black_scholes_greeks.py').read_bytes()).hexdigest(),
            'excluded_nonpositive_time_or_sigma':excluded,'freeze':freeze,
            'summary':summarize(rows),'rows':rows}
    opts.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report['summary'],indent=2))


if __name__=='__main__':main()
