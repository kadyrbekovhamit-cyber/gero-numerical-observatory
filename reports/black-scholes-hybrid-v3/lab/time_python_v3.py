"""Predeclared small sequential timing, executed only after accuracy evidence."""
from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time
from lab.black_scholes import stable_detailed
from lab.black_scholes_v2 import prices as v2
from lab.black_scholes_v3 import prices as v3
from lab.corpus import arguments
from lab.corpus_v3 import ROOT,verify

def measure(cases):
    args=[arguments(c) for c in cases]
    functions={'v0':stable_detailed,'v2':v2,'v3':v3}
    samples={key:[] for key in functions}
    for fn in functions.values():
        for a in args:
            fn(*a)
    for repeat in range(7):
        order=list(functions) if repeat%2==0 else list(reversed(functions))
        for key in order:
            start=time.perf_counter_ns()
            for _ in range(5):
                for a in args:
                    functions[key](*a)
            samples[key].append((time.perf_counter_ns()-start)/(5*len(args)))
    medians={key:statistics.median(v) for key,v in samples.items()}
    return {'cases':len(args),'raw_ns_per_call':samples,'median_ns_per_call':medians,
            'v2_over_v3':medians['v2']/medians['v3'],'v3_over_v0':medians['v3']/medians['v0']}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT/'evidence/python-timing-v3.json')
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Never overwrite timing evidence')
    freeze=verify()
    source=ROOT/'evidence/benchmark-v3.json'
    accuracy=json.loads(source.read_text())
    cases=json.loads((ROOT/'evidence/corpus-v3.json').read_text())['cases']
    fallback_ids={r['id'] for r in accuracy['rows'] if r.get('details',{}).get('v3',{}).get('method')=='v2_fallback'}
    groups={'all':cases}
    groups.update({f:[c for c in cases if c['family']==f] for f in sorted({c['family'] for c in cases})})
    if fallback_ids:
        groups['fallback_selected']=[c for c in cases if c['id'] in fallback_ids]
    result={'generated_utc':datetime.now(timezone.utc).isoformat(),
            'status':'descriptive timing of detailed Python APIs; no equal-ULP accuracy claim',
            'environment':freeze['environment'],'accuracy_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'warmup_batches':1,'timed_batches':7,'sequential_loops_per_batch':5,'alternating_order':True,
            'groups':{name:measure(group) for name,group in groups.items()},
            'limitations':['Single machine; synthetic strata, not market frequencies',
                           'Fallback subset selected by candidate path; overhead is descriptive',
                           'No native C++ timing or significance claim']}
    result['predeclared_success']=accuracy['summary']['all_v3_practical_gates_passed'] and result['groups']['all']['v2_over_v3']>2
    verify()
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'predeclared_success':result['predeclared_success'],
                      'groups':{k:{x:y for x,y in v.items() if x!='raw_ns_per_call'} for k,v in result['groups'].items()}},indent=2))

if __name__=='__main__':
    main()
