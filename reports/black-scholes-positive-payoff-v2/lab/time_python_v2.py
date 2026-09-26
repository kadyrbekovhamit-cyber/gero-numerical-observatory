"""Small post-accuracy timing of the two Python detailed-price APIs only.

Not an equal-accuracy benchmark and not a speed comparison with native Jaeckel.
"""
import hashlib
import json
from pathlib import Path
import platform
import statistics
import time

from lab.black_scholes import stable_detailed
from lab.black_scholes_v2 import prices
from lab.corpus import arguments

ROOT=Path(__file__).resolve().parents[1]


def main():
    output=ROOT/'evidence/python-timing-v2.json'
    if output.exists():
        raise FileExistsError('Do not overwrite timing evidence')
    cases=json.loads((ROOT/'evidence/corpus-v1.json').read_text())['cases'][::3]
    args=[arguments(c) for c in cases]
    functions={'v0_detailed':stable_detailed,'v2_detailed':prices}
    times={key:[] for key in functions}
    for function in functions.values():
        for values in args:
            function(*values)
    for repeat in range(5):
        order=list(functions) if repeat%2==0 else list(reversed(functions))
        for name in order:
            start=time.perf_counter_ns()
            for values in args:
                functions[name](*values)
            times[name].append((time.perf_counter_ns()-start)/len(args))
    result={'status':'post-accuracy descriptive timing; unequal achieved accuracy',
            'platform':platform.platform(),'cases':len(cases),'selection':'all v1 fixtures at indices 0,3,6,...',
            'warmup_batches_per_method':1,'timed_batches_per_method':5,
            'alternating_method_order':True,'nanoseconds_per_call':times,
            'median_nanoseconds_per_call':{key:statistics.median(values) for key,values in times.items()},
            'candidate_sha256':hashlib.sha256((ROOT/'lab/black_scholes_v2.py').read_bytes()).hexdigest(),
            'limitations':'Python API costs on this machine; no native C++ timing; no equal-accuracy or production throughput claim'}
    result['median_v2_over_v0']=result['median_nanoseconds_per_call']['v2_detailed']/result['median_nanoseconds_per_call']['v0_detailed']
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
