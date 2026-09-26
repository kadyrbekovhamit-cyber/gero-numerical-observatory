"""Portable post-result replay of the frozen Greek experiment.

The original receipt remains unchanged. This harness replaces only local
verification (no native price comparator is used by this experiment).
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):
    os.environ[key]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''

from lab.corpus_greeks_v1 import ROOT,dependency_hashes
from lab.run_greeks_v1 import evaluate_case,summarize


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_sources():
    freeze=json.loads((ROOT/'evidence/freeze-greeks-v1.json').read_text())
    for name,expected in freeze['source_sha256'].items():
        if sha(ROOT/name)!=expected:raise RuntimeError('Frozen source/input changed: '+name)
    if dependency_hashes()!=freeze['mpmath_source_sha256']:
        raise RuntimeError('Different mpmath source tree; audit the dependency change explicitly')
    return freeze


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    out=p.parse_args().output
    if out.exists():raise FileExistsError('Use a fresh replay output path')
    freeze=verify_sources()
    cases=json.loads((ROOT/'evidence/corpus-greeks-v1.json').read_text())['cases']
    rows=[evaluate_case(case) for case in cases]
    verify_sources()
    report={'generated_utc':datetime.now(timezone.utc).isoformat(),
            'status':'post-result portable replay; original numeric code, oracle, inputs and metrics unchanged',
            'freeze':freeze,'harness_sha256':sha(Path(__file__)),
            'current_environment':{'python':sys.version,'platform':platform.platform()},
            'summary':summarize(rows),'rows':rows}
    out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'cases':len(rows),'passed':report['summary']['passed_cases'],'output':str(out)},indent=2))


if __name__=='__main__':main()
