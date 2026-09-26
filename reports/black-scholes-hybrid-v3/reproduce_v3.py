"""Portable replay harness added AFTER the original v3 result.

Validate frozen source hashes, build the pinned comparator in a temporary
directory, substitute only verification and binary location, label the replay.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):
    os.environ[key]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''

from lab.corpus_v3 import ROOT,dependency_hashes
from lab.build_jackel import SOURCES
from lab import run_benchmark_v1 as native_adapter
from lab import run_benchmark_v3 as runner

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def verify_sources():
    freeze=json.loads((ROOT/'evidence/freeze-v3.json').read_text())
    for name,value in freeze['source_sha256'].items():
        if name=='build/jackel_cli':
            continue
        if sha(ROOT/name)!=value:
            raise RuntimeError('Frozen source/input changed: '+name)
    native=json.loads((ROOT/'evidence/jackel-build-v1.json').read_text())
    if sha(ROOT/'lab/jackel_cli.cpp')!=native['wrapper_sha256']:
        raise RuntimeError('Pinned native wrapper changed')
    if dependency_hashes()!=freeze['mpmath_source_sha256']:
        raise RuntimeError('Different mpmath source tree; explicitly audit this dependency change')
    return freeze

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Choose a fresh replay output')
    verify_sources()
    with tempfile.TemporaryDirectory(prefix='black-scholes-v3-replay-') as tmp:
        base=Path(tmp)
        out=base/'build'
        out.mkdir()
        compiler=os.environ.get('CXX','clang++')
        flags=['-std=c++11','-O2','-DNDEBUG','-ffp-contract=off']
        objects=[]
        for source in [ROOT/'vendor/jackel'/s for s in SOURCES]+[ROOT/'lab/jackel_cli.cpp']:
            obj=out/(source.stem+'.o')
            subprocess.run([compiler,*flags,'-c',str(source),'-o',str(obj)],check=True)
            objects.append(str(obj))
        binary=out/'jackel_cli'
        subprocess.run([compiler,*objects,'-o',str(binary)],check=True)
        environment=json.loads(subprocess.check_output([str(binary),'--environment'],text=True))
        if not all(environment.values()):
            raise RuntimeError('Unsupported native floating environment')
        native_adapter.ROOT=base
        runner.verify=verify_sources
        report=runner.run()
        report['status']='Post-result portable replay; original frozen algorithm/oracle/corpus unchanged'
        report['portable_replay']={
            'harness_sha256':sha(Path(__file__)),
            'compiler':subprocess.check_output([compiler,'--version'],text=True),
            'flags':flags,'native_binary_sha256':sha(binary),'floating_environment':environment,
            'changes':'Fresh native build location and source-only verifier; no numerical algorithm/oracle/corpus changes',
            'original_receipt_retained':True}
        args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        print(json.dumps({'cases':report['summary']['cases'],
                          'passed':report['summary']['all_v3_practical_gates_passed'],
                          'output':str(args.output)},indent=2))

if __name__=='__main__':
    main()
