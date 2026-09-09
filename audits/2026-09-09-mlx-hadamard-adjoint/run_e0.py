"""Run the additional measured energy step after build_and_test.py."""
from pathlib import Path
import json, os, subprocess
ROOT=Path(__file__).resolve().parent
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
records=json.loads((ROOT/'build-results.json').read_text())
base=records[0]['command']; idx=base.index('-o')
steps=[]
def run(command,label):
    p=subprocess.run(command,capture_output=True,text=True,timeout=60)
    (ROOT/(label+'.log')).write_text(p.stdout+p.stderr)
    steps.append(dict(label=label,command=command,returncode=p.returncode))
    (ROOT/'e0-build-results.json').write_text(json.dumps(steps,indent=2)+'\n')
    print(label,p.returncode,p.stdout,flush=True)
    return p
assert run(base[:idx]+['-o',str(ROOT/'e0_step.o'),'-c',str(ROOT/'e0_step.cpp')],'compile-e0').returncode==0
for label in ('before','after'):
    command=next(r['command'] for r in records if r['label']=='link-'+label).copy()
    command[1]=str(ROOT/'e0_step.o'); command[-1]=str(ROOT/('e0-'+label))
    assert run(command,'link-e0-'+label).returncode==0
    assert run([command[-1]],'run-e0-'+label).returncode==(1 if label=='before' else 0)
