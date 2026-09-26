"""Post-result portable replay of the exploratory 60-stencil/IV audit."""
import hashlib
import json
import os
from pathlib import Path
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
from lab import audit_sensitivities_v3 as audit
from lab.corpus_v3 import dependency_hashes


def verify_sources():
    root=Path(__file__).resolve().parent
    freeze=json.loads((root/'evidence/freeze-v3.json').read_text())
    for name in ('lab/black_scholes.py','lab/black_scholes_v2.py','lab/black_scholes_v3.py','lab/reference.py'):
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=freeze['source_sha256'][name]:
            raise RuntimeError('Frozen numerical source changed: '+name)
    prior=json.loads((root/'evidence/sensitivity-audit-v3.json').read_text())
    if hashlib.sha256((root/'lab/audit_sensitivities_v3.py').read_bytes()).hexdigest()!=prior['runner_sha256']:
        raise RuntimeError('Original exploratory runner changed')
    if dependency_hashes()!=freeze['mpmath_source_sha256']:
        raise RuntimeError('mpmath dependency differs')
    return freeze


if __name__=='__main__':
    audit.verify=verify_sources
    audit.main()
