"""Use only synthetic, initialized backing storage to test view isolation."""
import json
import os
from pathlib import Path
import resource
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
import mlx.core as mx
mx.set_default_device(mx.cpu)

rows=[]
for tail in ([101.,202.],[-31.,47.]):
    backing=mx.array([2.]+tail).reshape(3,1,1)
    mx.eval(backing)
    a=backing[:1]
    b=mx.array([3.,5.,7.]).reshape(3,1,1)
    ids=mx.array([0,1,2],dtype=mx.uint32)
    for sorted_indices in (False,True):
        f=lambda b:mx.gather_mm(a,b,rhs_indices=ids,sorted_indices=sorted_indices).sum()
        rows.append(dict(tail=tail,sorted_indices=sorted_indices,
                         visible_input=a.tolist(),loss=f(b).item(),
                         actual_db=mx.grad(f)(b).reshape(3).tolist(),expected_db=[2.,2.,2.]))
report=dict(device=str(mx.default_device()),cases=rows,
            scope='Only synthetic backing arrays in this process; no real data or cross-process testing.')
Path(__file__).with_name('broadcast-tail-results.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
