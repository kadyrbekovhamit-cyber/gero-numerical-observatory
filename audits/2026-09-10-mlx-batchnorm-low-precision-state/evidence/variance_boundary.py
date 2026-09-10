"""Even a representable biased variance is lost by summing before averaging."""
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import time
for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[key]="1"
resource.setrlimit(resource.RLIMIT_CPU,(30,30))
import mlx.core as mx
mx.set_default_device(mx.cpu)
ROOT=Path(__file__).resolve().parent
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,ROOT/path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.BatchNorm
started=time.process_time()
rows=[]
for mode,cls in [("before",load("boundary_before","upstream-python_mlx_nn_layers_normalization.py")),
                 ("after",load("boundary_after","patched-normalization.py"))]:
    bn=cls(1,momentum=.125,affine=False)
    first=bn(mx.array([[-192.],[192.]],dtype=mx.float16))
    first_state=bn.running_var.item()
    second=bn(mx.array([[-1.],[1.]],dtype=mx.float16))
    next_state=bn.running_var.item()
    bn.eval()
    evaluated=bn(mx.array([[-1.],[1.]],dtype=mx.float16))
    expected_first_state=.875+.125*2*192**2
    expected_next_state=.875*expected_first_state+.25
    expected_eval=[-1/math.sqrt(expected_next_state+1e-5),1/math.sqrt(expected_next_state+1e-5)]
    checks=dict(first_training=all(abs(a-b)<.002 for a,b in zip(first.reshape(-1).tolist(),[-1.,1.])),
                second_training=all(abs(a-b)<.002 for a,b in zip(second.reshape(-1).tolist(),[-1.,1.])),
                first_state=first_state==expected_first_state,
                second_state=next_state==expected_next_state,
                eval=all(math.isclose(a,b,rel_tol=3e-6) for a,b in zip(evaluated.reshape(-1).tolist(),expected_eval)))
    rows.append(dict(mode=mode,first_training=first.tolist(),first_state=first_state,
                     second_training=second.tolist(),second_state=next_state,eval_output=evaluated.tolist(),
                     expected_first_state=expected_first_state,expected_second_state=expected_next_state,
                     expected_eval=expected_eval,checks=checks))
def clean(x):
    if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [clean(v) for v in x]
    if isinstance(x,float) and not math.isfinite(x):return "NaN" if math.isnan(x) else "Infinity"
    return x
x=mx.array([-192.,192.],dtype=mx.float16)
squares=mx.square(x-mx.mean(x))
direct=dict(squares=squares.tolist(),sum_squares=mx.sum(squares).item(),
            biased_variance=mx.var(x).item(),expected_biased_variance=36864.,
            note="Sum overflows before division by two; training also fails.")
out=clean(dict(device=str(mx.default_device()),cpu_seconds=time.process_time()-started,cases=rows,direct_statistics=direct))
(ROOT/"variance-boundary-results.json").write_text(json.dumps(out,indent=2,allow_nan=False)+"\n")
print(json.dumps(out,indent=2))
assert all(rows[1]['checks'].values())
