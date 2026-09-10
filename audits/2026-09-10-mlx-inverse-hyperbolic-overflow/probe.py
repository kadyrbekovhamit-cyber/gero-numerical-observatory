"""Real MLX CPU probe with bounded one-thread synthetic inputs."""
import importlib.metadata
import json
import math
import os
from pathlib import Path
import resource
import time
for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"):os.environ[key]="1"
resource.setrlimit(resource.RLIMIT_CPU,(30,30))
import mlx.core as mx
mx.set_default_device(mx.cpu)
ROOT=Path(__file__).resolve().parent
def clean(v):
    if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
    if isinstance(v,list):return [clean(x) for x in v]
    if isinstance(v,float) and not math.isfinite(v):
        return "NaN" if math.isnan(v) else ("Infinity" if v>0 else "-Infinity")
    return v
start=time.process_time()
rows=[]
for dtype,large in ((mx.float32,1e20),(mx.float64,1e200),
                    (mx.float16,1000.),(mx.bfloat16,1e20)):
    for name in ("arcsinh","arccosh"):
        f=getattr(mx,name)
        near=1.125 if dtype in (mx.float16,mx.bfloat16) else 1.0001
        for value in ([0.,1.,large,-large] if name=="arcsinh" else [near,2.,large]):
            x=mx.array(value,dtype=dtype);q=x.item()
            expected=(1/math.hypot(q,1) if name=="arcsinh" else
                      (1/math.sqrt(q-1))/math.sqrt(q+1))
            _,j=mx.jvp(f,[x],[mx.ones_like(x)])
            _,v=mx.vjp(f,[x],[mx.ones_like(x)])
            rows.append(dict(op=name,dtype=str(dtype),input=q,value=f(x).item(),
                jvp=j[0].item(),vjp=v[0].item(),expected=expected))
compositions=[]
for dtype,power in ((mx.float32,80),(mx.float64,600)):
    c=mx.array(2.**power,dtype=dtype)
    t=mx.array(1.,dtype=dtype)
    for name in ("arcsinh","arccosh"):
        op=getattr(mx,name)
        f=lambda z:op(c*z)
        step=2.**(-6 if dtype==mx.float32 else -10)
        fd=(f(t+step).item()-f(t-step).item())/(2*step)
        compositions.append(dict(op=name,dtype=str(dtype),scale=c.item(),t=1.,
            value=f(t).item(),gradient=mx.grad(f)(t).item(),
            hessian=mx.grad(mx.grad(f))(t).item(),
            third=mx.grad(mx.grad(mx.grad(f)))(t).item(),
            finite_difference=fd,step=step,
            expected_gradient=1.,expected_hessian=-1.,expected_third=2.,
            reference_note="Finite-scale analytic corrections are below the precision shown."))
out=clean(dict(version=importlib.metadata.version("mlx"),device=str(mx.default_device()),
    cases=rows,compositions=compositions,cpu_seconds=time.process_time()-start))
(ROOT/"wheel-reproduction.json").write_text(json.dumps(out,indent=2,allow_nan=False)+"\n")
print(json.dumps(out,indent=2,allow_nan=False))
