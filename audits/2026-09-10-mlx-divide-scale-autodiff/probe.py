"""Small synthetic CPU-only MLX probe; one numerical thread."""
import importlib.metadata
import json
import math
import os
from pathlib import Path
import resource
import time
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
resource.setrlimit(resource.RLIMIT_CPU, (30, 30))
import mlx.core as mx
mx.set_default_device(mx.cpu)
ROOT = Path(__file__).resolve().parent
start = time.process_time()
def clean(x):
    if isinstance(x, dict): return {k:clean(v) for k,v in x.items()}
    if isinstance(x, (tuple,list)): return [clean(v) for v in x]
    if isinstance(x,float) and not math.isfinite(x):
        return "NaN" if math.isnan(x) else "Infinity" if x>0 else "-Infinity"
    return x
cases, compositions = [], []
for dtype, powers in ((mx.float16,(-14,10)),(mx.bfloat16,(-80,80)),
                      (mx.float32,(-80,80)),(mx.float64,(-600,600))):
    for power in (0,*powers):
        c=mx.array(math.ldexp(1.,power),dtype=dtype)
        one=mx.array(1.,dtype=dtype)
        _,v=mx.vjp(mx.divide,[c,c],[one])
        _,j=mx.jvp(lambda b:mx.divide(c,b),[c],[one])
        _,radial=mx.jvp(mx.divide,[c,c],[c,c])
        cases.append(dict(dtype=str(dtype),scale=c.item(),value=(c/c).item(),
            vjp=[x.item() for x in v],jvp_denominator=j[0].item(),
            radial_jvp=radial[0].item(),expected_vjp=[1/c.item(),-1/c.item()],
            expected_radial=0.))
        if dtype not in (mx.float32,mx.float64):continue
        f=lambda t:mx.divide(c,c*t)
        h=2.**(-6 if dtype==mx.float32 else -10)
        fd=(f(one+h).item()-f(one-h).item())/(2*h)
        compositions.append(dict(dtype=str(dtype),scale=c.item(),value=f(one).item(),
            gradient=mx.grad(f)(one).item(),hessian=mx.grad(mx.grad(f))(one).item(),
            third=mx.grad(mx.grad(mx.grad(f)))(one).item(),finite_difference=fd,
            step=h,expected_gradient=-1.,expected_hessian=2.,expected_third=-6.))
out=clean(dict(version=importlib.metadata.version("mlx"),device=str(mx.default_device()),
    cases=cases,compositions=compositions,cpu_seconds=time.process_time()-start))
(ROOT/"wheel-reproduction.json").write_text(json.dumps(out,indent=2,allow_nan=False)+"\n")
print(json.dumps(out,indent=2,allow_nan=False))
