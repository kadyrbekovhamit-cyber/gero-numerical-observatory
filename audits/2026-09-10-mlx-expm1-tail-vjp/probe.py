"""Small CPU-only probe; actual MLX, analytic derivatives and JVP/VJP parity."""
import cmath
import importlib.metadata
import json
import math
import os
from pathlib import Path
import resource
import time
for name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
resource.setrlimit(resource.RLIMIT_CPU,(30,30))
import mlx.core as mx
mx.set_default_device(mx.cpu)
ROOT=Path(__file__).resolve().parent
def clean(v):
    if isinstance(v,dict): return {k:clean(x) for k,x in v.items()}
    if isinstance(v,list): return [clean(x) for x in v]
    if isinstance(v,complex): return dict(real=clean(v.real),imag=clean(v.imag))
    if isinstance(v,float) and not math.isfinite(v):
        return "NaN" if math.isnan(v) else ("Infinity" if v>0 else "-Infinity")
    return v
start=time.process_time()
rows=[]
for dtype, points in ((mx.float32,[-20.,-18.,-10.,0.,1.]),
                      (mx.float64,[-40.,-20.,-1.,0.,1.])):
    for value in points:
        x=mx.array(value,dtype=dtype)
        _,fw=mx.jvp(mx.expm1,[x],[mx.ones_like(x)])
        _,bw=mx.vjp(mx.expm1,[x],[mx.ones_like(x)])
        rows.append(dict(dtype=str(dtype),input=x.item(),output=mx.expm1(x).item(),
            jvp=fw[0].item(),vjp=bw[0].item(),expected=math.exp(x.item())))
x=mx.array(-20.)
scaled=lambda z: 1e8*mx.expm1(z)
_,j=mx.jvp(scaled,[x],[mx.ones_like(x)])
_,v=mx.vjp(scaled,[x],[mx.ones_like(x)])
scaled_row=dict(input=-20.,jvp=j[0].item(),vjp=v[0].item(),expected=1e8*math.exp(-20.))
try:
    z=mx.array(.5+1j,dtype=mx.complex64)
    _,j=mx.jvp(mx.expm1,[z],[mx.ones_like(z)])
    complex_row=dict(status="accepted",jvp=j[0].item())
except ValueError as exc:
    complex_row=dict(status="unsupported, excluded from audit",message=str(exc))
out=clean(dict(version=importlib.metadata.version("mlx"),device=str(mx.default_device()),
    cases=rows,scaled=scaled_row,complex=complex_row,cpu_seconds=time.process_time()-start))
(ROOT/"wheel-reproduction.json").write_text(json.dumps(out,indent=2,allow_nan=False)+"\n")
print(json.dumps(out,indent=2,allow_nan=False))
