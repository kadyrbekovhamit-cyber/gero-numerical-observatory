"""Tiny CPU checks of smooth product derivatives, including zero inputs."""

import importlib.metadata
import json
import math
import os
from pathlib import Path
import resource

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
resource.setrlimit(resource.RLIMIT_CPU, (30, 30))

import mlx.core as mx

mx.set_default_device(mx.cpu)
ROOT = Path(__file__).resolve().parent


def clean(value):
    if isinstance(value, list):
        return [clean(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def hessian(fn, x):
    return mx.stack([mx.grad(lambda z: mx.grad(fn)(z)[i])(x) for i in range(x.size)])


cases=[]
for values in ([1.,2.,3.],[0.,2.,3.],[1.,0.,3.],[1.,2.,0.],[0.,0.,3.],[0.,0.,0.]):
    x=mx.array(values)
    a,b,c=values
    fn=lambda z: mx.sum(mx.cumprod(z))
    polynomial=lambda z: z[0]+z[0]*z[1]+z[0]*z[1]*z[2]
    try:
        row=dict(input=values,forward=fn(x).item(),gradient=clean(mx.grad(fn)(x).tolist()),
                 hessian=clean(hessian(fn,x).tolist()),
                 explicit_polynomial_hessian=clean(hessian(polynomial,x).tolist()),
                 expected_gradient=[1+b+b*c,a*(1+c),a*b],
                 expected_hessian=[[0,1+c,b],[1+c,0,a],[b,a,0]])
    except Exception as error:
        row=dict(input=values,error=repr(error))
    cases.append(row)
    print(json.dumps(row),flush=True)

(ROOT/'probe-results.json').write_text(json.dumps(dict(mlx=importlib.metadata.version('mlx'),device=str(mx.default_device()),cases=cases),indent=2,allow_nan=False)+'\n')
