"""Installed-wheel evidence, with independent finite differences and formulas."""
import importlib.metadata
import json
import math
import os
from pathlib import Path
import resource
import time
for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS",
          "VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ[k]="1"
resource.setrlimit(resource.RLIMIT_CPU,(30,30))
import mlx.core as mx
mx.set_default_device(mx.cpu)
ROOT=Path(__file__).resolve().parent

def clean(value):
    if isinstance(value,list):return [clean(x) for x in value]
    if isinstance(value,dict):return {k:clean(v) for k,v in value.items()}
    if isinstance(value,float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else ("Infinity" if value>0 else "-Infinity")
    return value


def main():
    start=time.process_time()
    scalar=lambda x:.5*mx.square(mx.logcumsumexp(x.reshape(1))).sum()
    x=mx.array(0.)
    e=1/1024
    one=dict(value=scalar(x).item(),gradient=mx.grad(scalar)(x).item(),
             second=mx.grad(mx.grad(scalar))(x).item(),expected_second=1.,
             finite_curvature=((scalar(x+e)-2*scalar(x)+scalar(x-e))/(e*e)).item())
    x=mx.array([0.,0.])
    target=mx.logcumsumexp(x)
    mx.eval(target)
    loss=lambda z:.5*mx.square(mx.logcumsumexp(z)-target).sum()
    grad=mx.grad(loss)
    hess=mx.stack([mx.grad(lambda z:grad(z)[i])(x) for i in range(2)])
    finite=mx.stack([(grad(x+e*mx.eye(2)[i])-grad(x-e*mx.eye(2)[i]))/(2*e) for i in range(2)])
    def b(cot):
        return mx.vjp(mx.logcumsumexp,[x],[cot])[1][0]
    cot=mx.zeros_like(x)
    cot_jac=mx.stack([mx.grad(lambda c:b(c)[i])(cot) for i in range(2)])
    result=dict(version=importlib.metadata.version("mlx"),device=str(mx.default_device()),
                singleton=one,two_element=dict(
                    target=target.tolist(),loss=loss(x).item(),gradient=grad(x).tolist(),
                    hessian=hess.tolist(),finite_hessian=finite.tolist(),
                    expected_hessian=[[1.25,.25],[.25,.25]],
                    derivative_of_vjp_wrt_cotangent=cot_jac.tolist(),
                    expected_derivative_of_vjp_wrt_cotangent=[[1.,.5],[0.,.5]]),
                cpu_seconds=time.process_time()-start)
    result=clean(result)
    (ROOT/"wheel-reproduction.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=="__main__": main()
