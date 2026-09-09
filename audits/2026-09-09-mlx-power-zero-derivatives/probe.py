"""Small CPU tests of fixed-exponent polynomial derivatives at zero."""
import importlib.metadata
import json
import math
import os
from pathlib import Path
import resource
import time

for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
resource.setrlimit(resource.RLIMIT_CPU,(30,30))
import mlx.core as mx
mx.set_default_device(mx.cpu)
ROOT=Path(__file__).resolve().parent


def clean(value):
    if isinstance(value,float) and not math.isfinite(value):return str(value)
    if isinstance(value,list):return [clean(x) for x in value]
    if isinstance(value,dict):return {k:clean(v) for k,v in value.items()}
    return value


def main():
    started=time.process_time()
    rows=[]
    for n in range(4):
        fn=lambda x,n=n:mx.power(x,mx.array(float(n)))
        derivatives=[]
        for order in range(5):
            expected=float(math.factorial(n)) if order==n else (1. if n==order==0 else 0.)
            derivatives.append(dict(order=order,actual=fn(mx.array(0.)).item(),expected=expected))
            if order<4:fn=mx.grad(fn)
        rows.append(dict(exponent=n,input=0.,derivatives=derivatives))
    exponents=mx.array([0.,1.,2.])
    coefficients=mx.array([1.,2.,3.])
    model=lambda x:mx.sum(coefficients*mx.power(x,exponents))
    direct=lambda x:1.+2.*x+3.*x*x
    loss=lambda x:0.5*(model(x)-4.)**2
    direct_loss=lambda x:0.5*(direct(x)-4.)**2
    x=mx.array(0.);h=mx.array(1/1024)
    gradient=mx.grad(loss)(x);reference=mx.grad(direct_loss)(x)
    example=dict(model=model(x).item(),model_gradient=mx.grad(model)(x).item(),
                 expected_model_gradient=2.,direct_gradient=mx.grad(direct)(x).item(),
                 model_finite_difference=((model(x+h)-model(x-h))/(2*h)).item(),
                 loss=loss(x).item(),loss_gradient=gradient.item(),expected_loss_gradient=-6.,
                 step_size=0.125,loss_after_step=loss(x-0.125*gradient).item(),
                 loss_after_reference_step=loss(x-0.125*reference).item())
    control=[]
    for a in (0.5,2.):
        for b in (0.,1.,2.5):
            f=lambda x,y:mx.power(x,y)
            ga=mx.grad(f,argnums=0)
            gab=mx.grad(lambda y:ga(mx.array(a),y))(mx.array(b)).item()
            control.append(dict(base=a,exponent=b,mixed_derivative=gab,expected=a**(b-1)*(1+b*math.log(a))))
    report=clean(dict(version=importlib.metadata.version('mlx'),device=str(mx.default_device()),
                      cases=rows,polynomial_example=example,mixed_controls=control,
                      cpu_seconds=time.process_time()-started))
    (ROOT/'probe-results.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,allow_nan=False),flush=True)


if __name__=='__main__':main()
