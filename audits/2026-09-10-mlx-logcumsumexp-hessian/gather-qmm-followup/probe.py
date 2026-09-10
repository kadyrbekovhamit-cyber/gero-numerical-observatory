"""Small synthetic probes; CPU only, one computational thread."""
import importlib.metadata
import json
import math
import os
from pathlib import Path
import resource
import time
for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS",
          "VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[k] = "1"
resource.setrlimit(resource.RLIMIT_CPU, (30,30))
import mlx.core as mx
mx.set_default_device(mx.cpu)
ROOT = Path(__file__).resolve().parent


def clean(x):
    if isinstance(x,list): return [clean(v) for v in x]
    if isinstance(x,dict): return {k:clean(v) for k,v in x.items()}
    return str(x) if isinstance(x,float) and not math.isfinite(x) else x


def hessian(f,x):
    return mx.stack([mx.grad(lambda z: mx.grad(f)(z)[i])(x) for i in range(x.size)])


def qmm():
    rows = []
    for gathered in (True,False):
        for transpose in (True,False):
            x = mx.array([1.]+[0.]*31).reshape(1,1,32)
            s,b = mx.ones((1,32,1)),mx.zeros((1,32,1))
            w = mx.full((1,32,4),0x11111111,dtype=mx.uint32)
            ids = mx.array([0],dtype=mx.uint32)
            ux = x
            us = x.reshape(s.shape)
            if not gathered:
                w,s,b,us = w[0],s[0],b[0],us[0]
            def f(x,s,b):
                if gathered:
                    return mx.gather_qmm(x,w,s,b,rhs_indices=ids,
                        transpose=transpose,group_size=32,bits=4).sum()
                return mx.quantized_matmul(x,w,s,b,transpose=transpose,group_size=32,bits=4).sum()
            for param in (1,2):
                try:
                    def gx(q):
                        return (mx.grad(f,argnums=0)(x,q,b)*ux).sum() if param==1 else (
                            mx.grad(f,argnums=0)(x,s,q)*ux).sum()
                    first = mx.grad(gx)(s if param==1 else b)
                    second = mx.grad(lambda z:(mx.grad(f,argnums=param)(z,s,b)*us).sum())(x)
                    eps = 1/1024
                    def perturbed(t):
                        return gx((s if param==1 else b)+t*us)
                    row = dict(gathered=gathered,transpose=transpose,param=param,
                               dx_then_param=(first*us).sum().item(),
                               param_then_dx=(second*ux).sum().item(),
                               finite_difference=((perturbed(eps)-perturbed(-eps))/(2*eps)).item(),
                               expected=1. if transpose else 32.)
                except Exception as e:
                    row = dict(gathered=gathered,transpose=transpose,param=param,error=repr(e))
                rows.append(row)
    return rows


def scan():
    rows = []
    for shift in (0.,1000.,100000.,100000000.):
        x = mx.array([shift,shift])
        f = lambda z: mx.logcumsumexp(z).sum()
        rows.append(dict(test="shift",shift=shift,forward=f(x).item(),
                         gradient=mx.grad(f)(x).tolist(),expected=[1.5,.5]))
    x = mx.array([0.,0.])
    for target_values in ([0.,math.log(2)],[0.,0.],[1.,1.]):
        target = mx.array(target_values)
        f = lambda z: .5*mx.square(mx.logcumsumexp(z)-target).sum()
        ref = lambda z: .5*((z[0]-target[0])**2+
                            (mx.log(mx.exp(z[0])+mx.exp(z[1]))-target[1])**2)
        try:
            rows.append(dict(test="loss_hessian",target=target_values,loss=f(x).item(),
                gradient=mx.grad(f)(x).tolist(),hessian=hessian(f,x).tolist(),
                reference=hessian(ref,x).tolist()))
        except Exception as e:
            rows.append(dict(test="loss_hessian",target=target_values,error=repr(e)))
    return rows


def main():
    start = time.process_time()
    result = clean(dict(version=importlib.metadata.version("mlx"),device=str(mx.default_device()),
                        qmm=qmm(),scan=scan()))
    result["cpu_seconds"] = time.process_time()-start
    (ROOT/"probe-results.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=="__main__":
    main()
