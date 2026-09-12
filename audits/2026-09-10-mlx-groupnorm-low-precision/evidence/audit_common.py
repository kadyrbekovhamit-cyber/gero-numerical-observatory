"""Shared one-thread CPU setup and an independent scalar GroupNorm reference."""
from decimal import Decimal, localcontext
import importlib.util
import math
import os
from pathlib import Path
import resource
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key]='1'
resource.setrlimit(resource.RLIMIT_CPU,(30,30))
import mlx.core as mx
import mlx.nn as nn
mx.set_default_device(mx.cpu)
ROOT=Path(__file__).resolve().parent
def load(name,file):
    spec=importlib.util.spec_from_file_location(name,ROOT/file)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.GroupNorm
CLASSES={'before':load('groupnorm_before','upstream-python_mlx_nn_layers_normalization.py'),
         'after':load('groupnorm_after','patched-normalization.py')}
TOL={mx.float16:.004,mx.bfloat16:.025,mx.float32:5e-6,mx.float64:1e-11}
def flat(x):return x.reshape(-1).tolist()
def clean(x):
    if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [clean(v) for v in x]
    if isinstance(x,float) and not math.isfinite(x):return 'NaN' if math.isnan(x) else 'Infinity' if x>0 else '-Infinity'
    return x
def add_check(rows,name,mode,actual,expected,rtol=0,atol=0):
    if not isinstance(actual,list):actual,expected=[actual],[expected]
    ok=len(actual)==len(expected) and all(math.isfinite(a) and abs(a-b)<=atol+rtol*abs(b) for a,b in zip(actual,expected))
    rows.append(dict(name=name,mode=mode,passed=ok,actual=actual,expected=expected,rtol=rtol,atol=atol))

def reference(x,groups,compatible,eps):
    batch,channels=x.shape[0],x.shape[-1]
    spatial=math.prod(x.shape[1:-1])
    values=flat(x)
    output=[0.]*len(values)
    blocks=[]
    # Groups are defined directly by channel membership, not array reshapes.
    for b in range(batch):
        for group in range(groups):
            selected=[c for c in range(channels)
                      if (c//(channels//groups) if compatible else c%groups)==group]
            indices=[(b*spatial+s)*channels+c for s in range(spatial) for c in selected]
            with localcontext() as ctx:
                ctx.prec=70
                v=[Decimal.from_float(float(values[i])) for i in indices]
                mean=sum(v)/len(v)
                var=sum((z-mean)**2 for z in v)/len(v)
                inv=(var+Decimal.from_float(eps)).sqrt()
                z=[float((a-mean)/inv) for a in v]
            for i,a in zip(indices,z):output[i]=a
            blocks.append(dict(indices=indices,z=z,inverse_std=1/float(inv)))
    return output,blocks

def input_data(shape,dtype,amplitude):
    total=math.prod(shape)
    values=[amplitude*(((i*7+i//shape[-1]*3)%13)-6) for i in range(total)]
    return mx.array(values,dtype=dtype).reshape(shape)

def summary(rows):
    return {mode:dict(checks=sum(r['mode']==mode for r in rows),
                      failed=sum(r['mode']==mode and not r['passed'] for r in rows)) for mode in CLASSES}
