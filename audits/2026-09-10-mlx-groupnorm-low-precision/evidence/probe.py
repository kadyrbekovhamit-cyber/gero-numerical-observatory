"""Tiny actual MLX GroupNorm probe; CPU and one numerical thread."""
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import resource
import time
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key]='1'
resource.setrlimit(resource.RLIMIT_CPU,(30,30))
import mlx.core as mx
import mlx.nn as nn
mx.set_default_device(mx.cpu)
ROOT=Path(__file__).resolve().parent
started=time.process_time()
rows=[]
for dtype in [mx.float16,mx.bfloat16,mx.float32,mx.float64]:
    for amplitude in [1.,128.,256.]:
        for compatible in [False,True]:
            x=mx.array([[[-amplitude,amplitude,-amplitude,amplitude]]],dtype=dtype)
            layer=nn.GroupNorm(1,4,affine=False,pytorch_compatible=compatible)
            out=layer(x)
            rows.append(dict(dtype=str(dtype),amplitude=amplitude,pytorch_compatible=compatible,
                             output=out.reshape(-1).tolist(),output_dtype=str(out.dtype),
                             expected=[-amplitude/math.sqrt(amplitude**2+layer.eps),
                                       amplitude/math.sqrt(amplitude**2+layer.eps)]*2))
layer=nn.GroupNorm(1,3)
x=mx.array([[[-256.,0.,512.]]],dtype=mx.float16)
weights=mx.array([[[-.5,2.,-1.]]])
loss=lambda model,x:mx.sum(model(x)*weights)
value,grads=nn.value_and_grad(layer,loss)(layer,x)
example=dict(input=x.tolist(),forward=layer(x).tolist(),loss=value.item(),gamma_gradient=grads['weight'].tolist())
def clean(x):
    if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,float) and not math.isfinite(x):return 'NaN' if math.isnan(x) else 'Infinity' if x>0 else '-Infinity'
    return x
result=clean(dict(version=importlib.metadata.version('mlx'),device=str(mx.default_device()),
                  cpu_seconds=time.process_time()-started,cases=rows,gradient_example=example))
(ROOT/'installed-groupnorm-source.py').write_text(inspect.getsource(nn.GroupNorm))
(ROOT/'wheel-reproduction.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps(result,indent=2,allow_nan=False))
