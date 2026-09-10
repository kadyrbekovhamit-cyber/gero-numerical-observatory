"""Small real-MLX CPU probe, with one numerical thread."""
import importlib.metadata
import inspect
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
import mlx.optimizers as optim
mx.set_default_device(mx.cpu)
ROOT=Path(__file__).resolve().parent
start=time.process_time()
def clean(x):
    if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [clean(v) for v in x]
    if isinstance(x,float) and not math.isfinite(x):
        return "NaN" if math.isnan(x) else "Infinity" if x>0 else "-Infinity"
    return x
rows=[]
for dtype,powers in ((mx.float16,(-14,10)),(mx.bfloat16,(-80,80)),
                     (mx.float32,(-80,80)),(mx.float64,(-600,600))):
    for power in (0,*powers):
        c=math.ldexp(1.,power)
        g=mx.array([3*c,4*c],dtype=dtype)
        q=g.tolist();reference=math.hypot(*q)
        factor=min(1.,1./(reference+1e-6))
        clipped,norm=optim.clip_grad_norm({"w":g},1.)
        updated=optim.SGD(learning_rate=.1).apply_gradients(
            clipped,{"w":mx.zeros_like(g)})
        rows.append(dict(dtype=str(dtype),power=power,input=q,expected_norm=reference,
            linalg_norm=mx.linalg.norm(g).item(),reported_norm=norm.item(),norm_dtype=str(norm.dtype),
            clipped=clipped["w"].tolist(),clipped_dtype=str(clipped["w"].dtype),
            expected_clipped=[x*factor for x in q],sgd_update=updated["w"].tolist(),
            expected_update=[-.1*x*factor for x in q]))
(ROOT/"installed-clip-source.py").write_text(inspect.getsource(optim.clip_grad_norm))
out=clean(dict(version=importlib.metadata.version("mlx"),device=str(mx.default_device()),
    cases=rows,cpu_seconds=time.process_time()-start))
(ROOT/"wheel-reproduction.json").write_text(json.dumps(out,indent=2,allow_nan=False)+"\n")
print(json.dumps(out,indent=2,allow_nan=False))
