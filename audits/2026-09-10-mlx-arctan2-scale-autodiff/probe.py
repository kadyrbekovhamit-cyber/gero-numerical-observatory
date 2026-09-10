"""Bounded synthetic MLX CPU test, one numerical thread."""
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
    if isinstance(x, dict): return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)): return [clean(v) for v in x]
    if isinstance(x, float) and not math.isfinite(x):
        return "NaN" if math.isnan(x) else "Infinity" if x > 0 else "-Infinity"
    return x
rows = []
compositions = []
for dtype, powers in ((mx.float16, (-14, 10)), (mx.bfloat16, (-80, 80)),
                      (mx.float32, (-80, 80)), (mx.float64, (-600, 600))):
    for power in (0, *powers):
        c = mx.array(math.ldexp(1., power), dtype=dtype)
        a, b = c, c
        one = mx.array(1., dtype=dtype)
        out, v = mx.vjp(mx.arctan2, [a, b], [one])
        _, j0 = mx.jvp(mx.arctan2, [a, b], [one, mx.zeros_like(one)])
        _, jr = mx.jvp(mx.arctan2, [a, b], [a, b])
        rows.append(dict(dtype=str(dtype), scale=c.item(), value=out[0].item(),
            vjp=[z.item() for z in v], jvp_first=j0[0].item(),
            radial_jvp=jr[0].item(), expected_vjp=[.5/c.item(), -.5/c.item()],
            expected_jvp=.5/c.item(), expected_radial_jvp=0.))
        if dtype not in (mx.float32, mx.float64):
            continue
        f = lambda t: mx.arctan2(c*t, c)
        step = 2.**(-6 if dtype == mx.float32 else -10)
        fd = (f(one+step).item()-f(one-step).item())/(2*step)
        compositions.append(dict(dtype=str(dtype), scale=c.item(), t=1.,
            value=f(one).item(), gradient=mx.grad(f)(one).item(),
            hessian=mx.grad(mx.grad(f))(one).item(),
            third=mx.grad(mx.grad(mx.grad(f)))(one).item(),
            finite_difference=fd, step=step,
            expected_gradient=.5, expected_hessian=-.5, expected_third=.5))
result = clean(dict(version=importlib.metadata.version("mlx"),
    device=str(mx.default_device()), cases=rows, compositions=compositions,
    cpu_seconds=time.process_time()-start))
(ROOT/"wheel-reproduction.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
print(json.dumps(result, indent=2, allow_nan=False))
