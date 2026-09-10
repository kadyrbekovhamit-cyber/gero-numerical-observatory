"""Small installed-wheel CPU check against Python's double-precision libm."""
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

def clean(x):
    if isinstance(x, dict): return {k: clean(v) for k, v in x.items()}
    if isinstance(x, list): return [clean(v) for v in x]
    if isinstance(x, float) and not math.isfinite(x):
        return "NaN" if math.isnan(x) else ("Infinity" if x > 0 else "-Infinity")
    return x

start = time.process_time()
values = [-700., -100., -89., -1., -1e-8, 0., 1e-8, .25, 1., 89., 100., 700.]
x = mx.array(values, dtype=mx.float64)
y = mx.exp(x)
actual = y.tolist()
result = clean(dict(version=importlib.metadata.version("mlx"),
    device=str(mx.default_device()), output_dtype=str(y.dtype),
    cases=[dict(input=v, actual=a, expected=math.exp(v),
                relative_error=abs(a/math.exp(v)-1.)) for v, a in zip(values, actual)],
    cpu_seconds=time.process_time()-start))
(ROOT/"wheel-reproduction.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
print(json.dumps(result, indent=2, allow_nan=False))
