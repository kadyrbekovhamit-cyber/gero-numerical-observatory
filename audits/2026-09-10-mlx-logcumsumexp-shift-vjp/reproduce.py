"""Tiny installed-wheel evidence: CPU only and one computation thread."""
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


def clean(value):
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
    return value


def main():
    start = time.process_time()
    rows = []
    for dtype in (mx.float32, mx.float64):
        for shift in (0., 1000., 100000., 1e8, -1e8, 1e20):
            x = mx.array([shift, shift], dtype=dtype)
            last = lambda z: mx.logcumsumexp(z)[-1]
            reduced = lambda z: mx.logsumexp(z)
            rows.append(dict(dtype=str(dtype), input=x.tolist(),
                scan_value=last(x).item(), reduce_value=reduced(x).item(),
                scan_gradient=mx.grad(last)(x).tolist(),
                reduce_gradient=mx.grad(reduced)(x).tolist(),
                expected=[.5, .5]))
    # A global center can damage an early prefix that had small inputs.
    x = mx.array([0., 0., 1e8])
    original = lambda z: mx.logcumsumexp(z)[1]
    def centered(z):
        shift = mx.stop_gradient(mx.max(z))
        return (mx.logcumsumexp(z-shift)+shift)[1]
    counterexample = dict(input=x.tolist(),
        original=mx.grad(original)(x).tolist(),
        global_center=mx.grad(centered)(x).tolist(), expected=[.5, .5, 0.])
    result = clean(dict(version=importlib.metadata.version("mlx"),
        device=str(mx.default_device()), cases=rows,
        naive_global_center_counterexample=counterexample,
        cpu_seconds=time.process_time()-start))
    (ROOT/"wheel-reproduction.json").write_text(
        json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
