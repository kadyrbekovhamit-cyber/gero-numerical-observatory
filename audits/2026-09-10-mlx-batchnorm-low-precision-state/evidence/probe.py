"""Real MLX normalization and running-state probe, sequential CPU only."""
import importlib.metadata
import inspect
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
import mlx.nn as nn
mx.set_default_device(mx.cpu)
ROOT = Path(__file__).resolve().parent
started = time.process_time()

def clean(value):
    if isinstance(value, dict): return {k: clean(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)): return [clean(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else "Infinity" if value > 0 else "-Infinity"
    return value

cases = []
for dtype in [mx.float16, mx.bfloat16, mx.float32, mx.float64]:
    bn = nn.BatchNorm(1, momentum=0.125, affine=False)
    expected_state = 1.0
    stages = []
    for amplitude in [256.0, 1.0, 1.0, 1.0]:
        x = mx.array([[-amplitude],[amplitude]], dtype=dtype)
        output = bn(x)
        expected_state = .875*expected_state+.125*2*amplitude**2
        stages.append(dict(amplitude=amplitude, output=output.tolist(),
                           expected_output=[[-amplitude/math.sqrt(amplitude**2+bn.eps)],
                                            [amplitude/math.sqrt(amplitude**2+bn.eps)]],
                           running_var=bn.running_var.tolist(), expected_running_var=expected_state,
                           running_dtype=str(bn.running_var.dtype), output_dtype=str(output.dtype)))
    bn.eval()
    evaluated = bn(mx.array([[-1.],[1.]], dtype=dtype))
    cases.append(dict(dtype=str(dtype), stages=stages, eval_output=evaluated.tolist(),
                      expected_eval=[[-1/math.sqrt(expected_state+bn.eps)], [1/math.sqrt(expected_state+bn.eps)]]))

# Neighboring GroupNorm observation only; separate from the running-state defect.
groups = []
for compatible in [False, True]:
    gn = nn.GroupNorm(1, 4, affine=False, pytorch_compatible=compatible)
    x = mx.array([[[-256.,256.,-256.,256.]]], dtype=mx.float16)
    groups.append(dict(pytorch_compatible=compatible, output=gn(x).tolist()))

(ROOT / "installed-batchnorm-source.py").write_text(inspect.getsource(nn.BatchNorm))
result = clean(dict(version=importlib.metadata.version("mlx"), device=str(mx.default_device()),
                    cpu_seconds=time.process_time()-started, batch_norm=cases, group_norm_neighbor=groups))
(ROOT / "wheel-reproduction.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
print(json.dumps(result, indent=2, allow_nan=False))
