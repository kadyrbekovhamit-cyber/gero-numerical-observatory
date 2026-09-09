"""Small CPU reproduction of Muon matrix/conv reshape consistency."""

import hashlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import resource
import types

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
resource.setrlimit(resource.RLIMIT_CPU, (30, 30))

import mlx.core as mx
import mlx.optimizers as released

mx.set_default_device(mx.cpu)
root = Path(__file__).resolve().parent
source = (root / "upstream-optimizers.py").read_bytes()
upstream = types.ModuleType("audited_optimizers")
exec(compile(source, str(root / "upstream-optimizers.py"), "exec"), upstream.__dict__)


def new_opt(cls):
    return cls(learning_rate=0.01, momentum=0.0, weight_decay=0.0, nesterov=False)


rows = []
for label, cls in (("released_0.32.2", released.Muon), ("pinned_main", upstream.Muon)):
    for shape in ((8, 1, 1, 2), (1, 1, 8, 2), (2, 2, 2, 2)):
        flat_shape = (shape[0], math.prod(shape[1:]))
        gradient = (mx.arange(math.prod(shape), dtype=mx.float32) + 1) / 16
        p4, p2 = mx.zeros(shape), mx.zeros(flat_shape)
        o4, o2 = new_opt(cls), new_opt(cls)
        u4 = o4.apply_gradients({"w": gradient.reshape(shape)}, {"w": p4})["w"]
        u2 = o2.apply_gradients({"w": gradient.reshape(flat_shape)}, {"w": p2})["w"]
        f4 = u4.reshape(flat_shape)
        scale4 = math.sqrt(max(1, shape[-2] / shape[-1]))
        scale2 = math.sqrt(max(1, flat_shape[0] / flat_shape[1]))
        rows.append({
            "implementation": label, "shape": shape, "flat_shape": flat_shape,
            "max_abs_difference": mx.max(mx.abs(f4-u2)).item(),
            "conv_update_norm": mx.linalg.norm(f4).item(),
            "matrix_update_norm": mx.linalg.norm(u2).item(),
            "observed_ratio": (mx.linalg.norm(f4)/mx.linalg.norm(u2)).item(),
            "predicted_ratio_from_shape": scale4/scale2,
            "conv_update": f4.tolist(), "matrix_update": u2.tolist(),
        })

# A 1x1 convolution and a linear map implement the same forward function.
x = mx.array([[[[1.0, -0.5], [0.25, 0.75]], [[-1.0, 0.5], [0.5, 1.0]]]])
w = ((mx.arange(16, dtype=mx.float32) + 1) / 32).reshape(8, 2)
loss2 = lambda z: mx.mean(mx.square(x @ z.T))
loss4 = lambda z: mx.mean(mx.square(mx.conv2d(x, z)))
y2 = x @ w.T
y4 = mx.conv2d(x, w.reshape(8, 1, 1, 2))
g2 = mx.grad(loss2)(w)
g4 = mx.grad(loss4)(w.reshape(8, 1, 1, 2))
updated2 = new_opt(upstream.Muon).apply_gradients({"w": g2}, {"w": w})["w"]
updated4 = new_opt(upstream.Muon).apply_gradients({"w": g4}, {"w": w.reshape(8, 1, 1, 2)})["w"]
functional = {
    "forward_max_difference": mx.max(mx.abs(y2-y4)).item(),
    "loss_linear": loss2(w).item(), "loss_conv": loss4(w.reshape(8, 1, 1, 2)).item(),
    "gradient_max_difference": mx.max(mx.abs(g2-g4.reshape(8, 2))).item(),
    "update_max_difference": mx.max(mx.abs(updated2-updated4.reshape(8, 2))).item(),
    "linear_step_norm": mx.linalg.norm(w-updated2).item(),
    "conv_step_norm": mx.linalg.norm(w-updated4.reshape(8, 2)).item(),
}
result = {
    "source": json.loads((root / "source-metadata.json").read_text()),
    "native_wheel": importlib.metadata.version("mlx"), "device": str(mx.default_device()),
    "upstream_optimizer_sha256": hashlib.sha256(source).hexdigest(),
    "released_muon_source_sha256": hashlib.sha256(inspect.getsource(released.Muon).encode()).hexdigest(),
    "thread_limits": {key: os.environ[key] for key in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")},
    "reshape_cases": rows, "real_conv_linear_case": functional,
}
(root / "probe-results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
print(json.dumps({"reshape_cases": [{k:v for k,v in row.items() if k not in ('conv_update','matrix_update')} for row in rows], "real_conv_linear_case": functional}, indent=2))
