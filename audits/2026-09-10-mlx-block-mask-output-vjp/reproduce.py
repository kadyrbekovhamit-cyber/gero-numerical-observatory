"""Synthetic installed-wheel reproduction. All operations run on one CPU stream."""
import importlib.metadata
import json
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


def main():
    started = time.process_time()
    a, b = mx.ones((2, 3)), mx.ones((3, 2))
    f = lambda mask: mx.block_masked_mm(a, b, 32, mask).sum()
    rows = []
    eps = 1 / 1024
    for value in (0., .5, 1., -2.):
        mask = mx.full((1, 1), value)
        rows.append(dict(mask=value, forward=f(mask).item(),
                         vjp=mx.grad(f)(mask).item(),
                         finite_difference=((f(mask+eps)-f(mask-eps))/(2*eps)).item(),
                         expected_gradient=12.))
    mask = mx.zeros((1, 1))
    loss = lambda m: .5 * mx.square(mx.block_masked_mm(a, b, 32, m) - 3).sum()
    g = mx.grad(loss)(mask)
    training = dict(initial=loss(mask).item(), actual_gradient=g.item(),
                    expected_gradient=-36.,
                    actual_next=loss(mask-g/36).item(),
                    expected_next=loss(mask+1).item())
    result = dict(version=importlib.metadata.version("mlx"), device=str(mx.default_device()),
                  rows=rows, training=training, cpu_seconds=time.process_time()-started)
    (ROOT / "wheel-reproduction.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
