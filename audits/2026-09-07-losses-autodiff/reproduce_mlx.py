"""Minimal reproduction against the installed official MLX package."""
import importlib.metadata
import json
import math
import platform

import mlx.core as mx
import mlx.nn as nn

rows = []
for device in ['cpu', 'gpu']:
    mx.set_default_device(getattr(mx, device))
    x = mx.array(20., mx.float32)
    zero = mx.array(0., mx.float32)
    one = mx.array(1., mx.float32)
    rows.append(dict(device=device,
        bce_positive=nn.losses.binary_cross_entropy(x, one).item(),
        bce_reflected=nn.losses.binary_cross_entropy(-x, zero).item(),
        bce_expected=math.log1p(math.exp(-20)),
        logaddexp_vjp_second=mx.grad(lambda z: mx.logaddexp(x, z))(zero).item(),
        logaddexp_vjp_swapped_first=mx.grad(lambda z: mx.logaddexp(z, x))(zero).item(),
        logaddexp_jvp_second=mx.jvp(lambda z: mx.logaddexp(x, z), [zero], [one])[1][0].item(),
        gradient_expected=1 / (1 + math.exp(20))))
print(json.dumps(dict(mlx=importlib.metadata.version('mlx'),
                      platform=platform.platform(), observations=rows), indent=2))
