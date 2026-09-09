"""Public MLX API reproduction; small synthetic arrays on CPU."""
import mlx.core as mx
from mlx.nn.losses import gaussian_nll_loss

mx.set_default_device(mx.cpu)
for mean, variance in [(300., 300.), (20., 300.), (0., .0001)]:
    x = mx.array([mean], dtype=mx.float16)
    y = mx.array([0.], dtype=mx.float16)
    v = mx.array([variance], dtype=mx.float16)
    loss = gaussian_nll_loss(x, y, v)
    derivative = mx.grad(lambda z: gaussian_nll_loss(x, y, z))(v)
    print('mean=', x.item(), 'variance=', v.item(),
          'loss=', loss.item(), 'dL/dvariance=', derivative.item())
