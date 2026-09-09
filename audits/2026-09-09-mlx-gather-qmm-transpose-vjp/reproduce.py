"""Small installed-wheel reproduction, CPU only, synthetic tensors."""
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import time

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
resource.setrlimit(resource.RLIMIT_CPU, (30, 30))
import mlx.core as mx
mx.set_default_device(mx.cpu)

ROOT = Path(__file__).resolve().parent


def main():
    started = time.process_time()
    x = mx.array([1.] + [0.] * 31).reshape(1, 1, 32)
    ids = mx.array([0], dtype=mx.uint32)
    w = mx.full((1, 32, 4), 0x11111111, dtype=mx.uint32)
    s, b = mx.ones((1, 32, 1)), mx.zeros((1, 32, 1))
    kwargs = dict(rhs_indices=ids, transpose=False, group_size=32, bits=4)
    f = lambda x, s, b: mx.gather_qmm(x, w, s, b, **kwargs)
    out, grads = mx.vjp(f, [x, s, b], [mx.ones((1, 1, 32))])
    mx.eval(*out, *grads)
    epsilon = 1 / 1024
    finite = {}
    for arg in ('scales', 'biases'):
        finite[arg] = []
        for coord in (0, 1):
            d = mx.array([float(i == coord) for i in range(32)]).reshape(s.shape)
            if arg == 'scales':
                num = f(x, s + epsilon*d, b).sum() - f(x, s - epsilon*d, b).sum()
            else:
                num = f(x, s, b + epsilon*d).sum() - f(x, s, b - epsilon*d).sum()
            finite[arg].append((num / (2*epsilon)).item())
    target = mx.array([2.] + [0.] * 31).reshape(1, 1, 32)
    loss = lambda bias: 0.5 * mx.square(f(x, s, bias) - target).sum()
    bg = mx.grad(loss)(b)
    target_gradient = mx.array([30.] + [0.] * 31).reshape(b.shape)
    rectangular = {}
    rw = mx.full((1, 32, 8), 0x11111111, dtype=mx.uint32)
    rs, rb = mx.ones((1, 32, 2)), mx.zeros((1, 32, 2))
    rf = lambda scale, bias: mx.gather_qmm(x, rw, scale, bias, **kwargs).sum()
    try:
        rg = mx.grad(lambda bias: rf(rs, bias))(rb)
        mx.eval(rg)
        rectangular['actual_bias_gradient_shape'] = list(rg.shape)
        rectangular['expected_bias_gradient_shape'] = list(rb.shape)
    except Exception as error:
        rectangular['bias_error'] = str(error)
    try:
        rg = mx.grad(lambda scale: rf(scale, rb))(rs)
        mx.eval(rg)
        rectangular['scale_gradient_shape'] = list(rg.shape)
    except Exception as error:
        rectangular['scale_error'] = str(error)
    result = dict(
        version=importlib.metadata.version('mlx'), device=str(mx.default_device()),
        square=dict(forward=out[0].reshape(-1).tolist(),
                    dx_first=grads[0].reshape(-1)[:4].tolist(),
                    ds=grads[1].reshape(-1).tolist(), db=grads[2].reshape(-1).tolist(),
                    expected_ds_db=[32.] + [0.]*31, finite_differences=finite),
        loss=dict(initial=loss(b).item(), actual_gradient=bg.reshape(-1).tolist(),
                  expected_gradient=target_gradient.reshape(-1).tolist(),
                  actual_next=loss(b-bg/32).item(),
                  oracle_next=loss(b-target_gradient/32).item()),
        rectangular=rectangular,
        cpu_seconds=time.process_time()-started,
    )
    (ROOT / 'wheel-reproduction.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(version=result['version'], device=result['device'],
        ds_first=result['square']['ds'][:4],
        expected_first=result['square']['expected_ds_db'][:4],
        finite_differences=finite,
        loss={k:v for k,v in result['loss'].items() if 'gradient' not in k},
        rectangular=rectangular, cpu_seconds=result['cpu_seconds']), indent=2))


if __name__ == '__main__':
    main()
