import importlib.util
import os
from pathlib import Path
import subprocess
import types

import mlx.core as mx
import mlx.nn as nn
import mpmath as mp
import numpy as np
import pytest

ROOT = Path(os.environ.get('AUDIT_MLX_SOURCE', Path(__file__).resolve().parents[2] / 'audit-targets/current-stack-2026-09-07/mlx'))
source = ROOT / 'python/mlx/nn/losses.py'
code = subprocess.check_output(['git', '-C', str(ROOT), 'show', 'HEAD:python/mlx/nn/losses.py'], text=True) if os.getenv('AUDIT_ORIGINAL') == '1' else source.read_text()
losses = types.ModuleType('audited_losses')
exec(compile(code, str(source), 'exec'), losses.__dict__)
mx.set_default_device(getattr(mx, os.environ.get('AUDIT_DEVICE', 'cpu')))
mp.mp.dps = 100

def as_np(value):
    return np.asarray(value.astype(mx.float32)).astype(np.float64)

@pytest.mark.parametrize('dtype_name,limit,rtol', [('float32', 80, 3e-6), ('float16', 8, .003), ('bfloat16', 80, .02)])
@pytest.mark.parametrize('shape', [(8,), (2, 4), (2, 2, 2)])
@pytest.mark.parametrize('target', [0., 1.])
def test_positive_tail_and_gradient(dtype_name, limit, rtol, shape, target):
    values = np.linspace(0, limit, 8) * (1 if target else -1)
    x = mx.array(values.reshape(shape), dtype=getattr(mx, dtype_name))
    y = mx.full(shape, target, x.dtype)
    rounded = as_np(x)
    expected = np.array([float(mp.log1p(mp.exp(-abs(mp.mpf(float(v)))))) for v in rounded.flat]).reshape(shape)
    gradient = np.array([float((1 - 2 * target) / (1 + mp.exp(abs(mp.mpf(float(v)))))) for v in rounded.flat]).reshape(shape)
    got = losses.binary_cross_entropy(x, y, reduction='none')
    grad = mx.grad(lambda z: losses.binary_cross_entropy(z, y, reduction='sum'))(x)
    np.testing.assert_allclose(as_np(got), expected, rtol=rtol, atol=0)
    np.testing.assert_allclose(as_np(grad), gradient, rtol=rtol, atol=0)

@pytest.mark.parametrize('dtype_name,rtol', [('float32', 3e-6), ('float16', .003), ('bfloat16', .02)])
@pytest.mark.parametrize('reduction', ['none', 'mean', 'sum'])
def test_weighted_binary_and_soft_labels(dtype_name, rtol, reduction):
    x = mx.array([-80., -5., -.2, 0., .2, 5., 80.], getattr(mx, dtype_name))
    y = mx.array([0., 1., .25, .5, .75, 0., 1.], x.dtype)
    weights = mx.array([1., 2., .5, 1., .5, 2., 1.], x.dtype)
    ref = np.array([float((mp.log1p(mp.exp(-abs(mp.mpf(float(a))))) + (a * (1-b) if a >= 0 else -a*b)) * w)
                    for a,b,w in zip(as_np(x), as_np(y), as_np(weights))])
    if reduction == 'sum':
        ref = ref.sum()
    elif reduction == 'mean':
        ref = ref.mean()
    np.testing.assert_allclose(as_np(losses.binary_cross_entropy(x, y, weights, reduction=reduction)), ref, rtol=rtol, atol=6e-8 if dtype_name == 'float16' else 0)

@pytest.mark.parametrize('shape', [(8,), (2, 4), (2, 2, 2)])
def test_batch_layout_and_label_symmetry(shape):
    x = mx.array([-40., -20., -5., -.5, .5, 5., 20., 40.]).reshape(shape)
    y = mx.array([0., 0., 1., 1., 0., 0., 1., 1.]).reshape(shape)
    actual = losses.binary_cross_entropy(x, y, reduction='none')
    reflected = losses.binary_cross_entropy(-x, 1-y, reduction='none')
    flattened = losses.binary_cross_entropy(x.flatten(), y.flatten(), reduction='none').reshape(shape)
    np.testing.assert_allclose(as_np(actual), as_np(reflected), rtol=3e-6, atol=0)
    np.testing.assert_array_equal(as_np(actual), as_np(flattened))

def test_upstream_bce_controls():
    # Run the existing upstream BCE test against the selected losses module.
    import sys
    sys.path.insert(0, str(ROOT / 'python/tests'))
    spec = importlib.util.spec_from_file_location('upstream_test_losses', ROOT / 'python/tests/test_losses.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = nn.losses.binary_cross_entropy
    nn.losses.binary_cross_entropy = losses.binary_cross_entropy
    try:
        case = module.TestLosses('test_binary_cross_entropy')
        case.test_binary_cross_entropy()
    finally:
        nn.losses.binary_cross_entropy = original
