import mlx.core as mx
import mpmath as mp
import numpy as np
import pytest

from load_mlx import load_activations

act = load_activations()
mp.mp.dps = 80

def reference(name, x):
    x = mp.mpf(float(x))
    if name == 'elu': return x if x > 0 else mp.expm1(x)
    if name == 'selu': return mp.mpf('1.0507') * (x if x > 0 else mp.mpf('1.67326') * mp.expm1(x))
    if name == 'gelu': return x * mp.erfc(-x / mp.sqrt(2)) / 2
    return x * (1 + mp.tanh(mp.sqrt(2/mp.pi) * (x + mp.mpf('0.044715') * x**3))) / 2

@pytest.mark.parametrize('dtype', [mx.float16, mx.float32])
@pytest.mark.parametrize('name, alpha', [('elu', 1.0), ('elu', 1.7), ('selu', None)])
def test_positive_elu_has_linear_derivative(dtype, name, alpha):
    x = mx.array([12., 100., 1000.], dtype=dtype)
    fn = (lambda v: act.elu(v, alpha)) if name == 'elu' else act.selu
    gradient = mx.grad(lambda v: fn(v).astype(mx.float32).sum())(x)
    expected = np.ones(3) * (1.0507 if name == 'selu' else 1)
    np.testing.assert_allclose(np.array(gradient), expected, rtol=1e-3, atol=1e-6)
    np.testing.assert_allclose(np.array(fn(x)), np.array(x) * expected, rtol=1e-3)

@pytest.mark.parametrize('dtype', [mx.float16, mx.float32])
def test_gelu_preserves_representable_positive_values(dtype):
    values = [32752., 40000., 65504.] if dtype == mx.float16 else [1e38, 2e38, 3e38]
    x = mx.array(values, dtype=dtype)
    np.testing.assert_array_equal(np.array(act.gelu(x)), np.array(x))

@pytest.mark.parametrize('dtype', [mx.float16, mx.float32])
def test_gelu_approx_large_gradient_is_finite(dtype):
    values = [-65504., -40000., 40000., 65504.] if dtype == mx.float16 else [-3e38, -1e20, 1e20, 3e38]
    x = mx.array(values, dtype=dtype)
    gradient = mx.grad(lambda v: act.gelu_approx(v).astype(mx.float32).sum())(x)
    np.testing.assert_array_equal(np.array(gradient), [0., 0., 1., 1.])

@pytest.mark.parametrize('dtype', [mx.float16, mx.float32])
@pytest.mark.parametrize('name', ['elu', 'selu', 'gelu', 'gelu_approx'])
def test_ordinary_inputs_match_high_precision_reference(dtype, name):
    x = mx.array([-6., -2., -0.5, -0.01, 0.01, 0.5, 2., 6.], dtype=dtype)
    values = x.tolist()
    expected = np.array([float(reference(name, value)) for value in values])
    derivatives = np.array([reference_derivative(name, value) for value in values])
    fn = getattr(act, name)
    rtol, atol = (4e-3, 4e-3) if dtype == mx.float16 else (1e-5, 3e-6)
    np.testing.assert_allclose(np.array(fn(x)), expected, rtol=rtol, atol=atol)
    actual = np.array(mx.grad(lambda v: fn(v).astype(mx.float32).sum())(x))
    np.testing.assert_allclose(actual, derivatives, rtol=rtol, atol=atol)

def reference_derivative(name, value):
    x = mp.mpf(float(value))
    if name == 'elu': return float(1 if x > 0 else mp.exp(x))
    if name == 'selu': return float(mp.mpf('1.0507') * (1 if x > 0 else mp.mpf('1.67326') * mp.exp(x)))
    if name == 'gelu': return float(mp.erfc(-x/mp.sqrt(2))/2 + x*mp.exp(-x*x/2)/mp.sqrt(2*mp.pi))
    c, a = mp.sqrt(2/mp.pi), mp.mpf('0.044715')
    t = mp.tanh(c*(x+a*x**3))
    return float((1+t)/2 + x*(1-t*t)*c*(1+3*a*x*x)/2)

@pytest.mark.parametrize('dtype', [mx.float16, mx.float32])
@pytest.mark.parametrize('name', ['elu', 'gelu', 'gelu_approx'])
def test_elementwise_batch_layout_invariance(dtype, name):
    x = mx.array([-20., -6., -2., -0.1, 0., 0.1, 2., 6., 12., 20., 100., 40000.], dtype=dtype)
    fn = getattr(act, name)
    flat = np.array(fn(x))
    batched = np.array(fn(x.reshape(3, 4).T).T).reshape(-1)
    np.testing.assert_allclose(flat, batched, rtol=1e-5, atol=1e-6)
