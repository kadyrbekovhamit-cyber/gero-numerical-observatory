"""Check the documented product-clamped cosine, including its derivatives."""
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

HERE = Path(__file__).resolve().parent
SOURCE = Path(os.environ.get('AUDIT_MLX_SOURCE', HERE.parents[1] / 'audit-targets/current-stack-2026-09-07/mlx'))
original_code = subprocess.check_output(['git', '-C', str(SOURCE), 'show', 'HEAD:python/mlx/nn/losses.py'], text=True)
original = types.ModuleType('original_losses')
exec(compile(original_code, 'original_losses.py', 'exec'), original.__dict__)
if os.getenv('AUDIT_ORIGINAL') == '1':
    cosine = original.cosine_similarity_loss
elif os.getenv('AUDIT_REPAIR') == 'source':
    patched=types.ModuleType('patched_losses')
    exec(compile((SOURCE/'python/mlx/nn/losses.py').read_text(),'patched_losses.py','exec'),patched.__dict__)
    cosine=patched.cosine_similarity_loss
else:
    from stable_cosine import cosine_similarity_loss as cosine
mx.set_default_device(getattr(mx, os.environ.get('AUDIT_DEVICE', 'cpu')))
mp.mp.dps = 100


def as_np(x):
    mx.eval(x)
    return np.asarray(x.astype(mx.float32)).astype(np.float64)


def oracle(a, b, eps=1e-8):
    a = [mp.mpf(float(x)) for x in a]
    b = [mp.mpf(float(x)) for x in b]
    d = sum(x * y for x, y in zip(a, b))
    n1, n2 = mp.sqrt(sum(x*x for x in a)), mp.sqrt(sum(y*y for y in b))
    den = max(n1*n2, mp.mpf(eps))
    if n1*n2 > eps:
        da = [y/den - d*x/(den*n1*n1) for x,y in zip(a,b)]
        db = [x/den - d*y/(den*n2*n2) for x,y in zip(a,b)]
    else:
        da, db = [y/den for y in b], [x/den for x in a]
    return float(d/den), np.array([float(x) for x in da]), np.array([float(x) for x in db])


@pytest.mark.parametrize('dtype,rtol', [('float32', 5e-6), ('float16', 0.005), ('bfloat16', 0.025)])
@pytest.mark.parametrize('shape,axis', [((2,4),1), ((1,2,4),-1), ((4,2),0)])
@pytest.mark.parametrize('case', ['ordinary', 'large', 'opposite', 'orthogonal', 'small', 'zero', 'unequal'])
def test_values_gradients_batch_axes(dtype, rtol, shape, axis, case):
    magnitude = 1000. if dtype == 'float16' else 1e20
    cases = {
        'ordinary': ([.5, .5, .2, .9], [.6, .4, .3, .8]),
        'large': ([magnitude, -magnitude, magnitude, -magnitude], [magnitude, magnitude, -magnitude, -magnitude/2]),
        'opposite': ([magnitude]*4, [-magnitude]*4),
        'orthogonal': ([magnitude, 0, 0, 0], [0, magnitude, 0, 0]),
        'small': ([1e-6]*4, [2e-6, -1e-6, 1e-6, 1e-6]),
        'zero': ([0.]*4, [1e-5, 2e-5, -1e-5, 1e-5]),
        'unequal': ([1e-3]*4, [1000., -1000., 1000., 1000.]),
    }
    a0,b0 = cases[case]
    a_np,b_np = np.tile(a0,(2,1)),np.tile(b0,(2,1))
    if axis == 0:
        a_np,b_np=a_np.T,b_np.T
    a=mx.array(a_np.reshape(shape),getattr(mx,dtype));b=mx.array(b_np.reshape(shape),a.dtype)
    aa,bb=np.moveaxis(as_np(a),axis,-1),np.moveaxis(as_np(b),axis,-1)
    triples=[oracle(x,y) for x,y in zip(aa.reshape(-1,4),bb.reshape(-1,4))]
    expected=np.array([t[0] for t in triples]).reshape(aa.shape[:-1])
    ga=np.moveaxis(np.array([t[1] for t in triples]).reshape(aa.shape),-1,axis)
    gb=np.moveaxis(np.array([t[2] for t in triples]).reshape(bb.shape),-1,axis)
    got=cosine(a,b,axis=axis)
    da,db=mx.grad(lambda x,y: mx.sum(cosine(x,y,axis=axis)),argnums=(0,1))(a,b)
    np.testing.assert_allclose(as_np(got),expected,rtol=rtol,atol=1e-7)
    np.testing.assert_allclose(as_np(da),ga,rtol=rtol,atol=6e-8 if dtype=='float16' else 1e-35)
    np.testing.assert_allclose(as_np(db),gb,rtol=rtol,atol=6e-8 if dtype=='float16' else 1e-35)
    assert got.dtype == a.dtype


@pytest.mark.parametrize('magnitude', [1e-30, 1e-20, 1e-6, 1., 1e20, 1e30])
@pytest.mark.parametrize('relation', ['self', 'opposite', 'reciprocal'])
def test_scale_range_and_symmetry(magnitude, relation):
    a=mx.array([[magnitude,-magnitude,magnitude,-magnitude]],mx.float32)
    b=-a if relation=='opposite' else (mx.array([[1/magnitude,-1/magnitude,1/magnitude,-1/magnitude]],mx.float32) if relation=='reciprocal' else a)
    ref,_,_=oracle(as_np(a)[0],as_np(b)[0])
    value=as_np(cosine(a,b))
    np.testing.assert_allclose(value,[ref],rtol=5e-6,atol=1e-35)
    np.testing.assert_allclose(value,as_np(cosine(b,a)),rtol=5e-6,atol=1e-35)


@pytest.mark.parametrize('eps', [1e-8,1e-4,0.5])
@pytest.mark.parametrize('reduction', ['none','mean','sum'])
def test_epsilon_product_semantics_and_reductions(eps,reduction):
    a=mx.array([[1e-6,0,0,0],[1e-3,2e-3,0,0],[1,0,0,0]])
    b=mx.array([[2e-6,0,0,0],[1e-3,1e-3,0,0],[.2,0,0,0]])
    refs=np.array([oracle(x,y,eps)[0] for x,y in zip(as_np(a),as_np(b))])
    ref=refs.mean() if reduction=='mean' else refs.sum() if reduction=='sum' else refs
    np.testing.assert_allclose(as_np(cosine(a,b,eps=eps,reduction=reduction)),ref,rtol=5e-6,atol=1e-12)


def test_existing_upstream_cosine_controls():
    import sys
    sys.path.insert(0,str(SOURCE/'python/tests'))
    spec=importlib.util.spec_from_file_location('upstream_test_losses',SOURCE/'python/tests/test_losses.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    saved=nn.losses.cosine_similarity_loss
    try:
        nn.losses.cosine_similarity_loss=cosine
        case=module.TestLosses(methodName='test_cosine_similarity_loss')
        case.setUp();case.test_cosine_similarity_loss();case.tearDown()
    finally:
        nn.losses.cosine_similarity_loss=saved


@pytest.mark.parametrize('sa,sb', [(1e-30,1e-30),(1e-20,1e20),(1e20,1e-20),(1e30,1e30)])
def test_extreme_scale_gradients(sa,sb):
    a=mx.array([[sa,-sa,sa,-sa]],mx.float32)
    b=mx.array([[sb,sb,-sb,-sb/2]],mx.float32)
    _,ga,gb=oracle(as_np(a)[0],as_np(b)[0])
    da,db=mx.grad(lambda x,y: mx.sum(cosine(x,y)),argnums=(0,1))(a,b)
    np.testing.assert_allclose(as_np(da),ga[None,:],rtol=1e-5,atol=1e-35)
    np.testing.assert_allclose(as_np(db),gb[None,:],rtol=1e-5,atol=1e-35)


@pytest.mark.parametrize('case',['ordinary','large','zero'])
def test_compiled_value_and_gradient(case):
    scale=1e20 if case=='large' else 1e-5 if case=='zero' else 1.
    a=mx.array([[0.,0.,0.,0.]]) if case=='zero' else mx.array([[scale,-scale,scale,-scale]])
    b=mx.array([[scale,scale,-scale,-scale/2]])
    ref,ga,_=oracle(as_np(a)[0],as_np(b)[0])
    value=mx.compile(lambda x,y: cosine(x,y))(a,b)
    derivative=mx.compile(mx.grad(lambda x,y: mx.sum(cosine(x,y))))(a,b)
    np.testing.assert_allclose(as_np(value),[ref],rtol=1e-5,atol=1e-7)
    np.testing.assert_allclose(as_np(derivative),ga[None,:],rtol=1e-5,atol=1e-35)


@pytest.mark.parametrize('eps',[0.,-1.])
def test_unclamped_nonzero_vectors(eps):
    a=mx.array([[1.,-1.,1.,-1.]])
    b=mx.array([[1.,1.,-1.,-.5]])
    ref,ga,_=oracle(as_np(a)[0],as_np(b)[0],eps)
    np.testing.assert_allclose(as_np(cosine(a,b,eps=eps)),[ref],rtol=1e-5)
    np.testing.assert_allclose(as_np(mx.grad(lambda x: mx.sum(cosine(x,b,eps=eps)))(a)),ga[None,:],rtol=1e-5,atol=1e-7)


def test_integer_input_result_type():
    a=mx.array([[1,1,-1,-1]],mx.int32)
    got=cosine(a,a)
    assert mx.issubdtype(got.dtype,mx.floating)
    np.testing.assert_array_equal(as_np(got),[1.])
