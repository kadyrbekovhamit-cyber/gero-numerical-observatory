import numpy as np
import pytest
from gero_stability.compare import compare, max_ulp, Tolerance


@pytest.mark.parametrize("a,b", [(np.nan, np.nan), (np.inf, np.inf), (np.inf, -np.inf), (1, np.nan)])
def test_nonfinite_is_never_a_pass(a, b):
    assert not compare(np.array([a], float), np.array([b], float))["passed"]


def test_matching_nan_is_contract_failure_not_backend_divergence():
    result = compare(np.array([np.nan]), np.array([np.nan]))
    assert not result["passed"]
    assert not result["numerical_divergence"]
    assert compare(np.array([np.inf]), np.array([-np.inf]))["numerical_divergence"]


def test_shape_does_not_broadcast_and_dtype_is_checked():
    assert compare(np.ones((1, 3)), np.ones(3))["reason"] == "shape_mismatch"
    assert compare(np.ones(3, np.float32), np.ones(3, np.float64))["reason"] == "dtype_mismatch"


def test_relative_error_uses_reference_and_absolute_floor():
    assert compare(np.array([1e-7]), np.array([0.]), Tolerance(0, 1e-6))["passed"]
    assert not compare(np.array([10.1]), np.array([10.]), Tolerance(1e-3, 0))["passed"]


def test_large_integers_do_not_lose_precision():
    assert not compare(np.array([2**62+1], dtype=np.int64), np.array([2**62], dtype=np.int64))["passed"]


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_ulp_adjacency_and_signed_zero(dtype):
    x = np.array([-1., 0., 1.], dtype=dtype)
    assert max_ulp(x, np.nextafter(x, np.full_like(x, np.inf))) == 1
    assert max_ulp(np.array([-0.], dtype=dtype), np.array([0.], dtype=dtype)) == 0


@pytest.mark.parametrize("a,b", [(-1, 0), (0, -1), (float("nan"), 0), (0, float("inf"))])
def test_invalid_tolerance(a, b):
    with pytest.raises(ValueError): Tolerance(a, b)
