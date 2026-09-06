"""Strict contracts: shape and dtype first; non-finite outputs never pass."""
from dataclasses import asdict, dataclass
import numpy as np


@dataclass(frozen=True)
class Tolerance:
    rtol: float
    atol: float

    def __post_init__(self):
        if not np.isfinite([self.rtol, self.atol]).all() or min(self.rtol, self.atol) < 0:
            raise ValueError("Tolerances must be finite and nonnegative")

    def dict(self):
        return asdict(self)


def tolerance_for(dtype):
    return {"float16": Tolerance(5e-3, 5e-4), "float32": Tolerance(1e-5, 1e-6),
            "float64": Tolerance(1e-12, 1e-13)}.get(str(np.dtype(dtype)), Tolerance(0, 0))


def max_ulp(actual, expected):
    """Ordered IEEE bit patterns; Python ints avoid uint64 subtraction overflow."""
    if actual.dtype != expected.dtype or actual.dtype.kind != "f" or actual.dtype.itemsize not in (2, 4, 8):
        return None
    uint = np.dtype(f"u{actual.dtype.itemsize}")
    sign = 1 << (8 * actual.dtype.itemsize - 1)
    def ordered(bits):
        bits = int(bits)
        return sign - (bits & (sign - 1)) if bits & sign else sign + bits
    valid = np.isfinite(actual) & np.isfinite(expected)
    if not valid.any(): return None
    a, b = actual[valid].view(uint), expected[valid].view(uint)
    return max((abs(ordered(x) - ordered(y)) for x, y in zip(a, b)), default=0)


def compare(actual, expected, tolerance=None, *, require_dtype=True):
    actual, expected = np.asarray(actual), np.asarray(expected)
    tol = tolerance or tolerance_for(expected.dtype)
    result = {"passed": False, "numerical_divergence": True, "reason": None, "actual_shape": list(actual.shape),
              "expected_shape": list(expected.shape), "actual_dtype": str(actual.dtype),
              "expected_dtype": str(expected.dtype), "tolerance": tol.dict()}
    if actual.shape != expected.shape:
        return dict(result, reason="shape_mismatch")
    if require_dtype and actual.dtype != expected.dtype:
        return dict(result, reason="dtype_mismatch")
    finite = np.isfinite(actual) & np.isfinite(expected)
    nonfinite = int((~finite).sum())
    if actual.dtype.kind in "iub" and expected.dtype.kind in "iub":
        # Do not coerce int64 to double before checking equality.
        bad = actual != expected
        error = np.abs(actual.astype(object) - expected.astype(object)).astype(np.float64)
        limit = np.zeros(actual.shape)
    else:
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            error = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
            limit = tol.atol + tol.rtol * np.abs(expected.astype(np.float64))
            bad = (error > limit) | ~finite
    def safe_max(values):
        if not values.size:
            return None
        val = float(np.max(values))
        return val if np.isfinite(val) else None
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        relative = error / np.maximum(np.abs(expected.astype(np.float64)), np.finfo(np.float64).tiny)
        budget = np.divide(error, limit, out=np.where(error == 0, 0., np.inf), where=limit > 0)
    indexes = np.argwhere(bad)
    first = indexes[0].tolist() if indexes.size else ([] if actual.ndim == 0 and bool(bad) else None)
    matching_nonfinite = (np.isnan(actual) & np.isnan(expected)) | (np.isinf(actual) & (actual == expected))
    return dict(result, passed=not bool(np.any(bad)), numerical_divergence=bool(np.any(bad & ~matching_nonfinite)),
                reason="nonfinite_output" if nonfinite else
                ("tolerance_exceeded" if np.any(bad) else None),
                elements=int(actual.size), mismatched=int(np.sum(bad)), nonfinite=nonfinite,
                max_abs=safe_max(error[finite]), max_rel=safe_max(relative[finite]),
                max_budget_ratio=safe_max(budget[finite]), max_ulp=max_ulp(actual, expected),
                first_mismatch=first)
