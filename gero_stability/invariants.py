"""Domain-aware contracts and metamorphic tests, independent of agreement."""
import numpy as np
from onnx import numpy_helper
from .backends import evaluator
from .compare import compare, tolerance_for


def output_checks(case, y, tolerance=None):
    x = case.feeds["x"].astype(np.float64)
    tol = tolerance or tolerance_for(y.dtype)
    checks = []
    def predicate(name, passed, **details):
        checks.append({"name": name, "passed": bool(passed), **details})
    def close(name, actual, expected):
        checks.append({"name": name, **compare(actual, np.asarray(expected, dtype=actual.dtype), tol)})
    predicate("finite_output", np.isfinite(y).all())
    if case.contract in ("probability", "masked_probability", "log_probability"):
        p = np.exp(y.astype(np.float64)) if case.contract == "log_probability" else y
        predicate("probability_range", np.isfinite(p).all() and np.all((p >= 0) & (p <= 1)))
        target = np.ones(p.shape[:-1])
        if case.contract == "masked_probability":
            mask = case.feeds["mask"]
            target = mask.any(axis=-1).astype(np.float64)
            predicate("masked_entries_exact_zero", np.all(y[~mask] == 0))
        close("probability_mass", p.sum(axis=-1, dtype=np.float64), target)
    elif case.contract == "layer_norm":
        centered = x - x.mean(axis=-1, keepdims=True)
        variance = np.mean(centered * centered, axis=-1, keepdims=True)
        oracle = centered / np.sqrt(variance + case.parameters["epsilon"])
        close("centered_float64_oracle", y, oracle)
        # With epsilon the expected variance is v/(v+epsilon), not exactly 1.
        close("normalized_variance", np.var(y.astype(np.float64), axis=-1, keepdims=True),
              variance / (variance + case.parameters["epsilon"]))
    elif case.contract == "unit_norm":
        # Scaled float64 formula avoids intermediate square overflow.
        scale = np.max(np.abs(x), axis=-1, keepdims=True)
        scaled = x / np.where(scale == 0, 1, scale)
        norm = np.sqrt(np.sum(scaled * scaled, axis=-1, keepdims=True))
        oracle = scaled / np.where(norm == 0, 1, norm)
        close("scaled_float64_oracle", y, oracle)
        close("l2_unit_or_zero", np.linalg.norm(y.astype(np.float64), axis=-1), np.any(x != 0, axis=-1).astype(float))
    elif case.contract == "mse":
        diff = x - case.feeds["target"].astype(np.float64)
        close("mse_float64_oracle", y, np.mean(diff * diff, axis=-1, keepdims=True))
        predicate("mse_nonnegative", np.all(y >= 0))
    elif case.contract == "cosine":
        target = case.feeds["target"].astype(np.float64)
        norm = np.linalg.norm(x, axis=-1, keepdims=True) * np.linalg.norm(target, axis=-1, keepdims=True)
        oracle = np.sum(x * target, axis=-1, keepdims=True) / np.maximum(norm, case.parameters["epsilon"])
        close("cosine_float64_oracle", y, oracle)
        predicate("cosine_range", np.all(np.abs(y) <= 1 + tol.atol + tol.rtol))
    elif case.contract == "dequantize":
        constants = {i.name: numpy_helper.to_array(i) for i in case.model.graph.initializer}
        oracle = (x - constants["zero_point"].astype(np.float64)) * constants["scale"].astype(np.float64)
        close("dequantize_affine_oracle", y, oracle)
    return checks


def metamorphic_checks(case, y, backend, optimization="disabled", tolerance=None):
    tol = tolerance or tolerance_for(y.dtype)
    checks = []
    def run(feeds):
        return evaluator(case.with_feeds(feeds).model, backend, optimization)(feeds)
    # Every input is per-example. Global batch metrics are deliberately excluded.
    n = case.feeds["x"].shape[0]
    pieces = [run({k: v[i:i+1] for k, v in case.feeds.items()}) for i in range(n)]
    checks.append({"name": "batch_partition", **compare(np.concatenate(pieces, axis=0), y, tol)})
    permutation = np.arange(n)[::-1]
    permuted = run({k: v[permutation] for k, v in case.feeds.items()})
    checks.append({"name": "batch_permutation", **compare(permuted, y[permutation], tol)})
    # Change rank while retaining the feature axis: [N,D] -> [1,N,D].
    shaped = run({k: v.reshape((1,) + v.shape) for k, v in case.feeds.items()})
    checks.append({"name": "batch_reshape", **compare(shaped.reshape(y.shape), y, tol)})
    duplicated = run({k: np.concatenate([v, v], axis=0) for k, v in case.feeds.items()})
    checks.append({"name": "batch_duplicate", **compare(duplicated, np.concatenate([y, y], axis=0), tol)})
    if case.equivalent is not None:
        other = evaluator(case.equivalent, backend, optimization)(case.feeds)
        checks.append({"name": "equivalent_graph", **compare(other, y, tol)})
    return checks
