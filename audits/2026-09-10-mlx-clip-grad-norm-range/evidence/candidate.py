"""Isolated finite-real gradient clipping prototype; no source-tree edits."""
import mlx.core as mx
from mlx.utils import tree_flatten, tree_map, tree_reduce


def clip_grad_norm(grads, max_norm):
    if max_norm < 0:
        raise ValueError(f"max_norm should be >=0, {max_norm} was provided instead")

    leaves = [g for _, g in tree_flatten(grads)]
    if any(mx.issubdtype(g.dtype, mx.complexfloating) for g in leaves):
        norm_squared = tree_reduce(lambda acc, g: acc + g.square().sum(), grads, 0.0)
        total_norm = mx.sqrt(norm_squared)
        normalizer = mx.minimum(max_norm / (total_norm + 1e-6), 1.0)
        return tree_map(lambda g: g * normalizer, grads), total_norm

    # Keep the public dtype promotion, but accumulate at least in float32.
    norm_dtype = mx.sqrt(tree_reduce(
        lambda acc, g: acc + mx.zeros((), dtype=g.dtype).sum(), grads, 0.0
    )).dtype
    acc_dtype = mx.float64 if norm_dtype == mx.float64 else mx.float32
    work = [g.astype(acc_dtype) for g in leaves if g.size]
    maximum = mx.array(0.0, dtype=acc_dtype)
    for g in work:
        maximum = mx.maximum(maximum, mx.max(mx.abs(g)))
    scale = mx.stop_gradient(mx.where(
        (maximum > 0) & mx.isfinite(maximum), maximum, 1.0
    ))
    squares = mx.array(0.0, dtype=acc_dtype)
    for g in work:
        squares = squares + mx.sum(mx.square(g / scale))
    root = mx.sqrt(mx.where(squares == 0, 1.0, squares))
    norm = mx.where(squares == 0, 0.0, scale * root)
    cap = mx.array(max_norm, dtype=acc_dtype)

    large = scale >= 1.0
    denominator = mx.where(large, 1.0, norm) + 1e-6
    small_factor = mx.minimum(cap / denominator, 1.0)
    active = ~(cap >= norm + 1e-6) | (mx.isinf(cap) & mx.isinf(norm))
    safe_cap = mx.where(active & large, cap, 1.0)
    large_scale = mx.where(large, scale, 1.0)
    large_root = mx.where(large, root + 1e-6 / large_scale, 1.0)

    def clip(g):
        value = g.astype(acc_dtype)
        # Divide the larger numerator factor by the large scale first.
        first = mx.where(mx.abs(value) >= safe_cap, value, safe_cap)
        second = mx.where(mx.abs(value) >= safe_cap, safe_cap, value)
        reduced = (first / large_scale) * (second / large_root)
        result = mx.where(large, mx.where(active, reduced, value), value * small_factor)
        return result.astype(mx.result_type(g.dtype, norm_dtype))

    return tree_map(clip, grads), norm.astype(norm_dtype)
