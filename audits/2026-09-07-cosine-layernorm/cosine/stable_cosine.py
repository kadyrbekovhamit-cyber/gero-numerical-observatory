"""Local repair proposal for real floating-point cosine similarity."""
import mlx.core as mx


def cosine_similarity_loss(x1, x2, axis=1, eps=1e-8, reduction='none'):
    out_dtype = mx.result_type(x1, x2)
    if not mx.issubdtype(out_dtype, mx.floating):
        out_dtype = mx.float32
    work_dtype = mx.result_type(x1, x2, mx.float32)
    a, b = x1.astype(work_dtype), x2.astype(work_dtype)
    s1 = mx.max(mx.abs(a), axis=axis, keepdims=True)
    s2 = mx.max(mx.abs(b), axis=axis, keepdims=True)
    s1 = mx.stop_gradient(mx.where(s1 > 0, s1, 1))
    s2 = mx.stop_gradient(mx.where(s2 > 0, s2, 1))
    u, v = a / s1, b / s2
    q1 = mx.sum(u * u, axis=axis, keepdims=True)
    q2 = mx.sum(v * v, axis=axis, keepdims=True)
    # Keep the inactive square-root path differentiable at zero vectors.
    n1 = mx.sqrt(mx.where(q1 > 0, q1, 1))
    n2 = mx.sqrt(mx.where(q2 > 0, q2, 1))
    dot = mx.sum(u * v, axis=axis, keepdims=True)
    if eps <= 0:
        value = dot / mx.where((q1 > 0) & (q2 > 0), n1 * n2, 0)
    else:
        scaled_eps = (eps / mx.maximum(s1, s2)) / mx.minimum(s1, s2)
        regular = (q1 > 0) & (q2 > 0) & (n1 * n2 > scaled_eps)
        normalized = dot / (n1 * n2)

        # In the clamped branch divide the smaller vector first.
        ca, cb = mx.where(regular, 0, a), mx.where(regular, 0, b)
        small = mx.where(s1 <= s2, ca, cb)
        large = mx.where(s1 <= s2, cb, ca)
        clamped = mx.sum((small / eps) * large, axis=axis, keepdims=True)
        value = mx.where(regular, normalized, clamped)
    loss = mx.squeeze(value, axis=axis).astype(out_dtype)
    if reduction == 'none':
        return loss
    if reduction == 'mean':
        return mx.mean(loss)
    if reduction == 'sum':
        return mx.sum(loss)
    raise ValueError('Invalid reduction')
