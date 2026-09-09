"""Bounded mitigation experiment, not a universal numerical-stability fix."""
import math
import mlx.core as mx


def gaussian_candidate(inputs, targets, vars, full=False, eps=1e-6,
                       reduction='mean'):
    if inputs.shape != targets.shape or inputs.shape != vars.shape:
        raise ValueError('Input, target and variance shapes must match.')
    x, y, v = (a.astype(mx.float32) for a in (inputs, targets, vars))
    v = mx.maximum(v, eps)
    residual = (x-y) / mx.sqrt(v)
    loss = 0.5 * (mx.log(v) + mx.square(residual))
    if full:
        loss = loss + 0.5 * math.log(2 * math.pi)
    if reduction == 'none':
        return loss
    if reduction == 'mean':
        return mx.mean(loss)
    if reduction == 'sum':
        return mx.sum(loss)
    raise ValueError('Invalid reduction.')
