def clip_grad_norm(grads, max_norm):
    """Clips the global norm of the gradients.

    This function ensures that the global norm of the gradients does not exceed
    ``max_norm``. It scales down the gradients proportionally if their norm is
    greater than ``max_norm``.

    Example:
        >>> grads = {"w1": mx.array([2, 3]), "w2": mx.array([1])}
        >>> clipped_grads, total_norm = clip_grad_norm(grads, max_norm=2.0)
        >>> print(clipped_grads)
        {"w1": mx.array([...]), "w2": mx.array([...])}

    Args:
        grads (dict): A dictionary containing the gradient arrays.
        max_norm (float): The maximum allowed global norm of the gradients.

    Returns:
        (dict, float): The possibly rescaled gradients and the original
        gradient norm.
    """
    if max_norm < 0:
        raise ValueError(f"max_norm should be >=0, {max_norm} was provided instead")

    norm_squared = tree_reduce(lambda acc, g: acc + g.square().sum(), grads, 0.0)
    total_norm = mx.sqrt(norm_squared)
    normalizer = mx.minimum(max_norm / (total_norm + 1e-6), 1.0)
    clipped_grads = tree_map(lambda g: g * normalizer, grads)
    return clipped_grads, total_norm
