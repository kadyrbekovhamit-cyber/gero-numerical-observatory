class GroupNorm(Module):
    r"""Applies Group Normalization [1] to the inputs.

    Computes the same normalization as layer norm, namely

    .. math::

        y = \frac{x - E[x]}{\sqrt{Var[x] + \epsilon}} \gamma + \beta,

    where :math:`\gamma` and :math:`\beta` are learned per feature dimension
    parameters initialized at 1 and 0 respectively. However, the mean and
    variance are computed over the spatial dimensions and each group of
    features. In particular, the input is split into num_groups across the
    feature dimension.

    The feature dimension is assumed to be the last dimension and the dimensions
    that precede it (except the first) are considered the spatial dimensions.

    [1]: https://arxiv.org/abs/1803.08494

    Args:
        num_groups (int): Number of groups to separate the features into
        dims (int): The feature dimensions of the input to normalize over
        eps (float): A small additive constant for numerical stability.
            Default: ``1e-5``.
        affine (bool): If True learn an affine transform to apply after the
            normalization. Default: ``True``.
        pytorch_compatible (bool): If True perform the group normalization in
            the same order/grouping as PyTorch. Default: ``False``.
    """

    def __init__(
        self,
        num_groups: int,
        dims: int,
        eps: float = 1e-5,
        affine: bool = True,
        pytorch_compatible: bool = False,
    ):
        super().__init__()
        if eps <= 0.0:
            raise ValueError(f"[GroupNorm] 'eps' must be positive but got {eps}.")
        if num_groups <= 0:
            raise ValueError(
                f"The number of groups ({num_groups}) must be a positive integer."
            )
        if dims <= 0:
            raise ValueError(
                f"The number of features ({dims}) must be a positive integer."
            )
        if dims % num_groups != 0:
            raise ValueError(
                f"The number of features ({dims}) must be evenly divisible"
                f" by the number of groups ({num_groups})."
            )
        if affine:
            self.bias = mx.zeros((dims,))
            self.weight = mx.ones((dims,))
        self.num_groups = num_groups
        self.dims = dims
        self.eps = eps
        self.pytorch_compatible = pytorch_compatible

    def _extra_repr(self):
        return (
            f"{self.num_groups}, {self.dims}, eps={self.eps}, "
            f"affine={'weight' in self}, pytorch_compatible={self.pytorch_compatible}"
        )

    def _pytorch_compatible_group_norm(self, x):
        num_groups = self.num_groups
        batch, *rest, dims = x.shape
        group_size = dims // num_groups

        # Split into groups
        x = x.reshape(batch, -1, num_groups, group_size)
        x = x.transpose(0, 2, 1, 3).reshape(batch, num_groups, -1)

        # Normalize
        x = mx.fast.layer_norm(x, eps=self.eps, weight=None, bias=None)

        x = x.reshape(batch, num_groups, -1, group_size)
        x = x.transpose(0, 2, 1, 3).reshape(batch, *rest, dims)
        return x

    def _group_norm(self, x):
        num_groups = self.num_groups
        batch, *rest, dims = x.shape

        # Split into groups
        x = x.reshape(batch, -1, num_groups)

        # Normalize
        means = mx.mean(x, axis=1, keepdims=True)
        var = mx.var(x, axis=1, keepdims=True)
        x = (x - means) * mx.rsqrt(var + self.eps)
        x = x.reshape(batch, *rest, dims)

        return x

    def __call__(self, x):
        group_norm = (
            self._pytorch_compatible_group_norm
            if self.pytorch_compatible
            else self._group_norm
        )
        x = group_norm(x)
        return (self.weight * x + self.bias) if "weight" in self else x
