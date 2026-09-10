class BatchNorm(Module):
    r"""Applies Batch Normalization over a 2D, 3D or 4D input.

    Computes

    .. math::

        y = \frac{x - E[x]}{\sqrt{Var[x] + \epsilon}} \gamma + \beta,

    where :math:`\gamma` and :math:`\beta` are learned per feature dimension
    parameters initialized at 1 and 0 respectively.

    The input shape is specified as ``NC`` or ``NLC``, where ``N`` is the
    batch, ``C`` is the number of features or channels, and ``L`` is the
    sequence length. The output has the same shape as the input. For
    four-dimensional arrays, the shape is ``NHWC``, where ``H`` and ``W`` are
    the height and width respectively.

    For more information on Batch Normalization, see the original paper `Batch
    Normalization: Accelerating Deep Network Training by Reducing Internal
    Covariate Shift <https://arxiv.org/abs/1502.03167>`_.

    Args:
        num_features (int): The feature dimension to normalize over.
        eps (float, optional): A small additive constant for numerical
            stability. Default: ``1e-5``.
        momentum (float, optional): The momentum for updating the running
            mean and variance. Default: ``0.1``.
        affine (bool, optional): If ``True``, apply a learned affine
            transformation after the normalization. Default: ``True``.
        track_running_stats (bool, optional): If ``True``, track the
            running mean and variance. Default: ``True``.

    Examples:
        >>> import mlx.core as mx
        >>> import mlx.nn as nn
        >>> x = mx.random.normal((5, 4))
        >>> bn = nn.BatchNorm(num_features=4, affine=True)
        >>> output = bn(x)
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ):
        super().__init__()
        if eps <= 0.0:
            raise ValueError(f"[BatchNorm] 'eps' must be positive but got {eps}.")

        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.track_running_stats = track_running_stats

        if affine:
            self.weight = mx.ones((num_features,))
            self.bias = mx.zeros((num_features,))

        if self.track_running_stats:
            self.running_mean = mx.zeros((num_features,))
            self.running_var = mx.ones((num_features,))
            self.freeze(keys=["running_mean", "running_var"], recurse=False)

    def unfreeze(self, *args, **kwargs):
        """Wrap unfreeze to make sure that running_mean and var are always
        frozen parameters."""
        super().unfreeze(*args, **kwargs)
        self.freeze(keys=["running_mean", "running_var"], recurse=False)

    def _extra_repr(self):
        return (
            f"{self.num_features}, eps={self.eps}, "
            f"momentum={self.momentum}, affine={'weight' in self}, "
            f"track_running_stats={self.track_running_stats}"
        )

    def _calc_stats(self, x: mx.array, ddof: int = 0) -> Tuple[mx.array, mx.array]:
        """
        Calculate the mean and variance of the input tensor across the batch
        and spatial dimensions.

        Args:
            x (array): Input tensor.
            ddof (int): Delta degrees of freedom for variance.

        Returns:
            tuple: Tuple containing mean and variance.
        """
        reduction_axes = tuple(range(0, x.ndim - 1))

        mean = mx.mean(x, axis=reduction_axes)
        var = mx.var(x, axis=reduction_axes, ddof=ddof)

        return mean, var

    def __call__(self, x: mx.array) -> mx.array:
        """
        Forward pass of BatchNorm.

        Args:
            x (array): Input tensor.

        Returns:
            array: Normalized output tensor.
        """
        if x.ndim < 2 or x.ndim > 4:
            raise ValueError(
                f"Expected input tensor to have 2, 3 or 4 dimensions, but got {x.ndim}"
            )

        if self.training:
            stats_size = 1
            for size in x.shape[:-1]:
                stats_size *= size
            if stats_size == 1:
                raise ValueError(
                    "BatchNorm training requires more than one value per channel."
                )

        # Calculate the mean and variance used to normalize the input x. If we
        # are in training mode update the running stats if needed.
        mean, var = self._calc_stats(x)
        if self.training and self.track_running_stats:
            mu = self.momentum
            _, running_var = self._calc_stats(x, ddof=1)
            self.running_mean = (1 - mu) * self.running_mean + mu * mean
            self.running_var = (1 - mu) * self.running_var + mu * running_var
        elif self.track_running_stats:
            mean = self.running_mean
            var = self.running_var

        x = (x - mean) * mx.rsqrt(var + self.eps)
        return (self.weight * x + self.bias) if "weight" in self else x
