from __future__ import annotations

from dataclasses import dataclass

import jax
from jax.random import PRNGKeyArray
from jaxtyping import Array, Float

from fzjax.initializers import Initializer
from fzjax.ptree import fzjax_dataclass

from .conv import Conv2dParams, conv2d


@fzjax_dataclass
@dataclass(frozen=True)
class TMConv2dParams(Conv2dParams):
    @classmethod
    def create(
        cls,
        in_filters: int,
        out_filters: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
        *,
        initializer: Initializer,
        rng: PRNGKeyArray,
        **kwargs,
    ) -> TMConv2dParams:
        assert not kwargs
        return Conv2dParams.create(
            in_filters=in_filters,
            out_filters=out_filters,
            kernel_size=kernel_size,
            stride=stride,
            groups=groups,
            padding="SAME",
            use_bias=False,
            initializer=initializer,
            rng=rng,
        )


@jax.jit
def tm_conv2d(
    params: TMConv2dParams,
    x: Float[Array, "N InC H W"],
    shifts: Float[Array, "N OutC H W"] | None,
    gains: Float[Array, "N OutC H W"] | None,
    biases: Float[Array, "N OutC H W"] | None,
) -> Float[Array, "N OutC H W"]:
    x = conv2d(params, x)
    if shifts is not None:
        x = x - shifts
    if gains is not None:
        x = x * (1 + gains)
    if biases is not None:
        x = x + biases

    return x
