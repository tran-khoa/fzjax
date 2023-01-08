from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Callable

import jax.numpy as jnp
from jax.random import PRNGKeyArray
from jaxtyping import Array, Float

from fzjax import fzjax_dataclass

from ..initializers import Initializer
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
        dtype: Any = jnp.float32,
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
            dtype=dtype,
            use_bias=False,
            initializer=initializer,
            rng=rng,
        )


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
        x = x - (1 + gains)
    if biases is not None:
        x = x - biases

    return x
