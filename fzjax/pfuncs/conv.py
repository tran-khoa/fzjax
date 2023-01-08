from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import jax.lax as lax
import jax.numpy as jnp
import jax.random
from jax.random import PRNGKeyArray
from jaxtyping import Array, Float

from fzjax import Differentiable, Meta, fzjax_dataclass
from fzjax.initializers import Initializer


@fzjax_dataclass
@dataclass(frozen=True)
class Conv2dParams:
    filters: Differentiable[Float[Array, "OutC InC K K"]]
    biases: Differentiable[Optional[Float[Array, "Out"]]]

    stride: Meta[int]
    groups: Meta[int]
    padding: str

    dtype: Meta[Any]

    @classmethod
    def create(
        cls,
        in_filters: int,
        out_filters: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
        padding: str = "SAME",
        dtype: Any = jnp.float32,
        *,
        use_bias: bool = False,
        initializer: Initializer,
        rng: PRNGKeyArray,
    ):
        wkey, bkey = jax.random.split(rng)
        filters = initializer(
            (out_filters, in_filters, kernel_size, kernel_size), rng=wkey
        )
        biases = None
        if use_bias:
            biases = initializer(
                (out_filters,), rng=bkey, pseudo_shape=(out_filters, in_filters)
            )

        return cls(filters, biases, stride, groups, padding, dtype)

    @property
    def in_filters(self):
        return self.filters.shape[1]

    @property
    def out_filters(self):
        return self.filters.shape[0]


def conv2d(
    params: Conv2dParams, x: Float[Array, "N InC InH InW"]
) -> Float[Array, "N OutC OutH OutW"]:
    x = lax.conv_general_dilated(
        lhs=x,
        rhs=params.filters,
        padding="SAME",
        window_strides=(params.stride, params.stride),
        dimension_numbers=("NCHW", "OIHW", "NCHW"),
    )
    if params.biases is not None:
        x += jnp.reshape(params.biases, (1, -1, 1, 1))

    return x
