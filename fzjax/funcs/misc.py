from __future__ import annotations

import typing

import jax.lax as lax
import jax.numpy as jnp

from fzjax.higher import pfunc_jit
from fzjax.ptree import Meta

if typing.TYPE_CHECKING:
    from jaxtyping import Array, Float


@pfunc_jit
def normalize(
    inputs: Float[Array, "*axes"], axis: Meta[int] = -1, eps: float = 1e-5
) -> Float[Array, "*axes"]:
    if axis == -1:
        axis = len(inputs.shape) - 1

    r_axes = [i for i, v in enumerate(inputs.shape) if v == axis]

    mean = jnp.mean(inputs, r_axes, keepdims=True)
    mean_of_squares = jnp.mean(jnp.square(inputs), r_axes, keepdims=True)
    var = mean_of_squares - jnp.square(mean)

    eps = lax.convert_element_type(eps, inputs.dtype)

    return (inputs - mean) * lax.rsqrt(var + eps)
