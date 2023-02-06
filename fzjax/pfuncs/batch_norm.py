from __future__ import annotations

import typing
from dataclasses import dataclass
from functools import partial

import chex
import jax
import jax.lax as lax
import jax.numpy as jnp

from fzjax.ptree import Differentiable, Meta, fzjax_dataclass
from fzjax.higher import pfunc_jit

if typing.TYPE_CHECKING:
    from jaxtyping import Array, Float, Integer


@fzjax_dataclass
class BatchNormStates:
    mean_average: Float[Array, "*axes"]
    var_average: Float[Array, "*axes"]
    mean_hidden: Float[Array, "*axes"]
    var_hidden: Float[Array, "*axes"]
    counter: Integer[Array, ""]


@fzjax_dataclass
@dataclass(frozen=True)
class BatchNormParams:
    scale: Differentiable[Float[Array, "*axes"]]
    offset: Differentiable[Float[Array, "*axes"]]

    states: BatchNormStates

    shape: Meta[tuple[int, ...]]
    decay_rate: Meta[float] = 0.999
    eps: Meta[float] = 1e-5

    @classmethod
    def create(
        cls,
        shape: tuple[int, ...],
        dtype: chex.ArrayDType = jnp.float32,
        decay_rate: float = 0.999,
        eps: float = 1e-5,
    ) -> BatchNormParams:
        """
        Args:
            shape: Shape of batch norm. Set to -1 for each axis that should be averaged over, else set to dimension.
            dtype: Input data type.
            decay_rate: Decay rate of the EMA of the means and variances.
            eps: A small float added to variance to avoid dividing by zero.
        """
        return BatchNormParams(
            scale=jnp.ones(shape, dtype),
            offset=jnp.zeros(shape, dtype),
            states=BatchNormStates(
                mean_average=jnp.zeros(shape, dtype),
                mean_hidden=jnp.zeros(shape, dtype),
                var_average=jnp.zeros(shape, dtype),
                var_hidden=jnp.zeros(shape, dtype),
                counter=jnp.zeros((1,), dtype=jnp.int32),
            ),
            shape=shape,
            decay_rate=decay_rate,
            eps=eps,
        )


@pfunc_jit
def batch_norm(
    params: BatchNormParams,
    inputs: Float[Array, "*axes"],
    update_stats: Meta[bool] = False,
    compute_stats: Meta[bool] = False,
) -> tuple[Float[Array, "*axes"], BatchNormStates]:
    r_axes = [i for i, v in enumerate(params.shape) if v == 1]

    new_state = params.states
    if compute_stats or update_stats:
        mean = jnp.mean(inputs, r_axes, keepdims=True)
        mean_of_squares = jnp.mean(jnp.square(inputs), r_axes, keepdims=True)
        var = mean_of_squares - jnp.square(mean)

        if update_stats:
            counter = params.states.counter + 1
            decay_rate = lax.convert_element_type(params.decay_rate, inputs.dtype)
            one = jnp.ones([], inputs.dtype)

            mean_hidden = params.states.mean_hidden * decay_rate + mean * (
                1 - decay_rate
            )
            var_hidden = params.states.var_hidden * decay_rate + var * (1 - decay_rate)

            mean_average = mean_hidden / (one - jnp.power(decay_rate, counter))
            var_average = var_hidden / (one - jnp.power(decay_rate, counter))

            new_state = BatchNormStates(
                mean_average=mean_average,
                var_average=var_average,
                mean_hidden=mean_hidden,
                var_hidden=var_hidden,
                counter=counter,
            )
    else:
        mean = params.states.mean_average.astype(inputs.dtype)
        var = params.states.var_average.astype(inputs.dtype)

    eps = lax.convert_element_type(params.eps, inputs.dtype)
    inv = params.scale * lax.rsqrt(var + eps)

    return (inputs - mean) * inv + params.offset, new_state
