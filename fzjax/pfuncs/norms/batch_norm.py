from __future__ import annotations

import typing
from dataclasses import dataclass
from typing import TypeVar

import chex
import jax.lax as lax
import jax.numpy as jnp

from fzjax.higher import pfunc_jit
from fzjax.pfuncs import Norm
from fzjax.ptree import Differentiable, Meta, fzjax_dataclass

if typing.TYPE_CHECKING:
    from jaxtyping import Array, Integer


T = TypeVar("T", bound=jnp.ndarray)


@pfunc_jit
def batch_norm(
    params: BatchNorm,
    inputs: T,
    update_stats: Meta[bool] = False,
    compute_stats: Meta[bool] = False,
) -> tuple[T, BatchNormStates]:
    r_axes = [i for i, v in enumerate(params.shape) if v == -1]

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


@fzjax_dataclass
class BatchNormStates:
    mean_average: jnp.ndarray
    var_average: jnp.ndarray
    mean_hidden: jnp.ndarray
    var_hidden: jnp.ndarray
    counter: Integer[Array, ""]


@fzjax_dataclass
@dataclass(frozen=True)
class BatchNorm(Norm):
    scale: Differentiable[jnp.ndarray]
    offset: Differentiable[jnp.ndarray]

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
    ) -> BatchNorm:
        """
        Args:
            shape: Shape of batch norm. Set to -1 for each axis that should be averaged over, else set to dimension.
            dtype: Input data type.
            decay_rate: Decay rate of the EMA of the means and variances.
            eps: A small float added to variance to avoid dividing by zero.
        """
        param_shape = [1 if a <= 0 else a for a in shape]

        return BatchNorm(
            scale=jnp.ones(param_shape, dtype),
            offset=jnp.zeros(param_shape, dtype),
            states=BatchNormStates(
                mean_average=jnp.zeros(param_shape, dtype),
                mean_hidden=jnp.zeros(param_shape, dtype),
                var_average=jnp.zeros(param_shape, dtype),
                var_hidden=jnp.zeros(param_shape, dtype),
                counter=jnp.zeros((1,), dtype=jnp.int32),
            ),
            shape=shape,
            decay_rate=decay_rate,
            eps=eps,
        )

    def __call__(
            self,
            inputs: T,
            update_stats: Meta[bool] = False,
            compute_stats: Meta[bool] = False,
    ) -> tuple[T, BatchNormStates]:
        return batch_norm(self, inputs, update_stats, compute_stats)
