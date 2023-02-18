from __future__ import annotations
from dataclasses import dataclass
from typing import TypeVar

import chex
import jax.lax as lax
import jax.numpy as jnp

from fzjax.higher import pfunc_jit
from fzjax.ptree import Differentiable, Meta, fzjax_dataclass


@fzjax_dataclass
@dataclass(frozen=True)
class LayerNormParams:
    """
    Parameters:
        scale:
        offset:

    Hyperparameters:
        norm_shape:
            Tuple of input shape.
            Set to -1 for each axis that should be averaged over, else set to axis size.
        eps:
            A small float added to variance to avoid dividing by zero.
    """
    scale: Differentiable[jnp.ndarray]
    offset: Differentiable[jnp.ndarray]

    norm_shape: Meta[tuple[int, ...]]
    eps: Meta[float] = 1e-5

    @classmethod
    def create(
        cls,
        norm_shape: tuple[int, ...],
        dtype: chex.ArrayDType = jnp.float32,
        eps: float = 1e-5,
    ) -> LayerNormParams:
        """
        Args:
            norm_shape:
                Tuple of input shape.
                Set to -1 for each axis that should be averaged over, else set to axis size.
            dtype: Input data type.
            eps: A small float added to variance to avoid dividing by zero.
        """
        param_shape = [1 if a <= 0 else a for a in norm_shape]

        return LayerNormParams(
            scale=jnp.ones(param_shape, dtype),
            offset=jnp.zeros(param_shape, dtype),
            norm_shape=norm_shape,
            eps=eps,
        )


T = TypeVar("T", bound=jnp.ndarray)


@pfunc_jit
def layer_norm(
    params: LayerNormParams,
    inputs: T,
) -> T:
    reduced_axes = [i for i, v in enumerate(params.norm_shape) if v > 0]
    mean = jnp.mean(inputs, axis=reduced_axes, keepdims=True)
    var = jnp.var(inputs, axis=reduced_axes, keepdims=True)

    eps = lax.convert_element_type(params.eps, inputs.dtype)
    inv = params.scale * lax.rsqrt(var + eps)

    return (inputs - mean) * inv + params.offset
