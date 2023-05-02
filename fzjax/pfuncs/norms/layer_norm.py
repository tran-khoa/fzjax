from __future__ import annotations

from dataclasses import dataclass
from typing import TypeVar

import chex
import jax.lax as lax
import jax.numpy as jnp

from fzjax.higher import pfunc_jit
from fzjax.pfuncs import Norm
from fzjax.ptree import Differentiable, Meta, fzjax_dataclass

T = TypeVar("T", bound=jnp.ndarray)


@fzjax_dataclass
@dataclass(frozen=True)
class LayerNorm(Norm):
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
        shape: tuple[int, ...],
        dtype: chex.ArrayDType = jnp.float32,
        eps: float = 1e-5,
    ) -> LayerNorm:
        """
        Args:
            shape:
                Tuple of input shape.
                Set to -1 for each axis that should be averaged over, else set to axis size.
            dtype: Input data type.
            eps: A small float added to variance to avoid dividing by zero.
        """
        param_shape = [1 if a <= 0 else a for a in shape]

        return LayerNorm(
            scale=jnp.ones(param_shape, dtype),
            offset=jnp.zeros(param_shape, dtype),
            norm_shape=shape,
            eps=eps,
        )

    def __call__(self, inputs: T, **kwargs) -> tuple[T, None]:
        return layer_norm(self, inputs), None


@pfunc_jit
def layer_norm(
    params: LayerNorm,
    inputs: T,
) -> T:
    reduced_axes = [i for i, v in enumerate(params.norm_shape) if v > 0]
    mean = jnp.mean(inputs, axis=reduced_axes, keepdims=True)
    var = jnp.var(inputs, axis=reduced_axes, keepdims=True)

    eps = lax.convert_element_type(params.eps, inputs.dtype)
    inv = params.scale * lax.rsqrt(var + eps)

    return (inputs - mean) * inv + params.offset
