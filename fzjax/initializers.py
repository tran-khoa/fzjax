from __future__ import annotations

import math
import typing
from dataclasses import dataclass
from typing import Protocol

import chex
import jax
import jax.numpy as jnp

if typing.TYPE_CHECKING:
    from jax.random import PRNGKeyArray


class Initializer(Protocol):
    def __call__(
        self,
        shape: tuple[int, ...],
        rng: PRNGKeyArray,
        *,
        pseudo_shape: tuple[int, ...] | None = None,
    ) -> jnp.ndarray:
        ...


def compute_fans(shape, fan_in_axes=None):
    """Adapted from PyTorch. Computes the number of input and output units for a weight shape."""
    if len(shape) < 1:
        fan_in = fan_out = 1
    elif len(shape) == 1:
        fan_in = fan_out = shape[0]
    elif len(shape) == 2:
        fan_in, fan_out = shape
    else:
        if fan_in_axes is not None:
            # Compute fan-in using user-specified fan-in axes.
            fan_in = math.prod([shape[i] for i in fan_in_axes])
            fan_out = math.prod(
                [s for i, s in enumerate(shape) if i not in fan_in_axes]
            )
        else:
            # If no axes specified, assume convolution kernels (2D, 3D, or more.)
            # kernel_shape: (..., input_depth, depth)
            receptive_field_size = math.prod(shape[:-2])
            fan_in = shape[-2] * receptive_field_size
            fan_out = shape[-1] * receptive_field_size
    return fan_in, fan_out


def calculate_gain(nonlinearity, param=None):
    r"""Adapted from PyTorch. Return the recommended gain value for the given nonlinearity function.
    The values are as follows:

    ================= ====================================================
    nonlinearity      gain
    ================= ====================================================
    Linear / Identity :math:`1`
    Conv{1,2,3}D      :math:`1`
    Sigmoid           :math:`1`
    Tanh              :math:`\frac{5}{3}`
    ReLU              :math:`\sqrt{2}`
    Leaky Relu        :math:`\sqrt{\frac{2}{1 + \text{negative\_slope}^2}}`
    SELU              :math:`\frac{3}{4}`
    ================= ====================================================

    .. warning::
        In order to implement `Self-Normalizing Neural Networks`_ ,
        you should use ``nonlinearity='linear'`` instead of ``nonlinearity='selu'``.
        This gives the initial weights a variance of ``1 / N``,
        which is necessary to induce a stable fixed point in the forward pass.
        In contrast, the default gain for ``SELU`` sacrifices the normalisation
        effect for more stable gradient flow in rectangular layers.

    Args:
        nonlinearity: the non-linear function (`nn.functional` name)
        param: optional parameter for the non-linear function
    """
    linear_fns = [
        "linear",
        "conv1d",
        "conv2d",
        "conv3d",
        "conv_transpose1d",
        "conv_transpose2d",
        "conv_transpose3d",
    ]
    if nonlinearity in linear_fns or nonlinearity == "sigmoid":
        return 1
    elif nonlinearity == "tanh":
        return 5.0 / 3
    elif nonlinearity == "relu":
        return math.sqrt(2.0)
    elif nonlinearity == "leaky_relu":
        if param is None:
            negative_slope = 0.01
        elif (
            not isinstance(param, bool)
            and isinstance(param, int)
            or isinstance(param, float)
        ):
            # True/False are instances of int, hence check above
            negative_slope = param
        else:
            raise ValueError("negative_slope {} not a valid number".format(param))
        return math.sqrt(2.0 / (1 + negative_slope**2))
    elif nonlinearity == "selu":
        return (
            3.0 / 4
        )  # Value found empirically (https://github.com/pytorch/pytorch/pull/50664)
    else:
        raise ValueError("Unsupported nonlinearity {}".format(nonlinearity))


@dataclass
class KaimingUniformInitializer(Initializer):
    in_axes: None | tuple[int, ...] = None
    a: float = 0
    mode: str = "fan_in"
    nonlinearity: str = "leaky_relu"
    dtype: chex.ArrayDType = jnp.float32

    def __call__(
        self,
        shape: tuple[int, ...],
        rng: PRNGKeyArray,
        *,
        pseudo_shape: tuple[int, ...] | None = None,
    ) -> jnp.ndarray:
        fan_in, fan_out = compute_fans(pseudo_shape or shape, fan_in_axes=self.in_axes)
        fan = fan_in if self.mode == "fan_in" else fan_out

        gain = calculate_gain(self.nonlinearity, self.a)
        std = gain / math.sqrt(fan)
        bound = math.sqrt(3.0) * std
        return jax.random.uniform(
            rng, shape, minval=-bound, maxval=bound, dtype=self.dtype
        )
