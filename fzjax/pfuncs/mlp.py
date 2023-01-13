from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any, Collection

import jax.random
from jax.random import PRNGKeyArray
from jaxtyping import Array, Float

from fzjax import Meta, funcs, fzjax_dataclass
from fzjax.initializers import Initializer
from fzjax.pfuncs import BatchNormParams, batch_norm
from fzjax.pfuncs.linear import LinearParams, linear


@fzjax_dataclass
@dataclass(frozen=True)
class MLPParams:
    linear_params: list[LinearParams]
    bn_params: list[BatchNormParams]

    activation: Meta[str]

    @classmethod
    def create(
        cls,
        in_features: int,
        out_features: Collection[int],
        use_bias: bool = True,
        use_bn: bool = True,
        bn_kwargs: dict[str, Any] | None = None,
        *,
        initializer: Initializer,
        activation: str = "relu",
        rng: PRNGKeyArray,
    ) -> MLPParams:

        if bn_kwargs is None:
            bn_kwargs = {}

        dim_in = in_features
        bn_params = []
        linear_params = []

        for dim in out_features:
            rng, lin_rng = jax.random.split(rng)
            linear_params.append(
                LinearParams.create(
                    dim_in, dim, use_bias=use_bias, initializer=initializer, rng=lin_rng
                )
            )
            if use_bn:
                bn_params.append(BatchNormParams.create(shape=(1, dim), **bn_kwargs))
            dim_in = dim
        return MLPParams(linear_params, bn_params, activation)


def mlp(
    params: MLPParams, inputs: Float[Array, "N InC"], is_training: bool = False
) -> tuple[Float[Array, "N OutC"], Any]:
    x = inputs
    bn_states = None
    for p_linear, p_bn in itertools.zip_longest(params.linear_params, params.bn_params):
        x = linear(p_linear, x)
        if p_bn is not None:
            x, bn_states = batch_norm(p_bn, x, is_training)
        x = funcs.activation(params.activation, x)
    return x, bn_states
