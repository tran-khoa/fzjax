from __future__ import annotations

import itertools
from dataclasses import dataclass
from functools import partial
from typing import Any

import jax.random
from jax.random import PRNGKeyArray
from jaxtyping import Array, Float

import fzjax.funcs as funcs
from fzjax.initializers import Initializer
from fzjax.ptree import Meta, fzjax_dataclass

from .batch_norm import BatchNormParams, batch_norm
from .linear import LinearParams, linear


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
        out_features: list[int],
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

        for dim in out_features[:-1]:
            rng, lin_rng = jax.random.split(rng)
            linear_params.append(
                LinearParams.create(
                    in_features=dim_in,
                    out_features=dim,
                    use_bias=use_bias,
                    initializer=initializer,
                    rng=lin_rng,
                )
            )
            if use_bn:
                bn_params.append(BatchNormParams.create(shape=(1, dim), **bn_kwargs))
            dim_in = dim

        rng, lin_rng = jax.random.split(rng)
        linear_params.append(
            LinearParams.create(
                in_features=dim_in,
                out_features=out_features[-1],
                use_bias=use_bias,
                initializer=initializer,
                rng=lin_rng,
            )
        )

        return MLPParams(
            linear_params=linear_params, bn_params=bn_params, activation=activation
        )


@partial(jax.jit, static_argnames="is_training")
def mlp(
    params: MLPParams, inputs: Float[Array, "N InC"], is_training: bool = False
) -> tuple[Float[Array, "N OutC"], Any]:
    x = inputs
    bn_states = []
    for p_linear, p_bn in itertools.zip_longest(
        params.linear_params[:-1], params.bn_params
    ):
        x = linear(p_linear, x)
        if p_bn is not None:
            x, bn_state = batch_norm(p_bn, x, is_training)
            bn_states.append(bn_state)
        x = funcs.activation(params.activation, x)
    x = linear(params.linear_params[-1], x)
    return x, bn_states
