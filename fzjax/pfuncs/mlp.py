from __future__ import annotations

import itertools
import typing
from dataclasses import dataclass
from typing import Any, Sequence, Union

import jax.random

import fzjax.funcs as funcs
from fzjax.ptree import Meta, fzjax_dataclass
from . import NormType

from ..higher import pfunc_jit
from .batch_norm import BatchNormParams, batch_norm
from .layer_norm import LayerNormParams, layer_norm
from .linear import LinearParams, linear

if typing.TYPE_CHECKING:
    from jax.random import PRNGKeyArray
    from jaxtyping import Array, Float

    from fzjax.initializers import Initializer

NormParams = Union[None, BatchNormParams, LayerNormParams]


@fzjax_dataclass
@dataclass(frozen=True)
class MLPParams:
    linear_params: tuple[LinearParams, ...]
    norm_params: tuple[NormParams, ...]

    activation: Meta[str]

    @classmethod
    def create(
        cls,
        in_features: int,
        out_features: Sequence[int],
        use_bias: bool = True,
        norm_type: NormType = NormType.BATCH_NORM,
        norm_kwargs: dict[str, Any] | None = None,
        *,
        initializer: Initializer,
        activation: str = "relu",
        rng: PRNGKeyArray,
    ) -> MLPParams:
        if not funcs.is_valid_activation(activation):
            raise ValueError(f"Activation '{activation}' is not registered.")
        if not out_features:
            raise ValueError("out_features must have at least one element.")

        if norm_kwargs is None:
            norm_kwargs = {}

        dim_in = in_features
        norm_params = []
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
            if norm_type == NormType.BATCH_NORM:
                norm_params.append(
                    BatchNormParams.create(shape=(1, dim), **norm_kwargs)
                )
            elif norm_type == NormType.LAYER_NORM:
                norm_params.append(
                    LayerNormParams.create(norm_shape=(-1, dim), **norm_kwargs)
                )
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
            linear_params=tuple(linear_params),
            norm_params=tuple(norm_params),
            activation=activation,
        )


@pfunc_jit
def mlp(
    params: MLPParams,
    inputs: Float[Array, "N InC"],
    update_bn_stats: Meta[bool] = False,
) -> tuple[Float[Array, "N OutC"], Any]:
    x = inputs
    bn_states = []
    for p_linear, p_norm in itertools.zip_longest(
        params.linear_params[:-1], params.norm_params
    ):
        x = linear(p_linear, x)
        if isinstance(p_norm, BatchNormParams):
            x, bn_state = batch_norm(p_norm, x, update_stats=update_bn_stats)
            bn_states.append(bn_state)
        elif isinstance(p_norm, LayerNormParams):
            x = layer_norm(p_norm, x)
        x = funcs.activation(params.activation, x)
    x = linear(params.linear_params[-1], x)
    return x, bn_states
