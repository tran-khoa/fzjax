from __future__ import annotations

import itertools
import typing
from dataclasses import dataclass
from typing import Any, Sequence, Optional

import jax.random

import fzjax.funcs as funcs
from fzjax.ptree import Meta, fzjax_dataclass

from ..higher import pfunc_jit
from .norms.common import Norm
from .linear import Linear

if typing.TYPE_CHECKING:
    from jax.random import PRNGKeyArray
    from jaxtyping import Array, Float

    from fzjax.initializers import Initializer


@fzjax_dataclass
@dataclass(frozen=True)
class MLP:
    linears: tuple[Linear, ...]
    norms: tuple[Norm, ...]

    activation: Meta[str]

    @classmethod
    def create(
        cls,
        in_features: int,
        out_features: Sequence[int],
        use_bias: bool = True,

        norm_type: Optional[type[Norm]] = None,
        norm_kwargs: dict[str, Any] | None = None,
        *,
        initializer: Initializer,
        activation: str = "relu",
        rng: PRNGKeyArray,
    ) -> MLP:
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
                Linear.create(
                    in_features=dim_in,
                    out_features=dim,
                    use_bias=use_bias,
                    initializer=initializer,
                    rng=lin_rng,
                )
            )
            if norm_type is not None:
                norm_params.append(
                    norm_type.create(shape=(-1, dim), **norm_kwargs)
                )

            dim_in = dim

        rng, lin_rng = jax.random.split(rng)
        linear_params.append(
            Linear.create(
                in_features=dim_in,
                out_features=out_features[-1],
                use_bias=use_bias,
                initializer=initializer,
                rng=lin_rng,
            )
        )

        return MLP(
            linears=tuple(linear_params),
            norms=tuple(norm_params),
            activation=activation,
        )

    def __call__(self,
                 inputs: Float[Array, "N InC"],
                 norm_kwargs: dict[str, Any]) -> tuple[Float[Array, "N OutC"], list[Any]]:
        return mlp(self, inputs, norm_kwargs)


@pfunc_jit
def mlp(
    params: MLP,
    inputs: Float[Array, "N InC"],
    norm_kwargs: dict[str, Any],
) -> tuple[Float[Array, "N OutC"], list[Any]]:
    x = inputs
    norm_states = []
    for p_linear, p_norm in itertools.zip_longest(
            params.linears[:-1], params.norms
    ):
        x = p_linear(x)
        x, norm_state = p_norm(x, **norm_kwargs)
        x = funcs.activation(params.activation, x)
        norm_states.append(norm_state)

    x = params.linears[-1](x)
    return x, norm_states
