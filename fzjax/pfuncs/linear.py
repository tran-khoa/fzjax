from __future__ import annotations

import typing
from dataclasses import dataclass
from typing import Optional

import jax.random

from fzjax.ptree import Differentiable, fzjax_dataclass

if typing.TYPE_CHECKING:
    from jax.random import PRNGKeyArray
    from jaxtyping import Array, Float

    from fzjax.initializers import Initializer


@fzjax_dataclass
@dataclass(frozen=True)
class LinearParams:
    weights: Differentiable[Float[Array, "OutC InC"]]
    biases: Differentiable[Optional[Float[Array, "OutC"]]]

    @classmethod
    def create(
        cls,
        in_features: int,
        out_features: int,
        use_bias: bool = True,
        *,
        initializer: Initializer,
        rng: PRNGKeyArray,
    ) -> LinearParams:
        wkey, bkey = jax.random.split(rng)

        return LinearParams(
            weights=initializer((in_features, out_features), wkey),
            biases=initializer(
                (out_features,), bkey, pseudo_shape=(out_features, in_features)
            )
            if use_bias
            else None,
        )


@jax.jit
def linear(
    params: LinearParams, x: Float[Array, "... InC"]
) -> Float[Array, "... OutC"]:
    x = x @ params.weights
    if params.biases is not None:
        x = x + params.biases
    return x
