from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import jax.random
from jax.random import PRNGKeyArray
from jaxtyping import Array, Float

from fzjax import Differentiable, fzjax_dataclass
from fzjax.initializers import Initializer


@fzjax_dataclass
@dataclass(frozen=True)
class LinearParams:

    weights: Differentiable[Float[Array, "OutC InC"]]
    biases: Differentiable[Union[Float[Array, "OutC"], None]]

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
            weights=initializer((out_features, in_features), wkey),
            biases=initializer(
                (out_features,), bkey, pseudo_shape=(out_features, in_features)
            ),
        )


def linear(
    params: LinearParams, x: Float[Array, "... InC"]
) -> Float[Array, "... OutC"]:
    x = params.weights @ x
    if params.biases is not None:
        x = x + params.biases
    return x
