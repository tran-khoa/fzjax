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
class Linear:
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
        bias_initializer: Optional[Initializer] = None,
        rng: PRNGKeyArray,
    ) -> Linear:
        wkey, bkey = jax.random.split(rng)

        if bias_initializer is None:
            bias_initializer = initializer

        return Linear(
            weights=initializer((in_features, out_features), wkey),
            biases=bias_initializer(
                (out_features,), bkey, pseudo_shape=(out_features, in_features)
            )
            if use_bias
            else None,
        )

    def __call__(self, x: Float[Array, "... InC"]) -> Float[Array, "... OutC"]:
        return linear(self, x)


@jax.jit
def linear(
    params: Linear, x: Float[Array, "... InC"]
) -> Float[Array, "... OutC"]:
    x = x @ params.weights
    if params.biases is not None:
        x = x + params.biases
    return x
