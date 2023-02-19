from __future__ import annotations

from typing import Callable, Protocol, TypeVar

import jax.nn

T = TypeVar("T")


class ActivationFunction(Protocol[T]):
    def __call__(self, x: T) -> T:
        ...


_registered_activations: dict[str, ActivationFunction] = {
    "identity": lambda x: x,
    "relu": jax.nn.relu,
    "relu6": jax.nn.relu6,
    "sigmoid": jax.nn.sigmoid,
    "softplus": jax.nn.softplus,
    "soft_sign": jax.nn.soft_sign,
    "silu": jax.nn.silu,
    "swish": jax.nn.swish,
    "log_sigmoid": jax.nn.log_sigmoid,
    "leaky_relu": jax.nn.leaky_relu,
    "hard_sigmoid": jax.nn.hard_sigmoid,
    "hard_silu": jax.nn.hard_silu,
    "hard_swish": jax.nn.hard_swish,
    "hard_tanh": jax.nn.hard_tanh,
    "elu": jax.nn.elu,
    "celu": jax.nn.celu,
    "selu": jax.nn.selu,
    "gelu": jax.nn.gelu,
    "glu": jax.nn.glu,
}


def is_valid_activation(func: str) -> bool:
    return func in _registered_activations


def activation(func: str, x: T) -> T:
    if func not in _registered_activations:
        raise ValueError(f"Invalid activation function {func}.")
    return _registered_activations[func](x)


def register_fzjax_activation(name: str) -> Callable[[T], T]:
    def inner(func: T) -> T:
        if name in _registered_activations:
            raise ValueError(f"Activation {name} already registered.")
        _registered_activations[name] = func
        return func

    return inner
