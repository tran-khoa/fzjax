from __future__ import annotations

import os
import typing
import warnings

import jax.numpy as jnp
import pytest
from jax.lib import xla_bridge

from fzjax.higher import pfunc_jit, pfunc_value_and_grad

if typing.TYPE_CHECKING:
    from fzjax.ptree import Differentiable, Donate


os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


def test_pfunc_jit():
    def f(donated: Donate, not_donated):
        return donated + not_donated

    jitted = pfunc_jit(f)
    donated_arr = jnp.zeros((1,))
    non_donated_arr = jnp.ones((1,))
    assert jitted(donated_arr, non_donated_arr).item() == 1.0
    if xla_bridge.get_backend().platform in ("gpu", "tpu"):
        with pytest.raises(RuntimeError):
            print(donated_arr)
    else:
        warnings.warn("Cannot test pfunc_jit donation on CPU.")

    assert non_donated_arr.item()


def test_pfunc_value_and_grad():
    def quadratic(x: Differentiable, y: Differentiable):
        return x**2 + y

    vag_func = pfunc_value_and_grad(quadratic, ["x"])

    value, grad = vag_func(2.0, 1.0)
    assert value.item() == 5.0
    assert grad["x"] == 4.0
    assert set(grad.keys()) == {"x"}

    vag_func2 = pfunc_value_and_grad(quadratic, ["x", "y"])
    value, grad = vag_func2(2.0, 1.0)

    assert value.item() == 5.0
    assert grad["x"] == 4.0
    assert grad["y"] == 1.0
