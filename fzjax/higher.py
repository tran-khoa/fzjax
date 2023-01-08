from __future__ import annotations

from typing import Any, Callable, Collection, Optional, TypeVar

import jax
from typing_extensions import Concatenate, ParamSpec

from fzjax.named_tree.utils import named_tree_differentiable, named_tree_update

P = TypeVar("P")
PS = ParamSpec("PS")

R = TypeVar("R")


def pfunc_value_and_grad(
    pfunc: Callable[Concatenate[P, PS], R],
    fields: Optional[Collection[str]] = None,
    *jax_args,
    **jax_kwargs,
):
    def target_func(
        p: dict[str, Any], params: P, *args: PS.args, **kwargs: PS.kwargs
    ) -> R:
        new_params = named_tree_update(params, p)
        return pfunc(new_params, *args, **kwargs)

    vag = jax.value_and_grad(target_func, *jax_args, argnums=0, **jax_kwargs)

    def new_vag(
        params: P, *args: PS.args, **kwargs: PS.kwargs
    ) -> tuple[R, dict[str, Any]]:
        diff_params = named_tree_differentiable(params, fields)
        return vag(diff_params, params, *args, **kwargs)

    return new_vag
