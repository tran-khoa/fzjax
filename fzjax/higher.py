from __future__ import annotations

from typing import (Any, Callable, Collection, Iterable, Optional, TypeVar,
                    Union)

import jax
from typing_extensions import Concatenate, ParamSpec

from fzjax.ptree import (JDC_META_MARKER, ptree_differentiable, ptree_filter,
                         ptree_update)

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
        new_params = ptree_update(params, p)
        return pfunc(new_params, *args, **kwargs)

    vag = jax.value_and_grad(target_func, *jax_args, argnums=0, **jax_kwargs)

    def new_vag(
        params: P, *args: PS.args, **kwargs: PS.kwargs
    ) -> tuple[R, dict[str, Any]]:
        diff_params = ptree_differentiable(params, fields)
        return vag(diff_params, params, *args, **kwargs)

    return new_vag


def pfunc_jit(
    pfunc: Callable[Concatenate[P, PS], R],
    donate_argpaths: Optional[Collection[str]] = None,
    donate_argnums: Union[int, Iterable[int]] = (),
    static_argnums: Union[int, Iterable[int], None] = None,
    **jax_kwargs,
):
    if not donate_argpaths:
        return jax.jit(pfunc, **jax_kwargs)

    if "static_argnames" in jax_kwargs:
        raise ValueError("static_argnames not supported!")

    if "donated_argnames" in jax_kwargs:
        raise ValueError("donated_argnames not supported!")

    def target_func(
        p: dict[str, Any], params: P, *args: PS.args, **kwargs: PS.kwargs
    ) -> R:
        merged_params = ptree_update(params, p)  # is this even necessary?
        return pfunc(merged_params, *args, **kwargs)

    if isinstance(donate_argnums, int):
        donate_argnums = (0, donate_argnums + 1)
    else:
        donate_argnums = (0,) + tuple(n + 1 for n in donate_argnums)

    if isinstance(static_argnums, int):
        static_argnums = static_argnums + 1
    elif isinstance(static_argnums, Iterable):
        static_argnums = tuple(n + 1 for n in static_argnums)

    def new_jit(params: P, *args: PS.args, **kwargs: PS.kwargs) -> R:
        donated = ptree_filter(
            params,
            lambda p, v: any(p.startswith(q) for q in donate_argpaths)
            and JDC_META_MARKER not in v.meta,
        )
        return jax.jit(
            target_func,
            static_argnums=static_argnums,
            donate_argnums=donate_argnums,
            **jax_kwargs,
        )(donated, params, *args, **kwargs)

    return new_jit
