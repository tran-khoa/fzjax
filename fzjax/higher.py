from __future__ import annotations

import dataclasses
from functools import wraps
from typing import Callable, Collection, TypeVar, Union

import jax
from chex import ArrayTree
from typing_extensions import ParamSpec

from fzjax.ptree import (
    AnnotationPredicate,
    Donate,
    SelectPredicate,
    ptree_differentiable,
    ptree_filter,
    ptree_update,
)
from fzjax.ptree.internal_helpers import fzjax_datacls_from_func, get_func_signature
from fzjax.ptree.utils import NonNullPredicate

P = TypeVar("P")
PS = ParamSpec("PS")

R = TypeVar("R", ArrayTree, tuple[ArrayTree, ArrayTree])

Selectors = Collection[str]


def pfunc_value_and_grad(
    pfunc: Callable[PS, R],
    diff_paths: Collection[str] = (),
    return_diff_params: bool = False,
    *jax_args,
    argnums: None = None,
    **jax_kwargs,
) -> Callable[
    PS,
    Union[
        tuple[R, dict[str, ArrayTree]], tuple[tuple[R, dict[str, ArrayTree]]], ArrayTree
    ],
]:
    """
    Create a function that evaluates both ``pfunc`` and the gradient of ``pfunc``.

    Args:
        pfunc: Function returning either a single pytree or a pair of pytrees.
        diff_paths: selectors (see below)
        return_diff_params: returns dictionary of differentiated parameters
        *jax_args: positional arguments passed to jax.value_and_grad
        argnums: throws exception if not None, use diff_paths instead
        **jax_kwargs: keyword arguments passed to jax.value_and_grad

    Annotations:
        Meta
        Differentiable
        NoDiff

    Selectors:
        Selectors start with one of the argument names, e.g.
        ```python
        def f(x, y, z):
            ...

        diff_paths = ["x.value", "y", "z"]
        ```

        If paths not given, all leaves marked as Differentiable will be selected.
        If paths given, only Differentiable leaves of nodes selected by given paths are selected.

        Annotations are always inherited to children and grandchildren.
        NoDiff overrides previous Differentiable annotations.


    Returns:
        Values and gradient of selected inputs.
    """

    if argnums is not None:
        raise ValueError("Cannot set argnums, use diff_paths instead.")

    def _new_vag_func(
        *args: PS.args, **kwargs: PS.kwargs
    ) -> Union[
        tuple[R, dict[str, ArrayTree]], tuple[tuple[R, dict[str, ArrayTree]]], ArrayTree
    ]:
        if getattr(_new_vag_func, "internal_func", None) is None:
            def _helper_func(_sel, _p):
                _p = ptree_update(_p, _sel)
                return pfunc(
                    **{f.name: getattr(_p, f.name) for f in dataclasses.fields(_p)}
                )
            _new_vag_func.internal_func = jax.value_and_grad(
                _helper_func, *jax_args, argnums=0, **jax_kwargs
            )

        if getattr(_new_vag_func, "datacls", None) is None:
            _new_vag_func.datacls = fzjax_datacls_from_func(pfunc)

        if getattr(_new_vag_func, "signature", None) is None:
            _new_vag_func.signature = list(get_func_signature(pfunc).keys())

        params = _new_vag_func.datacls(
            **{k: v for k, v in zip(_new_vag_func.signature, args)}, **kwargs
        )
        diff_params = ptree_differentiable(params, diff_paths)
        if return_diff_params:
            return _new_vag_func.internal_func(diff_params, params), diff_params
        return _new_vag_func.internal_func(diff_params, params)

    return _new_vag_func


def pfunc_jit(
    pfunc: Callable[PS, R],
    donate_paths: Selectors = (),
    *jax_args,
    static_argnums: None = None,
    static_argnames: None = None,
    donate_argnums: None = None,
    **jax_kwargs,
) -> Callable[PS, R]:
    if any(x is not None for x in (static_argnums, static_argnames, donate_argnums)):
        raise ValueError(
            "Cannot use *_argnums and *_argnames parameters, "
            "use *_paths and annotations instead."
        )

    @wraps(pfunc)
    def _new_jit_func(*args: PS.args, **kwargs: PS.kwargs) -> R:
        if getattr(_new_jit_func, "internal_func", None) is None:

            def _helper_func(_sel, _p):
                _p = ptree_update(_p, _sel)

                return pfunc(
                    **{f.name: getattr(_p, f.name) for f in dataclasses.fields(_p)}
                )

            _new_jit_func.internal_func = jax.jit(
                _helper_func, *jax_args, donate_argnums=0, **jax_kwargs
            )
            _new_jit_func.lower = _new_jit_func.internal_func.lower

        if getattr(_new_jit_func, "datacls", None) is None:
            _new_jit_func.datacls = fzjax_datacls_from_func(pfunc)

        if getattr(_new_jit_func, "signature", None) is None:
            _new_jit_func.signature = list(get_func_signature(pfunc).keys())

        params = _new_jit_func.datacls(
            **{k: v for k, v in zip(_new_jit_func.signature, args)}, **kwargs
        )

        predicate = NonNullPredicate() and (
            AnnotationPredicate(Donate) or SelectPredicate(donate_paths)
        )
        donate_params = ptree_filter(params, predicate)
        return _new_jit_func.internal_func(donate_params, params)

    return _new_jit_func
