from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Collection, Generic, Optional, TypeVar, Union

import jax
import jax.numpy as jnp
import optax
from chex import ArrayTree
from jaxtyping import Array, Float, PyTree
from typing_extensions import ParamSpec

from fzjax.ptree import (
    AnnotationPredicate,
    Donate,
    SelectPredicate,
    fzjax_dataclass, ptree_differentiable,
    ptree_filter,
    ptree_update,
)
from fzjax.ptree.internal_helpers import fzjax_datacls_from_func, get_func_signature
from fzjax.ptree.utils import NonNullPredicate

P = ParamSpec("P")
PS = ParamSpec("PS")

R = TypeVar("R", ArrayTree, tuple[ArrayTree, ArrayTree])

Selectors = Collection[str]


def rename(newname):
    def decorator(f):
        f.__name__ = newname
        return f

    return decorator


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
        if getattr(_new_vag_func, "jax_vag", None) is None:

            @rename(f"{pfunc.__name__}_vag_wrapper")
            def _wrapper(_sel, _p):
                _p = ptree_update(_p, _sel)
                return pfunc(
                    **{f.name: getattr(_p, f.name) for f in dataclasses.fields(_p)}
                )

            _new_vag_func.jax_vag = jax.value_and_grad(
                _wrapper, *jax_args, argnums=0, **jax_kwargs
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
            return _new_vag_func.jax_vag(diff_params, params), diff_params
        return _new_vag_func.jax_vag(diff_params, params)

    return _new_vag_func


FuncT = TypeVar("FuncT", bound=Callable)


def pfunc_jit(
    pfunc: FuncT,
    donate_paths: Selectors = (),
    *jax_args,
    static_argnums: None = None,
    static_argnames: None = None,
    donate_argnums: None = None,
    **jax_kwargs,
) -> FuncT:
    if any(x is not None for x in (static_argnums, static_argnames, donate_argnums)):
        raise ValueError(
            "Cannot use *_argnums and *_argnames parameters, "
            "use *_paths and annotations instead."
        )

    @wraps(pfunc)
    def _jit_pfunc(*args, **kwargs):
        if getattr(_jit_pfunc, "jax_jit", None) is None:

            @rename(f"{pfunc.__name__}_jit_wrapper")
            def _wrapper(_sel, _p):
                _p = ptree_update(_p, _sel)

                return pfunc(
                    **{f.name: getattr(_p, f.name) for f in dataclasses.fields(_p)}
                )

            _jit_pfunc.jax_jit = jax.jit(
                _wrapper, *jax_args, donate_argnums=0, **jax_kwargs
            )
            _jit_pfunc.lower = _jit_pfunc.jax_jit.lower

        if getattr(_jit_pfunc, "datacls", None) is None:
            _jit_pfunc.datacls = fzjax_datacls_from_func(pfunc)

        if getattr(_jit_pfunc, "signature", None) is None:
            _jit_pfunc.signature = list(get_func_signature(pfunc).keys())

        params = _jit_pfunc.datacls(
            **{k: v for k, v in zip(_jit_pfunc.signature, args)}, **kwargs
        )
        predicate = NonNullPredicate() and (
            AnnotationPredicate(Donate) or SelectPredicate(donate_paths)
        )
        donate_params = ptree_filter(params, predicate)

        return _jit_pfunc.jax_jit(donate_params, params)

    return _jit_pfunc


@fzjax_dataclass
@dataclass(frozen=True)
class BackpropOut:
    updated: dict[str, PyTree]
    extra: Any
    loss: Float[Array, ""]
    opt_state: optax.OptState


@dataclass
class BackpropFn(Generic[P]):
    value_and_grad: Callable[P, tuple[tuple[Any], Any]]
    optimizer: optax.GradientTransformation
    has_aux: bool

    def __call__(self,
                 lr: Union[jnp.ndarray, float],
                 opt_state: Optional[optax.OptState] = None,
                 **kwargs: P.kwargs) -> BackpropOut:
        optimizer = optax.chain(self.optimizer, optax.scale(-lr))

        extra = None
        if self.has_aux:
            ((loss, extra), grads), diff_params = self.value_and_grad(**kwargs)
        else:
            (loss, grads), diff_params = self.value_and_grad(**kwargs)

        if opt_state is None:
            opt_state = optimizer.init(diff_params)
        updates, opt_state = optimizer.update(grads, opt_state)
        args_updated = set(k.split(".")[0] for k in updates.keys())

        updated = optax.apply_updates(diff_params, updates)
        updated = ptree_update({k: v for k, v in kwargs.items() if k in args_updated},
                               updated)

        return BackpropOut(
            updated=updated,
            extra=extra,
            loss=loss,
            opt_state=opt_state,
        )


def pfunc_update_backprop(
    loss_fn: Callable[P, Union[Float[Array, ""], tuple[Float[Array, ""], Any]]],
    optimizer: optax.GradientTransformation,
    diff_paths: Selectors = (),
    donate_paths: Selectors = (),
    has_aux: bool = False,
) -> BackpropFn[P]:

    loss_fn = pfunc_jit(loss_fn, donate_paths,)

    return BackpropFn(
        value_and_grad=pfunc_value_and_grad(loss_fn,
                                            diff_paths=diff_paths,
                                            return_diff_params=True,
                                            has_aux=has_aux),
        optimizer=optimizer,
        has_aux=has_aux,
    )
