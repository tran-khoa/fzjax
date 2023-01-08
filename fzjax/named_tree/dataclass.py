from __future__ import annotations

import collections
import dataclasses
import functools
import sys
import types
from types import (MethodDescriptorType, MethodWrapperType,
                   WrapperDescriptorType)
from typing import Annotated, Any, Container, TypeVar

import jax.numpy as jnp
from jax import tree_util

from fzjax.named_tree import registry
from fzjax.named_tree.registry import FlattenNode

try:
    # Attempt to import flax for serialization. The exception handling lets us drop
    # flax from our dependencies.
    from flax import serialization
except ImportError:
    serialization = None  # type: ignore

JDC_META_MARKER = "__fzjax_pytree_static_field__"
JDC_DIFF_MARKER = "__fzjax_pytree_differentiable_field__"

# Stolen from here: https://github.com/google/jax/issues/10476
StaticT = TypeVar("StaticT")
Meta = Static = Annotated[StaticT, JDC_META_MARKER]

DiffT = TypeVar("DiffT", jnp.ndarray, Container[jnp.ndarray])
Diff = Differentiable = Annotated[DiffT, JDC_DIFF_MARKER]

T = TypeVar("T")


def fzjax_dataclass(cls: type[T]) -> type[T]:
    """
    This function is a modification of jax_dataclasses "_register_pytree_dataclass".
    This version can transforms classes to dataclasses only if necessary.

    Splitting jax_dataclasses.pytree_dataclass into @dataclass and @as_pytree "fixes"
    PyCharm's handling of non-standard dataclasses.
    """

    if not dataclasses.is_dataclass(cls):
        cls = dataclasses.dataclass(cls)

    # Determine which fields are static and part of the treedef, and which should be
    # registered as child nodes.
    child_node_field_names: list[str] = []
    static_field_names: list[str] = []

    # We don't directly use field.type for postponed evaluation; we want to make sure
    # that our types are interpreted as proper types and not as (string) forward
    # references.
    #
    # Note that there are ocassionally situations where the @jdc.pytree_dataclass
    # decorator is called before a referenced type is defined; to suppress this error,
    # we resolve missing names to our subscriptible placeohlder object.
    type_from_name = get_type_hints_partial(cls, include_extras=True)  # type: ignore
    fields_metadata: dict[str, tuple[str]] = {}

    for field in dataclasses.fields(cls):
        if not field.init:
            continue

        field_type = type_from_name[field.name]

        fields_metadata[field.name] = tuple(
            getattr(field_type, "__metadata__", tuple())
        )

        # Two ways to mark a field as static: either via the Static[] type or
        # jdc.static_field().
        if JDC_META_MARKER in fields_metadata[field.name]:
            static_field_names.append(field.name)
            continue
        if field.metadata.get(JDC_META_MARKER, False):
            static_field_names.append(field.name)
            continue

        child_node_field_names.append(field.name)

    # Define flatten, unflatten operations: this simple converts our dataclass to a list
    # of fields.
    def _flatten(obj):
        children = tuple(getattr(obj, key) for key in child_node_field_names)
        treedef = tuple(getattr(obj, key) for key in static_field_names)
        return children, treedef

    def _unflatten(treedef, children):
        return cls(
            **dict(zip(child_node_field_names, children)),
            **{key: tdef for key, tdef in zip(static_field_names, treedef)},
        )

    tree_util.register_pytree_node(cls, _flatten, _unflatten)

    # Define flatten, unflatten operations for named tree.
    def _named_flatten(obj):
        return FlattenNode(
            keys=tuple(fields_metadata.keys()),  # noqa
            vals=tuple(getattr(obj, k) for k in fields_metadata.keys()),
            meta=tuple(fields_metadata[k] for k in fields_metadata.keys()),
        )

    def _named_unflatten(node: FlattenNode):
        return cls(**dict(zip(node.keys, node.vals)))

    registry.register_named_pytree_node(
        cls,
        _named_flatten,
        _named_unflatten,
        fallback_unflatten_fn=lambda node: dict(zip(node.keys, node.vals)),
    )

    # Serialization: this is mostly copied from `flax.struct.dataclass`.
    if serialization is not None:

        def _to_state_dict(x: T):
            state_dict = {
                name: serialization.to_state_dict(getattr(x, name))
                for name in child_node_field_names
            }
            return state_dict

        def _from_state_dict(x: T, state: dict):
            # Copy the state so we can pop the restored fields.
            state = state.copy()
            updates = {}
            for name in child_node_field_names:
                if name not in state:
                    raise ValueError(
                        f"Missing field {name} in state dict while restoring"
                        f" an instance of {cls.__name__}"
                    )
                value = getattr(x, name)
                value_state = state.pop(name)
                updates[name] = serialization.from_state_dict(value, value_state)
            if state:
                names = ",".join(state.keys())
                raise ValueError(
                    f'Unknown field(s) "{names}" in state dict while'
                    f" restoring an instance of {cls.__name__}"
                )
            return dataclasses.replace(x, **updates)

        serialization.register_serialization_state(
            cls, _to_state_dict, _from_state_dict
        )

    return cls


class _UnresolvableForwardReference:
    def __class_getitem__(cls, item) -> type[_UnresolvableForwardReference]:
        """__getitem__ passthrough, for supporting generics."""
        return _UnresolvableForwardReference


_allowed_types = (
    types.FunctionType,
    types.BuiltinFunctionType,
    types.MethodType,
    types.ModuleType,
    WrapperDescriptorType,
    MethodWrapperType,
    MethodDescriptorType,
)


@functools.lru_cache(maxsize=128)
def get_type_hints_partial(obj, include_extras=False) -> dict[str, Any]:
    """Adapted from typing.get_type_hints(), but aimed at suppressing errors from not
    (yet) resolvable forward references.

    This function should only be used to search for fields that are annotated with
    `jdc.Static[]`.

    For example:

        @jdc.pytree_dataclass
        class A:
            x: B
            y: jdc.Static[bool]

        @jdc.pytree_dataclass
        class B:
            x: jnp.ndarray

    Note that the type annotations of `A` need to be parsed by the `pytree_dataclass`
    decorator in order to register the static field, but `B` is not yet defined when the
    decorator is run. We don't actually care about the details of the `B` annotation, so
    we replace it in our annotation dictionary with a dummy value.

    Differences:
        1. `include_extras` must be True.
        2. Only supports types.
        3. Doesn't throw an error when a name is not found. Instead, replaces the value
           with `_UnresolvableForwardReference`.
    """
    assert include_extras

    # Replace any unresolvable names with _UnresolvableForwardReference.
    base_globals: dict[str, Any] = collections.defaultdict(
        lambda: _UnresolvableForwardReference
    )
    base_globals.update(__builtins__)  # type: ignore

    # Classes require a special treatment.
    if isinstance(obj, type):
        hints = {}
        for base in reversed(obj.__mro__):
            ann = base.__dict__.get("__annotations__", {})
            if len(ann) == 0:
                continue

            base_globals.update(sys.modules[base.__module__].__dict__)

            for name, value in ann.items():
                if value is None:
                    value = type(None)
                if isinstance(value, str):
                    value = eval(value, base_globals)
                hints[name] = value
        return hints

    nsobj = obj
    # Find globalns for the unwrapped object.
    while hasattr(nsobj, "__wrapped__"):
        nsobj = nsobj.__wrapped__
    base_globals.update(getattr(nsobj, "__globals__", {}))

    hints = getattr(obj, "__annotations__", None)  # type: ignore
    if hints is None:
        # Return empty annotations for something that _could_ have them.
        if isinstance(obj, _allowed_types):
            return {}
        else:
            raise TypeError(
                "{!r} is not a module, class, method, or function.".format(obj)
            )
    hints = dict(hints)
    for name, value in hints.items():
        if value is None:
            value = type(None)
        if isinstance(value, str):
            value = eval(value, base_globals)
        hints[name] = value
    return hints
