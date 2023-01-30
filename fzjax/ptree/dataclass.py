from __future__ import annotations

import dataclasses
from typing import TypeVar

from chex import mappable_dataclass
from jax import tree_util
from typing_extensions import dataclass_transform

from .annotations import JDC_META_MARKER
from .internal_helpers import get_type_hints_partial

try:
    # Attempt to import flax for serialization. The exception handling lets us drop
    # flax from our dependencies.
    from flax import serialization
except ImportError:
    serialization = None  # type: ignore


T = TypeVar("T")


@dataclass_transform()
def fzjax_dataclass(cls: type[T]) -> type[T]:
    """
    This function is a modification of jax_dataclasses "_register_pytree_dataclass".
    This version can transforms classes to dataclasses only if necessary.

    Since this dataclass uses chex's mappable_dataclass, the inialializer only accepts
    keyword arguments (kw_only=True).

    Splitting jax_dataclasses.pytree_dataclass into @dataclass and @as_pytree "fixes"
    PyCharm's handling of non-standard dataclasses.
    """

    if not dataclasses.is_dataclass(cls):
        cls = dataclasses.dataclass(cls)

    # Makes dataclass compatible with dm-tree via the chex library
    cls = mappable_dataclass(cls)
    # We remove `collection.abc.Mapping` mixin methods here to allow
    # fields with these names.
    for attr in ("values", "keys", "get", "items"):
        setattr(cls, attr, None)  # redefine
        delattr(cls, attr)  # delete

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

        fields_metadata[field.name] = tuple(getattr(field_type, "__metadata__", ()))

        # Handle Meta[] annotation by moving metadata into PyTreeDef
        if JDC_META_MARKER in fields_metadata[field.name]:
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

    # Serialization: this is mostly copied from `flax.struct.dataclass`.
    if serialization is not None:

        def _to_state_dict(x: T):
            return {
                name: serialization.to_state_dict(getattr(x, name))
                for name in child_node_field_names
            }

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
                updates[name] = serialization.from_state_dict(value, value_state, name)
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
