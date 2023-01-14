from __future__ import annotations

from typing import Annotated, Container, TypeVar

import jax.numpy as jnp

###########
# Markers #
###########
JDC_META_MARKER = "__fzjax_pytree_static_field__"
JDC_DIFF_MARKER = "__fzjax_pytree_differentiable_field__"
JDC_NODIFF_MARKER = "__fzjax_pytree_nondifferentiable_field__"

# Stolen from here: https://github.com/google/jax/issues/10476

###############
# Annotations #
###############
MetaT = TypeVar("MetaT")
Meta = Static = Annotated[MetaT, JDC_META_MARKER]

DiffT = TypeVar("DiffT", jnp.ndarray, Container[jnp.ndarray])
Diff = Differentiable = Annotated[DiffT, JDC_DIFF_MARKER]

# Explicitly marks node and its children as non-differentiable, overriding Differentiable
NoDiffT = TypeVar("NoDiffT", jnp.ndarray, Container[jnp.ndarray])
NoDiff = Annotated[NoDiffT, JDC_NODIFF_MARKER]


def post_process_annotations(_annotations: tuple[str]) -> tuple[str]:

    # If nodiff detected, remove diff marker.
    if JDC_NODIFF_MARKER in _annotations:
        return tuple(m for m in _annotations if m != JDC_DIFF_MARKER)

    if JDC_META_MARKER in _annotations and JDC_DIFF_MARKER in _annotations:
        raise ValueError("Node is marked as both Meta and Differentiable!")

    return _annotations
