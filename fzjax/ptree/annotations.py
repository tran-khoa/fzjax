from __future__ import annotations

from typing import Annotated, Container, TypeVar

import jax.numpy as jnp

########################
# Order of annotations #
########################
# X -> Y: X overrides Y
# i.e. if a leaf would have annotations {X, Y}, the leaf will only have {X}
# Meta     -> Diff, Donate
# NoDiff   -> Diff


###########
# Markers #
###########
JDC_META_MARKER = "__fzjax_pytree_static_field__"
JDC_DIFF_MARKER = "__fzjax_pytree_differentiable_field__"
JDC_NODIFF_MARKER = "__fzjax_pytree_nondifferentiable_field__"
JDC_DONATE_MARKER = "__fzjax_pytree_donate_field__"

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

DonateT = TypeVar("DonateT")
Donate = Annotated[DonateT, JDC_DONATE_MARKER]

ANNOTATIONS = {
    "Meta": Meta,
    "Static": Meta,
    "Diff": Diff,
    "Differentiable": Differentiable,
    "NoDiff": NoDiff,
    "Donate": Donate,
}


def post_process_annotations(_annotations: tuple[str]) -> tuple[str]:

    # If nodiff detected, remove diff marker.
    if JDC_NODIFF_MARKER in _annotations:
        _annotations = tuple(m for m in _annotations if m != JDC_DIFF_MARKER)

    elif JDC_META_MARKER in _annotations and JDC_DIFF_MARKER in _annotations:
        _annotations = tuple(m for m in _annotations if m != JDC_DIFF_MARKER)

    elif JDC_META_MARKER in _annotations and JDC_DONATE_MARKER in _annotations:
        _annotations = tuple(m for m in _annotations if m != JDC_DONATE_MARKER)

    return _annotations
