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


def post_process_annotations(annotations: tuple[str, ...]) -> tuple[str, ...]:
    """
    Returns a version of annotations that coheres to "Order of annotations".
    """

    if JDC_NODIFF_MARKER in annotations or JDC_META_MARKER in annotations:
        annotations = tuple(m for m in annotations if m != JDC_DIFF_MARKER)

    if JDC_META_MARKER in annotations:
        annotations = tuple(m for m in annotations if m != JDC_DONATE_MARKER)

    return annotations
