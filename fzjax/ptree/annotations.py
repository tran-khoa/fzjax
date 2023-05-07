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
STATIC = "__fzjax_pytree_static_field__"
DIFF = "__fzjax_pytree_differentiable_field__"
NODIFF = "__fzjax_pytree_nondifferentiable_field__"
DONATE = "__fzjax_pytree_donate_field__"

# Stolen from here: https://github.com/google/jax/issues/10476
# and here: https://github.com/brentyi/jax_dataclasses/blob/main/jax_dataclasses/_dataclasses.py

###############
# Annotations #
###############
MetaT = TypeVar("MetaT")
Meta = Static = Annotated[MetaT, STATIC]

DiffT = TypeVar("DiffT", jnp.ndarray, Container[jnp.ndarray])
Diff = Differentiable = Annotated[DiffT, DIFF]

# Explicitly marks node and its children as non-differentiable, overriding Differentiable
NoDiffT = TypeVar("NoDiffT", jnp.ndarray, Container[jnp.ndarray])
NoDiff = Annotated[NoDiffT, NODIFF]

DonateT = TypeVar("DonateT")
Donate = Annotated[DonateT, DONATE]

ANNOTATIONS = {
    "Meta": Meta,
    "Static": Meta,
    "Diff": Diff,
    "Differentiable": Differentiable,
    "NoDiff": NoDiff,
    "Donate": Donate,
}

ANNOTATION_MARKERS = {
    "Meta": STATIC,
    "Static": STATIC,
    "Diff": DIFF,
    "Differentiable": DIFF,
    "NoDiff": NODIFF,
    "Donate": NODIFF,
}


def register_annotation(name: str,
                        annote_type: type[Annotated],
                        marker: str):
    ANNOTATIONS[name] = annote_type
    ANNOTATION_MARKERS[name] = marker


def post_process_annotations(annotations: tuple[str, ...]) -> tuple[str, ...]:
    """
    Returns a version of annotations that coheres to "Order of annotations".
    """

    if NODIFF in annotations or STATIC in annotations:
        annotations = tuple(m for m in annotations if m != DIFF)

    if STATIC in annotations:
        annotations = tuple(m for m in annotations if m != DONATE)

    return annotations
