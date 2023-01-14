from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pytest

from fzjax.ptree import (JDC_DIFF_MARKER, JDC_META_MARKER, AnnotatedLeaf,
                         Differentiable, Meta, fzjax_dataclass,
                         ptree_differentiable, ptree_flatten, ptree_unflatten,
                         ptree_update)


@fzjax_dataclass
@dataclass(frozen=True)
class Params:
    nested: Optional[Params]
    param: Differentiable[list[float]]
    state: float
    meta: Meta[int]


@pytest.fixture
def dummies():
    flat_obj = Params(nested=None, param=[0.1, 0.4], state=0.2, meta=1)
    nested_first_order = Params(nested=flat_obj, param=[0.1], state=0.2, meta=1)
    nested_second_order = Params(
        nested=nested_first_order, param=[0.1, 0.2], state=0.2, meta=1
    )
    return flat_obj, nested_first_order, nested_second_order


def test_flatten_unflatten(dummies):
    flattened = ptree_flatten(dummies[2])

    ref_flattened = {
        "meta": AnnotatedLeaf(val=1, annotations=(JDC_META_MARKER,)),
        "state": AnnotatedLeaf(val=0.2, annotations=tuple()),
        "param.1": AnnotatedLeaf(val=0.2, annotations=(JDC_DIFF_MARKER,)),
        "param.0": AnnotatedLeaf(val=0.1, annotations=(JDC_DIFF_MARKER,)),
        "nested.meta": AnnotatedLeaf(val=1, annotations=(JDC_META_MARKER,)),
        "nested.state": AnnotatedLeaf(val=0.2, annotations=tuple()),
        "nested.param.0": AnnotatedLeaf(val=0.1, annotations=(JDC_DIFF_MARKER,)),
        "nested.nested.meta": AnnotatedLeaf(val=1, annotations=(JDC_META_MARKER,)),
        "nested.nested.state": AnnotatedLeaf(val=0.2, annotations=tuple()),
        "nested.nested.param.1": AnnotatedLeaf(val=0.4, annotations=(JDC_DIFF_MARKER,)),
        "nested.nested.param.0": AnnotatedLeaf(val=0.1, annotations=(JDC_DIFF_MARKER,)),
        "nested.nested.nested": AnnotatedLeaf(val=None, annotations=tuple()),
    }
    assert flattened == ref_flattened

    unflattened = ptree_unflatten(dummies[2], flattened)
    assert dummies[2] == unflattened


def test_update(dummies):
    flat2_obj = Params(nested=None, param=0.3, state=0.4, meta=2)
    result = ptree_update(
        dummies[2], {"nested.nested": flat2_obj, "param": 7.0, "nested.param": 8.0}
    )
    assert result.nested.nested == flat2_obj
    assert result.param == 7.0
    assert result.nested.param == 8.0


def test_params_differentiable(dummies):
    result = ptree_differentiable(dummies[2], {"nested.nested"}, return_values=True)
    assert result == {"nested.nested.param.0": 0.1, "nested.nested.param.1": 0.4}
