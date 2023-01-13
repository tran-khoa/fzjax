from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pytest

from fzjax import Differentiable, Meta, fzjax_dataclass
from fzjax.named_tree.registry import (JDC_DIFF_MARKER, JDC_META_MARKER,
                                       FlattenLeaf, named_flatten,
                                       named_unflatten)
from fzjax.named_tree.utils import named_tree_differentiable, named_tree_update


@fzjax_dataclass
@dataclass(frozen=True)
class Params:
    nested: Optional[Params]
    param: Differentiable[list[float]]
    state: float
    meta: Meta[int]


@pytest.fixture
def dummies():
    flat_obj = Params(None, [0.1, 0.4], 0.2, 1)
    nested_first_order = Params(flat_obj, [0.1], 0.2, 1)
    nested_second_order = Params(nested_first_order, [0.1, 0.2], 0.2, 1)
    return flat_obj, nested_first_order, nested_second_order


def test_flatten_unflatten(dummies):
    flattened, classes = named_flatten(dummies[2])

    ref_flattened = {
        "meta": FlattenLeaf(val=1, meta=(JDC_META_MARKER,)),
        "state": FlattenLeaf(val=0.2, meta=tuple()),
        "param.1": FlattenLeaf(val=0.2, meta=(JDC_DIFF_MARKER,)),
        "param.0": FlattenLeaf(val=0.1, meta=(JDC_DIFF_MARKER,)),
        "nested.meta": FlattenLeaf(val=1, meta=(JDC_META_MARKER,)),
        "nested.state": FlattenLeaf(val=0.2, meta=tuple()),
        "nested.param.0": FlattenLeaf(val=0.1, meta=(JDC_DIFF_MARKER,)),
        "nested.nested.meta": FlattenLeaf(val=1, meta=(JDC_META_MARKER,)),
        "nested.nested.state": FlattenLeaf(val=0.2, meta=tuple()),
        "nested.nested.param.1": FlattenLeaf(val=0.4, meta=(JDC_DIFF_MARKER,)),
        "nested.nested.param.0": FlattenLeaf(val=0.1, meta=(JDC_DIFF_MARKER,)),
        "nested.nested.nested": FlattenLeaf(val=None, meta=tuple()),
    }
    assert flattened == ref_flattened

    unflattened = named_unflatten(flattened, classes)
    assert dummies[2] == unflattened


def test_update(dummies):
    flat2_obj = Params(None, 0.3, 0.4, 2)
    result = named_tree_update(
        dummies[2], {"nested.nested": flat2_obj, "param": 7.0, "nested.param": 8.0}
    )
    assert result.nested.nested == flat2_obj
    assert result.param == 7.0
    assert result.nested.param == 8.0


def test_params_differentiable(dummies):
    result = named_tree_differentiable(dummies[2], {"nested.nested"}, flat_pytree=True)
    assert result == {"nested.nested.param.0": 0.1, "nested.nested.param.1": 0.4}
