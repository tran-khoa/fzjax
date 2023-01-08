from __future__ import annotations

import itertools
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Generic, NamedTuple, Protocol, TypeVar

FallbackT = TypeVar("FallbackT", list, tuple, dict)
T = TypeVar("T")
_T = TypeVar("_T")


@dataclass(frozen=True)
class FlattenLeaf(Generic[_T]):
    val: _T
    meta: tuple[str] = tuple()

    def with_val(self, val: _T):
        return FlattenLeaf(val, self.meta)


class FlattenNode(NamedTuple):
    keys: tuple[str | int]
    vals: tuple[Any]
    meta: tuple[tuple[str]] = []


class FlattenFn(Protocol[_T]):
    def __call__(self, obj: _T) -> FlattenNode:
        ...


class UnflattenFn(Protocol[_T]):
    def __call__(self, node: FlattenNode) -> _T:
        ...


class _RegistryEntry(NamedTuple):
    flatten: FlattenFn
    unflatten: UnflattenFn
    fallback_unflatten_fn: UnflattenFn | None = None


_registry: dict[type, _RegistryEntry] = {
    tuple: _RegistryEntry(
        lambda xs: FlattenNode(tuple(range(len(xs))), xs),
        lambda xs: tuple(v for _, v in sorted(zip(xs.keys, xs.vals))),
    ),
    list: _RegistryEntry(
        lambda xs: FlattenNode(tuple(range(len(xs))), xs),
        lambda xs: list(v for _, v in sorted(zip(xs.keys, xs.vals))),
    ),
    dict: _RegistryEntry(
        lambda xs: FlattenNode(tuple(xs.keys()), tuple(xs.values())),
        lambda xs: dict(zip(xs.keys(), xs.values())),
    ),
}


def register_named_pytree_node(
    nodetype: type[T],
    flatten_fn: FlattenFn[T],
    unflatten_fn: UnflattenFn[T],
    fallback_unflatten_fn: UnflattenFn[FallbackT] | None = None,
):
    _registry[nodetype] = _RegistryEntry(
        flatten_fn, unflatten_fn, fallback_unflatten_fn
    )


def named_flatten(
    obj: Any,
) -> tuple[dict[str, FlattenLeaf], dict[str | None, type | None]]:
    stack = [(None, obj, tuple())]

    flattened = {}
    clz = {}

    def _handle(_obj):
        if type(_obj) not in _registry:
            return FlattenLeaf(_obj)

        return _registry[type(_obj)].flatten(_obj)

    while stack:
        prefix, _obj, meta = stack.pop()
        _node = _handle(_obj)

        if isinstance(_node, FlattenLeaf):
            _node = FlattenLeaf(val=_node.val, meta=meta + _node.meta)
            flattened[prefix] = _node
            clz[prefix] = None
        else:
            assert isinstance(_node, FlattenNode)
            clz[prefix] = type(_obj)
            for node_prefix, node_obj, node_meta in itertools.zip_longest(
                reversed(_node.keys), reversed(_node.vals), reversed(_node.meta)
            ):
                if prefix is not None:
                    node_prefix = f"{prefix}.{node_prefix}"
                stack.append((node_prefix, node_obj, meta + (node_meta or tuple())))

    return flattened, clz


def named_unflatten(
    d: dict[str, FlattenLeaf], clz: dict[str, Any], with_fallback: bool = False
):
    depth_clz = list(
        sorted(
            (
                ((len(prefix.split(".")) if prefix else 0), (prefix, _clz))
                for prefix, _clz in clz.items()
            )
        )
    )

    children: dict[str | None, dict[str, Any]] = defaultdict(lambda: defaultdict(dict))
    while depth_clz:
        _, (prefix, _clz) = depth_clz.pop()

        if _clz is not None:
            _unflatten = _registry[_clz].unflatten
            if with_fallback and (
                _fallback_unflatten := _registry[_clz].fallback_unflatten_fn
            ):
                _unflatten = _fallback_unflatten

            flatten_node = FlattenNode(
                keys=tuple(children[prefix].keys()),  # noqa
                vals=tuple(children[prefix].values()),
            )
            obj = _unflatten(flatten_node)
        else:
            obj = d[prefix].val

        if prefix is None:
            assert not depth_clz
            return obj

        split = prefix.split(".")
        if len(split) == 1:
            children[None][prefix] = obj
        else:
            children[".".join(split[:-1])][split[-1]] = obj

    raise RuntimeError()
