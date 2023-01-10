from __future__ import annotations

from typing import Annotated, Any, Callable, Collection, TypeVar

from .registry import FlattenLeaf, JDC_DIFF_MARKER, named_flatten, named_unflatten

T = TypeVar("T")


def named_tree_map(
    f: Callable[[Any], Any], tree: Any, *rest: Any, preserve_structure: bool = True
):
    """
    Instead of working on flattened PyTrees, we work on named PyTrees,
    where items of dicts/dataclasses are addressed by their key/property name
    and lists/tuples are addressed by their index.
    Other objects are handled as leaves.

    This allows the *rest trees to only comprise a subset of the main tree.

    Args:
        f:
        tree:
        *rest:
        preserve_structure:
            Whether to preserve dataclasses (and later: namedtuples).
            This is necessary when only subsets of these objects are returned.

    """
    raise NotImplementedError()


def named_tree_update(obj: T, changes: dict[str, Any]) -> T:
    fmap, clz = named_flatten(obj)
    for k, v in changes.items():
        if k not in fmap:
            with_prefix = {_k for _k in fmap.keys() if _k.startswith(k)}
            with_prefix_clz = {_k for _k in clz.keys() if _k and _k.startswith(k)}
            if not with_prefix:
                raise ValueError(f"Unknown key {k}!")

            for x in with_prefix:
                fmap.pop(x)
            for x in with_prefix_clz:
                clz.pop(x)
            new_fmap, new_clz = named_flatten(v)
            fmap.update({(f"{k}.{kk}" if kk else k): vv for kk, vv in new_fmap.items()})
            clz.update({(f"{k}.{kk}" if kk else k): vv for kk, vv in new_clz.items()})
        else:
            fmap[k] = fmap[k].with_val(v)
    return named_unflatten(fmap, clz)


def named_tree_filter(
    obj: Any, predicate: Callable[[str, FlattenLeaf], bool],
    as_flatten_leaves: bool = False
) -> dict[str, Any]:
    fmap, clz = named_flatten(obj)
    fmap = {k: v for k, v in fmap.items() if predicate(k, v)}
    clz = {
        k: v
        for k, v in clz.items()
        if k is None or any(fk.startswith(k) for fk in fmap.keys())
    }

    if not as_flatten_leaves:
        return {k: v.val for k, v in fmap.items()}
    return named_unflatten(fmap, clz, with_fallback=True)


def named_tree_by_annotation(
    obj: Any, annotation: type[Annotated]
) -> dict[str, Any]:
    annotations_str = annotation.__metadata__[0]
    return named_tree_filter(
        obj, lambda _, v: annotations_str in v.meta
    )


def named_tree_differentiable(
    obj: T, subset: Collection[str] | None = None, flat_pytree: bool = False
) -> dict[str, Any]:
    def predicate(prefix: str, leaf: FlattenLeaf):
        return (
            JDC_DIFF_MARKER in leaf.meta
            and (not subset or any(prefix.startswith(q) for q in subset))
            and (leaf.val is not None)
        )

    return named_tree_filter(obj, predicate)
