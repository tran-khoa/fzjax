from __future__ import annotations

import dataclasses
from collections import defaultdict
from dataclasses import dataclass
from typing import (
    Annotated,
    Any,
    Callable,
    Collection,
    Generic,
    Optional,
    TypeVar,
    Union,
)

import tree

from .annotations import JDC_DIFF_MARKER, JDC_NODIFF_MARKER, post_process_annotations
from .internal_helpers import get_type_hints_partial

FallbackT = TypeVar("FallbackT", list, tuple, dict)
T = TypeVar("T")
_T = TypeVar("_T")


@dataclass(frozen=False, init=False)
class AnnotatedLeaf(Generic[_T]):
    val: _T
    meta: tuple[str]

    def __init__(self, val: _T, annotations: tuple[str] = tuple()):
        self.val = val
        self.meta = post_process_annotations(annotations)


def ptree_flatten(obj: Any, with_annotations: bool = True) -> dict[str, AnnotatedLeaf]:
    flattened = tree.flatten_with_path(obj)
    flattened = {".".join(map(str, path)): val for path, val in flattened}

    if not with_annotations:
        return flattened

    # maps annotation -> paths
    annotated_paths: dict[str, list[tuple[str, ...]]] = defaultdict(list)

    def register_annotation_prefix(prefix: tuple[str, ...], _obj):
        if dataclasses.is_dataclass(_obj):
            type_from_name = get_type_hints_partial(type(_obj), include_extras=True)  # type: ignore
            for field in dataclasses.fields(_obj):
                if not field.init:
                    continue

                field_type = type_from_name[field.name]
                for annotation in list(getattr(field_type, "__metadata__", list())):
                    annotated_paths[annotation].append(prefix + (field.name,))

    def annotate(path: str, val: Any) -> AnnotatedLeaf:
        found_annotations = set()
        for annotation, prefixes in annotated_paths.items():
            if any(path.startswith(prefix) for prefix in prefixes):
                found_annotations.add(annotation)
        return AnnotatedLeaf(val, annotations=tuple(found_annotations))

    tree.traverse_with_path(register_annotation_prefix, obj)
    annotated_paths = {
        annotation: set(".".join(map(str, path)) for path in paths)
        for annotation, paths in annotated_paths.items()
    }

    return {path: annotate(path, val) for path, val in flattened.items()}


def ptree_unflatten(
    structure: Any,
    flattened: Union[
        dict[str, Union[AnnotatedLeaf, Any]], list[Union[AnnotatedLeaf, Any]]
    ],
) -> Any:
    if isinstance(flattened, list):
        return [v.val if isinstance(v, AnnotatedLeaf) else v for v in flattened]
    else:
        assert isinstance(flattened, dict)
        flattened = [
            v.val if isinstance(v, AnnotatedLeaf) else v for v in flattened.values()
        ]
    return tree.unflatten_as(structure, flattened)


def ptree_update(obj: T, changes: dict[str, Any]) -> T:
    remaining = set(changes.keys())

    changes = {k: v if v is not None else tree.MAP_TO_NONE for k, v in changes.items()}

    def update(path: tuple, _obj: Any):
        path_str = ".".join(map(str, path))

        # Stop depth traversal if current path is not a prefix of any needle
        if not any(change.startswith(path_str) for change in changes):
            return _obj if _obj is not None else tree.MAP_TO_NONE

        # Found replacement, stopping depth traversal
        if path_str in changes:
            remaining.remove(path_str)
            return changes[path_str]

        # Continue
        return None

    return tree.traverse_with_path(update, obj)


def ptree_filter(
    obj: Any,
    predicate: Callable[[str, AnnotatedLeaf], bool],
    return_values: bool = True,
) -> dict[str, Union[AnnotatedLeaf, Any]]:
    """
    Returns a flattened ptree for leaves if predicate(path, val) is True.

    Args:
        obj: ptree to be filtered
        predicate:
            Accepts the path as a dot-string and an AnnotatedLeaf,
            returns True for leaves that should be included.
        return_values: whether to return the values or AnnotatedLeaf objects

    Returns:
        flattened dict
    """
    fmap = ptree_flatten(obj)
    fmap = {
        k: v.val if return_values else v for k, v in fmap.items() if predicate(k, v)
    }
    return fmap


def ptree_select(
    obj: Any, paths: Optional[Collection[str]] = None, return_values: bool = True
):
    return ptree_filter(
        obj, lambda p, _: any(p.startswith(q) for q in paths), return_values
    )


def ptree_by_annotation(
    obj: Any, annotation: type[Annotated], return_values: bool = True
) -> dict[str, AnnotatedLeaf]:
    annotations_str = annotation.__metadata__[0]
    return ptree_filter(obj, lambda _, v: annotations_str in v.meta, return_values)


def ptree_differentiable(
    obj: T, subset: Collection[str] | None = None, return_values: bool = True
) -> dict[str, AnnotatedLeaf]:
    def predicate(prefix: str, leaf: AnnotatedLeaf):
        return (
            JDC_DIFF_MARKER in leaf.meta
            and JDC_NODIFF_MARKER not in leaf.meta
            and (not subset or any(prefix.startswith(q) for q in subset))
            and (leaf.val is not None)
        )

    return ptree_filter(obj, predicate, return_values)
