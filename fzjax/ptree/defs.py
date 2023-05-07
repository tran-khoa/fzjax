import copy
import dataclasses
from dataclasses import dataclass, field
from typing import Any, Optional, Union

from fzjax.ptree.internal_helpers import get_type_hints_partial


JPath = tuple[Union[str, int], ...]


@dataclass(frozen=True)
class Marker:
    ...

StaticM = Marker()
DiffM = Marker()
NoDiffM = Marker()
DonateM = Marker()


def update_markers(parent_markers: set[Marker], child_markers: set[Marker]) -> set[Marker]:
    markers = parent_markers.union(child_markers)

    if NoDiffM in markers or StaticM in markers:
        markers.difference_update({DiffM})

    if StaticM in markers:
        markers.difference_update({DonateM})

    return markers


@dataclass(frozen=True)
class Struct:
    _data: dict[tuple, Any] = field(init=False)
    _markers: dict[tuple, Any] = field(init=False)

    def __getattribute__(self, name: str) -> Any:
        return super().__getattribute__(name)

    @property
    def struct_data(self) -> dict[tuple, Any]:
        return copy.copy(self._data)

    @property
    def struct_markers(self) -> dict[tuple, Any]:
        return copy.copy(self._markers)

    def __post_init__(self):
        type_from_name = get_type_hints_partial(type(self), include_extras=True)

        _data, _markers = {}, {}
        for field in dataclasses.fields(self):
            v = getattr(self, field.name)
            _data[(field.name,)] = v
            _markers[(field.name,)] = set(getattr(type_from_name[field.name], "__metadata__", ()))

            if isinstance(v, Struct):
                for p, vv in v.struct_data:
                    _data[(field.name, *p)] = vv
                for p, mm in v.struct_markers:
                    _data[(field.name, *p)] = update_markers(_markers[(field.name,)], mm)

        object.__setattr__(self, "_data", _data)
        object.__setattr__(self, "_markers", _markers)


class Tuple:
    ...