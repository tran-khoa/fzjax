from abc import ABC, abstractmethod
from typing import Any, TypeVar

import jax.numpy as jnp
from typing_extensions import Self

from fzjax.ptree import Meta

T = TypeVar("T", bound=jnp.ndarray)


class Norm(ABC):
    @abstractmethod
    def __call__(self,
                 inputs: T,
                 update_stats: Meta[bool] = False,
                 compute_stats: Meta[bool] = False,
                 ) -> tuple[T, Any]:
        ...

    @classmethod
    @abstractmethod
    def create(cls, shape: tuple[int, ...], **kwargs) -> Self:
        ...
