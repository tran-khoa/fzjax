from __future__ import annotations

from enum import Enum, auto
from typing import Union

from .batch_norm import BatchNormParams, BatchNormStates, batch_norm
from .conv import Conv2dParams, conv2d
from .layer_norm import LayerNormParams, layer_norm
from .linear import LinearParams, linear
from .mlp import MLPParams, mlp
from .task_modulated import TMConv2dParams, tm_conv2d


NormParams = Union[None, BatchNormParams, LayerNormParams]


class NormType(Enum):
    NONE = auto()
    BATCH_NORM = auto()
    LAYER_NORM = auto()
