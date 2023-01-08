from __future__ import annotations

from typing import Sequence

import jax.lax as lax
import jax.numpy as jnp


def _pool_infer_shape(
    x: jnp.ndarray,
    size: int | Sequence[int],
    channel_axis: int | None = -1,
) -> tuple[int, ...]:
    """
    Adapted from dm-haiku. Infer shape for pooling window or strides.
    """
    if isinstance(size, int):
        if channel_axis and not 0 <= abs(channel_axis) < x.ndim:
            raise ValueError(f"Invalid channel axis {channel_axis} for {x.shape}")
        if channel_axis and channel_axis < 0:
            channel_axis = x.ndim + channel_axis
        return (1,) + tuple(size if d != channel_axis else 1 for d in range(1, x.ndim))
    elif len(size) < x.ndim:
        # Assume additional dimensions are batch dimensions.
        return (1,) * (x.ndim - len(size)) + tuple(size)
    else:
        assert x.ndim == len(size)
        return tuple(size)


def max_pool(
    value: jnp.ndarray,
    window_shape: int | Sequence[int],
    strides: int | Sequence[int],
    padding: str,
    channel_axis: int | None = -1,
) -> jnp.ndarray:
    """
    Adapted from dm-haiku. Max pool.

    Args:
        value: Value to pool.
        window_shape: Shape of the pooling window, an int or same rank as value.
        strides: Strides of the pooling window, an int or same rank as value.
        padding: Padding algorithm. Either ``VALID`` or ``SAME``.
        channel_axis: Axis of the spatial channels for which pooling is skipped,
        used to infer ``window_shape`` or ``strides`` if they are an integer.

    Returns:
        Pooled result. Same rank as value.
    """
    if padding not in ("SAME", "VALID"):
        raise ValueError(f"Invalid padding '{padding}', must be 'SAME' or 'VALID'.")

    window_shape = _pool_infer_shape(value, window_shape, channel_axis)
    strides = _pool_infer_shape(value, strides, channel_axis)

    return lax.reduce_window(value, -jnp.inf, lax.max, window_shape, strides, padding)
