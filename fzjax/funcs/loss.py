from __future__ import annotations

import typing
from typing import Union, Sequence

import jax
import jax.numpy as jnp

from ..higher import pfunc_jit
from ..ptree import Static
from .misc import normalize

if typing.TYPE_CHECKING:
    from jaxtyping import Array, Float, Integer


@jax.jit
def softmax_cross_entropy(logits: Float[Array, "N C"], labels: Integer[Array, "N"]):
    one_hot: Float[Array, "N C"] = jax.nn.one_hot(labels, logits.shape[-1])
    return -jnp.sum(jax.nn.log_softmax(logits) * one_hot, axis=-1)


@jax.jit
def mse_loss(preds: Float[Array, "N"], labels: Integer[Array, "N"]):
    return jnp.mean(jnp.square(preds - labels))


@pfunc_jit
def cosine_similarity(
    x1: Float[Array, "*"],
    x2: Float[Array, "*"],
    axis: Static[Union[None, int, Sequence[int]]] = 1,
    eps: float = 1e-8,
) -> Float[Array, "*"]:
    """
    Computes the cosine similarity of two tensors over the given axes.

    Parameters
    ----------
    x1: First tensor.
    x2: Second tensor.
    axis: Axis or sequence of axes to compute cosine similarity over.
    eps: Epsilon offset in the denominator

    Returns
    -------
        Cosine similarity between x1 and x2.
    """
    x1_sqnorm = jnp.sum(jnp.square(x1), axis=axis, keepdims=True)
    x2_sqnorm = jnp.sum(jnp.square(x2), axis=axis, keepdims=True)

    x1_sqnorm = jnp.clip(x1_sqnorm, a_min=eps * eps)
    x2_sqnorm = jnp.clip(x2_sqnorm, a_min=eps * eps)

    x1_rsqnorm = jax.lax.rsqrt(x1_sqnorm)
    x2_rsqnorm = jax.lax.rsqrt(x2_sqnorm)

    return jnp.sum((x1 * x1_rsqnorm) * (x2 * x2_rsqnorm), axis=axis)


@jax.jit
def simclr(projs: Float[Array, "NA C"], temperature: float = 0.1) -> Float[Array, ""]:
    """
    SimCLR loss function as proposed in
    "A Simple Framework for Contrastive Learning of Visual Representations" (Chen et al., 2020)

    Implementation adapted from
        https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/JAX/tutorial17/SimCLR.html

    Args:
        projs: (batch_size, dim)
            Projected representations.

            Expects the first half of the batch to correspond to the first view of each image,
            and second half to the second view of each image, i.e. for images 1...N:

            [1A ... NA 1B ... NB]
        temperature:
            Softmax rescaling, lower values force more extreme attraction/repulsion.

    Returns:
        SimCLR loss funtion
    """

    sim_mat: Float[Array, "N N"] = cosine_similarity(
        projs[:, None, :], projs[None, :, :], axis=-1
    )
    sim_mat /= temperature

    # mask out diagonal
    diag_range = jnp.arange(projs.shape[0], dtype=jnp.int32)
    sim_mat = sim_mat.at[diag_range, diag_range].set(-9e15)

    # positive examples
    shifted_diag = jnp.roll(diag_range, projs.shape[0] // 2)
    pos_logits = sim_mat[diag_range, shifted_diag]

    nll = -pos_logits + jax.nn.logsumexp(sim_mat, axis=-1)
    return nll.mean()


def barlow_twins(
    projs: Float[Array, "NA C"], weight_off_diagonal: float = 5 * 10**-3
):
    """
    Barlow Twins loss function as proposed in
    "Barlow Twins: Self-Supervised Learning via Redundancy Reduction" (Zbontar et al., 2021)

    Args:
        projs: (batch_size, dim)
            Projected representations.

            Expects the first half of the batch to correspond to the first view of each image,
            and second half to the second view of each image, i.e. for images 1...N:

            [1A ... NA 1B ... NB]
        weight_off_diagonal:
            Weighting (lambda) of the redundancy reduction term, equation 1.

    Returns:
        SimCLR loss funtion
    """
    batch_size: int = projs.shape[0] // 2
    num_channels: int = projs.shape[1]

    projs = projs.reshape((2, batch_size, num_channels))
    projs = jax.vmap(normalize)(projs)

    ccm = jnp.matmul(projs.T, projs) / batch_size
    ccm = jnp.square(ccm - jnp.identity(num_channels, dtype=ccm.dtype))

    diag_range = jnp.arange(projs.shape[0], dtype=jnp.int32)
    ccm = ccm.at[diag_range, diag_range].multiply(weight_off_diagonal)
    return jnp.sum(ccm)
