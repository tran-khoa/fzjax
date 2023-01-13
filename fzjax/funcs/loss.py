import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Integer


def softmax_cross_entropy(logits: Float[Array, "N C"], labels: Integer[Array, "N"]):
    one_hot: Float[Array, "N C"] = jax.nn.one_hot(labels, logits.shape[-1])
    return -jnp.sum(jax.nn.log_softmax(logits) * one_hot, axis=-1)


def mse_loss(preds: Float[Array, "N"], labels: Integer[Array, "N"]):
    return jnp.mean(jnp.square(preds - labels))


def cosine_similarity(
    x1: Float[Array, "*"],
    x2: Float[Array, "*"],
    axis: int = 1,
    eps: float = 1e-8,
) -> Float[Array, "*"]:

    x1_squared_norm = jnp.sum(jnp.square(x1), axis=axis, keepdims=True)
    x2_squared_norm = jnp.sum(jnp.square(x2), axis=axis, keepdims=True)

    x1_squared_norm = jnp.clip(x1_squared_norm, a_min=eps * eps)
    x2_squared_norm = jnp.clip(x2_squared_norm, a_min=eps * eps)

    x1_norm = jnp.sqrt(x1_squared_norm)
    x2_norm = jnp.sqrt(x2_squared_norm)

    return jnp.sum((x1 / x1_norm) * (x2 / x2_norm), axis=axis)


def simclr(projs: Float[Array, "N C"], temperature: float = 0.1) -> Float[Array, ""]:
    """
    SimCLR loss function as proposed in
    "A Simple Framework for Contrastive Learning of Visual Representations" (Chen et al., 2020)

    Implementation adapted from
        https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/JAX/tutorial17/SimCLR.html

    Args:
        projs: (batch_size, dim)
            Projected representations.
        temperature:
            Softmax rescaling, lower values force more extreme attraction/repulsion.

    Returns:
        SimCLR loss funtion
    """
    projs = projs[jnp.concatenate([jnp.arange(projs.shape[0] // 2) * 2, jnp.arange(projs.shape[0] // 2) * 2 + 1])]

    sim_mat: Float[Array, "N N"] = cosine_similarity(
        projs[:, None, :], projs[None, :, :]
    )
    sim_mat /= temperature

    # mask out diagonal
    diag_range = jnp.arange(projs.shape[0], dtype=jnp.int32)
    sim_mat = sim_mat.at[diag_range, diag_range].set(-9e15)

    # positive examples
    shifted_diag = jnp.roll(diag_range, projs.shape[0] // 2)
    pos_logits = sim_mat[diag_range, shifted_diag]

    nll = -pos_logits + jax.nn.logsumexp(sim_mat, axis=-1)
    nll = nll.mean()
    return nll
