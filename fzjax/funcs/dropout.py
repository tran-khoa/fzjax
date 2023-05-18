import jax.numpy as jnp
from jax.random import PRNGKeyArray, bernoulli

from fzjax.higher import pfunc_jit
from fzjax.ptree import Meta


@pfunc_jit
def dropout(x: jnp.ndarray, p: Meta[float], rng: PRNGKeyArray):
    keep = bernoulli(rng, p, x.shape)
    return jnp.where(keep, x / p, 0)
