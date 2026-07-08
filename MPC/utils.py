import jax
from jax import numpy as jnp


def count_parameters(params):
    return sum(jnp.size(p) for p in jax.tree_util.tree_leaves(params))
