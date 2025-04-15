from pep import build_A_matrix_with_xstar

import jax.numpy as jnp
import jax
jax.config.update('jax_disable_jit', True)

alpha = jnp.array([0.1, 0.2, 0.3])
beta = jnp.array([0.4, 0.5, 0.6])

build_A_matrix_with_xstar(alpha, beta)