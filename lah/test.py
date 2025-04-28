from pep import *

import jax.numpy as jnp
import jax

jax.config.update('jax_disable_jit', True)


mu = 1/10
L = 1
num_iter = 10
inv_cond = mu/L

# default theoretical choice for Nesterov (strongly convex): alpha = 1/L, beta = (1-sqrt(q))/(1+sqrt(q)) where q = mu/L
alpha = jnp.repeat(1/L, num_iter)
beta = jnp.repeat((1-inv_cond**0.5)/(1+inv_cond**0.5), num_iter)

"""
A = build_A_matrix_with_xstar(alpha, beta)
layer_nesterov = create_nesterov_strcvx_pep_sdp_layer(mu, L, num_iter)
sol_nesterov = layer_nesterov(A)

print(sol_nesterov)
pepit_nesterov(mu, L, np.column_stack((alpha, beta)))
"""

# print("==========================================")

# a = alpha
# layer_proxgd = create_proxgd_pep_sdp_layer(0, L, num_iter)
# sol_proxgd = layer_proxgd(a)

# print(sol_proxgd[1][-2] + sol_proxgd[2][-2] - sol_proxgd[1][-1] - sol_proxgd[2][-1])

A = build_A_matrix_prox_with_xstar(alpha, beta)
layer = create_proxgd_pep_sdp_layer(L, 10)
layer(A, solver_args={"solve_method": "CLARABEL", "verbose": True})
params = np.array(jnp.vstack([alpha, beta]).T)
pepit_quadprox_accel_gd(0, L, params, None)

"""
print("==========================================")

layer_quadgd = create_quadmin_pep_sdp_layer(mu, L, num_iter)
sol_quadgd = layer_quadgd(a)

print(sol_quadgd)
"""