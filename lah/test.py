from pep import build_A_matrix_with_xstar, create_nesterov_strcvx_pep_sdp_layer, pepit_nesterov, create_nesterov_pep_sdp_layer
import numpy as np

import jax.numpy as jnp
import jax

jax.config.update('jax_disable_jit', True)


mu = 1 #1/10
L = 10
num_iter = 10
inv_cond = mu/L

t_params = jnp.ones(num_iter)
t = 1
for i in range(0, num_iter):
    t = i #.5 * (1 + jnp.sqrt(1 + 4 * t ** 2))
    t_params = t_params.at[i].set(t/(t+3)) #(jnp.log(t))
beta = t_params

# default theoretical choice for Nesterov (strongly convex): alpha = 1/L, beta = (1-sqrt(q))/(1+sqrt(q)) where q = mu/L
alpha = jnp.repeat(1/L, num_iter)
beta = jnp.repeat((1-inv_cond**0.5)/(1+inv_cond**0.5), num_iter)


A = build_A_matrix_with_xstar(alpha, beta)



layer_nesterov = create_nesterov_pep_sdp_layer(L, num_iter)
G, F = layer_nesterov(A, solver_args={"solve_method": "CLARABEL", "verbose": True})

print('our layer:', F[-2] - F[-1])
pepit_tau = pepit_nesterov(0, L, np.column_stack((alpha, beta)))
print('pep value:', pepit_tau)
print("==========================================")
import pdb
pdb.set_trace()

# layer_nesterov = create_nesterov_strcvx_pep_sdp_layer(mu, L, num_iter)
# G, F = layer_nesterov(A, solver_args={"solve_method": "CLARABEL", "verbose": False})

# print('our layer:', F[-2] - F[-1])
# pepit_tau = pepit_nesterov(mu, L, np.column_stack((alpha, beta)))
# print('pep value:', pepit_tau)

# print("==========================================")

# a = alpha
# layer_proxgd = create_proxgd_pep_sdp_layer(0, L, num_iter)
# sol_proxgd = layer_proxgd(a, solver_args={"solve_method": "CLARABEL", "verbose": True})

# print(sol_proxgd[1][-2] + sol_proxgd[2][-2] - sol_proxgd[1][-1] - sol_proxgd[2][-1])




# print("==========================================")

# layer_quadgd = create_quadmin_pep_sdp_layer(mu, L, num_iter)
# sol_quadgd = layer_quadgd(a, solver_args={"solve_method": "CLARABEL", "verbose": True})

# print(sol_quadgd)
