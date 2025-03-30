from functools import partial

import cvxpy as cp
import jax.numpy as jnp
import jax.scipy as jsp
from jax import grad, lax, vmap
from jax.tree_util import tree_map

from lah.utils.generic_utils import python_fori_loop, unvec_symm, vec_symm


def fixed_point_gd(z, P, c, gd_step):
    grad = P @ z + c
    return z - gd_step * grad


def fixed_point_nesterov_str_cvx(z, y, i, P, c, gd_step, beta):
    y_next = fixed_point_gd(z, P, c, gd_step)
    z_next = y_next + beta * (y_next - y)
    i_next = i + 1
    return z_next, y_next, i_next


def fp_eval_lah_nesterov_str_cvx_gd(i, val, supervised, z_star, P, c, gd_steps, betas):
    z, y, t, loss_vec, z_all, obj_diffs = val
    z_next, y_next, t_next = fixed_point_nesterov_str_cvx(z, y, t, P, c, gd_steps[i], betas[i])
    if supervised:
        diff = jnp.linalg.norm(z - z_star)
    else:
        diff = jnp.linalg.norm(z_next - z)
    loss_vec = loss_vec.at[i].set(diff)
    z_all = z_all.at[i, :].set(z_next)
    obj = .5 * y_next @ P @ y_next + c @ y_next
    opt_obj = .5 * z_star @ P @ z_star + c @ z_star
    obj_diffs = obj_diffs.at[i].set(obj - opt_obj)
    return z_next, y_next, t_next, loss_vec, z_all, obj_diffs


def fp_train_lah_nesterov_str_cvx_gd(i, val, supervised, z_star, P, c, gd_steps, betas):
    z, y, t, loss_vec = val
    z_next, y_next, t_next = fixed_point_nesterov_str_cvx(z, y, t, P, c, gd_steps[i], betas[i])
    diff = jnp.linalg.norm(z_next - z_star)
    loss_vec = loss_vec.at[i].set(diff)
    return z_next, y_next, t_next, loss_vec


def k_steps_eval_lah_nesterov_gd(k, z0, q, params, P, supervised, z_star, jit):
    iter_losses = jnp.zeros(k)
    z_all_plus_1 = jnp.zeros((k + 1, z0.size))
    z_all_plus_1 = z_all_plus_1.at[0, :].set(z0)
    fp_eval_partial = partial(fp_eval_lah_nesterov_str_cvx_gd,
                              supervised=supervised,
                              z_star=z_star,
                              P=P,
                              c=q,
                              gd_steps=params[:,0],
                              betas=params[:,1]
                              )
    z_all = jnp.zeros((k, z0.size))
    obj_diffs = jnp.zeros(k)
    y0 = z0
    t0 = 0
    val = z0, y0, t0, iter_losses, z_all, obj_diffs
    start_iter = 0
    if jit:
        out = lax.fori_loop(start_iter, k, fp_eval_partial, val)
    else:
        out = python_fori_loop(start_iter, k, fp_eval_partial, val)
    z_final, y_final, t_final, iter_losses, z_all, obj_diffs = out
    z_all_plus_1 = z_all_plus_1.at[1:, :].set(z_all)
    return z_final, iter_losses, z_all_plus_1, obj_diffs


def k_steps_train_lah_nesterov_gd(k, z0, q, params, P, supervised, z_star, jit):
    iter_losses = jnp.zeros(k)

    fp_train_partial = partial(fp_train_lah_nesterov_str_cvx_gd,
                               supervised=supervised,
                               z_star=z_star,
                               P=P,
                               c=q,
                               gd_steps=params[:,0],
                              betas=params[:,1]
                               )
    val = z0, z0, 0, iter_losses
    start_iter = 0
    if jit:
        out = lax.fori_loop(start_iter, k, fp_train_partial, val)
    else:
        out = python_fori_loop(start_iter, k, fp_train_partial, val)
    z_final, y_final, t_final, iter_losses = out
    return z_final, y_final, t_final, iter_losses
