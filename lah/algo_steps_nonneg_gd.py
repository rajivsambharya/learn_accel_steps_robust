from functools import partial

import cvxpy as cp
import jax.numpy as jnp
import jax.scipy as jsp
from jax import grad, lax, vmap
from jax.tree_util import tree_map

from lah.utils.generic_utils import python_fori_loop, unvec_symm, vec_symm




def k_steps_eval_lah_nonneg_gd(k, z0, q, params, A, safeguard_step, supervised, z_star, jit):
    iter_losses = jnp.zeros(k)
    z_all_plus_1 = jnp.zeros((k + 1, z0.size))
    z_all_plus_1 = z_all_plus_1.at[0, :].set(z0)

    fp_eval_partial = partial(fp_eval_lah_nonneg_gd_safeguard,
                              supervised=supervised,
                              z_star=z_star,
                              A=A,
                              safeguard_step=safeguard_step,
                              c=q,
                              nonneg_gd_steps=params
                              )
    z_all = jnp.zeros((k, z0.size))
    obj_diffs = jnp.zeros(k)
    val = z0, iter_losses, z_all, obj_diffs, False
    start_iter = 0
    if jit:
        out = lax.fori_loop(start_iter, k, fp_eval_partial, val)
    else:
        out = python_fori_loop(start_iter, k, fp_eval_partial, val)
    z_final, iter_losses, z_all, obj_diffs, iter = out
    z_all_plus_1 = z_all_plus_1.at[1:, :].set(z_all)
    return z_final, iter_losses, z_all_plus_1, obj_diffs


def k_steps_train_lah_nonneg_gd(k, z0, q, params, A, supervised, z_star, jit):
    iter_losses = jnp.zeros(k)

    fp_train_partial = partial(fp_train_lah_nonneg_gd,
                               supervised=supervised,
                               z_star=z_star,
                               A=A,
                               c=q,
                               nonneg_gd_steps=params
                               )
    val = z0, iter_losses
    start_iter = 0
    if jit:
        out = lax.fori_loop(start_iter, k, fp_train_partial, val)
    else:
        out = python_fori_loop(start_iter, k, fp_train_partial, val)
    z_final, iter_losses = out
    return z_final, iter_losses


def fp_eval_lah_nonneg_gd(i, val, supervised, z_star, A, c, nonneg_gd_steps):
    z, loss_vec, z_all, obj_diffs = val
    z_next = fixed_point_nonneg_gd(z, A, c, nonneg_gd_steps[i])
    diff = jnp.linalg.norm(z - z_star)
    loss_vec = loss_vec.at[i].set(diff)
    obj = .5 * jnp.linalg.norm(A @ z - c) ** 2
    opt_obj = .5 * jnp.linalg.norm(A @ z_star - c) ** 2
    obj_diffs = obj_diffs.at[i].set(obj - opt_obj)
    z_all = z_all.at[i, :].set(z_next)
    return z_next, loss_vec, z_all, obj_diffs


def fp_eval_lah_nonneg_gd_safeguard(i, val, supervised, z_star, A, safeguard_step, c, nonneg_gd_steps):
    z, loss_vec, z_all, obj_diffs, safeguard = val

    # next_safeguard = lax.cond(obj_diffs[i-1] > 2 * obj_diffs[i-2], lambda _: True, lambda _: safeguard, operand=None)
    # step_size = lax.cond(next_safeguard, lambda _: safeguard_step, lambda _: nonneg_gd_steps[i], operand=None)

    # z_next = fixed_point_nonneg_gd(z, A, c, lambd, step_size) #nonneg_gd_steps[i])

    z_next = fixed_point_nonneg_gd(z, A, c, lambd, nonneg_gd_steps[i])
    diff = jnp.linalg.norm(z - z_star)
    
    obj = .5 * jnp.linalg.norm(A @ z - c) ** 2 
    opt_obj = .5 * jnp.linalg.norm(A @ z_star - c) ** 2

    next_obj = .5 * jnp.linalg.norm(A @ z_next - c) ** 2


    # next_safeguard = lax.cond(next_obj - opt_obj < 0, lambda _: True, lambda _: safeguard, operand=None)
    next_safeguard = lax.cond(next_obj - opt_obj > 1 * (obj - opt_obj), lambda _: True, lambda _: safeguard, operand=None)
    z_next_final = lax.cond(next_safeguard, lambda _: fixed_point_nonneg_gd(z, A, c, safeguard_step), lambda _: z_next, operand=None)

    # next_obj = next_obj #* (iter % 10 == 0)
    # do safeguard
    # if next_obj > obj:
    #     z_next = fixed_point_nonneg_gd(z, A, c, lambd, safeguard_step)
    # z_next = lax.cond(iter % 10 == 0, lambda _: z_next, lambda _: fixed_point_nonneg_gd(z, A, c, lambd, safeguard_step), operand=None)
    
    # next_safeguard = lax.cond(next_obj - opt_obj > obj_diffs[i-10], lambda _: True, lambda _: safeguard, operand=None)
    # z_next_final = lax.cond(next_safeguard, lambda _: z_next, lambda _: fixed_point_nonneg_gd(z, A, c, lambd, safeguard_step), operand=None)

    
    # safeguard or z_next != z_next_final

    # update vectors carrying over
    loss_vec = loss_vec.at[i].set(diff)
    obj_diffs = obj_diffs.at[i].set(obj - opt_obj)
    # z_all = z_all.at[i, :].set(z_next_final)
    z_all = z_all.at[i, :].set(z_next_final)

    next_safeguard = safeguard
    

    return z_next_final, loss_vec, z_all, obj_diffs, next_safeguard


def fp_train_lah_nonneg_gd(i, val, supervised, z_star, A, c, nonneg_gd_steps):
    z, loss_vec = val
    z_next = fixed_point_nonneg_gd(z, A, c, nonneg_gd_steps[i])
    diff = jnp.linalg.norm(z_next - z_star) ** 2
    # diff = jnp.linalg.norm(z_next - z) ** 2
    loss_vec = loss_vec.at[i].set(diff)
    return z_next, loss_vec



def fixed_point_nonneg_gd(z, A, b, nonneg_gd_step):
    return jnp.clip(z + nonneg_gd_step * A.T.dot(b - A.dot(z)), a_min=0)


def soft_threshold(z, alpha):
    """
    soft-thresholding function for nonneg_gd
    """
    return jnp.clip(jnp.abs(z) - alpha, a_min=0) * jnp.sign(z)


def k_steps_train_nonneg_gd_accel(k, z0, q, A, params, supervised, z_star, jit):
    iter_losses = jnp.zeros(k)

    fp_train_partial = partial(fp_train_nonneg_gd_accel,
                               supervised=supervised,
                               z_star=z_star,
                               A=A,
                               b=q,
                               nonneg_gd_step=nonneg_gd_step
                               )
    val = z0, z0, 1, iter_losses
    start_iter = 0
    if jit:
        out = lax.fori_loop(start_iter, k, fp_train_partial, val)
    else:
        out = python_fori_loop(start_iter, k, fp_train_partial, val)
    z_final, y_final, t_final, iter_losses = out
    return z_final, iter_losses


def k_steps_eval_fnonneg_gd(k, z0, q, params, A, supervised, z_star, jit):
    iter_losses = jnp.zeros(k)
    obj_diffs = jnp.zeros(k)
    z_all_plus_1 = jnp.zeros((k + 1, z0.size))
    z_all_plus_1 = z_all_plus_1.at[0, :].set(z0)
    fp_eval_partial = partial(fp_eval_fnonneg_gd,
                              supervised=supervised,
                              z_star=z_star,
                              A=A,
                              b=q,
                              nonneg_gd_step=params
                              )
    z_all = jnp.zeros((k, z0.size))
    val = z0, z0, 1, iter_losses, obj_diffs, z_all
    start_iter = 0
    if jit:
        out = lax.fori_loop(start_iter, k, fp_eval_partial, val)
    else:
        out = python_fori_loop(start_iter, k, fp_eval_partial, val)
    z_final, y_final, t_final, iter_losses, obj_diffs, z_all = out
    z_all_plus_1 = z_all_plus_1.at[1:, :].set(z_all)
    return z_final, iter_losses, z_all_plus_1, obj_diffs


def k_steps_eval_nonneg_gd_accel_l2ws(k, z0, q, nonneg_gd_step, momentum_step, A, supervised, z_star, jit):
    iter_losses = jnp.zeros(k)
    obj_diffs = jnp.zeros(k)
    z_all_plus_1 = jnp.zeros((k + 1, z0.size))
    z_all_plus_1 = z_all_plus_1.at[0, :].set(z0)
    fp_eval_partial = partial(fp_eval_nonneg_gd_accel_l2ws,
                              supervised=supervised,
                              z_star=z_star,
                              A=A,
                              b=q,
                              nonneg_gd_step=nonneg_gd_step,
                              momentum_step=momentum_step
                              )
    z_all = jnp.zeros((k, z0.size))
    val = z0, z0, 1, iter_losses, obj_diffs, z_all
    start_iter = 0
    if jit:
        out = lax.fori_loop(start_iter, k, fp_eval_partial, val)
    else:
        out = python_fori_loop(start_iter, k, fp_eval_partial, val)
    z_final, y_final, t_final, iter_losses, obj_diffs, z_all = out
    z_all_plus_1 = z_all_plus_1.at[1:, :].set(z_all)
    return z_final, iter_losses, z_all_plus_1, obj_diffs


def k_steps_train_nonneg_gd_accel_l2ws(k, z0, q, nonneg_gd_step, momentum_step, A, supervised, z_star, jit):
    iter_losses = jnp.zeros(k)
    obj_diffs = jnp.zeros(k)
    z_all_plus_1 = jnp.zeros((k + 1, z0.size))
    z_all_plus_1 = z_all_plus_1.at[0, :].set(z0)
    fp_eval_partial = partial(fp_train_nonneg_gd_accel_l2ws,
                              supervised=supervised,
                              z_star=z_star,
                              A=A,
                              b=q,
                              nonneg_gd_step=nonneg_gd_step,
                              momentum_step=momentum_step
                              )
    z_all = jnp.zeros((k, z0.size))
    val = z0, z0, 1, iter_losses, obj_diffs
    start_iter = 0
    if jit:
        out = lax.fori_loop(start_iter, k, fp_eval_partial, val)
    else:
        out = python_fori_loop(start_iter, k, fp_eval_partial, val)
    z_final, y_final, t_final, iter_losses, obj_diffs = out
    return z_final, iter_losses #, z_all_plus_1, obj_diffs


def fixed_point_fnonneg_gd(z, y, t, A, b, nonneg_gd_step):
    """
    applies the fnonneg_gd fixed point operator
    """
    z_next = fixed_point_nonneg_gd(y, A, b, nonneg_gd_step)
    t_next = .5 * (1 + jnp.sqrt(1 + 4 * t ** 2))
    y_next = z_next + (t - 1) / t_next * (z_next - z)
    return z_next, y_next, t_next


def fp_train_nonneg_gd_accel(i, val, supervised, z_star, A, b, nonneg_gd_step):
    z, y, t, loss_vec, obj_diffs = val
    z_next, y_next, t_next = fixed_point_nonneg_gd_accel(z, y, t, A, b, nonneg_gd_step)
    if supervised:
        diff = jnp.linalg.norm(z - z_star)
    else:
        diff = jnp.linalg.norm(z_next - z)
    loss_vec = loss_vec.at[i].set(diff)
    return z_next, y_next, t_next, loss_vec, obj_diffs


def fp_train_nonneg_gd_accel_l2ws(i, val, supervised, z_star, A, b, nonneg_gd_step, momentum_step):
    z, y, t, loss_vec, obj_diffs = val
    # z_next, y_next, t_next = fixed_point_nonneg_gd_accel(z, y, t, A, b, nonneg_gd_step, momentum_step)
    z_next, y_next, t_next = fixed_point_nonneg_gd_accel(z, y, momentum_step, A, b, nonneg_gd_step)
    if supervised:
        diff = jnp.linalg.norm(z - z_star)
    else:
        diff = jnp.linalg.norm(z_next - z)
    loss_vec = loss_vec.at[i].set(diff)
    return z_next, y_next, t_next, loss_vec, obj_diffs


def fp_eval_nonneg_gd_accel_l2ws(i, val, supervised, z_star, A, b, nonneg_gd_step, momentum_step):
    z, y, t, loss_vec, obj_diffs, z_all = val
    z_next, y_next, t_next = fixed_point_nonneg_gd_accel(z, y, momentum_step, A, b, nonneg_gd_step)

    diff = jnp.linalg.norm(z_next - z)
    loss_vec = loss_vec.at[i].set(diff)

    obj = .5 * jnp.linalg.norm(A @ z - b) ** 2
    opt_obj = .5 * jnp.linalg.norm(A @ z_star - b) ** 2
    obj_diffs = obj_diffs.at[i].set(obj - opt_obj)

    z_all = z_all.at[i, :].set(z_next)
    return z_next, y_next, t_next, loss_vec, obj_diffs, z_all


def fp_eval_fnonneg_gd(i, val, supervised, z_star, A, b, nonneg_gd_step):
    z, y, t, loss_vec, obj_diffs, z_all = val
    z_next, y_next, t_next = fixed_point_fnonneg_gd(z, y, t, A, b, nonneg_gd_step[i])

    diff = jnp.linalg.norm(z_next - z)
    loss_vec = loss_vec.at[i].set(diff)

    obj = .5 * jnp.linalg.norm(A @ z - b) ** 2
    opt_obj = .5 * jnp.linalg.norm(A @ z_star - b) ** 2
    obj_diffs = obj_diffs.at[i].set(obj - opt_obj)

    z_all = z_all.at[i, :].set(z_next)
    return z_next, y_next, t_next, loss_vec, obj_diffs, z_all


def k_steps_train_lah_nonneg_gd_accel(k, z0, q, params, A, supervised, z_star, jit):
    iter_losses = jnp.zeros(k)

    fp_train_partial = partial(fp_train_lah_nonneg_gd_accel,
                               supervised=supervised,
                               z_star=z_star,
                               A=A,
                               c=q,
                               nonneg_gd_steps=params
                               )
    val = z0, z0, 1, iter_losses
    start_iter = 0
    if jit:
        out = lax.fori_loop(start_iter, k, fp_train_partial, val)
    else:
        out = python_fori_loop(start_iter, k, fp_train_partial, val)
    z_final, y_final, t_final, iter_losses = out
    return z_final, iter_losses


def fp_train_lah_nonneg_gd_accel(i, val, supervised, z_star, A, c, nonneg_gd_steps):
    z, y, t, loss_vec = val
    # z_next = fixed_point_fnonneg_gd(z, A, c, lambd, nonneg_gd_steps[i])
    z_next, y_next, t_next = fixed_point_nonneg_gd_accel(z, y, nonneg_gd_steps[i,1], A, c, nonneg_gd_steps[i,0])
    diff = jnp.linalg.norm(z_next - z_star) ** 2
    # diff = jnp.linalg.norm(z_next - z) ** 2
    obj = .5 * jnp.linalg.norm(A @ z - c) ** 2
    opt_obj = .5 * jnp.linalg.norm(A @ z_star - c) ** 2
    
    if supervised:
        loss_vec = loss_vec.at[i].set(diff)
    else:
        loss_vec = loss_vec.at[i].set(obj - opt_obj)
    # loss_vec = loss_vec.at[i].set(obj - opt_obj)
    return z_next, y_next, t_next, loss_vec


def k_steps_eval_lah_nonneg_gd_accel(k, z0, q, params, A, supervised, z_star, jit):
    iter_losses = jnp.zeros(k)
    z_all_plus_1 = jnp.zeros((k + 1, z0.size))
    z_all_plus_1 = z_all_plus_1.at[0, :].set(z0)

    fp_eval_partial = partial(fp_eval_lah_nonneg_gd_accel,
                              supervised=supervised,
                              z_star=z_star,
                              A=A,
                              c=q,
                              nonneg_gd_steps=params
                              )
    z_all = jnp.zeros((k, z0.size))
    obj_diffs = jnp.zeros(k)
    val = z0, z0, 1, iter_losses, obj_diffs, z_all
    start_iter = 0
    if jit:
        out = lax.fori_loop(start_iter, k, fp_eval_partial, val)
    else:
        out = python_fori_loop(start_iter, k, fp_eval_partial, val)
    z_final, y_final, t_final, iter_losses, obj_diffs, z_all = out
    z_all_plus_1 = z_all_plus_1.at[1:, :].set(z_all)
    return z_final, iter_losses, z_all_plus_1, obj_diffs


def fp_eval_lah_nonneg_gd_accel(i, val, supervised, z_star, A, c, nonneg_gd_steps):
    z, y, t, loss_vec, obj_diffs, z_all = val
    z_next, y_next, t_next = fixed_point_nonneg_gd_accel(z, y, nonneg_gd_steps[i,1], A, c, nonneg_gd_steps[i,0])
    if supervised:
        # diff = 10 * jnp.log10(jnp.linalg.norm(z - z_star) ** 2 / jnp.linalg.norm(z_star) ** 2)
        diff = jnp.linalg.norm(z - z_star)
    else:
        diff = jnp.linalg.norm(z_next - z)
    loss_vec = loss_vec.at[i].set(diff)

    obj = .5 * jnp.linalg.norm(A @ z - c) ** 2
    opt_obj = .5 * jnp.linalg.norm(A @ z_star - c) ** 2
    obj_diffs = obj_diffs.at[i].set(obj - opt_obj)

    z_all = z_all.at[i, :].set(z_next)
    return z_next, y_next, t_next, loss_vec, obj_diffs, z_all

def fixed_point_nonneg_gd_accel(z, y, beta, A, b, nonneg_gd_step):
    """
    applies the fnonneg_gd fixed point operator
    """
    z_next = fixed_point_nonneg_gd(y, A, b, nonneg_gd_step)
    # t_next = .5 * (1 + jnp.sqrt(1 + 4 * t ** 2))
    t_next = beta
    y_next = z_next + beta * (z_next - z)
    return z_next, y_next, t_next

# Soft thresholding (proximal operator for L1 norm)
# def soft_thresholding(x, thresh):
#     return jnp.sign(x) * jnp.maximum(jnp.abs(x) - thresh, 0.0)

# Gradient of smooth term 0.5 ||Az - b||^2
def grad_smooth(z, A, b):
    return A.T @ (A @ z - b)

# Objective: smooth + L1
def full_obj(z, A, b, lam):
    return 0.5 * jnp.sum((A @ z - b)**2) + lam * jnp.sum(jnp.abs(z))

# Proximal gradient step with backtracking
def fixed_point_prox_gd_backtracking(z, A, b, lam, eta_init, beta, alpha):
    grad = grad_smooth(z, A, b)
    eta = eta_init

    def cond_fn(val):
        eta, _ = val
        # z_new = soft_thresholding(z - eta * grad, eta * lam)
        z_new = jnp.clip(z - eta * grad, a_min=0)
        f_new = 0.5 * jnp.sum((A @ z_new - b)**2)
        f_old = 0.5 * jnp.sum((A @ z - b)**2)
        inner_prod = jnp.dot(grad, z_new - z)
        quad_term = (1 / (2 * eta)) * jnp.sum((z_new - z) ** 2)

        return f_new > f_old + inner_prod + quad_term

    def body_fn(val):
        eta, _ = val
        return eta * beta, None

    eta_final, _ = lax.while_loop(cond_fn, body_fn, (eta, None))
    # z_next = soft_thresholding(z - eta_final * grad, eta_final * lam)
    z_next = jnp.clip(z - eta_final * grad, a_min=0)
    return z_next, eta_final

# Single iteration body
def fp_eval_lasso_backtracking(i, val, supervised, z_star, A, b, beta, alpha):
    z, t, loss_vec, z_all, obj_diffs, eta_init = val
    z_next, eta_next = fixed_point_prox_gd_backtracking(z, A, b, eta_init, beta, alpha)
    t_next = t + 1

    diff = jnp.linalg.norm(z - z_star) if supervised else jnp.linalg.norm(z_next - z)
    loss_vec = loss_vec.at[i].set(diff)
    z_all = z_all.at[i, :].set(z_next)

    obj = full_obj(z_next, A, b)
    opt_obj = full_obj(z_star, A, b)
    obj_diffs = obj_diffs.at[i].set(obj - opt_obj)

    return z_next, t_next, loss_vec, z_all, obj_diffs, eta_next

# Top-level evaluation loop
def k_steps_eval_lasso_backtracking(k, z0, A, q, eta0, supervised, z_star, jit, beta=0.9, alpha=0.1):
    iter_losses = jnp.zeros(k)
    z_all_plus_1 = jnp.zeros((k + 1, z0.size)).at[0, :].set(z0)
    z_all = jnp.zeros((k, z0.size))
    obj_diffs = jnp.zeros(k)
    t0 = 0

    val = z0, t0, iter_losses, z_all, obj_diffs, eta0

    fp_eval_partial = partial(fp_eval_lasso_backtracking,
                              supervised=supervised,
                              z_star=z_star,
                              A=A,
                              b=q,
                              beta=beta,
                              alpha=alpha)

    if jit:
        out = lax.fori_loop(0, k, fp_eval_partial, val)
    else:
        for i in range(k):
            val = fp_eval_partial(i, val)
        out = val

    z_final, _, iter_losses, z_all, obj_diffs, _ = out
    z_all_plus_1 = z_all_plus_1.at[1:, :].set(z_all)

    return z_final, iter_losses, z_all_plus_1, obj_diffs








# Gradient of the smooth part
def grad_smooth(z, A, b):
    return A.T @ (A @ z - b)

# Full objective
def full_obj(z, A, b, lam):
    return 0.5 * jnp.sum((A @ z - b)**2) + lam * jnp.sum(jnp.abs(z))

# Soft thresholding
def soft_thresholding(x, tau):
    return jnp.sign(x) * jnp.maximum(jnp.abs(x) - tau, 0.0)

# Backtracking on Nesterov update
def fixed_point_prox_nesterov_backtracking(z, y, A, b, lam, eta_init, beta, shrink):
    eta = eta_init
    # grad_y = grad_smooth((1 - eta) * z + eta * v, A, b)

    def cond_fn(val):
        eta, _ = val
        # theta = eta
        # y = (1 - theta) * z + theta * v
        
        # x_next = soft_thresholding(y - eta * grad, eta * lam)
        # f_x_next = 0.5 * jnp.sum((A @ x_next - b) ** 2)
        # f_y = 0.5 * jnp.sum((A @ y - b) ** 2)
        # return f_x_next > f_y - (eta / 2.0) * jnp.sum(grad ** 2)

        grad = grad_smooth(y, A, b)
        z_new = soft_thresholding(y - eta * grad, eta * lam)
        
        y_new = z_new + beta * (z_new - z)
        f_old = 0.5 * jnp.sum((A @ z_new - b)**2)
        f_new = 0.5 * jnp.sum((A @ y_new - b)**2)
        inner_prod = jnp.dot(grad, z_new - y_new)
        quad_term = (1 / (2 * eta)) * jnp.sum((z_new - y_new) ** 2)

        return f_new > f_old + inner_prod + quad_term

    def body_fn(val):
        eta, _ = val
        return eta * shrink, None

    eta_final, _ = lax.while_loop(cond_fn, body_fn, (eta, None))

    # Compute final update
    theta = eta_final
    # y = (1 - theta) * z + theta * 0
    grad = grad_smooth(z, A, b)
    z_next = soft_thresholding(y - eta_final * grad, eta_final * lam)
    y_next = z_next + beta * (z_next - z)
    

    return z_next, y_next, eta_final

# Single iteration
def fp_eval_lasso_nesterov_backtracking(i, val, supervised, z_star, A, b, shrink):
    z, y, t, loss_vec, z_all, obj_diffs, eta_init = val
    # beta = (i - 2) / (i + 1)
    t_next = t + 1
    beta = (t_next - 1) / (t_next + 2)

    z_next, y_next, eta_next = fixed_point_prox_nesterov_backtracking(z, y, A, b, eta_init, beta, shrink)

    # t_next = t + 1
    # y_next = z + (t_next - 1) / (t_next + 2) * (z_next - z)

    diff = jnp.linalg.norm(z - z_star) if supervised else jnp.linalg.norm(z_next - z)
    loss_vec = loss_vec.at[i].set(diff)
    z_all = z_all.at[i, :].set(z_next)

    obj = full_obj(z_next, A, b)
    opt_obj = full_obj(z_star, A, b)
    obj_diffs = obj_diffs.at[i].set(obj - opt_obj)

    return z_next, y_next, t_next, loss_vec, z_all, obj_diffs, eta_next

# Top-level loop
def k_steps_eval_lasso_nesterov_backtracking(k, z0, A, q, eta0, supervised, z_star, jit, shrink=0.95):
    iter_losses = jnp.zeros(k)
    z_all_plus_1 = jnp.zeros((k + 1, z0.size)).at[0, :].set(z0)
    z_all = jnp.zeros((k, z0.size))
    obj_diffs = jnp.zeros(k)
    t0 = 1
    v0 = z0

    val = z0, v0, t0, iter_losses, z_all, obj_diffs, eta0

    fp_eval_partial = partial(fp_eval_lasso_nesterov_backtracking,
                              supervised=supervised,
                              z_star=z_star,
                              A=A,
                              b=q,
                              shrink=shrink)

    if jit:
        out = lax.fori_loop(0, k, fp_eval_partial, val)
    else:
        for i in range(k):
            val = fp_eval_partial(i, val)
        out = val

    z_final, _, _, iter_losses, z_all, obj_diffs, _ = out
    z_all_plus_1 = z_all_plus_1.at[1:, :].set(z_all)

    return z_final, iter_losses, z_all_plus_1, obj_diffs


def fixed_point_fnonneg_gd_backtracking__(x_km1, x_km2, t_km1, theta_km1, mu0_k, A, b, beta):
    # def fnonneg_gd_step(x_km1, x_km2, t_km1, theta_km1):
    #     t_k = (1 + jnp.sqrt(1 + 4 * t_km1**2)) / 2
    #     theta_k = t_km1 / t_k
    #     y_k = x_km1 + theta_k * (x_km1 - x_km2)
    #     return t_k, y_k
    def fnonneg_gd_step(x_km1, x_km2, t_k, theta_k):
        t_kp1 = (1 + jnp.sqrt(1 + 4 * (theta_k ** 2) * t_k ** 2)) / 2
        y_kp1 = x_km1 + (t_k - 1) / t_kp1 * (x_km1 - x_km2)
        return t_kp1, y_kp1


    def grad_smooth(z):
        return A.T @ (A @ z - b)

    def full_obj(z):
        return 0.5 * jnp.sum((A @ z - b) ** 2) #+ lam * jnp.sum(jnp.abs(z))

    def Q_mu(p, y, grad_y, mu):
        return (full_obj(y) +
                jnp.dot(grad_y, p - y) +
                (1 / (2 * mu)) * jnp.sum((p - y) ** 2)) #+ lam * jnp.sum(jnp.abs(p)))

    def prox_step(y, grad_y, mu):
        # return soft_thresholding(y - mu * grad_y, mu * lam)
        return jnp.clip(y - mu * grad_y, a_min=0)

    mu_k = mu0_k
    t_k, y_k = fnonneg_gd_step(x_km1, x_km2, t_km1, theta_km1)

    grad_y = grad_smooth(y_k)
    p_mu_y = prox_step(y_k, grad_y, mu_k)

    def cond_fn(val):
        mu_k, _, _ = val
        p_mu_y = prox_step(y_k, grad_y, mu_k)
        return full_obj(p_mu_y) > Q_mu(p_mu_y, y_k, grad_y, mu_k)

    def body_fn(val):
        mu_k, theta_km1, _ = val
        mu_k_new = mu_k * beta
        theta_km1_new = theta_km1 / beta
        return mu_k_new, theta_km1_new, prox_step(y_k, grad_y, mu_k_new)

    mu_k, theta_km1, x_k = lax.while_loop(cond_fn, body_fn, (mu_k, theta_km1, p_mu_y))

    # Post-update
    mu0_kp1 = mu_k  # next iteration’s initial mu
    theta_k = mu_k / mu0_kp1
    t_k, y_kp1 = fnonneg_gd_step(x_k, x_km1, t_k, theta_k)

    return x_k, x_km1, t_k, theta_k, mu0_kp1

def fp_eval_fnonneg_gd_backtracking__(i, val, supervised, z_star, A, b, beta):
    x_km1, x_km2, t_km1, theta_km1, loss_vec, z_all, obj_diffs, mu0_k = val

    x_k, x_km1_new, t_k, theta_k, mu0_kp1 = fixed_point_fnonneg_gd_backtracking__(
        x_km1, x_km2, t_km1, theta_km1, mu0_k, A, b, beta
    )

    diff = jnp.linalg.norm(x_k - z_star) if supervised else jnp.linalg.norm(x_k - x_km1)
    loss_vec = loss_vec.at[i].set(diff)
    z_all = z_all.at[i, :].set(x_k)

    obj = 0.5 * jnp.sum((A @ x_k - b) ** 2) #+ lam * jnp.sum(jnp.abs(x_k))
    opt_obj = 0.5 * jnp.sum((A @ z_star - b) ** 2) #+ lam * jnp.sum(jnp.abs(z_star))
    obj_diffs = obj_diffs.at[i].set(obj - opt_obj)

    return x_k, x_km1_new, t_k, theta_k, loss_vec, z_all, obj_diffs, mu0_kp1

def k_steps_eval_fnonneg_gd_backtracking__(k, z0, A, q, eta0, supervised, z_star, jit, beta=0.8):
    d = z0.size
    iter_losses = jnp.zeros(k)
    z_all = jnp.zeros((k, d))
    obj_diffs = jnp.zeros(k)

    x_km2 = z0
    x_km1 = z0
    t_km1 = 0.0
    theta_km1 = 1.0
    mu0_k = eta0

    val = x_km1, x_km2, t_km1, theta_km1, iter_losses, z_all, obj_diffs, mu0_k

    partial_eval = partial(fp_eval_fnonneg_gd_backtracking__,
                           supervised=supervised,
                           z_star=z_star,
                           A=A,
                           b=q,
                           beta=beta)

    if jit:
        out = lax.fori_loop(0, k, partial_eval, val)
    else:
        for i in range(k):
            val = partial_eval(i, val)
        out = val

    x_final, _, _, _, iter_losses, z_all, obj_diffs, _ = out
    z_all_plus_1 = jnp.zeros((k + 1, d)).at[0, :].set(z0).at[1:, :].set(z_all)

    return x_final, iter_losses, z_all_plus_1, obj_diffs
