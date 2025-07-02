from functools import partial

import cvxpy as cp
import jax.numpy as jnp
import jax.scipy as jsp
from jax import grad, lax, vmap
from jax.tree_util import tree_map

from lah.utils.generic_utils import python_fori_loop, unvec_symm, vec_symm


def k_steps_eval_lah_logisticgd(k, z0, q, params, num_points, safeguard_step, supervised, z_star, jit, safeguard=True):
    iter_losses = jnp.zeros(k)
    z_all_plus_1 = jnp.zeros((k + 1, z0.size))
    z_all_plus_1 = z_all_plus_1.at[0, :].set(z0)

    X_flat = q[:num_points * 784]
    X = jnp.reshape(X_flat, (num_points, 784))
    y = q[num_points * 784:]

    if safeguard:
        fp_eval_partial = partial(fp_eval_lah_logisticgd_safeguard,
                              supervised=supervised,
                              z_star=z_star,
                              X=X,
                              y=y,
                              safeguard_step=safeguard_step,
                              gd_steps=params
                              )
    else:
        fp_eval_partial = partial(fp_eval_lah_logisticgd,
                                supervised=supervised,
                                z_star=z_star,
                                X=X,
                                y=y,
                                gd_steps=params
                                )
    # nesterov
    fp_eval_nesterov_partial = partial(fp_eval_nesterov_logisticgd,
                            supervised=supervised,
                            z_star=z_star,
                            X=X,
                            y=y,
                            gd_steps=safeguard_step #params
                            )
    
    z_all = jnp.zeros((k, z0.size))
    obj_diffs = jnp.zeros(k)
    val = z0, iter_losses, z_all, obj_diffs

    switch_2_nesterov = False

    if switch_2_nesterov:
        # run learned for 100
        start_iter = 0
        if jit:
            out = lax.fori_loop(start_iter, k, fp_eval_partial, val)
        else:
            out = python_fori_loop(start_iter, k, fp_eval_partial, val)
        z_final, iter_losses, z_all, obj_diffs = out

        # run nesterov for the rest
        start_iter = 0
        iter_losses_nesterov = jnp.zeros(k - 100)
        z_all_nesterov = jnp.zeros((k - 100, z0.size))
        obj_diffs_nesterov = jnp.zeros(k - 100)
        val = z_final, z_final, 0, iter_losses_nesterov, z_all_nesterov, obj_diffs_nesterov
        if jit:
            out_nesterov = lax.fori_loop(start_iter, k - 100, fp_eval_nesterov_partial, val)
        else:
            out_nesterov = python_fori_loop(start_iter, k - 100, fp_eval_nesterov_partial, val)
        z_final_nesterov, y_final_nesterov, t_final_nesterov, iter_losses_nesterov, z_all_nesterov, obj_diffs_nesterov = out_nesterov

        # z_final, y_final, t_final, iter_losses, z_all, obj_diffs

        # stitch everything together
        z_final = z_final_nesterov

        z_all = z_all.at[100:, :].set(z_all_nesterov)
        iter_losses = iter_losses.at[100:].set(iter_losses_nesterov)
        obj_diffs = obj_diffs.at[100:].set(obj_diffs_nesterov)

        z_all_plus_1 = z_all_plus_1.at[1:, :].set(z_all)
    else:
        z_all = jnp.zeros((k, z0.size))
        obj_diffs = jnp.zeros(k)
        safeguard = False
        prev_obj = 1e10
        val = z0, iter_losses, z_all, obj_diffs #, safeguard, prev_obj
        start_iter = 0
        if jit:
            out = lax.fori_loop(start_iter, k, fp_eval_partial, val)
        else:
            out = python_fori_loop(start_iter, k, fp_eval_partial, val)
        # z_final, iter_losses, z_all, obj_diffs, safeguard, prev_obj = out
        z_final, iter_losses, z_all, obj_diffs = out
        z_all_plus_1 = z_all_plus_1.at[1:, :].set(z_all)
    return z_final, iter_losses, z_all_plus_1, obj_diffs


def k_steps_train_lah_logisticgd(k, z0, q, params, num_points, supervised, z_star, jit):
    iter_losses = jnp.zeros(k)

    X_flat = q[:num_points * 784]
    X = jnp.reshape(X_flat, (num_points, 784))
    y = q[num_points * 784:]

    fp_train_partial = partial(fp_train_lah_logisticgd,
                               supervised=supervised,
                               z_star=z_star,
                               X=X,
                               y=y,
                               gd_steps=params
                               )
    val = z0, iter_losses
    start_iter = 0
    if jit:
        out = lax.fori_loop(start_iter, k, fp_train_partial, val)
    else:
        out = python_fori_loop(start_iter, k, fp_train_partial, val)
    z_final, iter_losses = out
    return z_final, iter_losses

def fp_eval_lah_logisticgd(i, val, supervised, z_star, X, y, gd_steps):
    z, loss_vec, z_all, obj_diffs = val
    z_next = fixed_point_logisticgd(z, X, y, gd_steps[i])
    diff = jnp.linalg.norm(z - z_star)
    loss_vec = loss_vec.at[i].set(diff)
    w, b = z[:-1], z[-1]
    obj = compute_loss(y, sigmoid(X @ w + b))
    w_star, b_star = z_star[:-1], z_star[-1]
    opt_obj = compute_loss(y, sigmoid(X @ w_star + b_star))


    obj_diffs = obj_diffs.at[i].set(obj - opt_obj)
    z_all = z_all.at[i, :].set(z_next)
    return z_next, loss_vec, z_all, obj_diffs


def fp_eval_lah_logisticgd_safeguard(i, val, supervised, z_star, X, y, safeguard_step, gd_steps):
    # z, loss_vec, z_all, obj_diffs = val
    z, loss_vec, z_all, obj_diffs, safeguard, prev_obj = val
    z_next = fixed_point_logisticgd(z, X, y, gd_steps[i])
    diff = jnp.linalg.norm(z - z_star)
    loss_vec = loss_vec.at[i].set(diff)

    w, b = z[:-1], z[-1]
    obj = compute_loss(y, sigmoid(X @ w + b))

    w_star, b_star = z_star[:-1], z_star[-1]
    opt_obj = compute_loss(y, sigmoid(X @ w_star + b_star))

    w_next, b_next = z_next[:-1], z_next[-1]
    next_obj = compute_loss(y, sigmoid(X @ w_next + b_next))

    # z_next = lax.cond(next_obj < obj, lambda _: z_next, lambda _: fixed_point_logisticgd(z, X, y, safeguard_step), operand=None)

    next_safeguard = lax.cond(next_obj - opt_obj > 20 * (obj - opt_obj), lambda _: True, lambda _: safeguard, operand=None)
    # next_safeguard = lax.cond(next_obj - opt_obj > 10 * prev_obj, lambda _: True, lambda _: safeguard, operand=None)
    z_next_final = lax.cond(next_safeguard, lambda _: fixed_point_logisticgd(z, X, y, safeguard_step), lambda _: z_next, operand=None)


    obj_diffs = obj_diffs.at[i].set(obj - opt_obj)
    z_all = z_all.at[i, :].set(z_next)
    # return z_next, loss_vec, z_all, obj_diffs
    return z_next_final, loss_vec, z_all, obj_diffs, next_safeguard, obj - opt_obj


def fp_train_lah_logisticgd(i, val, supervised, z_star, X, y, gd_steps):
    z, loss_vec = val
    z_next = fixed_point_logisticgd(z, X, y, gd_steps[i])
    diff = jnp.linalg.norm(z_next - z_star) ** 2
    loss_vec = loss_vec.at[i].set(diff)
    return z_next, loss_vec


def fixed_point_logisticgd(z, X, y, gd_step):
    w, b = z[:-1], z[-1]
    logits = jnp.clip(X @ w + b, -20, 20)
    y_hat = sigmoid(logits)
    
    dw, db = compute_gradient(X, y, y_hat)
    w_next = w - gd_step * dw #(dw + .01 * w)
    b_next = b - gd_step * db #(db + .01 * b)
    z_next = jnp.hstack([w_next, b_next])
    return z_next


def sigmoid(x):
    return 1 / (1 + jnp.exp(-x))


def compute_gradient(X, y, y_hat):
    m = y.shape[0]
    dw = 1/m * jnp.dot(X.T, (y_hat - y))
    db = 1/m * jnp.sum(y_hat - y)
    return dw, db


def compute_loss(y, y_hat):
    m = y.shape[0]
    return -1/m * (jnp.dot(y, jnp.log(1e-6 + y_hat)) + jnp.dot((1 - y), jnp.log(1 + 1e-6 - y_hat)))


def k_steps_eval_logisticgd(k, z0, q, gd_step, num_points, supervised, z_star, jit, safeguard=True):
    iter_losses = jnp.zeros(k)
    z_all_plus_1 = jnp.zeros((k + 1, z0.size))
    z_all_plus_1 = z_all_plus_1.at[0, :].set(z0)

    X_flat = q[:num_points * 784]
    X = jnp.reshape(X_flat, (num_points, 784))
    y = q[num_points * 784:]

    # if safeguard:
    #     fp_eval_partial = partial(fp_eval_logisticgd_safeguard,
    #                           supervised=supervised,
    #                           z_star=z_star,
    #                           X=X,
    #                           y=y,
    #                           safeguard_step=safeguard_step,
    #                           gd_steps=params
    #                           )
    # else:
    fp_eval_partial = partial(fp_eval_logisticgd,
                            supervised=supervised,
                            z_star=z_star,
                            X=X,
                            y=y,
                            gd_step=gd_step
                            )
    
    z_all = jnp.zeros((k, z0.size))
    obj_diffs = jnp.zeros(k)
    val = z0, iter_losses, z_all, obj_diffs
    start_iter = 0
    if jit:
        out = lax.fori_loop(start_iter, k, fp_eval_partial, val)
    else:
        out = python_fori_loop(start_iter, k, fp_eval_partial, val)
    z_final, iter_losses, z_all, obj_diffs = out
    z_all_plus_1 = z_all_plus_1.at[1:, :].set(z_all)
    return z_final, iter_losses, z_all_plus_1, obj_diffs


def k_steps_train_logisticgd(k, z0, q, gd_step, num_points, supervised, z_star, jit):
    iter_losses = jnp.zeros(k)

    X_flat = q[:num_points * 784]
    X = jnp.reshape(X_flat, (num_points, 784))
    y = q[num_points * 784:]

    fp_train_partial = partial(fp_train_logisticgd,
                               supervised=supervised,
                               z_star=z_star,
                               X=X,
                               y=y,
                               gd_step=gd_step
                               )
    val = z0, iter_losses
    start_iter = 0
    if jit:
        out = lax.fori_loop(start_iter, k, fp_train_partial, val)
    else:
        out = python_fori_loop(start_iter, k, fp_train_partial, val)
    z_final, iter_losses = out
    return z_final, iter_losses


def fp_eval_logisticgd(i, val, supervised, z_star, X, y, gd_step):
    z, loss_vec, z_all, obj_diffs = val
    z_next = fixed_point_logisticgd(z, X, y, gd_step)
    diff = jnp.linalg.norm(z - z_star)
    loss_vec = loss_vec.at[i].set(diff)
    w, b = z[:-1], z[-1]
    obj = compute_loss(y, sigmoid(X @ w + b))
    w_star, b_star = z_star[:-1], z_star[-1]
    opt_obj = compute_loss(y, sigmoid(X @ w_star + b_star))


    obj_diffs = obj_diffs.at[i].set(obj - opt_obj)
    z_all = z_all.at[i, :].set(z_next)
    return z_next, loss_vec, z_all, obj_diffs


def fp_train_logisticgd(i, val, supervised, z_star, X, y, gd_step):
    z, loss_vec = val
    z_next = fixed_point_logisticgd(z, X, y, gd_step)
    diff = jnp.linalg.norm(z_next - z_star)
    loss_vec = loss_vec.at[i].set(diff)
    return z_next, loss_vec


def k_steps_eval_lm_logisticgd(k, z0, q, params, num_points, supervised, z_star, jit, safeguard=False):
    iter_losses = jnp.zeros(k)
    z_all_plus_1 = jnp.zeros((k + 1, z0.size))
    z_all_plus_1 = z_all_plus_1.at[0, :].set(z0)

    X_flat = q[:num_points * 784]
    X = jnp.reshape(X_flat, (num_points, 784))
    y = q[num_points * 784:]

    if safeguard:
        fp_eval_partial = partial(fp_eval_lm_logisticgd_safeguard,
                              supervised=supervised,
                              z_star=z_star,
                              X=X,
                              y=y,
                              safeguard_step=safeguard_step,
                              gd_steps=params
                              )
    else:
        fp_eval_partial = partial(fp_eval_lm_logisticgd,
                                supervised=supervised,
                                z_star=z_star,
                                X=X,
                                y=y,
                                gd_step=params
                                )
    
    z_all = jnp.zeros((k, z0.size))
    obj_diffs = jnp.zeros(k)
    val = z0, iter_losses, z_all, obj_diffs
    start_iter = 0
    if jit:
        out = lax.fori_loop(start_iter, k, fp_eval_partial, val)
    else:
        out = python_fori_loop(start_iter, k, fp_eval_partial, val)
    z_final, iter_losses, z_all, obj_diffs = out
    z_all_plus_1 = z_all_plus_1.at[1:, :].set(z_all)
    return z_final, iter_losses, z_all_plus_1, obj_diffs


def k_steps_train_lm_logisticgd(k, z0, q, params, num_points, supervised, z_star, jit):
    iter_losses = jnp.zeros(k)

    X_flat = q[:num_points * 784]
    X = jnp.reshape(X_flat, (num_points, 784))
    y = q[num_points * 784:]

    fp_train_partial = partial(fp_train_lm_logisticgd,
                               supervised=supervised,
                               z_star=z_star,
                               X=X,
                               y=y,
                               gd_step=params
                               )
    val = z0, iter_losses
    start_iter = 0
    if jit:
        out = lax.fori_loop(start_iter, k, fp_train_partial, val)
    else:
        out = python_fori_loop(start_iter, k, fp_train_partial, val)
    z_final, iter_losses = out
    return z_final, iter_losses

def fp_eval_lm_logisticgd(i, val, supervised, z_star, X, y, gd_step):
    z, loss_vec, z_all, obj_diffs = val
    z_next = fixed_point_logisticgd_lm(z, X, y, gd_step)
    diff = jnp.linalg.norm(z - z_star)
    loss_vec = loss_vec.at[i].set(diff)
    w, b = z[:-1], z[-1]
    obj = compute_loss(y, sigmoid(X @ w + b))
    w_star, b_star = z_star[:-1], z_star[-1]
    opt_obj = compute_loss(y, sigmoid(X @ w_star + b_star))


    obj_diffs = obj_diffs.at[i].set(obj - opt_obj)
    z_all = z_all.at[i, :].set(z_next)
    return z_next, loss_vec, z_all, obj_diffs


def fp_train_lm_logisticgd(i, val, supervised, z_star, X, y, gd_step):
    z, loss_vec = val
    z_next = fixed_point_logisticgd_lm(z, X, y, gd_step)
    diff = jnp.linalg.norm(z_next - z_star)
    loss_vec = loss_vec.at[i].set(diff)
    return z_next, loss_vec


def fixed_point_logisticgd_lm(z, X, y, gd_step):
    w, b = z[:-1], z[-1]
    y_hat = sigmoid(X @ w + b)
    dw, db = compute_gradient(X, y, y_hat)
    w_next = w - gd_step[:-1] * dw #(dw + .01 * w)
    b_next = b - gd_step[-1] * db #(db + .01 * b)
    z_next = jnp.hstack([w_next, b_next])
    return z_next

def k_steps_train_nesterov_l2ws_logisticgd(k, z0, q, gd_step, num_points, supervised, z_star, jit, safeguard=True):
    iter_losses = jnp.zeros(k)

    X_flat = q[:num_points * 784]
    X = jnp.reshape(X_flat, (num_points, 784))
    y = q[num_points * 784:]

    fp_train_partial = partial(fp_train_nesterov_l2ws_logisticgd,
                               supervised=supervised,
                               z_star=z_star,
                               X=X,
                               y=y,
                               gd_step=gd_step
                               )
    t0 = 0
    val = z0, z0, t0, iter_losses
    start_iter = 0
    if jit:
        out = lax.fori_loop(start_iter, k, fp_train_partial, val)
    else:
        out = python_fori_loop(start_iter, k, fp_train_partial, val)
    z_final, y_final, t_final, iter_losses = out
    return z_final, iter_losses

def k_steps_eval_nesterov_l2ws_logisticgd(k, z0, q, gd_step, num_points, supervised, z_star, jit, safeguard=True):
    iter_losses = jnp.zeros(k)
    z_all_plus_1 = jnp.zeros((k + 1, z0.size))
    z_all_plus_1 = z_all_plus_1.at[0, :].set(z0)

    X_flat = q[:num_points * 784]
    X = jnp.reshape(X_flat, (num_points, 784))
    y = q[num_points * 784:]

    fp_eval_partial = partial(fp_eval_nesterov_logisticgd,
                            supervised=supervised,
                            z_star=z_star,
                            X=X,
                            y=y,
                            gd_step=gd_step
                            )
    
    z_all = jnp.zeros((k, z0.size))
    obj_diffs = jnp.zeros(k)
    val = z0, z0, 0, iter_losses, z_all, obj_diffs
    start_iter = 0
    if jit:
        out = lax.fori_loop(start_iter, k, fp_eval_partial, val)
    else:
        out = python_fori_loop(start_iter, k, fp_eval_partial, val)
    z_final, y_final, t_final, iter_losses, z_all, obj_diffs = out
    z_all_plus_1 = z_all_plus_1.at[1:, :].set(z_all)
    return z_final, iter_losses, z_all_plus_1, obj_diffs



def k_steps_eval_nesterov_logisticgd(k, z0, q, params, num_points, supervised, z_star, jit, safeguard=True):
    iter_losses = jnp.zeros(k)
    z_all_plus_1 = jnp.zeros((k + 1, z0.size))
    z_all_plus_1 = z_all_plus_1.at[0, :].set(z0)

    X_flat = q[:num_points * 784]
    X = jnp.reshape(X_flat, (num_points, 784))
    y = q[num_points * 784:]

    fp_eval_partial = partial(fp_eval_nesterov_logisticgd,
                            supervised=supervised,
                            z_star=z_star,
                            X=X,
                            y=y,
                            gd_steps=params
                            )
    
    z_all = jnp.zeros((k, z0.size))
    obj_diffs = jnp.zeros(k)
    val = z0, z0, 0, iter_losses, z_all, obj_diffs
    start_iter = 0
    if jit:
        out = lax.fori_loop(start_iter, k, fp_eval_partial, val)
    else:
        out = python_fori_loop(start_iter, k, fp_eval_partial, val)
    z_final, y_final, t_final, iter_losses, z_all, obj_diffs = out
    z_all_plus_1 = z_all_plus_1.at[1:, :].set(z_all)
    return z_final, iter_losses, z_all_plus_1, obj_diffs


def fp_eval_nesterov_logisticgd(i, val, supervised, z_star, X, y, gd_step):
    z, v, t, loss_vec, z_all, obj_diffs = val
    z_next, v_next, t_next = fixed_point_nesterov_logisticgd(z, v, t, X, y, gd_step)
    diff = jnp.linalg.norm(z - z_star)
    loss_vec = loss_vec.at[i].set(diff)
    w, b = z[:-1], z[-1]
    obj = compute_loss(y, sigmoid(X @ w + b))
    w_star, b_star = z_star[:-1], z_star[-1]
    opt_obj = compute_loss(y, sigmoid(X @ w_star + b_star))

    obj_diffs = obj_diffs.at[i].set(obj - opt_obj)
    z_all = z_all.at[i, :].set(z_next)
    return z_next, v_next, t_next, loss_vec, z_all, obj_diffs


def fp_train_nesterov_l2ws_logisticgd(i, val, supervised, z_star, X, y, gd_step):
    z, v, t, loss_vec = val
    z_next, v_next, t_next = fixed_point_nesterov_logisticgd(z, v, t, X, y, gd_step)
    diff = jnp.linalg.norm(z - z_star)
    loss_vec = loss_vec.at[i].set(diff)
    w, b = z[:-1], z[-1]
    obj = compute_loss(y, sigmoid(X @ w + b))
    w_star, b_star = z_star[:-1], z_star[-1]
    opt_obj = compute_loss(y, sigmoid(X @ w_star + b_star))

    # obj_diffs = obj_diffs.at[i].set(obj - opt_obj)
    # z_all = z_all.at[i, :].set(z_next)
    return z_next, v_next, t_next, loss_vec


def fixed_point_nesterov_logisticgd(z, v, t, X, y, gd_step):
    v_next = fixed_point_logisticgd(z, X, y, gd_step)
    # t_next = .5 * (1 + jnp.sqrt(1 + 4 * t ** 2))
    z_next = v_next + t/ (t + 3) * (v_next - v)
    t_next = t + 1
    return z_next, v_next, t_next






def fixed_point_lah_nesterov_logisticgd(z, v, t, X, y, gd_step, beta):
    v_next = fixed_point_logisticgd(z, X, y, gd_step)
    z_next = v_next + beta * (v_next - v)
    t_next = t + 1
    return z_next, v_next, t_next

def fp_eval_lah_nesterov_logistic(i, val, supervised, z_star, X, y_label, gd_steps, betas):
    z, y, t, loss_vec, z_all, obj_diffs = val
    z_next, y_next, t_next = fixed_point_lah_nesterov_logisticgd(z, y, t, X, y_label, gd_steps[i], jnp.clip(betas[i], a_min=0, a_max=100)) #beta_vals = np.clip(params[:,1], a_min=0, a_max=0.999)
    if supervised:
        diff = jnp.linalg.norm(z - z_star)
    else:
        diff = jnp.linalg.norm(z_next - z)
    
    loss_vec = loss_vec.at[i].set(diff)
    w, b = z[:-1], z[-1]

    obj = compute_loss(y_label, sigmoid(X @ w + b))
    w_star, b_star = z_star[:-1], z_star[-1]
    opt_obj = compute_loss(y_label, sigmoid(X @ w_star + b_star))


    obj_diffs = obj_diffs.at[i].set(obj - opt_obj)
    z_all = z_all.at[i, :].set(z_next)

    obj_diffs = obj_diffs.at[i].set(obj - opt_obj)
    return z_next, y_next, t_next, loss_vec, z_all, obj_diffs


def fp_train_lah_nesterov_logistic(i, val, supervised, z_star, X, y_label, gd_steps, betas):
    z, y, t, loss_vec = val
    z_next, y_next, t_next = fixed_point_lah_nesterov_logisticgd(z, y, t, X, y_label, gd_steps[i], jnp.clip(betas[i], a_min=0, a_max=100))
    # diff = jnp.linalg.norm(z_next - z_star)
    # diff = jnp.linalg.norm(z_next - z)
    w, b = z[:-1], z[-1]

    obj = compute_loss(y_label, sigmoid(X @ w + b))
    w_star, b_star = z_star[:-1], z_star[-1]
    opt_obj = compute_loss(y_label, sigmoid(X @ w_star + b_star))

    loss_vec = loss_vec.at[i].set(obj - opt_obj)
    return z_next, y_next, t_next, loss_vec


def k_steps_eval_lah_nesterov_gd(k, z0, q, params, num_points, supervised, z_star, jit):
    X_flat = q[:num_points * 784]
    X = jnp.reshape(X_flat, (num_points, 784))
    y_label = q[num_points * 784:]
    iter_losses = jnp.zeros(k)
    z_all_plus_1 = jnp.zeros((k + 1, z0.size))
    z_all_plus_1 = z_all_plus_1.at[0, :].set(z0)
    fp_eval_partial = partial(fp_eval_lah_nesterov_logistic,
                              supervised=supervised,
                              z_star=z_star,
                              X=X,
                              y_label=y_label,
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


def k_steps_train_lah_nesterov_gd(k, z0, q, params, num_points, supervised, z_star, jit):
    X_flat = q[:num_points * 784]
    X = jnp.reshape(X_flat, (num_points, 784))
    y_label = q[num_points * 784:]
    iter_losses = jnp.zeros(k)

    fp_train_partial = partial(fp_train_lah_nesterov_logistic,
                               supervised=supervised,
                               z_star=z_star,
                               X=X,
                               y_label=y_label,
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


def k_steps_eval_adam_logistic(k, z0, q, num_points, step_size, beta1, beta2, epsilon, supervised, z_star, jit):
    X_flat = q[:num_points * 784]
    X = jnp.reshape(X_flat, (num_points, 784))
    y_label = q[num_points * 784:]
    iter_losses = jnp.zeros(k)
    z_all_plus_1 = jnp.zeros((k + 1, z0.size))
    z_all_plus_1 = z_all_plus_1.at[0, :].set(z0)
    fp_eval_partial = partial(fp_eval_adam_logistic,
                              supervised=supervised,
                              z_star=z_star,
                              X=X,
                              y_label=y_label,
                              step_size=step_size,
                              beta1=beta1,
                              beta2=beta2,
                              epsilon=epsilon
                              )
    z_all = jnp.zeros((k, z0.size))
    obj_diffs = jnp.zeros(k)
    # y0 = z0
    t0 = 1
    m_w, v_w, m_b, v_b = jnp.zeros(784), jnp.zeros(784), jnp.zeros(1), jnp.zeros(1)
    val = z0, t0, iter_losses, z_all, obj_diffs, m_w, v_w, m_b, v_b
    
    start_iter = 0
    if jit:
        out = lax.fori_loop(start_iter, k, fp_eval_partial, val)
    else:
        out = python_fori_loop(start_iter, k, fp_eval_partial, val)
    z_final, t_final, iter_losses, z_all, obj_diffs, m_w, v_w, m_b, v_b = out
    z_all_plus_1 = z_all_plus_1.at[1:, :].set(z_all)
    return z_final, iter_losses, z_all_plus_1, obj_diffs


def fp_eval_adam_logistic(i, val, supervised, z_star, X, y_label, step_size, beta1, beta2, epsilon):
    z, t, loss_vec, z_all, obj_diffs, m_w, v_w, m_b, v_b = val
    # z_next, y_next, t_next = fixed_point_lah_nesterov_logisticgd(z, y, t, X, y_label, gd_steps[i], jnp.clip(betas[i], a_min=0, a_max=100)) #beta_vals = np.clip(params[:,1], a_min=0, a_max=0.999)
    z_next, m_w_next, v_w_next, m_b_next, v_b_next = fixed_point_logistic_adam(z, X, y_label, step_size, beta1, beta2, epsilon, m_w, v_w, m_b, v_b, t)
    t_next = t + 1
    
    if supervised:
        diff = jnp.linalg.norm(z - z_star)
    else:
        diff = jnp.linalg.norm(z_next - z)
    
    loss_vec = loss_vec.at[i].set(diff)
    w, b = z[:-1], z[-1]

    obj = compute_loss(y_label, sigmoid(X @ w + b))
    w_star, b_star = z_star[:-1], z_star[-1]
    opt_obj = compute_loss(y_label, sigmoid(X @ w_star + b_star))


    obj_diffs = obj_diffs.at[i].set(obj - opt_obj)
    z_all = z_all.at[i, :].set(z_next)

    return z_next, t_next, loss_vec, z_all, obj_diffs, m_w_next, v_w_next, m_b_next, v_b_next


def fixed_point_logistic_adam(z, X, y, step_size, beta1, beta2, epsilon, m_w, v_w, m_b, v_b, t):
    w, b = z[:-1], z[-1]
    logits = jnp.clip(X @ w + b, -20, 20)
    y_hat = sigmoid(logits)
    
    dw, db = compute_gradient(X, y, y_hat)
    
    # Update moments
    m_w = beta1 * m_w + (1 - beta1) * dw
    v_w = beta2 * v_w + (1 - beta2) * (dw ** 2)
    
    m_b = beta1 * m_b + (1 - beta1) * db
    v_b = beta2 * v_b + (1 - beta2) * (db ** 2)
    
    # Bias correction
    m_w_hat = m_w / (1 - beta1 ** t)
    v_w_hat = v_w / (1 - beta2 ** t)
    
    m_b_hat = m_b / (1 - beta1 ** t)
    v_b_hat = v_b / (1 - beta2 ** t)
    
    # Update weights and bias
    w_next = w - step_size * m_w_hat / (jnp.sqrt(v_w_hat) + epsilon)
    b_next = b - step_size * m_b_hat / (jnp.sqrt(v_b_hat) + epsilon)
    
    z_next = jnp.hstack([w_next, b_next])
    return z_next, m_w, v_w, m_b, v_b



def k_steps_eval_adagrad_logistic(k, z0, q, num_points, step_size, supervised, z_star, jit):
    X_flat = q[:num_points * 784]
    X = jnp.reshape(X_flat, (num_points, 784))
    y_label = q[num_points * 784:]
    iter_losses = jnp.zeros(k)
    z_all_plus_1 = jnp.zeros((k + 1, z0.size))
    z_all_plus_1 = z_all_plus_1.at[0, :].set(z0)
    fp_eval_partial = partial(fp_eval_adagrad_logistic,
                              supervised=supervised,
                              z_star=z_star,
                              X=X,
                              y_label=y_label,
                              step_size=step_size
                              )
    z_all = jnp.zeros((k, z0.size))
    obj_diffs = jnp.zeros(k)
    # y0 = z0
    t0 = 1
    G_diag = jnp.zeros(785)
    val = z0, t0, iter_losses, z_all, obj_diffs, G_diag
    
    start_iter = 0
    if jit:
        out = lax.fori_loop(start_iter, k, fp_eval_partial, val)
    else:
        out = python_fori_loop(start_iter, k, fp_eval_partial, val)
    z_final, t_final, iter_losses, z_all, obj_diffs, G_diag = out
    z_all_plus_1 = z_all_plus_1.at[1:, :].set(z_all)
    return z_final, iter_losses, z_all_plus_1, obj_diffs


def fp_eval_adagrad_logistic(i, val, supervised, z_star, X, y_label, step_size):
    z, t, loss_vec, z_all, obj_diffs, G_diag = val
    # z_next, y_next, t_next = fixed_point_lah_nesterov_logisticgd(z, y, t, X, y_label, gd_steps[i], jnp.clip(betas[i], a_min=0, a_max=100)) #beta_vals = np.clip(params[:,1], a_min=0, a_max=0.999)
    z_next, G_diag_next = fixed_point_logistic_adagrad(z, X, y_label, step_size, G_diag)
    t_next = t + 1
    
    if supervised:
        diff = jnp.linalg.norm(z - z_star)
    else:
        diff = jnp.linalg.norm(z_next - z)
    
    loss_vec = loss_vec.at[i].set(diff)
    w, b = z[:-1], z[-1]

    obj = compute_loss(y_label, sigmoid(X @ w + b))
    w_star, b_star = z_star[:-1], z_star[-1]
    opt_obj = compute_loss(y_label, sigmoid(X @ w_star + b_star))


    obj_diffs = obj_diffs.at[i].set(obj - opt_obj)
    z_all = z_all.at[i, :].set(z_next)

    return z_next, t_next, loss_vec, z_all, obj_diffs, G_diag_next


def fixed_point_logistic_adagrad(z, X, y, step_size, G_diag):
    w, b = z[:-1], z[-1]
    logits = jnp.clip(X @ w + b, -20, 20)
    y_hat = sigmoid(logits)
    
    dw, db = compute_gradient(X, y, y_hat)
    
    G_w_diag = G_diag[:-1]
    G_b_diag = G_diag[-1]
    
    G_w_diag_next = G_w_diag + dw * dw
    G_b_diag_next = G_b_diag + db * db
    G_diag_next = jnp.hstack([G_w_diag_next, G_b_diag_next])
    
    # Update moments
    # m_w = beta1 * m_w + (1 - beta1) * dw
    # v_w = beta2 * v_w + (1 - beta2) * (dw ** 2)
    
    # m_b = beta1 * m_b + (1 - beta1) * db
    # v_b = beta2 * v_b + (1 - beta2) * (db ** 2)
    
    # # Bias correction
    # m_w_hat = m_w / (1 - beta1 ** t)
    # v_w_hat = v_w / (1 - beta2 ** t)
    
    # m_b_hat = m_b / (1 - beta1 ** t)
    # v_b_hat = v_b / (1 - beta2 ** t)
    
    # Update weights and bias
    w_next = w - step_size * dw * (G_w_diag_next + 1e-8) ** (-.5)
    b_next = b - step_size * db * (G_b_diag_next + 1e-8) ** (-.5)
    
    z_next = jnp.hstack([w_next, b_next])
    return z_next, G_diag_next


def sigmoid(x):
    return 1 / (1 + jnp.exp(-x))

def compute_loss(y, y_hat):
    eps = 1e-8
    return -jnp.mean(y * jnp.log(y_hat + eps) + (1 - y) * jnp.log(1 - y_hat + eps))

def compute_gradient(X, y, y_hat):
    error = y_hat - y
    dw = X.T @ error / X.shape[0]
    db = jnp.mean(error)
    return dw, db

def fixed_point_logistic_backtracking(z, X, y, eta_init, beta, alpha):
    w, b = z[:-1], z[-1]
    logits = jnp.clip(X @ w + b, -20, 20)
    y_hat = sigmoid(logits)
    loss = compute_loss(y, y_hat)
    dw, db = compute_gradient(X, y, y_hat)
    grad = jnp.hstack([dw, db])
    eta = eta_init

    def cond_fn(val):
        eta, _ = val
        z_new = z - eta * grad
        w_new, b_new = z_new[:-1], z_new[-1]
        logits_new = jnp.clip(X @ w_new + b_new, -20, 20)
        y_hat_new = sigmoid(logits_new)
        loss_new = compute_loss(y, y_hat_new)
        rhs = loss - alpha * eta * jnp.dot(grad, grad)
        return loss_new > rhs

    def body_fn(val):
        eta, _ = val
        return eta * beta, None

    eta_final, _ = lax.while_loop(cond_fn, body_fn, (eta, None))
    z_next = z - eta_final * grad
    return z_next, eta_final

def fp_eval_logistic_backtracking(i, val, supervised, z_star, X, y_label, eta_init, beta, alpha):
    z, t, loss_vec, z_all, obj_diffs, _ = val
    z_next, eta_next = fixed_point_logistic_backtracking(z, X, y_label, eta_init, beta, alpha)
    t_next = t + 1
    diff = jnp.linalg.norm(z - z_star) if supervised else jnp.linalg.norm(z_next - z)
    loss_vec = loss_vec.at[i].set(diff)
    z_all = z_all.at[i, :].set(z_next)

    w, b = z_next[:-1], z_next[-1]
    obj = compute_loss(y_label, sigmoid(X @ w + b))
    w_star, b_star = z_star[:-1], z_star[-1]
    opt_obj = compute_loss(y_label, sigmoid(X @ w_star + b_star))
    obj_diffs = obj_diffs.at[i].set(obj - opt_obj)

    return z_next, t_next, loss_vec, z_all, obj_diffs, eta_next

def k_steps_eval_logistic_backtracking(k, z0, q, num_points, eta0, supervised, z_star, jit, beta=0.5, alpha=0.1):
    X_flat = q[:num_points * 784]
    X = jnp.reshape(X_flat, (num_points, 784))
    y_label = q[num_points * 784:]
    iter_losses = jnp.zeros(k)
    z_all_plus_1 = jnp.zeros((k + 1, z0.size)).at[0, :].set(z0)
    z_all = jnp.zeros((k, z0.size))
    obj_diffs = jnp.zeros(k)
    t0 = 1

    val = z0, t0, iter_losses, z_all, obj_diffs, eta0

    fp_eval_partial = partial(fp_eval_logistic_backtracking,
                              supervised=supervised,
                              z_star=z_star,
                              X=X,
                              y_label=y_label,
                              eta_init=eta0,
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