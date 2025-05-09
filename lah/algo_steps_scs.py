from functools import partial

import cvxpy as cp
import jax.numpy as jnp
import jax.scipy as jsp
from jax import grad, lax, vmap
from jax.tree_util import tree_map
from lah.algo_steps import lin_sys_solve

from lah.utils.generic_utils import python_fori_loop, unvec_symm, vec_symm


def k_steps_eval_lah_accel_scs(k, z0, q, params, proj, P, A, idx_mapping, supervised, z_star, jit, hsde, zero_cone_size,
                      custom_loss=None, lightweight=False):
    """
    if k = 500 we store u_1, ..., u_500 and z_0, z_1, ..., z_500
        which is why we have all_z_plus_1
    """
    all_u, all_z = jnp.zeros((k, z0.size)), jnp.zeros((k, z0.size))
    all_z_plus_1 = jnp.zeros((k + 1, z0.size))
    all_z_plus_1 = all_z_plus_1.at[0, :].set(z0)
    all_v = jnp.zeros((k, z0.size))
    iter_losses = jnp.zeros(k)
    dist_opts = jnp.zeros(k)
    primal_residuals, dual_residuals = jnp.zeros(k), jnp.zeros(k)
    m, n = A.shape
    scalar_params, all_factors, scaled_vecs = params[0], params[1], params[2]
    alphas = jnp.exp(scalar_params[:, 2])
    tau_factors = scalar_params[:, 3] #jnp.exp(scalar_params[:, 3])
    betas = jnp.exp(scalar_params[:, 4])
    factors1, factors2 = all_factors
    verbose = not jit
    # import pdb
    # pdb.set_trace()

    # if hsde:
    #     # first step: iteration 0
    #     # we set homogeneous = False for the first iteration
    #     #   to match the SCS code which has the global variable FEASIBLE_ITERS
    #     #   which is set to 1
    #     homogeneous = False

    #     z_next, u, u_tilde, v = fixed_point_hsde_peaceman(
    #         z0, homogeneous, q, factors1[0, :, :], factors2[0, :], proj, scaled_vecs[0, :], alphas[0], tau_factors[0], verbose=verbose)
    #     all_z = all_z.at[1, :].set(z_next)
    #     all_u = all_u.at[1, :].set(u)
    #     all_v = all_v.at[1, :].set(v)
    #     iter_losses = iter_losses.at[0].set(jnp.linalg.norm(z_next - z0))
    #     dist_opts = dist_opts.at[0].set(jnp.linalg.norm((z0[:-1] - z_star)))

    #     x, y, s = extract_sol(u, v, n, False)
    #     pr = jnp.linalg.norm(A @ x + s - q[n:])
    #     dr = jnp.linalg.norm(A.T @ y + P @ x + q[:n])
    #     primal_residuals = primal_residuals.at[0].set(pr)
    #     dual_residuals = dual_residuals.at[0].set(dr)
    #     z0 = z_next

    fp_eval_partial = partial(fp_eval_lah_accel_scs, q_r=q, z_star=z_star, all_factors=all_factors,
                              proj=proj, P=P, A=A, idx_mapping=idx_mapping,
                              c=None, b=None, hsde=hsde,
                              homogeneous=False, scaled_vecs=scaled_vecs, alphas=alphas, betas=betas, tau_factors=tau_factors,
                              custom_loss=custom_loss,
                              verbose=verbose)
    val = z0, z0, iter_losses, all_z, all_u, all_v, primal_residuals, dual_residuals, dist_opts
    # start_iter = 1 if hsde else 0
    start_iter = 0
    if jit:
        out = lax.fori_loop(start_iter, k, fp_eval_partial, val)
    else:
        out = python_fori_loop(start_iter, k, fp_eval_partial, val)
    z_final, z_penult, iter_losses, all_z, all_u, all_v, primal_residuals, dual_residuals, dist_opts = out
    all_z_plus_1 = all_z_plus_1.at[1:, :].set(all_z)
    return z_final, iter_losses, all_z_plus_1, primal_residuals, dual_residuals, all_u, all_v, dist_opts


def fp_eval_lah_accel_scs(i, val, q_r, z_star, all_factors, proj, P, A, idx_mapping, c, b, hsde, homogeneous, 
                      scaled_vecs, alphas, betas, tau_factors,
            lightweight=False, custom_loss=None, verbose=False):
    """
    q_r = r if hsde else q_r = q
    homogeneous tells us if we set tau = 1.0 or use the root_plus method
    """
    m, n = A.shape
    z, z_prev, loss_vec, all_z, all_u, all_v, primal_residuals, dual_residuals, dist_opts = val

    r = q_r
    factors1, factors2 = all_factors
    idx = idx_mapping[i]
    z_next, u, u_tilde, v = fixed_point_hsde(
        z, z_prev, homogeneous, r, factors1[idx, :, :], factors2[idx, :], proj, scaled_vecs[idx, :], alphas[idx], betas[idx],
        verbose=verbose)
    # z0 = 0 * z_peaceman
    # z0 = z0.at[-1].set(1)
    # import pdb
    # pdb.set_trace()
    # z_next = z_dr + betas[idx] * (z_dr - z_prev)
    # alpha = 1 / (i+2)
    
    # z_next =  alpha * z0 + (1-alpha) * z_peaceman
    # z_next = z_peaceman
    # import pdb
    # pdb.set_trace()
    
    dist_opt = jnp.linalg.norm(z[:-1] / z[-1] - z_star)
    diff = jnp.linalg.norm(z_next / z_next[-1] - z / z[-1])

    loss_vec = loss_vec.at[i].set(diff)

    # primal and dual residuals
    if not lightweight:
        # x, y, s = extract_sol(u, v, n, hsde)
        x, y, s = extract_sol(all_u[i-1,:], all_v[i-1,:], n, hsde)
        pr = jnp.linalg.norm(A @ x + s - q_r[n:])
        dr = jnp.linalg.norm(A.T @ y + P @ x + q_r[:n])
        primal_residuals = primal_residuals.at[i].set(pr)
        dual_residuals = dual_residuals.at[i].set(dr)
        dist_opts = dist_opts.at[i].set(dist_opt)
    all_z = all_z.at[i, :].set(z_next)
    all_u = all_u.at[i, :].set(u)
    all_v = all_v.at[i, :].set(v)
    return z_next, z, loss_vec, all_z, all_u, all_v, primal_residuals, dual_residuals, dist_opts
    # return z_next, z_prev, loss_vec, all_z, all_u, all_v, primal_residuals, dual_residuals, dist_opts



def k_steps_train_lah_accel_scs(k, z0, q, params, P, A, idx_mapping, supervised, z_star, proj, jit, hsde):
    iter_losses = jnp.zeros(k)
    # scale_vec = get_scale_vec(rho_x, scale, m, n, zero_cone_size, hsde=hsde)

    scalar_params, all_factors, scaled_vecs = params[0], params[1], params[2]
    alphas = jnp.exp(scalar_params[:, 2])
    tau_factors = scalar_params[:, 3]
    betas = jnp.exp(scalar_params[:, 4])
    factors1, factors2 = all_factors

    fp_train_partial = partial(fp_train_lah_accel_scs, q_r=q, all_factors=all_factors,
                               supervised=supervised, P=P, A=A, idx_mapping=idx_mapping,
                               z_star=z_star, proj=proj, hsde=hsde,
                               homogeneous=False, scaled_vecs=scaled_vecs, alphas=alphas, betas=betas,
                               tau_factors=tau_factors)
    # if hsde:
    #     # first step: iteration 0
    #     # we set homogeneous = False for the first iteration
    #     #   to match the SCS code which has the global variable FEASIBLE_ITERS
    #     #   which is set to 1
    #     homogeneous = False
    #     z_next, u, u_tilde, v = fixed_point_hsde_peaceman(
    #         z0, homogeneous, q, factors1[0, :, :], 
    #         factors2[0, :], proj, scaled_vecs[0, :], alphas[0], tau_factors[0])
    #     iter_losses = iter_losses.at[0].set(jnp.linalg.norm(z_next - z0))
    #     z0 = z_next
    val = z0, z0, iter_losses
    # start_iter = 1 if hsde else 0
    start_iter = 0
    if jit:
        out = lax.fori_loop(start_iter, k, fp_train_partial, val)
    else:
        out = python_fori_loop(start_iter, k, fp_train_partial, val)
    z_final, z_prev_final, iter_losses = out
    return z_final, iter_losses


def fp_train_lah_accel_scs(i, val, q_r, all_factors, P, A, idx_mapping, supervised, z_star, proj, hsde, homogeneous, 
                       scaled_vecs, alphas, betas, tau_factors):
    """
    q_r = r if hsde else q_r = q
    homogeneous tells us if we set tau = 1.0 or use the root_plus method
    """
    z, z_prev, loss_vec = val
    r = q_r
    factors1, factors2 = all_factors
    idx = idx_mapping[i]

    # z_peaceman, u, u_tilde, v = fixed_point_hsde_peaceman(z, homogeneous, r, factors1[idx, :, :], 
    #                                          factors2[idx, :], proj, scaled_vecs[idx, :], alphas[idx], tau_factors[idx])
    # z_next = alphas[idx] * z_prev + betas[idx] * z + (1 - alphas[idx] - betas[idx]) * z_peaceman
    
    z_next, u, u_tilde, v = fixed_point_hsde(
        z, z_prev, homogeneous, r, factors1[idx, :, :], factors2[idx, :], proj, scaled_vecs[idx, :], alphas[idx], betas[idx])
    
    # add acceleration
    # z_next = (1 - betas[i, 0]) * z_next + betas[i, 0] * z
    m, n = A.shape

    if supervised:
        # x, y, s = extract_sol(u, v, n, hsde)
        # pr = jnp.linalg.norm(A @ x + s - q_r[n:])
        # dr = jnp.linalg.norm(A.T @ y + P @ x + q_r[:n])
        # diff = jnp.linalg.norm(z_next[:-1] / z_next[-1] - z_star)
        diff = jnp.linalg.norm(z_next[:-1] / z_next[-1] - z[:-1] / z[-1])
        # diff = jnp.linalg.norm(z_next[:-1] - z_star) # / z[-1] - z_star)
    else:
        diff = 0 #jnp.linalg.norm(z_next / z_next[-1] - z / z[-1])
    loss_vec = loss_vec.at[i].set(diff)
    return z_next, z, loss_vec
    # return z_next, z_prev, loss_vec



def fixed_point_hsde_peaceman(z_init, homogeneous, r, factor1, factor2, proj, scale_vec, alpha, tau_factor, lah=True, verbose=False):
    """
    implements 1 iteration of algorithm 5.1 in https://arxiv.org/pdf/2004.02177.pdf

    the names of the variables are a bit different compared with that paper

    we have
    u_tilde = (w_tilde, tau_tilde)
    u = (w, tau)
    z = (mu, eta)

    they have
    u_tilde = (z_tilde, tau_tilde)
    u = (z, tau)
    w = (mu, eta)

    tau_tilde, tau, eta are all scalars
    w_tilde, w, mu all have size (m + n)

    r = (I + M)^{-1} q
    requires the inital eta > 0

    if homogeneous, we normalize z s.t. ||z|| = sqrt(m + n + 1)
        and we do the root_plus calculation for tau_tilde
    else
        no normalization
        tau_tilde = 1 (bias towards feasibility)
    """
    tau_factor = 1
    # homogeneous = False
    if homogeneous:
        z_init = z_init / jnp.linalg.norm(z_init) * jnp.sqrt(z_init.size)

    # z = (mu, eta)
    mu, eta = z_init[:-1], z_init[-1]

    # u_tilde, tau_tilde update

    # non identity DR scaling
    rhs = jnp.multiply(scale_vec, mu)
    factor = (factor1, factor2)
    p = lin_sys_solve(factor, rhs)

    if lah:
        r = lin_sys_solve(factor, r)

    # non identity DR scaling
    # p = jnp.multiply(scale_vec, p)
    if homogeneous:
        tau_tilde = root_plus(mu, eta, p, r, scale_vec, tau_factor)
    else:
        tau_tilde = 1.0
    w_tilde = p - r * tau_tilde
    # check_for_nans(w_tilde)

    # u, tau update
    w_temp = 2 * w_tilde - mu
    w = proj(w_temp)
    tau = jnp.clip(2 * tau_tilde - eta, a_min=0)

    # mu, eta update
    # mu = mu + alpha * (w - w_tilde)
    # eta = eta + alpha * (tau - tau_tilde)
    mu = 2*w - w_temp #w_tilde
    eta = 2*tau - (2 * tau_tilde - eta) #tau_tilde

    # concatenate for z, u
    z = jnp.concatenate([mu, jnp.array([eta])])
    u = jnp.concatenate([w, jnp.array([tau])])
    u_tilde = jnp.concatenate([w_tilde, jnp.array([tau_tilde])])

    # for s extraction - not needed for algorithm
    full_scaled_vec = jnp.concatenate([scale_vec, jnp.array([tau_factor])])
    v = jnp.multiply(full_scaled_vec,  u + z_init - 2 * u_tilde)

    # z and u have size (m + n + 1)
    # v has shape (m + n)

    return z, u, u_tilde, v


def create_M(P, A):
    """
    create the matrix M in jax
    M = [ P   A.T
         -A   0  ]
    """
    m, n = A.shape
    M = jnp.zeros((n + m, n + m))
    M = M.at[:n, :n].set(P)
    M = M.at[:n, n:].set(A.T)
    M = M.at[n:, :n].set(-A)
    return M


def lin_sys_solve(factor, b):
    """
    solves the linear system
    Ax = b
    where factor is the lu factorization of A
    """
    return jsp.linalg.lu_solve(factor, b)
    # return jsp.sparse.linalg.cg(factor, b)


def proj(input, n, zero_cone_int, nonneg_cone_int, soc_proj_sizes, soc_num_proj, sdp_row_sizes,
         sdp_vector_sizes, sdp_num_proj):
    """
    projects the input onto a cone which is a cartesian product of the zero cone,
        non-negative orthant, many second order cones, and many positive semidefinite cones

    Assumes that the ordering is as follows
    zero, non-negative orthant, second order cone, psd cone
    ============================================================================
    SECOND ORDER CONE
    soc_proj_sizes: list of the sizes of the socp projections needed
    soc_num_proj: list of the number of socp projections needed for each size

    e.g. 50 socp projections of size 3 and 1 projection of size 100 would be
    soc_proj_sizes = [3, 100]
    soc_num_proj = [50, 1]
    ============================================================================
    PSD CONE
    sdp_proj_sizes: list of the sizes of the sdp projections needed
    sdp_vector_sizes: list of the sizes of the sdp projections needed
    soc_num_proj: list of the number of socp projections needed for each size

    e.g. 3 sdp projections of size 10, 10, and 100 would be
    sdp_proj_sizes = [10, 100]
    sdp_vector_sizes = [55, 5050]
    sdp_num_proj = [2, 1]
    """
    nonneg = jnp.clip(input[n + zero_cone_int: n + zero_cone_int + nonneg_cone_int], a_min=0)
    projection = jnp.concatenate([input[:n + zero_cone_int], nonneg])

    # soc setup
    num_soc_blocks = len(soc_proj_sizes)

    # avoiding doing inner product using jax so that we can jit
    soc_total = sum(i[0] * i[1] for i in zip(soc_proj_sizes, soc_num_proj))
    soc_bool = num_soc_blocks > 0

    # sdp setup
    num_sdp_blocks = len(sdp_row_sizes)
    sdp_total = sum(i[0] * i[1] for i in zip(sdp_vector_sizes, sdp_num_proj))
    sdp_bool = num_sdp_blocks > 0

    if soc_bool:
        socp = jnp.zeros(soc_total)
        soc_input = input[n+zero_cone_int+nonneg_cone_int:n +
                          zero_cone_int+nonneg_cone_int + soc_total]

        # iterate over the blocks
        start = 0
        for i in range(num_soc_blocks):
            # calculate the end point
            end = start + soc_proj_sizes[i] * soc_num_proj[i]

            # extract the right soc_input
            curr_soc_input = lax.dynamic_slice(
                soc_input, (start,), (soc_proj_sizes[i] * soc_num_proj[i],))

            # reshape so that we vmap all of the socp projections of the same size together
            curr_soc_input_reshaped = jnp.reshape(
                curr_soc_input, (soc_num_proj[i], soc_proj_sizes[i]))
            curr_soc_out_reshaped = soc_proj_single_batch(curr_soc_input_reshaped)
            curr_socp = jnp.ravel(curr_soc_out_reshaped)

            # place in the correct location in the socp vector
            socp = socp.at[start:end].set(curr_socp)

            # update the start point
            start = end

        projection = jnp.concatenate([projection, socp])
    if sdp_bool:
        sdp_proj = jnp.zeros(sdp_total)
        sdp_input = input[n + zero_cone_int + nonneg_cone_int + soc_total:]

        # iterate over the blocks
        start = 0
        for i in range(num_sdp_blocks):
            # calculate the end point
            end = start + sdp_vector_sizes[i] * sdp_num_proj[i]

            # extract the right sdp_input
            curr_sdp_input = lax.dynamic_slice(
                sdp_input, (start,), (sdp_vector_sizes[i] * sdp_num_proj[i],))

            # reshape so that we vmap all of the sdp projections of the same size together
            curr_sdp_input_reshaped = jnp.reshape(
                curr_sdp_input, (sdp_num_proj[i], sdp_vector_sizes[i]))
            curr_sdp_out_reshaped = sdp_proj_batch(curr_sdp_input_reshaped, sdp_row_sizes[i])
            curr_sdp = jnp.ravel(curr_sdp_out_reshaped)

            # place in the correct location in the sdp vector
            sdp_proj = sdp_proj.at[start:end].set(curr_sdp)

            # update the start point
            start = end

        projection = jnp.concatenate([projection, sdp_proj])
    return projection


def count_num_repeated_elements(vector):
    """
    given a vector, outputs the frequency in a row

    e.g. vector = [5, 5, 10, 10, 5]

    val_repeated = [5, 10, 5]
    num_repeated = [2, 2, 1]
    """
    m = jnp.r_[True, vector[:-1] != vector[1:], True]
    counts = jnp.diff(jnp.flatnonzero(m))
    unq = vector[m[:-1]]
    out = jnp.c_[unq, counts]

    val_repeated = out[:, 0].tolist()
    num_repeated = out[:, 1].tolist()
    return val_repeated, num_repeated


def soc_proj_single(input):
    """
    input is a single vector
        input = (s, y) where y is a vector and s is a scalar
    then we call soc_projection
    """
    # break into scalar and vector parts
    y, s = input[1:], input[0]

    # do the projection
    pi_y, pi_s = soc_projection(y, s)

    # stitch the pieces back together
    return jnp.append(pi_s, pi_y)


# def check_for_nans(matrix):
#     if jnp.isnan(matrix).any():
#         raise ValueError("Input matrix contains NaNs")

def check_for_nans(matrix):
    # Use lax.cond to handle the check
    has_nans = jnp.isnan(matrix).any()
    return lax.cond(has_nans, lambda _: ValueError("Input matrix contains NaNs"), lambda _: matrix, None)



def sdp_proj_single(x, n):
    """
    x_proj = argmin_y ||y - x||_2^2
                s.t.   y is psd
    x is a vector with shape (n * (n + 1) / 2)

    we need to pass in n to jit this function
        we could extract dim from x.shape theoretically,
        but we cannot jit a function
        whose values depend on the size of inputs
    """
    # print('sdp_proj_single', jax.numpy.isnan(x).max())
    # check_for_nans(x)

    # convert vector of size (n * (n + 1) / 2) to matrix of shape (n, n)
    X = unvec_symm(x, n)
    

    # do the eigendecomposition of X
    evals, evecs = jnp.linalg.eigh(X)

    # clip the negative eigenvalues
    evals_plus = jnp.clip(evals, 0, jnp.inf)

    # put the projection together with non-neg eigenvalues
    X_proj = evecs @ jnp.diag(evals_plus) @ evecs.T

    # vectorize the matrix
    x_proj = vec_symm(X_proj)
    return x_proj


def soc_projection(x, s):
    """
    returns the second order cone projection of (x, s)
    (y, t) = Pi_{K}(x, s)
    where K = {y, t | ||y||_2 <= t}

    the second order cone admits a closed form solution

    (y, t) = alpha (x, ||x||_2) if ||x|| >= |s|
             (x, s) if ||x|| <= |s|, s >= 0
             (0, 0) if ||x|| <= |s|, s <= 0

    where alpha = (s + ||x||_2) / (2 ||x||_2)

    case 1: ||x|| >= |s|
    case 2: ||x|| >= |s|
        case 2a: ||x|| >= |s|, s >= 0
        case 2b: ||x|| <= |s|, s <= 0

    """
    x_norm = jnp.linalg.norm(x)

    def case1_soc_proj(x, s):
        # case 1: y_norm >= |s|
        val = (s + x_norm) / (2 * x_norm + 1e-10)
        t = val * x_norm
        y = val * x
        return y, t

    def case2_soc_proj(x, s):
        # case 2: y_norm <= |s|
        # case 2a: s > 0
        def case2a(x, s):
            return x, s

        # case 2b: s < 0
        def case2b(x, s):
            return (0.0*jnp.zeros(x.size), 0.0)
        return lax.cond(s >= 0, case2a, case2b, x, s)
    return lax.cond(x_norm >= jnp.abs(s), case1_soc_proj, case2_soc_proj, x, s)

def root_plus(mu, eta, p, r, scale_vec, tau_factor):
    """
    mu, p, r are vectors each with size (m + n)
    eta is a scalar

    A step that solves the linear system
    (I + M)z + q tau = mu^k
    tau^2 - tau(eta^k + z^Tq) - z^T M z = 0
    where z in reals^d and tau > 0

    Since M is monotone, z^T M z >= 0
    Quadratic equation will have one non-negative root and one non-positive root

    solve by substituting z = p^k - r tau
        where r = (I + M)^{-1} q
        and p^k = (I + M)^{-1} mu^k

    the result is a closed-form solution involving the quadratic formula
        we take the positive root
    """
    r_scaled = jnp.multiply(r, scale_vec)
    a = tau_factor + r @ r_scaled
    b = mu @ r_scaled - 2 * r_scaled @ p - eta * tau_factor
    c = jnp.multiply(p, scale_vec) @ (p - mu)
    return (-b + jnp.sqrt(b ** 2 - 4 * a * c)) / (2 * a)

def extract_sol(u, v, n, hsde):
    if hsde:
        tau = u[-1] #+ 1e-10
        x, y, s = u[:n] / tau, u[n:-1] / tau, v[n:-1] / tau
    else:
        # x, y, s = u[:n], u[n:], v[n:]
        x, y, s = u[:n], u[n:-1], v[n:-1]
    return x, y, s


# provides vmapped versions of the projections for the soc and psd cones
soc_proj_single_batch = vmap(soc_proj_single, in_axes=(0), out_axes=(0))
sdp_proj_batch = vmap(sdp_proj_single, in_axes=(0, None), out_axes=(0))


def fixed_point_hsde(z_init, z_prev, homogeneous, r, factor1, factor2, proj, scale_vec, alpha, beta, lah=True, verbose=False):
    """
    implements 1 iteration of algorithm 5.1 in https://arxiv.org/pdf/2004.02177.pdf

    the names of the variables are a bit different compared with that paper

    we have
    u_tilde = (w_tilde, tau_tilde)
    u = (w, tau)
    z = (mu, eta)

    they have
    u_tilde = (z_tilde, tau_tilde)
    u = (z, tau)
    w = (mu, eta)

    tau_tilde, tau, eta are all scalars
    w_tilde, w, mu all have size (m + n)

    r = (I + M)^{-1} q
    requires the inital eta > 0

    if homogeneous, we normalize z s.t. ||z|| = sqrt(m + n + 1)
        and we do the root_plus calculation for tau_tilde
    else
        no normalization
        tau_tilde = 1 (bias towards feasibility)
    """
    tau_factor = 1
    # homogeneous = False
    if homogeneous:
        z_init = z_init / jnp.linalg.norm(z_init) * jnp.sqrt(z_init.size)

    # z = (mu, eta)
    mu, eta = z_init[:-1], z_init[-1]

    # u_tilde, tau_tilde update

    # non identity DR scaling
    rhs = jnp.multiply(scale_vec, mu)
    factor = (factor1, factor2)
    p = lin_sys_solve(factor, rhs)

    if lah:
        r = lin_sys_solve(factor, r)

    # non identity DR scaling
    # p = jnp.multiply(scale_vec, p)
    if homogeneous:
        tau_tilde = root_plus(mu, eta, p, r, scale_vec, tau_factor)
    else:
        tau_tilde = 1.0
    w_tilde = p - r * tau_tilde
    # check_for_nans(w_tilde)

    # u, tau update
    w_temp = 2 * w_tilde - mu
    w = proj(w_temp)
    tau = jnp.clip(2 * tau_tilde - eta, a_min=0)

    # mu, eta update
    # mu = 2 * w - w_temp #w_tilde
    # mu = mu + 2 * (w - w_tilde)
    mu = mu + alpha * (w - w_tilde)
    eta = 1 #eta + alpha * (tau - tau_tilde)

    # concatenate for z, u
    z = jnp.concatenate([mu, jnp.array([eta])])
    u = jnp.concatenate([w, jnp.array([tau])])
    u_tilde = jnp.concatenate([w_tilde, jnp.array([tau_tilde])])
    
    # beta = 0
    # z = z + beta * (z - z_prev)
    z = (1 - beta) * z + beta * z_prev

    # for s extraction - not needed for algorithm
    full_scaled_vec = jnp.concatenate([scale_vec, jnp.array([tau_factor])])
    v = jnp.multiply(full_scaled_vec,  u + z_init - 2 * u_tilde)

    # z and u have size (m + n + 1)
    # v has shape (m + n)
    if verbose:
        print('pre-solve u_tilde', rhs)
        print('u_tilde', u_tilde)
        print('u', u)
        print('z', z)
    # import pdb
    # pdb.set_trace()
    return z, u, u_tilde, v