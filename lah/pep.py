from functools import partial

import cvxpy as cp
import numpy as np
import jax.numpy as jnp
import jax.scipy as jsp
from jax import grad, lax, vmap
from jax.tree_util import tree_map
from PEPit import PEP
from PEPit.functions import SmoothStronglyConvexFunction, SmoothConvexFunction, ConvexFunction
from PEPit.operators import SymmetricLinearOperator, LipschitzOperator
from PEPit.primitive_steps import proximal_step
from lah.utils.generic_utils import python_fori_loop, unvec_symm, vec_symm
from cvxpylayers.jax import CvxpyLayer
import dill
import os
import hydra


def pepit_fixed_point(params):
    num_iters = params[:,0].size
    relaxation_sizes = params[:,0]
    averaged_sizes = relaxation_sizes / 2
    beta_vals = params[:,1]
    
    # Instantiate PEP
    problem = PEP()

    # Declare a non expansive operator
    A = problem.declare_function(LipschitzOperator, L=1.)

    # Start by defining its unique optimal point xs = x_*
    xs, _, _ = A.fixed_point()

    # Then define the starting point x0 of the algorithm
    x0 = problem.set_initial_point()

    # Set the initial constraint that is the difference between x0 and xs
    problem.set_initial_condition((x0 - xs) ** 2 <= 1)
    
    # Compute n steps of the accelerated proximal gradient method starting from x0
    x_new = x0
    y = x0
    for i in range(num_iters):
        alpha = averaged_sizes[i]
        beta = beta_vals[i]
        x_old = x_new
        x_new = (1 - alpha) * y + alpha * A.gradient(y)
        y = x_new + beta * (x_new - x_old)

    # Set the performance metric to the function value accuracy
    problem.set_performance_metric((y - A.gradient(y)) ** 2)
    
    # Solve the PEP
    verbose = 1
    pepit_verbose = max(verbose, 0)
    try:
        pepit_tau = problem.solve(verbose=pepit_verbose, solver=cp.MOSEK)
    except Exception as e:
        print('exception', e)
        pepit_tau = 0
    
    # Print conclusion if required
    verbose = 0
    if verbose:
        print('*** Example file:'
            ' worst-case performance of the Accelerated Proximal Gradient Method in function values***')
        print('\tPEPit guarantee:\t f(x_n)-f_* <= {:.6} ||x0 - xs||^2'.format(pepit_tau))
        # print('\tTheoretical guarantee:\t f(x_n)-f_* <= {:.6} ||x0 - xs||^2'.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method (and the reference theoretical value)
    return pepit_tau


def pepit_nesterov(mu, L, params):
    """
    params is an array of shape (num_iters, 2)
    - the first column is the vector of step sizes
    - the second column is the vector of momentum sizes
    """
    num_iters = params[:,0].size
    step_sizes = params[:,0]
    beta_vals = params[:,1]
    
    # Instantiate PEP
    problem = PEP()

    # Declare a strongly convex smooth function and a convex function
    f = problem.declare_function(SmoothStronglyConvexFunction, mu=mu, L=L)

    # Start by defining its unique optimal point xs = x_* and its function value Fs = F(x_*)
    xs = f.stationary_point()
    fs = f(xs)

    # Then define the starting point x0
    x0 = problem.set_initial_point()

    # Set the initial constraint that is the distance between x0 and x^*
    problem.set_initial_condition((x0 - xs) ** 2 <= 1)

    # Compute n steps of the accelerated proximal gradient method starting from x0
    x_new = x0
    y = x0
    for i in range(num_iters):
        alpha = step_sizes[i]
        beta = beta_vals[i]
        x_old = x_new
        x_new = y - alpha * f.gradient(y)
        y = x_new + beta * (x_new - x_old)

    # Set the performance metric to the function value accuracy
    problem.set_performance_metric((f(y)) - fs)

    # Solve the PEP
    verbose = 1
    pepit_verbose = max(verbose, 0)
    try:
        pepit_tau = problem.solve(verbose=pepit_verbose, solver=cp.MOSEK)
    except Exception as e:
        print('exception', e)
        pepit_tau = 0

    # Compute theoretical guarantee (for comparison)
    n = num_iters
    if mu == 0:
        theoretical_tau = 2 * L / (n ** 2 + 5 * n + 2)  # tight, see [2], Table 1 (column 1, line 1)
    else:
        theoretical_tau = 2 * L / (n ** 2 + 5 * n + 2)  # not tight (bound for smooth convex functions)
        # print('Warning: momentum is tuned for non-strongly convex functions.')

    # Print conclusion if required
    verbose = 0
    if verbose:
        print('*** Example file:'
            ' worst-case performance of the Accelerated Proximal Gradient Method in function values***')
        print('\tPEPit guarantee:\t f(x_n)-f_* <= {:.6} ||x0 - xs||^2'.format(pepit_tau))
        print('\tTheoretical guarantee:\t f(x_n)-f_* <= {:.6} ||x0 - xs||^2'.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method (and the reference theoretical value)
    return pepit_tau

def pepit_accel_gd(mu, L, params, quad, prox, obj):
    """
    params is an array of shape (num_iters, 2)
    - the first column is the vector of step sizes
    - the second column is the vector of momentum sizes

    quad is either True or False
    prox is either True or False
    obj is either "dist" or "func"
    """
    num_iters = params[:,0].size
    step_sizes = params[:,0]
    beta_vals = params[:,1]
    
    # Instantiate PEP
    problem = PEP()

    # Declare a strongly convex smooth function and a convex function
    
    if quad:
        f = problem.declare_function(SymmetricLinearOperator, mu = mu, L = L)
    else:
        f = problem.declare_function(SmoothStronglyConvexFunction, mu = mu, L = L)
    
    if prox:
        h = problem.declare_function(ConvexFunction)
        F = f + h
    else:
        F = f

    # Start by defining its unique optimal point xs = x_* and its function value Fs = F(x_*)
    xs = F.stationary_point()
    Fs = F(xs)

    # Then define the starting point x0
    x0 = problem.set_initial_point()

    # Set the initial constraint that is the distance between x0 and x_*
    problem.set_initial_condition((x0 - xs) ** 2 <= 1)

    # Compute n steps of the accelerated proximal gradient method starting from x0
    x_new = x0
    y = x0
    if prox:
        for i in range(num_iters):
            alpha = step_sizes[i]
            beta = beta_vals[i]
            x_old = x_new
            x_new, _, hx_new = proximal_step(y - alpha * f.gradient(y), h, alpha)
            y = x_new + beta * (x_new - x_old)
    else:
        for i in range(num_iters):
            alpha = step_sizes[i]
            beta = beta_vals[i]
            x_old = x_new
            x_new = y - alpha * f.gradient(y)
            y = x_new + beta * (x_new - x_old)

    # Set the performance metric to the function value accuracy
    if obj == "dist":
        problem.set_performance_metric((x_new - xs)**2)
    elif obj == "func":
        problem.set_performance_metric(F(x_new) - Fs)

    # Solve the PEP
    verbose = 1
    pepit_verbose = max(verbose, 0)
    try:
        pepit_tau = problem.solve(verbose=pepit_verbose, solver=cp.MOSEK)
    except Exception as e:
        print('exception', e)
        pepit_tau = 0

    # Print conclusion if required
    verbose = 1
    if verbose != -1:
        print('\tPEPit guarantee:\t f(x_n)-f_* <= {:.6} ||x0 - xs||^2'.format(pepit_tau))

    # Return the worst-case guarantee of the evaluated method
    if obj == "dist":
        return pepit_tau ** .5
    return pepit_tau


def create_nesterov_pep_sdp_layer(L, num_iters, cache=True, save=True):
    """
    creates the cvxpylayer for nesterovs method for the non-strongly convex case
    """
    k = num_iters
    A = cp.Parameter((k+2, k+3))
    
    # Define Gram matrix G and function values F
    G = cp.Variable((k + 3, k + 3), PSD=True)  # Gram of [x0, x*, g0, ..., g_{k-1}, g_k]
    F = cp.Variable(k + 2)  # f(x_0), ..., f(x_k), f(x_*)

    constraints = []

    # Interpolation constraints: for i,j in {0, ..., k, *}
    for i in range(k + 2):  # includes x_0,...,x_k, y_k, x^*
        for j in range(k + 2):
            if i != j:
                Ai = A[i]
                Aj = A[j]

                # Determine gradient indices
                def grad_idx(idx):
                    if idx <= k: # - 1:
                        return 2 + idx  # g_0 to g_{k-1}
                    else:  # x^* has zero gradient
                        return None

                gi_idx = grad_idx(i)
                gj_idx = grad_idx(j)

                # Gradient terms
                if gi_idx is None and gj_idx is None:
                    continue  # both gradients zero → vacuous

                # <g_j, x_i - x_j>
                if gj_idx is not None:
                    delta_x = Ai - Aj
                    g_j_dot_delta_x = cp.sum(cp.multiply(delta_x, G[gj_idx, :]))
                else:
                    g_j_dot_delta_x = 0.0

                # ||g_i - g_j||^2
                if gi_idx is None:
                    grad_diff_norm_sq = G[gj_idx, gj_idx]
                elif gj_idx is None:
                    grad_diff_norm_sq = G[gi_idx, gi_idx]
                else:
                    grad_diff_norm_sq = (
                        G[gi_idx, gi_idx]
                        - 2 * G[gi_idx, gj_idx]
                        + G[gj_idx, gj_idx]
                    )

                # Interpolation inequality
                constraints.append(
                    F[i] >= F[j] + g_j_dot_delta_x + (1 / (2 * L)) * grad_diff_norm_sq
                )

    # Optional: normalize ||x_0 - x^*||^2 = 1
    constraints.append(G[0,0] + G[1,1] - 2*G[0,1] == 1)

    # Objective: maximize worst-case f(x_k) - f(x^*)
    objective = cp.Maximize(F[-2] - F[-1])  # A[-1] is x^*
    
    prob = cp.Problem(objective, constraints)
    
    # check if the dill file exists in the cache
    # if it does exist, then load it
    orig_cwd = hydra.utils.get_original_cwd()
    filepath = f'{orig_cwd}/layers/smooth_nesterov/{L:.3f}_{num_iters}.dill'
    parameters = [A]
    variables = [G, F]
    return check_cache(filepath, prob, parameters, variables, cache=cache, save=save)
    

def check_cache(filepath, prob, parameters, variables, cache=True, save=True):
    if cache:
        if os.path.exists(filepath):
            # Load
            with open(filepath, 'rb') as f:
                cvxpylayer = dill.load(f)
            return cvxpylayer
    
    cvxpylayer = CvxpyLayer(prob, parameters=parameters, variables=variables)

    # Save
    if save:
        # if os.path.exists(filepath):
        orig_cwd = hydra.utils.get_original_cwd()
        layers_folder = f'{orig_cwd}/layers'
        if not os.path.exists(layers_folder):
            os.mkdir(layers_folder)
        with open(filepath, 'wb') as f:
            dill.dump(cvxpylayer, f)

    return cvxpylayer


def build_A_matrix_with_xstar(alpha_list, beta_list):
    """
    JAX-friendly version

    Returns A: jnp.ndarray of shape (k+2, k+3), where rows are:
    - A[0] to A[k]: x_0 to x_k
    - A[k+1]: x_star

    Basis: [x_0, x_star, g_0, ..., g_{k-1}, g_k]
    """
    k = alpha_list.shape[0]
    n_basis = k + 3  # [x0, x*, g_0, ..., g_{k-1}, g_k]
    A0 = jnp.zeros((k + 2, n_basis))

    idx_x0 = 0
    idx_xstar = 1
    idx_g = lambda t: 2 + t

    # Initialize A[0] = x0
    A0 = A0.at[0, idx_x0].set(1.0)

    def body(i, A):
        x_i = A[i]
        x_im1 = A[i - 1] if i > 0 else A[0]

        # Build y_i and y_{i-1}
        y_i = x_i.at[idx_g(i)].add(-alpha_list[i])
        if i == 0:
            y_im1 = x_im1 #x_im1.at[idx_g(i)].add(-alpha_list[i])
        else:
            y_im1 = x_im1.at[idx_g(i - 1)].add(-alpha_list[i - 1])

        x_ip1 = (1 + beta_list[i]) * y_i - beta_list[i] * y_im1
        return A.at[i + 1].set(x_ip1)

    A = lax.fori_loop(0, k, body, A0)

    # x_star
    A = A.at[k + 1, idx_xstar].set(1.0)

    return A

def build_A_matrix_prox_with_xstar(alpha_list, beta_list):
    """
    JAX-friendly version

    Returns A: jnp.ndarray of shape (2*k+2, 2*k+8), where rows are:
    - A[0] to A[k]: x_0 to x_k
    - A[k+1] to A[2k]: y_1 to y_k (as linear combination of x_i)
    - A[2k+1]: x_star

    Basis: [x_0, x_star, g_0, ..., g_k, g^*, s_0, ..., s_k, s^*, g_(y_k), y_k], in A the last two basis are never used
    """
    k = alpha_list.shape[0]
    n_basis = 2*k + 8  # [x_0, x_star, g_0, ..., g_k, g^*, s_0, ..., s_k, s^*, g_(y_k), y_k]
    A0 = jnp.zeros((2*k + 2, n_basis))

    idx_x0 = 0
    idx_xstar = 1
    idx_g = lambda t: 2 + t 
    idx_s = lambda t: k + 4 + t 

    # Initialize A[0] = x0
    A0 = A0.at[0, idx_x0].set(1.0)

    def body(i, A):
        x_i = A[i]

        # Build y_i and y_{i-1}
        y_ip1 = x_i.at[idx_g(i)].add(-alpha_list[i]).at[idx_s(i+1)].add(-alpha_list[i])
        if i == 0:
            y_i = x_i
        else:
            y_i = A[i+k]

        x_ip1 = (1 + beta_list[i]) * y_ip1 - beta_list[i] * y_i
        return A.at[i + 1].set(x_ip1).at[i+k+1].set(y_ip1)

    A = lax.fori_loop(0, k, body, A0)

    # x_star
    A = A.at[2*k + 1, idx_xstar].set(1.0)

    return A

def build_A_matrix_op_with_xstar(alpha_list, beta_list):
    """
    JAX-friendly version

    Returns A: jnp.ndarray of shape (k+2, k+3), where rows are:
    - A[0] to A[k]: x_0 to x_k
    - A[k+1]: x_star

    Basis: [x_0, x_star, q_0, ..., q_{k-1}, q_k]
    """
    k = alpha_list.shape[0]
    n_basis = k + 3  # [x0, x*, q_0, ..., q_{k-1}, q_k]
    A0 = jnp.zeros((k + 2, n_basis))

    idx_x0 = 0
    idx_xstar = 1
    idx_q = lambda t: 2 + t

    # Initialize A[0] = x0
    A0 = A0.at[0, idx_x0].set(1.0)

    def body(i, A):
        x_i = A[i]
        x_im1 = A[i - 1] if i > 0 else A[0]

        # Build y_i and y_{i-1}
        y_i = x_i - 0.5 * alpha_list[i] * (x_i.at[idx_q(i)].add(-1))
        if i == 0:
            y_im1 = x_im1 #x_im1.at[idx_g(i)].add(-alpha_list[i])
        else:
            y_im1 = x_im1 - 0.5 * alpha_list[i-1] * (x_im1.at[idx_q(i-1)].add(-1))

        x_ip1 = (1 + beta_list[i]) * y_i - beta_list[i] * y_im1
        return A.at[i + 1].set(x_ip1)

    A = lax.fori_loop(0, k, body, A0)

    # x_star
    A = A.at[k + 1, idx_xstar].set(1.0)

    return A


def create_nesterov_strcvx_pep_sdp_layer(mu, L, num_iters):

    """
    creates the cvxpylayer for nesterovs method for the strongly convex case
    """
    k = num_iters
    A = cp.Parameter((k+2, k+3))

    # Define Gram matrix G and function values F
    G = cp.Variable((2*k + 3, 2*k + 3), PSD=True)  # Gram of [x_0, x^*, g_0, ..., g_k, x_1, ..., x_k]
    F = cp.Variable(k + 2)  # f(x_0), ..., f(x_k), f(x^*)

    constraints = []

    # Determine gradient indices
    def grad_idx(idx):
        if idx <= k: 
            return 2 + idx  # g_0 to g_k
        else:  # x^* has zero gradient
            return None

    # Determine iterate indices
    def iter_idx(idx):
        if idx >= 1 and idx <= k:
            return k + 2 + idx
        elif idx == 0:
            return 0
        else:
            return 1

    # Interpolation constraints: for i,j in {0, ..., k, *}
    for i in range(k + 2):  # includes x_0,...,x_k, x^*
        for j in range(k + 2):
            if i != j:
                # Indices
                gi_idx = grad_idx(i)
                gj_idx = grad_idx(j)

                xi_idx = iter_idx(i)
                xj_idx = iter_idx(j)

                # Gradient terms
                if gi_idx is None and gj_idx is None:
                    continue  # both gradients zero → vacuous

                # <g_j, x_i - x_j>
                if gj_idx is not None:
                    gj_dot_delta_x = G[gj_idx, xi_idx] - G[gj_idx, xj_idx]
                else:
                    gj_dot_delta_x = 0.0

                # ||g_i - g_j||^2
                if gi_idx is None:
                    grad_diff_norm_sq = G[gj_idx, gj_idx]
                elif gj_idx is None:
                    grad_diff_norm_sq = G[gi_idx, gi_idx]
                else:
                    grad_diff_norm_sq = (
                        G[gi_idx, gi_idx]
                        - 2 * G[gi_idx, gj_idx]
                        + G[gj_idx, gj_idx]
                    )

                # ||x_i - x_j - (1/L)(g_i - g_j)||^2
                xi_minus_xj_square = G[xi_idx, xi_idx] - 2 * G[xi_idx, xj_idx] + G[xj_idx, xj_idx]
                
                if gi_idx is None:
                    gd_diff_norm_sq = xi_minus_xj_square + (2/L) * gj_dot_delta_x + (1/L)**2 * grad_diff_norm_sq
                elif gj_idx is None:
                    gd_diff_norm_sq = xi_minus_xj_square - (2/L) * (G[gi_idx, xi_idx] - G[gi_idx, xj_idx]) + (1/L)**2 * grad_diff_norm_sq
                else:
                    gd_diff_norm_sq = xi_minus_xj_square - (2/L) * (G[gi_idx, xi_idx] - G[gi_idx, xj_idx] - gj_dot_delta_x) + (1/L)**2 * grad_diff_norm_sq

                # Interpolation inequality
                constraints.append(
                    F[i] >= F[j] + gj_dot_delta_x + (1 / (2 * L)) * grad_diff_norm_sq + mu / 2 / (1 - mu/L) * gd_diff_norm_sq
                )
    
    # Equality constraint: express x_1, ... x_k as a function of x_0, g_0, ... g_k using A
    # it seems that the last row of A is not required?
    for i in range(1, k+1):
        Ai = A[i].T
        current_idx = iter_idx(i)
        constraints.append(G[:, range(k+3)] @ Ai == G[:, current_idx])

    # Optional: normalize ||x_0 - x^*||^2 = 1
    constraints.append(G[0,0] + G[1,1] - 2*G[0,1] == 1)

    # Objective: maximize worst-case f(x_k) - f(x^*)
    objective = cp.Maximize(F[-2] - F[-1])  # A[-1] is x^*
    prob = cp.Problem(objective, constraints)
    
    cvxpylayer = CvxpyLayer(prob, parameters=[A], variables=[G, F])
    
    return cvxpylayer

def create_quad_pep_sdp_layer(mu, L, num_iters):
    """
    create cvxpylayer for quadratic min: min_z (1/2) z^T Q z 
    result will be in terms of distance to optimality
    """

    k = num_iters
    A = cp.Parameter((k+2, k+3))

    # Define Gram matrix G
    G = cp.Variable((2*k + 3, 2*k + 3), PSD=True)  # Gram of [x_0, x_*, g_0, g_1, ..., g_k, x_1, ..., x_k]

    constraints = []

    iter_idx_vec = [0] + list(range(k+3, 2*k+3))  # index of x_0, x_1, ..., x_k in Gram matrix G 
    iter_idx_with_star_vec = iter_idx_vec + [1]  # index of x_0, x_1, ..., x_k, x_* in Gram matrix G 
    grad_idx_vec = range(2, k+3)  # index of g_0, g_1, ..., g_k in Gram matrix G

    # Semidefinite constraints
    left_mult_mat = np.zeros((2*k+3, k+1))
    right_mult_mat = np.zeros((2*k+3, k+1))
    for i in range(k+1):
        left_mult_mat[iter_idx_vec[i], i] = -mu
        left_mult_mat[grad_idx_vec[i], i] = 1
        right_mult_mat[iter_idx_vec[i], i] = L
        right_mult_mat[grad_idx_vec[i], i] = -1

    constraints.append(left_mult_mat.T @ G @ right_mult_mat >> 0)

    # Symmetry constraint
    for i in range(k+1):
        for j in range(k+1):
            constraints.append(G[iter_idx_vec[i], grad_idx_vec[j]] == G[iter_idx_vec[j], grad_idx_vec[i]])

    # Equality constraint: two equivalent definition for x_0, x_1, ..., x_k, x_* (the conditions with respect to x_0 and x_* are redundant)
    constraints.append(G[:, range(k+3)] @ A.T == G[:, iter_idx_with_star_vec])

    # Equality constraint: x_* = 0
    constraints.append(G[:, 1] == 0) 

    # Optional: normalize ||x_0 - x_*||^2 = 1
    constraints.append(G[0,0] - 2*G[0, 1] + G[1, 1] == 1)

    # Objective: maximize worst-case ||x_k||^2
    objective = cp.Maximize(G[2*k+2, 2*k+2])

    # If objective is function value then 
    # objective = cp.Maximize(0.5 * G[k+2, 2*k+2])  
    prob = cp.Problem(objective, constraints)
    
    cvxpylayer = CvxpyLayer(prob, parameters=[A], variables=[G])
    
    return cvxpylayer


def create_quadprox_pep_sdp_layer(mu, L, num_iters):
    """
    create cvxpylayer for quadratic + proximal min: min_z (1/2) z^T Q z + h(z) 
    result will be in terms of distance to optimality
    """

    k = num_iters
    A = cp.Parameter((2*k+2, 2*k+8))

    # Define Gram matrix G
    G = cp.Variable((3*k + 8, 3*k + 8), PSD=True)  # Gram of [x_0, x^*, g_0, ..., g_k, g^*, s_0, ..., s_k, s^*, x_1, ..., x_k, g_(y_k), y_k]
    H = cp.Variable(k + 2)  # h(y_0), ..., h(y_k), h(y^*)

    constraints = []

    iter_idx_vec = [0] + list(range(2*k+6, 3*k+6)) + [3*k + 7]  # index of x_0, x_1, ..., x_k, y_k in Gram matrix G 
    iter_idx_with_star_vec = iter_idx_vec + [1]  # index of x_0, x_1, ..., x_k, y_k, x_* in Gram matrix G 
    grad_idx_with_star_vec = list(range(2, k+3)) + [3*k + 6, k+3] # index of g_0, g_1, ..., g_k, g_(y_k), g_* in Gram matrix G
    subgrad_idx_with_star_vec = range(k+5, 2*k+6) # index of s_1, ..., s_k, s_* in Gram matrix G

    y_idx_in_A = range(k+1, 2*k+2)  # index of y_1, ..., y_k, x_* in A 

    # Semidefinite constraints
    left_mult_mat = np.zeros((3*k+8, k+3))
    right_mult_mat = np.zeros((3*k+8, k+3))
    for i in range(k+3):
        left_mult_mat[iter_idx_with_star_vec[i], i] = -mu
        left_mult_mat[grad_idx_with_star_vec[i], i] = 1
        right_mult_mat[iter_idx_with_star_vec[i], i] = L
        right_mult_mat[grad_idx_with_star_vec[i], i] = -1

    constraints.append(left_mult_mat.T @ G @ right_mult_mat >> 0)

    # Symmetry constraints
    for i in range(k+3):
        for j in range(k+3):
            constraints.append(G[iter_idx_with_star_vec[i], grad_idx_with_star_vec[j]] == G[iter_idx_with_star_vec[j], grad_idx_with_star_vec[i]])

    # Interpolation constraints
    for i in range(k+1):
        for j in range(k+1):
            # Indices for iterates y
            A_yi = A[y_idx_in_A[i]]
            A_yj = A[y_idx_in_A[j]]

            # Indices for subgradients
            sj_idx = subgrad_idx_with_star_vec[j]

            # <s_j, y_i - y_j>
            delta_y = A_yi - A_yj
            sj_dot_delta_y = cp.sum(cp.multiply(delta_y, G[sj_idx, range(2*k+8)]))

            # Interpolation inequality for H
            constraints.append(
                H[i] >= H[j] + sj_dot_delta_y
            )

    # Equality constraint: two equivalent definition for x_0, x_1, ..., x_k, y_k (the conditions with respect to x_0 is redundant)
    iter_idx_in_A = list(range(k+1)) + [2*k]
    constraints.append(G[:, range(2*k+8)] @ A[iter_idx_in_A, :].T == G[:, iter_idx_vec])

    # Equality constraint: g_* + s_* = 0
    constraints.append(G[:, k + 3] + G[:, 2*k + 5] == 0) 

    # Optional: normalize ||x_0 - x_*||^2 = 1
    constraints.append(G[0,0] - 2*G[0, 1] + G[1, 1] == 1)

    # Objective: maximize worst-case ||y_k - x_*||^2
    objective = cp.Maximize(G[3*k + 7, 3*k + 7] - 2*G[3*k + 7, 1] + G[1, 1])

    prob = cp.Problem(objective, constraints)
    
    cvxpylayer = CvxpyLayer(prob, parameters=[A], variables=[G, H])
    
    return cvxpylayer
    






def create_proxgd_pep_sdp_layer(L, num_iters, cache=True, save=True):
    """
    create cvxpylayer for quadratic min: min_z f(z) + h(z)
    where f is convex and L smooth
    h is convex
    bound in terms of function values
    """

    k = num_iters
    
    A = cp.Parameter((2*k + 2, 2*k + 8))

    # Define Gram matrix G and function values F, H
    G = cp.Variable((2*k + 8, 2*k + 8), PSD=True)  # Gram of [x_0, x^*, g_0, ..., g_k, g^*, s_0, ..., s_k, s_*, g_(y_k), y_k]
    F = cp.Variable(k + 3)  # f(x_0), ..., f(x_k), f(y_k), f(x^*)
    H = cp.Variable(k + 2)  # h(y_0), ..., h(y_k), h(y^*)

    constraints = []

    # Determine gradient indices: g_(x_0), ..., g_(x_k), g_(y_k), g_*
    def grad_idx(idx):
        if idx != k+1:
            return 2 + idx  # g_0 to g^*
        else:
            return 2*k+6

    # Determine x indices in A [x_0, ..., x_k, y_1, ... y_k, x_*]^T
    def x_idx(idx):
        if idx == k+1:
            return 2*k
        elif idx == k+2:
            return 2*k+1
        else:
            return idx
    
    # Determine subgradient indices: s_(y_0), ..., s_(y_k), s_*
    def subgrad_idx(idx):
        return k + 4 + idx  # s_0 to s^*

    # Determine y indices in A [x_0, ..., x_k, y_1, ... y_k, x_*]^T
    def y_idx(idx):
        if idx == 0:
            return 0
        elif idx == k+1:
            return 2*k+1
        else:
            return idx + k


    # Interpolation constraints: for i, j in {0, ..., k, *}
    for i in range(k + 3):  
        for j in range(k + 3):
            if i != j:
                # Indices for iterates x and y_k
                A_xi = A[x_idx(i)]
                A_xj = A[x_idx(j)]

                # Indices for gradients
                gi_idx = grad_idx(i)
                gj_idx = grad_idx(j)

                # <g_j, x_i - x_j>
                delta_x = A_xi - A_xj
                gj_dot_delta_x = cp.sum(cp.multiply(delta_x, G[gj_idx, :]))

                # ||g_i - g_j||^2
                grad_diff_norm_sq = (
                    G[gi_idx, gi_idx]
                    - 2 * G[gi_idx, gj_idx]
                    + G[gj_idx, gj_idx]
                )

                # Interpolation inequality for F
                constraints.append(
                    F[i] >= F[j] + gj_dot_delta_x + (1 / (2 * L)) * grad_diff_norm_sq
                )
    
    for i in range(k+2):
        for j in range(k+2):
                # Indices for iterates y
                A_yi = A[y_idx(i)]
                A_yj = A[y_idx(j)]

                # Indices for subgradients
                sj_idx = subgrad_idx(j)

                # <s_j, y_i - y_j>
                delta_y = A_yi - A_yj
                sj_dot_delta_y = cp.sum(cp.multiply(delta_y, G[sj_idx, :]))

                # Interpolation inequality for H
                constraints.append(
                    H[i] >= H[j] + sj_dot_delta_y
                )

    # Equality constraint: g^* + s^* = 0
    opt_grad_idx = grad_idx(k+1)
    opt_subgrad_idx = subgrad_idx(k+1)
    constraints.append(G[:, opt_grad_idx] == -G[:, opt_subgrad_idx])

    # Equality constraint: y_k
    constraints.append(G @ A[2*k, :].T == G[:, 2*k+7])

    # Optional: normalize ||x_0 - x^*||^2 = 1
    constraints.append(G[0,0] + G[1,1] - 2*G[0,1] == 1)

    # Objective: maximize worst-case f(y_k) + h(y_k) - f(x^*) - h(x^*)
    objective = cp.Maximize(F[-2] + H[-2] - F[-1] - H[-1])
    prob = cp.Problem(objective, constraints)
    
    
    orig_cwd = hydra.utils.get_original_cwd()
    filepath = f'{orig_cwd}/layers/proxgd_nesterov/{L:.3f}_{num_iters}.dill'
    parameters = [A]
    variables = [G, F, H]
    return check_cache(filepath, prob, parameters, variables, cache=cache, save=save)





def create_admm_pep_sdp_layer(num_iters):
    """
    performance metric: ||z - S(z)||^2 where S is 1-Lipschitz
    """

    k = num_iters
    A = cp.Parameter((k+2, k+3))

    # Define Gram matrix G and function values F
    G = cp.Variable((2*k + 3, 2*k + 3), PSD=True)  # Gram of [x_0, x^*, q_0, ..., q_k, x_1, ..., x_k]
    F = cp.Variable(k + 2)  # f(x_0), ..., f(x_k), f(x^*)

    constraints = []

    # Determine gradient indices
    def op_idx(idx):
        if idx <= k: 
            return 2 + idx  # q_0 to q_k
        else:  # q^* = x^*
            return 1

    # Determine iterate indices
    def iter_idx(idx):
        if idx >= 1 and idx <= k:
            return k + 2 + idx
        elif idx == 0:
            return 0
        else:
            return 1

    # Interpolation constraints: for i,j in {0, ..., k, *}
    for i in range(k + 2):  # includes x_0,...,x_k, x^*
        for j in range(k + 2):
            if i != j:
                # Indices
                qi_idx = op_idx(i)
                qj_idx = op_idx(j)

                xi_idx = iter_idx(i)
                xj_idx = iter_idx(j)

                # ||g_i - g_j||^2
                op_diff_norm_sq = (
                    G[qi_idx, qi_idx]
                    - 2 * G[qi_idx, qj_idx]
                    + G[qj_idx, qj_idx]
                )

                # ||x_i - x_j - (1/L)(g_i - g_j)||^2
                xi_minus_xj_sq = G[xi_idx, xi_idx] - 2 * G[xi_idx, xj_idx] + G[xj_idx, xj_idx]

                # Interpolation inequality
                constraints.append(
                    op_diff_norm_sq <= xi_minus_xj_sq
                )
    
    # Equality constraint: express x_1, ... x_k as a function of x_0, g_0, ... g_k using A
    # it seems that the last row of A is not required?
    for i in range(1, k+1):
        Ai = A[i].T
        current_idx = iter_idx(i)
        constraints.append(G[:, range(k+3)] @ Ai == G[:, current_idx])

    # Optional: normalize ||x_0 - x^*||^2 = 1
    constraints.append(G[0,0] + G[1,1] - 2*G[0,1] == 1)

    # Objective: maximize worst-case ||x_k - q_k||^2
    objective = cp.Maximize(G[2*k+2, 2*k+2] - 2*G[2*k+2, k+2] + G[k+2, k+2])
    prob = cp.Problem(objective, constraints)
    
    cvxpylayer = CvxpyLayer(prob, parameters=[A], variables=[G])
    
    return cvxpylayer






def create_proxgd_strcvx_pep_sdp_layer(mu, L, num_iters):
    """
    create cvxpylayer for quadratic min: min_z f(z) + g(z)
    where f is mu strongly convex and L smooth (mu = 0 is possible)
    g is convex
    bound in terms of function values

    =======================================================
    Currently only works for GD (i.e., no momentum)
    =======================================================

    """

    k = num_iters
    
    a = cp.Parameter(k)

    # Define Gram matrix G and function values F, H
    G = cp.Variable((3*k + 6, 3*k + 6), PSD=True)  # Gram of [x_0, x^*, g_0, ..., g_k, g^*, s_0, ..., s_k, s^*, x_1, ..., x^k]
    F = cp.Variable(k + 2)  # f(x_0), ..., f(x_k), f(x^*)
    H = cp.Variable(k + 2)  # h(x_0), ..., h(x_k), h(x^*)

    constraints = []

    # Determine gradient indices
    def grad_idx(idx):
        return 2 + idx  # g_0 to g^*

    # Determine subgradient indices
    def subgrad_idx(idx):
        return k + 4 + idx  # s_0 to s^*

    # Determine iterate indices
    def iter_idx(idx):
        if idx >= 1 and idx <= k:
            return 2*k + 5 + idx # x_1, ..., x_k
        elif idx == 0:
            return 0 # x_0
        else:
            return 1 # x_*

    # Interpolation constraints: for i, j in {0, ..., k, *}
    for i in range(k + 2):  # includes x_0, ..., x_k, x^*
        for j in range(k + 2):
            if i != j:
                # Indices for iterates
                xi_idx = iter_idx(i)
                xj_idx = iter_idx(j)

                # Indices for gradients
                gi_idx = grad_idx(i)
                gj_idx = grad_idx(j)

                # <g_j, x_i - x_j>
                gj_dot_delta_x = G[gj_idx, xi_idx] - G[gj_idx, xj_idx]

                # ||g_i - g_j||^2
                grad_diff_norm_sq = (
                    G[gi_idx, gi_idx]
                    - 2 * G[gi_idx, gj_idx]
                    + G[gj_idx, gj_idx]
                )

                # ||x_i - x_j - (1/L)(g_i - g_j)||^2
                xi_minus_xj_square = G[xi_idx, xi_idx] - 2 * G[xi_idx, xj_idx] + G[xj_idx, xj_idx]
                gd_diff_norm_sq = xi_minus_xj_square - (2/L) * (G[gi_idx, xi_idx] - G[gi_idx, xj_idx] - G[gj_idx, xi_idx] + G[gj_idx, xj_idx]) + (1/L)**2 * grad_diff_norm_sq

                # Interpolation inequality for F
                constraints.append(
                    F[i] >= F[j] + gj_dot_delta_x + (1 / (2 * L)) * grad_diff_norm_sq + mu / 2 / (1 - mu/L) * gd_diff_norm_sq
                )

                # Indices for subgradients
                sj_idx = subgrad_idx(j)

                # <h_j, x_i - x_j>
                sj_dot_delta_x = G[sj_idx, xi_idx] - G[sj_idx, xj_idx]

                constraints.append(
                    H[i] >= H[j] + sj_dot_delta_x
                )

    # Equality constraint: express x_1, ... x_k as a function of x_0, g_0, ..., g_k, s_0, ..., s_k
    # x_i = x_{i-1} - alpha_{i-1}(g_{i-1} + s_i)
    for i in range(1, k+1):
        current_iter_idx = iter_idx(i)
        prev_iter_idx = iter_idx(i-1)
        prev_grad_idx = grad_idx(i-1)
        current_subgrad_idx = subgrad_idx(i)
        stepsize = a[i-1]
        constraints.append(G[:, current_iter_idx] == G[:, prev_iter_idx] - stepsize * (G[:, prev_grad_idx] + G[:, current_subgrad_idx]))

    # Equality constraint: g^* + s^* = 0
    opt_grad_idx = grad_idx(k+1)
    opt_subgrad_idx = subgrad_idx(k+1)
    constraints.append(G[:, opt_grad_idx] == -G[:, opt_subgrad_idx])

    # Optional: normalize ||x_0 - x^*||^2 = 1
    constraints.append(G[0,0] + G[1,1] - 2*G[0,1] == 1)

    # Objective: maximize worst-case f(x_k) + h(x_k) - f(x^*) - h(x^*)
    objective = cp.Maximize(F[-2] + H[-2] - F[-1] - H[-1])
    prob = cp.Problem(objective, constraints)
    
    cvxpylayer = CvxpyLayer(prob, parameters=[a], variables=[G, F, H])
    
    return cvxpylayer