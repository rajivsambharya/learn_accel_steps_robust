from functools import partial

import cvxpy as cp
import jax.numpy as jnp
import jax.scipy as jsp
from jax import grad, lax, vmap
from jax.tree_util import tree_map
from PEPit import PEP
from PEPit.functions import SmoothStronglyConvexFunction
from PEPit.operators import SymmetricLinearOperator
from PEPit.primitive_steps import proximal_step
from lah.utils.generic_utils import python_fori_loop, unvec_symm, vec_symm
from cvxpylayers.jax import CvxpyLayer


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
    verbose = 1
    if verbose != -1:
        print('*** Example file:'
            ' worst-case performance of the Accelerated Proximal Gradient Method in function values***')
        print('\tPEPit guarantee:\t f(x_n)-f_* <= {:.6} ||x0 - xs||^2'.format(pepit_tau))
        print('\tTheoretical guarantee:\t f(x_n)-f_* <= {:.6} ||x0 - xs||^2'.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method ( and the reference theoretical value)
    return pepit_tau, theoretical_tau



def pepit_quad_accel_gd(mu, L, params):
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
    f = problem.declare_function(SymmetricLinearOperator, mu=mu, L=L)

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
    # problem.set_performance_metric((f(y)) - fs)
    problem.set_performance_metric((y - xs)**2)

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
    verbose = 1
    if verbose != -1:
        print('*** Example file:'
            ' worst-case performance of the Accelerated Proximal Gradient Method in function values***')
        print('\tPEPit guarantee:\t f(x_n)-f_* <= {:.6} ||x0 - xs||^2'.format(pepit_tau))
        print('\tTheoretical guarantee:\t f(x_n)-f_* <= {:.6} ||x0 - xs||^2'.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method ( and the reference theoretical value)
    return pepit_tau, theoretical_tau



def pepit_quadprox_accel_gd(mu, L, params):
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
    f = problem.declare_function(SymmetricLinearOperator, mu=mu, L=L)

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
        x_new = proximal_step(y - alpha * f.gradient(y), f2, alpha)
        y = x_new + beta * (x_new - x_old)

    # Set the performance metric to the function value accuracy
    problem.set_performance_metric((f(y)) - fs)
    # problem.set_performance_metric((y - xs)**2)

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
    verbose = 1
    if verbose != -1:
        print('*** Example file:'
            ' worst-case performance of the Accelerated Proximal Gradient Method in function values***')
        print('\tPEPit guarantee:\t f(x_n)-f_* <= {:.6} ||x0 - xs||^2'.format(pepit_tau))
        print('\tTheoretical guarantee:\t f(x_n)-f_* <= {:.6} ||x0 - xs||^2'.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method ( and the reference theoretical value)
    return pepit_tau, theoretical_tau


def create_nesterov_pep_sdp_layer(L, num_iters):
    """
    ordering: x*, x_0,...,x_{p-1}, s*,s_0,..,s_{p-1}
    """
    k = num_iters
    A = cp.Parameter((k+3, k+3))
    
    # Define Gram matrix G and function values F
    G = cp.Variable((k + 3, k + 3), PSD=True)  # Gram of [x0, x*, g0, ..., g_{k-1}, g_f]
    F = cp.Variable(k + 3)  # f(x_0), ..., f(x_k), f(y_k), f(x_*)

    constraints = []

    # Interpolation constraints: for i,j in {0, ..., k}, plus g_f
    for i in range(k + 3):  # includes x_0,...,x_k, y_k, x^*
        for j in range(k + 3):
            if i != j:
                Ai = A[i]
                Aj = A[j]

                # Determine gradient indices
                def grad_idx(idx):
                    if idx <= k: # - 1:
                        return 2 + idx         # g_0 to g_{k-1}
                    elif idx == k + 1: #idx == k or idx == k + 1:
                        return 2 + k           # g_f (for x_k and y_k)
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
    objective = cp.Maximize(F[-3] - F[-1])  # A[-1] is x^*
    prob = cp.Problem(objective, constraints)
    
    cvxpylayer = CvxpyLayer(prob, parameters=[A], variables=[G, F])
    
    return cvxpylayer


def build_A_matrix_with_yk_and_xstar(alpha_list, beta_list):
    """
    JAX-friendly version of build_A_matrix_with_yk_and_xstar.

    Returns A: jnp.ndarray of shape (k+3, k+3), where rows are:
    - A[0] to A[k]: x_0 to x_k
    - A[k+1]: y_k
    - A[k+2]: x_star

    Basis: [x_0, x_star, g_0, ..., g_{k-1}, g_f]
    """
    k = alpha_list.shape[0]
    n_basis = k + 3  # [x0, x*, g_0, ..., g_{k-1}, g_f]
    A0 = jnp.zeros((k + 3, n_basis))

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
            y_im1 = x_im1.at[idx_g(i)].add(-alpha_list[i])
        else:
            y_im1 = x_im1.at[idx_g(i - 1)].add(-alpha_list[i - 1])

        x_ip1 = (1 + beta_list[i]) * y_i - beta_list[i] * y_im1
        return A.at[i + 1].set(x_ip1)

    A = lax.fori_loop(0, k, body, A0)

    # y_k = (1 + beta_k) * x_k - beta_k * x_{k-1}
    x_k = A[k]
    x_km1 = A[k - 1] if k >= 1 else A[0]
    beta_k = beta_list[k - 1]
    y_k = (1 + beta_k) * x_k - beta_k * x_km1
    A = A.at[k + 1].set(y_k)

    # x_star
    A = A.at[k + 2, idx_xstar].set(1.0)

    return A
