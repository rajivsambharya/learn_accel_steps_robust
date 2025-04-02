from functools import partial

import jax.numpy as jnp
from jax import random, vmap

import numpy as np

from lah.algo_steps_logistic import k_steps_eval_lah_nesterov_gd, k_steps_train_lah_nesterov_gd, k_steps_eval_nesterov_logisticgd, compute_gradient
from lah.l2o_model import L2Omodel
from PEPit import PEP
from PEPit.functions import SmoothStronglyConvexFunction
import cvxpy as cp
from cvxpylayers.jax import CvxpyLayer
from jax import lax


class LAHAccelLOGISTICGDmodel(L2Omodel):
    def __init__(self, **kwargs):
        super(LAHAccelLOGISTICGDmodel, self).__init__(**kwargs)

    def initialize_algo(self, input_dict):
        self.factor_static = None
        self.algo = 'lah_accel_logisticgd'
        self.factors_required = False
        num_points = input_dict['num_points']
        num_weights = 785
        self.q_mat_train, self.q_mat_test = input_dict['q_mat_train'], input_dict['q_mat_test']
        
        self.n = num_weights
        self.output_size = self.n


        # get the smoothness parameter
        first_prob_data = self.q_mat_train[0, :]
        X0_flat, y0 = first_prob_data[:-num_points], first_prob_data[-num_points:]
        X0 = jnp.reshape(X0_flat, (num_points, num_weights - 1))

        covariance_matrix = np.dot(X0.T, X0) / num_points
    
        # Compute the maximum eigenvalue of the covariance matrix
        evals, evecs = jnp.linalg.eigh(covariance_matrix)

        self.smooth_param = jnp.max(evals) / 4

        self.k_steps_train_fn = partial(k_steps_train_lah_nesterov_gd, num_points=num_points,
                                        jit=self.jit)
        self.k_steps_eval_fn = partial(k_steps_eval_lah_nesterov_gd, num_points=num_points, 
                                       jit=self.jit)
        self.nesterov_eval_fn = partial(k_steps_eval_nesterov_logisticgd, num_points=num_points,
                                       jit=self.jit)

        self.out_axes_length = 5

        N = self.q_mat_train.shape[0]
        self.lah_train_inputs = jnp.zeros((N, num_weights))

        e2e_loss_fn = self.create_end2end_loss_fn
        
        self.num_pep_iters = 20

        # end-to-end loss fn for silver evaluation
        self.loss_fn_eval_silver = e2e_loss_fn(bypass_nn=False, diff_required=False, 
                                               special_algo='silver')
        
        # end-to-end loss fn for silver evaluation
        self.loss_fn_eval_conj_grad = e2e_loss_fn(bypass_nn=False, diff_required=False, 
                                               special_algo='conj_grad')

        self.num_const_steps = input_dict.get('num_const_steps', 1)

        self.num_points = num_points
        
        self.pep_layer = self.create_nesterov_pep_sdp_layer(self.num_pep_iters)


    def compute_single_gradient(self, z, q):
        num_points = self.num_points
        X_flat = q[:num_points * 784]
        X = jnp.reshape(X_flat, (num_points, 784))
        y = q[num_points * 784:]

        w, b = z[:-1], z[-1]
        y_hat = sigmoid(X @ w + b)
        dw, db = compute_gradient(X, y, y_hat)
        return jnp.hstack([dw, db])


    def compute_gradients(self, batch_inputs, batch_q_data):
        # Use vmap to vectorize compute_single_gradient over the batch dimensions
        batched_compute_gradient = vmap(self.compute_single_gradient, in_axes=(0, 0))
        return batched_compute_gradient(batch_inputs, batch_q_data)
    

    def transform_params(self, params, n_iters):
        # n_iters = params[0].size
        transformed_params = jnp.zeros((n_iters, 2))
        transformed_params = jnp.clip(transformed_params.at[:n_iters - 1, :].set(jnp.exp(params[0][:n_iters - 1, :])), a_max=50000)
        transformed_params = transformed_params.at[n_iters - 1, :].set(2 / self.smooth_param * sigmoid(params[0][n_iters - 1, :]))
        # transformed_params = transformed_params.at[n_iters - 1, 0].set(jnp.exp(params[0][n_iters - 1, 0]))
        return transformed_params

    def perturb_params(self):
        # init step-varying params
        noise = jnp.array(np.clip(np.random.normal(size=(self.step_varying_num, 1)), a_min=1e-5, a_max=1e0)) * 0.00001
        step_varying_params = jnp.log(noise + 2 / (self.smooth_param + self.str_cvx_param)) * jnp.ones((self.step_varying_num, 1))

        # init steady_state_params
        steady_state_params = sigmoid_inv(self.smooth_param / (self.smooth_param + self.str_cvx_param)) * jnp.ones((1, 1))

        self.params = [jnp.vstack([step_varying_params, steady_state_params])]


    def set_params_for_nesterov(self):
        # self.params = [jnp.log(1 / self.smooth_param * jnp.ones((self.step_varying_num + 1, 1)))]
        nesterov_step = 1 / self.smooth_param # 4 / (3 * self.smooth_param + self.str_cvx_param)
        self.params = [jnp.log(nesterov_step * jnp.ones((self.step_varying_num + 1, 1)))]


    def set_params_for_silver(self):
        silver_steps = 4096
        # kappa = self.smooth_param / self.str_cvx_param

        silver_step_sizes = compute_silver_steps(silver_steps) / self.smooth_param
        params = jnp.ones((silver_steps, 1))
        params = params.at[:silver_steps - 1, 0].set(jnp.array(silver_step_sizes))
        params = params.at[silver_steps - 1, 0].set(1 / (self.smooth_param))
        self.params = [params]



    def init_params(self):
        # init step-varying params
        step_varying_params = jnp.log(1 / (self.smooth_param)) * jnp.ones((self.step_varying_num, 2))
        # step_varying_params =  jnp.ones((self.step_varying_num, 1)) * jnp.log(1.0) # jnp.log(0.1) # 
        
        t_params = jnp.ones(self.step_varying_num)
        t = 1
        for i in range(0, self.step_varying_num):
            t = i #.5 * (1 + jnp.sqrt(1 + 4 * t ** 2))
            t_params = t_params.at[i].set(t/(t+3)) #(jnp.log(t))
        # beta_params = convert_t_to_beta(t_params)
        # step_varying_params = step_varying_params.at[:, 1].set(jnp.log(beta_params))
        # step_varying_params = step_varying_params.at[1:, 1].set(jnp.log(beta_params.mean()))
        step_varying_params = step_varying_params.at[:, 1].set(jnp.log(t_params))

        # init steady_state_params
        steady_state_params = 0 * jnp.ones((1, 2)) #sigmoid_inv(0) * jnp.ones((1, 1))
        self.params = [jnp.vstack([step_varying_params, steady_state_params])]
        
        
    def pep_cvxpylayer(self, params):
        step_sizes = params[:self.num_pep_iters,0]
        momentum_sizes = params[:self.num_pep_iters,1]
        
        A_param = build_A_matrix_with_yk_and_xstar(step_sizes, momentum_sizes)
        G, H = self.pep_layer(A_param, solver_args={"solve_method": "CLARABEL", "verbose": True})

        return H[-3] - H[-1]


    def create_end2end_loss_fn(self, bypass_nn, diff_required, special_algo='gd'):
        supervised = True  # self.supervised and diff_required
        loss_method = self.loss_method

        def predict(params, input, q, iters, z_star, key, factor):
            z0 = input
            if diff_required:
                n_iters = key #self.train_unrolls if key else 1
                if n_iters == 1:
                    # for steady_state training
                    stochastic_params = 2 / self.smooth_param * sigmoid(params[0][:n_iters, :])
                else:
                    # for step-varying training
                    stochastic_params = jnp.exp(params[0][:n_iters, :])
            else:
                if special_algo == 'silver':
                    stochastic_params = params[0]
                else:
                    n_iters = key #min(iters, 51)
                    # stochastic_params = jnp.zeros((n_iters, 1))
                    # stochastic_params = stochastic_params.at[:n_iters - 1, 0].set(jnp.exp(params[0][:n_iters - 1, 0]))
                    # stochastic_params = stochastic_params.at[n_iters - 1, 0].set(2 / self.smooth_param * sigmoid(params[0][n_iters - 1, 0]))
                    stochastic_params = self.transform_params(params, n_iters)

            if self.train_fn is not None:
                train_fn = self.train_fn
            else:
                train_fn = self.k_steps_train_fn
            if self.eval_fn is not None:
                eval_fn = self.eval_fn
            else:
                eval_fn = self.k_steps_eval_fn

            # stochastic_params = params[0][:n_iters, 0]
            if bypass_nn:
                # use nesterov's acceleration
                eval_out = self.nesterov_eval_fn(k=iters,
                                   z0=z0,
                                   q=q,
                                #    params=stochastic_params,
                                    params=1/self.smooth_param,
                                   supervised=supervised,
                                   z_star=z_star)
                z_final, iter_losses, z_all_plus_1 = eval_out[0], eval_out[1], eval_out[2]
                angles = None
            else:
                if diff_required:
                    z_final, y_final, t_final, iter_losses = train_fn(k=iters,
                                                    z0=z0,
                                                    q=q,
                                                    params=stochastic_params,
                                                    supervised=supervised,
                                                    z_star=z_star)
                else:
                    eval_out = eval_fn(k=iters,
                                       z0=z0,
                                       q=q,
                                       params=stochastic_params,
                                       supervised=supervised,
                                       z_star=z_star)
                    z_final, iter_losses, z_all_plus_1 = eval_out[0], eval_out[1], eval_out[2]
                    angles = None

            loss = self.final_loss(loss_method, z_final,
                                   iter_losses, supervised, z0, z_star)

            if diff_required:
                return loss
            else:
                return_out = (loss, iter_losses, z_all_plus_1,
                              angles) + eval_out[3:]
                return return_out
        loss_fn = self.predict_2_loss(predict, diff_required)
        return loss_fn
    
    
    def create_nesterov_pep_sdp_layer(self, num_iters):
        """
        ordering: x*, x_0,...,x_{p-1}, s*,s_0,..,s_{p-1}
        """
        k = num_iters
        A = cp.Parameter((k+3, k+3))
        L = self.smooth_param
        
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
        
        # beta_list = jnp.array([i / (i+3) for i in range(num_iters)])

        return cvxpylayer
    
    def pep_clarabel(self, params):
        alpha_list = params[:,0]
        beta_list = params[:,1]
        k = len(alpha_list)
        L = self.smooth_param

        # Build coefficient matrix A: rows for x_0,...,x_k, y_k, x_star
        A = cp.Parameter((k+3, k+3))
        A_val = build_A_matrix_with_yk_and_xstar(alpha_list, beta_list)
        

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

        A.value = A_val
        prob = cp.Problem(objective, constraints)
        prob.solve(verbose=True, solver=cp.MOSEK)

        return prob.value
    
    
    def pepit_nesterov_check(self, params):
        num_iters = params[:,0].size
        step_sizes = params[:,0]
        beta_vals = params[:,1]
        
        # Instantiate PEP
        problem = PEP()
        mu = 0
        L = self.smooth_param._value

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




def compute_silver_steps(num_steps):
    rho = 1 + np.sqrt(2)
    schedule = [1 + rho**((k & -k).bit_length()-2) for k in range(1, num_steps)]
    return np.array(schedule)

def sigmoid(x):
    return 1 / (1 + jnp.exp(-x))

def sigmoid_inv(beta):
    return jnp.log(beta / (1 - beta))

def convert_t_to_beta(t_vals):
    beta_vals = jnp.ones(t_vals.size)
    for i in range(1, t_vals.size):
        beta_vals = beta_vals.at[i-1].set((t_vals[i-1] - 1) / t_vals[i])
    return beta_vals


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
