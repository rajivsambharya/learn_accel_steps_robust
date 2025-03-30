from functools import partial

import jax.numpy as jnp
from jax import random, vmap

import numpy as np

from lah.algo_steps_logistic import k_steps_eval_lah_nesterov_gd, k_steps_train_lah_nesterov_gd, k_steps_eval_nesterov_logisticgd, compute_gradient
from lah.l2ws_model import L2WSmodel
from lah.utils.nn_utils import calculate_pinsker_penalty, compute_single_param_KL
from PEPit import PEP
from PEPit.functions import SmoothStronglyConvexFunction
from PEPit.functions import ConvexFunction, ConvexIndicatorFunction
from PEPit.primitive_steps import proximal_step
import cvxpy as cp


class LAHAccelLOGISTICGDmodel(L2WSmodel):
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

        # cond_num = self.smooth_param / self.str_cvx_param

        self.k_steps_train_fn = partial(k_steps_train_lah_nesterov_gd, num_points=num_points,
                                        jit=self.jit)
        self.k_steps_eval_fn = partial(k_steps_eval_lah_nesterov_gd, num_points=num_points, 
                                       jit=self.jit)
        self.nesterov_eval_fn = partial(k_steps_eval_nesterov_logisticgd, num_points=num_points,
                                       jit=self.jit)
        # self.conj_grad_eval_fn = partial(k_steps_eval_conj_grad, num_points=num_points,
        #                                jit=self.jit)
        self.out_axes_length = 5

        N = self.q_mat_train.shape[0]
        self.lah_train_inputs = jnp.zeros((N, num_weights))



        e2e_loss_fn = self.create_end2end_loss_fn



        # end-to-end loss fn for silver evaluation
        self.loss_fn_eval_silver = e2e_loss_fn(bypass_nn=False, diff_required=False, 
                                               special_algo='silver')
        
        # end-to-end loss fn for silver evaluation
        self.loss_fn_eval_conj_grad = e2e_loss_fn(bypass_nn=False, diff_required=False, 
                                               special_algo='conj_grad')

        # end-to-end added fixed warm start eval - bypasses neural network
        # self.loss_fn_fixed_ws = e2e_loss_fn(bypass_nn=True, diff_required=False)

        self.num_const_steps = input_dict.get('num_const_steps', 1)

        self.num_points = num_points

        if self.train_unrolls == 1:
            self.train_case = 'one_step_grad'

    def compute_single_gradient(self, z, q):
        num_points = self.num_points
        X_flat = q[:num_points * 784]
        X = jnp.reshape(X_flat, (num_points, 784))
        y = q[num_points * 784:]

        w, b = z[:-1], z[-1]
        y_hat = sigmoid(X @ w + b)
        dw, db = compute_gradient(X, y, y_hat)
        return jnp.hstack([dw, db])

    # def compute_gradients(self, batch_inputs, batch_q_data):
    #     # TODO
    #     return gradients

    def compute_gradients(self, batch_inputs, batch_q_data):
        # Use vmap to vectorize compute_single_gradient over the batch dimensions
        batched_compute_gradient = vmap(self.compute_single_gradient, in_axes=(0, 0))
        return batched_compute_gradient(batch_inputs, batch_q_data)
    
    
    def pep_cvxpylayer(self, params):
        # params = params.at[0,1].set(1)
        step_sizes = params[:,0]
        # G, H = self.pep_layer(step_sizes, solver_args={"solve_method": "SCS", "verbose": True})

        beta = params[:,1]
        beta_sq = beta ** 2
        k = step_sizes.size
        G, H = self.pep_layer(step_sizes, beta, beta_sq, solver_args={"solve_method": "CLARABEL", "verbose": True}) #, "max_iters": 10000}) # , 

        return H[-1] - H[0]
    
    def pep_clarabel(self, params):
        num_iters = params[:,0].size
        step_sizes = params[:,0]
        beta_vals = params[:,1]
        beta_sq_vals = beta_vals ** 2
        proj = True
        step_sizes_param = cp.Parameter(num_iters)
        beta_param = cp.Parameter(num_iters)
        beta_param_sq = cp.Parameter(num_iters)

        k = num_iters #step_sizes.size
        G = cp.Variable((3*(k+1)+2, 3*(k+1)+2), symmetric=True) # gram matrix
        H = cp.Variable(k+2)
        if not proj:
            F = cp.Variable(k+1)

        tol = 0
        constraints = [G >> tol*np.eye(3*(k+1)+2)]

        # denom
        constraints.append(G[0, 0] - 2*G[0,1] + G[1, 1] == 1)
        
        # make y0 = z0
        y_ind_start = 2*(k+1)+2
        constraints.append(G[0, 0] - 2*G[0,y_ind_start] + G[y_ind_start, y_ind_start] == 0)

        # num
        x_k_ind = get_x_index_subdiff(k, k)
        print('x_k_ind', x_k_ind)
        
        # subgradient inequalities: ∂h(x_i)^T (x_i - x_j) >= 0
        # (y_{i-1} - x_i)^T (x_i - x_j) >= 0 -- based on triples (x_i, y_{i-1}, x_j)
        # x_list_subgrad, s_list_subgrad = collect_subgrad_subdiff_indices(k, 1)
        y_list, x_list, s_list = collect_nesterov_subdiff_indices(k)
        print('y_list', y_list)
        print('x_list', x_list)
        print('s_list', s_list)
        for i in range(len(x_list)):
            y_i_ind = y_list[i]
            x_i_ind = x_list[i]
            s_i_ind = s_list[i]
            if x_i_ind == 1:
                x_i_prev_ind = x_i_ind
                alpha_i = 1
            elif x_i_ind == 3:
                x_i_prev_ind = 0
                alpha_i = step_sizes_param[0]
            else:
                x_i_prev_ind = x_i_ind - 1
                alpha_i = step_sizes_param[i-1]
            for j in range(len(y_list)):
                y_j_ind = y_list[j]
                if y_j_ind != y_i_ind:
                    if proj:
                        # constraints.append(G[s_i_ind, x_i_ind] - G[s_i_ind, x_j_ind] >= 0)
                        constraints.append(G[x_i_ind, y_i_ind] - G[x_i_ind, y_j_ind] -G[y_i_ind, y_i_ind] +  G[y_i_ind, y_j_ind] -alpha_i*G[s_i_ind, y_i_ind] +alpha_i* G[s_i_ind, y_j_ind] >= 0)
                    else:
                        constraints.append((F[j] - F[i])  + G[s_i_ind, x_i_ind] - G[s_i_ind, x_j_ind] >= 0)
                    # import pdb
                    # pdb.set_trace()
                    
            # constraints.append(G[x_i_prev_ind, x_i_ind] - G[x_i_prev_ind, 0] -G[x_i_ind, x_i_ind] +  G[x_i_ind, 0] -alpha_i*G[s_i_ind, x_i_ind] +alpha_i* G[s_i_ind, 0] >= 0)


        # x_list, x_next_list, s_list, step_size_list = collect_linear_subdiff_indices(k, [step_sizes_param]) # Q u_i = v_i for i =1, ..., len(x_list) (these are diff)
        x_list, s_list = collect_linear_grad_indices(k)
        print('x_list', x_list)
        # print('x_next_list', x_next_list)
        print('s_list', s_list)
        # print('step_size_list', step_size_list)

        # u_i^T v_j = u_j^T v_i (i neq j) where u_i = x_i i=0,...,k-1 and v_j = 1 / alpha_j (x_j - y_j)
        for i in range(len(x_list)):
            x_i_ind = x_list[i]
            # x_next_i_ind = x_next_list[i]
            s_i_ind = s_list[i]
            # alpha_i = step_size_list[i]
            # if i == 0:
            #     alpha_i_sq = 1
            # else:
            #     alpha_i_sq = cross_alpha_ij[i-1, i-1]
            for j in range(len(x_list)):
                if i != j:
                    x_j_ind = x_list[j]
                    s_j_ind = s_list[j]

                    constraints.append((H[i] - H[j]) >=  G[s_j_ind, x_i_ind] - G[s_j_ind, x_j_ind] + 1  / (2 * self.smooth_param) * (G[s_i_ind, s_i_ind] + G[s_j_ind, s_j_ind] - 2 * G[s_i_ind, s_j_ind])) 
                    
        for i in range(k):
            beta_ind = i
            x_i_ind = 3 + i
            y_i_ind = y_ind_start + i + 1
            y_prev_i_ind = 0 if y_i_ind == 3 else y_i_ind - 1
            constraints.append(G[x_i_ind, x_i_ind] + (beta_param_sq[beta_ind] + 2*beta_param[beta_ind] + 1)  * G[y_i_ind, y_i_ind] + beta_param_sq[beta_ind] * G[y_prev_i_ind, y_prev_i_ind] + 2 * beta_param[beta_ind] * G[y_prev_i_ind, x_i_ind] - 2 * (beta_param[beta_ind] + beta_param_sq[beta_ind]) * G[y_prev_i_ind, y_i_ind] -2 * (beta_param[beta_ind] + 1) * G[x_i_ind, y_i_ind] == 0)

        prob = cp.Problem(cp.Maximize(H[-1] - H[0]), constraints)
        # step_sizes_param.value = np.ones(num_iters) / np.array(self.smooth_param)

        beta_param.value = beta_vals #np.ones(num_iters)
        beta_param_sq.value = (beta_param.value)**2 #np.ones(num_iters)
        step_sizes_param.value = step_sizes
        
        try:
            prob.solve(verbose=True, solver=cp.CLARABEL)
            return prob.value
        except Exception as e:
            print('exception clarabel', e)
            return 0


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
        # sigmoid_inv(beta)


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
                if special_algo == 'silver' or special_algo == 'conj_grad':
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
            if special_algo == 'conj_grad':
                eval_out = self.conj_grad_eval_fn(k=iters,
                                   z0=z0,
                                   q=q,
                                   params=stochastic_params,
                                   supervised=supervised,
                                   z_star=z_star)
                z_final, iter_losses, z_all_plus_1 = eval_out[0], eval_out[1], eval_out[2]
                angles = None
            elif bypass_nn:
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

            # penalty_loss = calculate_pinsker_penalty(self.N_train, params, self.b, self.c, self.delta)
            # loss = loss + self.penalty_coeff * penalty_loss

            if diff_required:
                return loss
            else:
                return_out = (loss, iter_losses, z_all_plus_1,
                              angles) + eval_out[3:]
                return return_out
        loss_fn = self.predict_2_loss(predict, diff_required)
        return loss_fn
    
    def pep_clarabel(self, params):
        alpha_list = params[:,0]
        beta_list = params[:,1]
        k = len(alpha_list)
        L = self.smooth_param

        # Build coefficient matrix A: rows for x_0,...,x_k, y_k, x_star
        A = build_A_matrix_with_yk_and_xstar(alpha_list, beta_list)

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
        prob.solve(verbose=True, solver=cp.MOSEK)

        return prob.value
    
    
    def pepit_nesterov_check(self, params):
        num_iters = params[:,0].size
        step_sizes = params[:,0]
        beta_vals = np.clip(params[:,1], a_min=0, a_max=.999)
        
        # Instantiate PEP
        problem = PEP()
        mu = 0
        L = self.smooth_param._value

        # Declare a strongly convex smooth function and a convex function
        f = problem.declare_function(SmoothStronglyConvexFunction, mu=mu, L=L)
        # h = problem.declare_function(ConvexIndicatorFunction)
        # h = problem.declare_function(ConvexFunction)
        # F = f #+ h

        # Start by defining its unique optimal point xs = x_* and its function value Fs = F(x_*)
        xs = f.stationary_point() #F.stationary_point()
        fs = f(xs)

        # Then define the starting point x0
        x0 = problem.set_initial_point()

        # Set the initial constraint that is the distance between x0 and x^*
        problem.set_initial_condition((x0 - xs) ** 2 <= 1)

        # Compute n steps of the accelerated proximal gradient method starting from x0
        x_new = x0
        y = x0
        for i in range(num_iters):
            # if i < num_iters:
            #     alpha = step_sizes[i]
            #     beta = beta_vals[i]
            # else:
            #     alpha = 1 / L
            #     beta = i / (i + 3)
            alpha = step_sizes[i]
            beta = beta_vals[i]
            x_old = x_new
            x_new = y - alpha * f.gradient(y)
            y = x_new + beta * (x_new - x_old)

        # Set the performance metric to the function value accuracy
        # problem.set_performance_metric((f(x_new)) - fs)
        problem.set_performance_metric((f(y)) - fs)
        
        # import pdb
        # pdb.set_trace()

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
    Returns A matrix of shape (k+3, k+3), where:
    - A[0] to A[k] represent x_0 to x_k
    - A[k+1] represents y_k
    - A[k+2] represents x_star

    Basis: [x_0, x_star, g_0, ..., g_{k-1}, g_f]
    """
    k = len(alpha_list)
    assert len(beta_list) == k, "Length of beta_list must match alpha_list"

    A = np.zeros((k + 3, k + 3))  # Rows: x_0 to x_k, y_k, x_star
    idx_x0 = 0
    idx_xstar = 1
    idx_g = lambda t: 2 + t
    idx_gf = 2 + k  # g_f = ∇f(y_k)

    # x_0 = [1, 0, 0, ..., 0]
    A[0, idx_x0] = 1.0

    # Build x_1 to x_k
    for i in range(k):
        # Construct y_i = x_i + β_i (x_i - x_{i-1})
        x_i = A[i]
        x_im1 = A[i - 1] if i > 0 else A[0]
        # y_i = (1 + beta_list[i]) * x_i - beta_list[i] * x_im1

        # x_{i+1} = y_i - α_i ∇f(y_i)
        # x_ip1 = y_i.copy()
        # x_ip1[idx_g(i)] -= alpha_list[i]
        y_i = x_i.copy()
        y_i[idx_g(i)] -= alpha_list[i]
        y_im1 = x_im1.copy()
        if i == 0:
            y_im1[idx_g(i)] -= alpha_list[i]
        else:
            y_im1[idx_g(i-1)] -= alpha_list[i-1]

        # A[i + 1] = x_ip1  # Store x_{i+1}
        A[i + 1] = (1 + beta_list[i]) * y_i - beta_list[i] * y_im1

    # Final extrapolated point: y_k = x_k + β_k (x_k - x_{k-1})
    x_k = A[k]
    x_km1 = A[k - 1] if k >= 1 else A[0]
    beta_k = beta_list[k - 1]
    y_k = (1 + beta_k) * x_k - beta_k * x_km1
    A[k + 1] = y_k  # Store y_k

    # x_star
    A[k + 2, idx_xstar] = 1.0

    return A
