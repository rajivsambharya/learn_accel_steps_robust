from functools import partial

import jax.numpy as jnp
from jax import random

import numpy as np

from lah_accel.algo_steps_nonneg_gd import k_steps_eval_lah_nonneg_gd_accel, k_steps_train_lah_nonneg_gd_accel, k_steps_eval_fnonneg_gd_backtracking__
from lah_accel.l2o_model import L2Omodel
from lah_accel.utils.nn_utils import calculate_pinsker_penalty, compute_single_param_KL

from jax import vmap, jit
import cvxpy as cp
from cvxpylayers.jax import CvxpyLayer

from PEPit import PEP
from PEPit.functions import SmoothStronglyConvexFunction
from PEPit.functions import ConvexFunction, ConvexIndicatorFunction
from PEPit.operators import SymmetricLinearOperator
from PEPit.primitive_steps import proximal_step
from lah_accel.pep import create_proxgd_pep_sdp_layer, build_A_matrix_prox_with_xstar, pepit_nesterov, create_quadprox_pep_sdp_layer, pepit_accel_gd


class LAHNONNEGGDAccelmodel(L2Omodel):
    def __init__(self, **kwargs):
        super(LAHNONNEGGDAccelmodel, self).__init__(**kwargs)

    def initialize_algo(self, input_dict):
        self.factor_static = None
        self.algo = 'lah_nonneg_gd'
        self.factors_required = False
        self.q_mat_train, self.q_mat_test = input_dict['c_mat_train'], input_dict['c_mat_test']
        # self.q_mat_train, self.q_mat_test = input_dict['b_mat_train'], input_dict['b_mat_test']
        # D, W = input_dict['D'], input_dict['W']
        A = input_dict['A']
        # lambd = input_dict['lambd']
        self.A = A
        # self.lambd = lambd

        # self.D, self.W = D, W
        self.m, self.n = A.shape
        self.output_size = self.n

        # evals, evecs = jnp.linalg.eigh(D.T @ D)
        # lambd = 0.1
        # self.ista_step = lambd / evals.max()
        # p = jnp.diag(P)
        # cond_num = jnp.max(p) / jnp.min(p)
        evals, evecs = jnp.linalg.eigh(A.T @ A)
        self.evals = evals

        self.str_cvx_param = jnp.min(evals)
        self.smooth_param = jnp.max(evals)

        cond_num = self.smooth_param / self.str_cvx_param

        safeguard_step = 1 / self.smooth_param

        # self.k_steps_train_fn = partial(k_steps_train_lah_ista, lambd=lambd, A=A,
        #                                 jit=self.jit)
        # self.k_steps_eval_fn = partial(k_steps_eval_lah_ista, lambd=lambd, A=A, safeguard_step=safeguard_step,
        #                                jit=self.jit)
        self.accel = False #input_dict['accel']
        self.supervised = not self.accel
        if self.accel or True:
            self.k_steps_train_fn = partial(k_steps_train_lah_nonneg_gd_accel,  A=A,
                                            jit=self.jit)
            self.k_steps_eval_fn = partial(k_steps_eval_lah_nonneg_gd_accel, A=A,
                                        jit=self.jit)
        else:
            self.k_steps_train_fn = partial(k_steps_train_lah_nonneg_gd, lambd=lambd, A=A,
                                            jit=self.jit)
            self.k_steps_eval_fn = partial(k_steps_eval_lah__nonneg_gd, lambd=lambd, A=A, safeguard_step=safeguard_step,
                                        jit=self.jit)
        self.backtracking_eval_fn = partial(k_steps_eval_fnonneg_gd_backtracking__, eta0=10.0, A=A,
                                       jit=self.jit)

        self.out_axes_length = 5

        N = self.q_mat_train.shape[0]
        self.lah_train_inputs = jnp.zeros((N, self.n))
        
         #20)
        # self.pep_layer = self.create_quad_prox_pep_sdp_layer(10)
        # self.pep_layer = self.create_quad_prox_pep_sdp_layer(self.train_unrolls)
        # self.pep_layer = self.create_nesterov_pep_sdp_layer(self.train_unrolls)
        self.num_pep_iters = 20 #self.train_unrolls

        # e2e_loss_fn = self.create_end2end_loss_fn
        # self.pep_layer = create_proxgd_pep_sdp_layer(self.smooth_param, self.num_pep_iters)
        # self.pep_layer = create_quadprox_pep_sdp_layer(self.str_cvx_param, self.smooth_param, self.num_pep_iters)

        # end-to-end loss fn for silver evaluation
        e2e_loss_fn = self.create_end2end_loss_fn
        self.loss_fn_eval_backtracking = e2e_loss_fn(bypass_nn=False, diff_required=False, 
                                               special_algo='backtracking')



    def f(self, z, c):
        return .5 * jnp.linalg.norm(self.A @ z - c) ** 2 + self.lambd * jnp.linalg.norm(z, ord=1)

    # def compute_avg_opt(self):
    #     batch_f = vmap(self.f, in_axes=(0, 0), out_axes=(0))
    #     opt_vals = batch_f(self.z_stars_train, self.theta)

    def transform_params(self, params, n_iters):
        # transformed_params = jnp.zeros((n_iters, params[0].shape[1]))
        # transformed_params = transformed_params.at[:, :].set(jnp.exp(params[0][:, :]))
        return jnp.exp(params[0][:, :])
        # return transformed_params

    def perturb_params(self):
        k = self.eval_unrolls

        # init step-varying params
        noise = jnp.array(np.clip(np.random.normal(size=(k, 2)), a_min=-1, a_max=1)) * 0.01
        # step_varying_params = jnp.log(noise + 2 / (self.smooth_param)) * jnp.ones((self.step_varying_num, 1))
        step_varying_params = noise + self.params[0][:k, :]

        # init steady_state_params
        # steady_state_params = sigmoid_inv(self.smooth_param / (self.smooth_param + self.str_cvx_param)) * jnp.ones((1, 2))
        steady_state_params = 0 * jnp.ones((1, 2))

        self.params = [jnp.vstack([step_varying_params, steady_state_params])]


    def set_params_for_nesterov(self):
        self.init_params()
        # self.params = [jnp.log(1 / self.smooth_param * jnp.ones((self.step_varying_num + 1, 1)))]


    def pep_cvxpylayer(self, params):
        step_sizes = params[:self.num_pep_iters,0]
        momentum_sizes = params[:self.num_pep_iters,1]
        
        A_param = build_A_matrix_prox_with_xstar(step_sizes, momentum_sizes)
        # G, F, H = self.pep_layer(A_param, solver_args={"solve_method": "SCS", "verbose": True})
        G, H = self.pep_layer(A_param, solver_args={"solve_method": "SCS", "verbose": True})
        # return F[-2] + H[-2] - F[-1] - H[-1]
        k = self.num_pep_iters
        # return G[2*k+2, 2*k+2] ** .5
        return G[3*k + 7, 3*k + 7] - 2*G[3*k + 7, 1] + G[1, 1]

    def init_params(self):
        # k = self.step_varying_num
        k = self.eval_unrolls
        

        # t_params = jnp.ones(k)
        # t = 1
        # for i in range(1, k):
        #     t = .5 * (1 + jnp.sqrt(1 + 4 * t ** 2))
        #     t_params = t_params.at[i].set(t) #(jnp.log(t))
        # beta_params = convert_t_to_beta(t_params)
        # step_varying_params = step_varying_params.at[:, 1].set(jnp.log(beta_params))
        if self.accel:
            # init step-varying params
            step_varying_params = jnp.log(1. / self.smooth_param) * jnp.ones((k, 2))
            kappa = self.smooth_param / self.str_cvx_param
            step_varying_params = step_varying_params.at[:, 1].set(jnp.log((kappa**.5-1) / (kappa**.5+1)))
            # step_varying_params = step_varying_params.at[:, 1].set(jnp.log(1e-3))
            steady_state_params = jnp.array([jnp.log(1. / self.smooth_param), jnp.log((kappa**.5-1) / (kappa**.5+1))])
        else:
            step_varying_params = jnp.log(1. / (self.smooth_param)) * jnp.ones((k, 2))
            # step_varying_params = jnp.log(2. / (self.smooth_param + self.str_cvx_param)) * jnp.ones((k, 2))
            step_varying_params = step_varying_params.at[:, 1].set(jnp.log(1e-8))
            steady_state_params = jnp.array([jnp.log(2. / (self.smooth_param + self.str_cvx_param)), jnp.log(1e-8)])

        # init steady_state_params
        # steady_state_params = 0 * jnp.ones((1, 2))
        

        self.params = [jnp.vstack([step_varying_params, steady_state_params])]
        
        # sigmoid_inv(beta)

    

    def create_end2end_loss_fn(self, bypass_nn, diff_required, special_algo='gd'):
        supervised = self.supervised  # self.supervised and diff_required
        loss_method = self.loss_method

        @partial(jit, static_argnames=['iters', 'key'])
        def predict(params, input, q, iters, z_star, key, factor):
            if diff_required:
                z0 = input
            else:
                z0 = input #+ 10000
            
            # params[0] = params[0].at[0,1].set(jnp.log(1))
            if diff_required:
                n_iters = key #self.train_unrolls if key else 1

                if n_iters == 1:
                    # for steady_state training
                    stochastic_params = 2 / self.smooth_param * sigmoid(params[0][:n_iters, 0])
                else:
                    # for step-varying training
                    # params[0] = params[0].at[-1,0].set()
                    stochastic_params = jnp.exp(params[0][:n_iters, :])
                    # stochastic_params = stochastic_params.at[-1,0].set(1 / self.smooth_param)
                    # stochastic_params = stochastic_params.at[-1,1].set(.9)
            else:
                if special_algo == 'silver' or special_algo == 'conj_grad':
                    stochastic_params = params[0]
                else:
                    n_iters = key #min(iters, 51)
                    # stochastic_params = jnp.zeros((n_iters, 1))
                    # stochastic_params = stochastic_params.at[:n_iters - 1, 0].set(jnp.exp(params[0][:n_iters - 1, 0]))
                    # stochastic_params = stochastic_params.at[n_iters - 1, 0].set(2 / self.smooth_param * sigmoid(params[0][n_iters - 1, 0]))
                    # stochastic_params = self.transform_params(params, n_iters)
                    stochastic_params = jnp.exp(params[0])
            if not self.accel:
                stochastic_params = stochastic_params.at[:,1].set(0)

            # stochastic_params = jnp.clip(stochastic_params, a_max=0.3)
            if self.train_fn is not None:
                train_fn = self.train_fn
            else:
                train_fn = self.k_steps_train_fn
            if self.eval_fn is not None:
                eval_fn = self.eval_fn
            else:
                eval_fn = self.k_steps_eval_fn


            # stochastic_params = params[0][:n_iters, 0]
            if special_algo == 'backtracking':
                eval_out = self.backtracking_eval_fn(k=iters,
                                   z0=z0,
                                   q=q,
                                #    params=stochastic_params[:,0],
                                   supervised=supervised,
                                   z_star=z_star)
                z_final, iter_losses, z_all_plus_1 = eval_out[0], eval_out[1], eval_out[2]
                angles = None
            elif bypass_nn:
                # use nesterov's acceleration
                eval_out = self.backtracking_eval_fn(k=iters,
                                   z0=z0,
                                   q=q,
                                #    params=stochastic_params[:,0],
                                   supervised=supervised,
                                   z_star=z_star)
                z_final, iter_losses, z_all_plus_1 = eval_out[0], eval_out[1], eval_out[2]
                angles = None
            else:
                if self.accel:
                    params = stochastic_params
                else:
                    params = stochastic_params #[:,:1]
                # import pdb
                # pdb.set_trace()
                if diff_required:
                    if self.accel:
                        z_final, iter_losses = train_fn(k=iters,
                                                        z0=z0,
                                                        # y0=z0,
                                                        q=q,
                                                        params=params,
                                                        supervised=supervised,
                                                        z_star=z_star)
                    else:
                        z_final, iter_losses = train_fn(k=iters,
                                                        z0=z0,
                                                        q=q,
                                                        params=params,
                                                        supervised=supervised,
                                                        z_star=z_star)
                else:
                    
                    eval_out = eval_fn(k=iters,
                                       z0=z0,
                                       q=q,
                                       params=params,
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
    
    def pepit_nesterov_check(self, params):
        return pepit_accel_gd(self.str_cvx_param._value, self.smooth_param._value, params, True, True, 'dist')
        # return pepit_accel_gd(self.str_cvx_param._value, self.smooth_param._value, params, True, True, 'func')
    
        
    def pepit_nesterov_check_old(self, params):
        num_iters = params[:,0].size
        step_sizes = params[:,0]
        beta_vals = params[:,1]
        
        # Instantiate PEP
        problem = PEP()
        mu = 0
        L = self.smooth_param._value

        # Declare a strongly convex smooth function and a convex function
        # f = problem.declare_function(SmoothStronglyConvexFunction, mu=mu, L=L)
        f = problem.declare_function(SymmetricLinearOperator, mu=mu, L=L)
        # h = problem.declare_function(ConvexIndicatorFunction)
        h = problem.declare_function(ConvexFunction)
        F = f + h

        # Start by defining its unique optimal point xs = x_* and its function value Fs = F(x_*)
        xs = F.stationary_point()
        Fs = F(xs)

        # Then define the starting point x0
        x0 = problem.set_initial_point()

        # Set the initial constraint that is the distance between x0 and x^*
        problem.set_initial_condition((x0 - xs) ** 2 <= 1)

        # Compute n steps of the accelerated proximal gradient method starting from x0
        x_new = x0
        y = x0
        for i in range(num_iters):
            x_old = x_new
            x_new, _, hx_new = proximal_step(y - step_sizes[i] * f.gradient(y), h, step_sizes[i])
            y = x_new + beta_vals[i] * (x_new - x_old)

        # Set the performance metric to the function value accuracy
        problem.set_performance_metric((f(x_new) + hx_new) - Fs)
        
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



def sigmoid(x):
    return 1 / (1 + jnp.exp(-x))

def sigmoid_inv(beta):
    return jnp.log(beta / (1 - beta))

def get_x_index_subdiff(iter, k, traj=0):
    return get_x_index(traj, iter, k)

def get_y_index_subdiff(iter, k, traj=0):
    return iter + 2*(k+1) + 2

def get_x_index(traj, iter, k):
    if iter == 0:
        return 0
    return 3 + traj * 2 * k + iter - 1

def get_s_index(iter, k, traj):
    return 3 + iter + k - 1 + traj * 2 * k

def collect_linear_grad_indices(k):
    x_list = [1] + [0] + [j for j in range(3, k+3)]
    s_list = [2] + [k+3] + [j for j in range(k+4, 2*k+4)]
    return x_list, s_list

def collect_linear_subdiff_indices(k, step_sizes_list, num_traj=1):
    # first embed the optimal solution
    x_list, x_next_list, s_list, step_size_list = [1], [1], [2], [1]

    # iterate over both trajectories
    for traj in range(num_traj):
        for j in range(0, k+1): # we start from x_1,x_2,...,x_k (ignore x0)
            x_list.append(get_x_index_subdiff(j, k, traj))
            x_next_list.append(get_x_index_subdiff(j+1, k, traj))
            s_list.append(get_s_index(j+1, k, traj))
            step_size_list.append(step_sizes_list[traj][j])

    return x_list, x_next_list, s_list, step_size_list

def collect_subgrad_subdiff_indices(k, num_traj=1):
    # first embed the optimal solution
    x_list, s_list, step_size_list = [1], [2], [1]

    # iterate over both trajectories
    for traj in range(num_traj):
        for j in range(1, k+1): # we start from x_1,x_2,...,x_k (ignore x0)
            x_list.append(get_x_index_subdiff(j, k, traj))
            s_list.append(get_s_index(j, k, traj)) # but for y we get the previous value

    return x_list, s_list

def collect_nesterov_subdiff_indices(k, num_traj=1):
    # first embed the optimal solution
    y_list, x_list, s_list, step_size_list = [1], [1], [2], [1]

    # iterate over both trajectories
    for traj in range(num_traj):
        for j in range(1, k+1): # we start from x_1,x_2,...,x_k (ignore x0)
            y_list.append(get_y_index_subdiff(j, k, traj))
            x = get_x_index_subdiff(j-1, k, traj)
            
            x_list.append(x)
            s_list.append(get_s_index(j, k, traj)) # but for y we get the previous value

    return y_list, x_list, s_list

def convert_t_to_beta(t_vals):
    beta_vals = jnp.ones(t_vals.size)
    for i in range(1, t_vals.size):
        beta_vals = beta_vals.at[i-1].set((t_vals[i-1] - 1) / t_vals[i])
    # import pdb
    # pdb.set_trace()
    return beta_vals

