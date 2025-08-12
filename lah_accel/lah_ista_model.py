from functools import partial

import jax.numpy as jnp
from jax import random

import numpy as np

from lah_accel.algo_steps_ista import k_steps_eval_lah_ista, k_steps_train_lah_ista, k_steps_eval_fista, k_steps_eval_lah_fista, k_steps_train_lah_fista, k_steps_eval_lasso_backtracking, k_steps_eval_lasso_nesterov_backtracking, k_steps_eval_fista_backtracking__
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
from lah_accel.pep import create_proxgd_pep_sdp_layer, build_A_matrix_prox_with_xstar, pepit_nesterov


class LAHISTAmodel(L2Omodel):
    def __init__(self, **kwargs):
        super(LAHISTAmodel, self).__init__(**kwargs)

    def initialize_algo(self, input_dict):
        self.factor_static = None
        self.algo = 'lah_ista'
        self.factors_required = False
        self.q_mat_train, self.q_mat_test = input_dict['c_mat_train'], input_dict['c_mat_test']
        # self.q_mat_train, self.q_mat_test = input_dict['b_mat_train'], input_dict['b_mat_test']
        # D, W = input_dict['D'], input_dict['W']
        A = input_dict['A']
        lambd = input_dict['lambd']
        self.A = A
        self.lambd = lambd

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
        self.accel = input_dict['accel']
        self.supervised = not self.accel
        if self.accel:
            self.k_steps_train_fn = partial(k_steps_train_lah_fista, lambd=lambd, A=A,
                                            jit=self.jit)
            self.k_steps_eval_fn = partial(k_steps_eval_lah_fista, lambd=lambd, A=A,
                                        jit=self.jit)
        else:
            self.k_steps_train_fn = partial(k_steps_train_lah_ista, lambd=lambd, A=A,
                                            jit=self.jit)
            self.k_steps_eval_fn = partial(k_steps_eval_lah_ista, lambd=lambd, A=A, safeguard_step=safeguard_step,
                                        jit=self.jit)
        self.backtracking_eval_fn = partial(k_steps_eval_fista_backtracking__, lambd=lambd, eta0=10.0, A=A,
                                       jit=self.jit)

        self.out_axes_length = 5

        N = self.q_mat_train.shape[0]
        self.lah_train_inputs = jnp.zeros((N, self.n))
        
         #20)
        # self.pep_layer = self.create_quad_prox_pep_sdp_layer(10)
        # self.pep_layer = self.create_quad_prox_pep_sdp_layer(self.train_unrolls)
        # self.pep_layer = self.create_nesterov_pep_sdp_layer(self.train_unrolls)
        self.num_pep_iters = self.train_unrolls

        # e2e_loss_fn = self.create_end2end_loss_fn
        self.pep_layer = create_proxgd_pep_sdp_layer(self.smooth_param, self.num_pep_iters)

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
        # init step-varying params
        noise = jnp.array(np.clip(np.random.normal(size=(self.step_varying_num, 1)), a_min=-1, a_max=1)) * 0.01
        # step_varying_params = jnp.log(noise + 2 / (self.smooth_param)) * jnp.ones((self.step_varying_num, 1))
        step_varying_params = noise + self.params[0][:self.step_varying_num, :]

        # init steady_state_params
        steady_state_params = sigmoid_inv(self.smooth_param / (self.smooth_param + self.str_cvx_param)) * jnp.ones((1, 1))

        self.params = [jnp.vstack([step_varying_params, steady_state_params])]


    def set_params_for_nesterov(self):
        self.init_params()
        # self.params = [jnp.log(1 / self.smooth_param * jnp.ones((self.step_varying_num + 1, 1)))]
        
    # def pep_cvxpylayer(self, params):
    #     """
    #     1: create layer (once, with params)
    #     2: do forward pass
    #     """
    #     mu_str_cvx = 0
    #     L_smooth = 0
        
    #     # forward pass, i.e., solve the sdp
    #     sol = layer(params)
        
    #     # get the actual loss
        
        
    #     return loss


    # def set_params_for_silver(self):
    #     silver_steps = 128
    #     kappa = self.smooth_param / self.str_cvx_param
    #     silver_step_sizes = compute_silver_steps(kappa, silver_steps) / self.smooth_param
    #     params = jnp.ones((silver_steps + 1, 1))
    #     params = params.at[:silver_steps, 0].set(jnp.array(silver_step_sizes))
    #     params = params.at[silver_steps, 0].set(2 / (self.smooth_param + self.str_cvx_param))

    #     self.params = [params]
        # step_varying_params = jnp.log(params[:self.step_varying_num, :1])
        # steady_state_params = sigmoid_inv(params[self.step_varying_num:, :1] * self.smooth_param / 2)
        # self.params = [jnp.vstack([step_varying_params, steady_state_params])]
        
    def pep_cvxpylayer(self, params):
        step_sizes = params[:self.num_pep_iters,0]
        momentum_sizes = params[:self.num_pep_iters,1]
        
        A_param = build_A_matrix_prox_with_xstar(step_sizes, momentum_sizes)
        G, F, H = self.pep_layer(A_param, solver_args={"solve_method": "SCS", "verbose": True})
        return F[-2] + H[-2] - F[-1] - H[-1]
        # return H[-2] - H[-1]
        # penalty = 0
        # beta = params[:,1]
        # theta_vals = 1 / beta
        # for i in range(1, params[:,0].size):
        #     lhs = (1 - theta_vals[i]) / theta_vals[i]**2
        #     rhs = 1 / theta_vals[i-1]**2
        #     penalty += jnp.clip(lhs - rhs, a_min=0, a_max=1000)
        # import pdb
        # pdb.set_trace()
        return penalty

    def init_params(self):
        # k = self.step_varying_num
        k = self.eval_unrolls
        
        # init step-varying params
        step_varying_params = jnp.log(1. / self.smooth_param) * jnp.ones((k, 2))

        t_params = jnp.ones(k)
        t = 1
        for i in range(1, k):
            t = .5 * (1 + jnp.sqrt(1 + 4 * t ** 2))
            t_params = t_params.at[i].set(t) #(jnp.log(t))
        beta_params = convert_t_to_beta(t_params)
        step_varying_params = step_varying_params.at[:, 1].set(jnp.log(beta_params))

        # init steady_state_params
        steady_state_params = 0 * jnp.ones((1, 2))

        self.params = [jnp.vstack([step_varying_params, steady_state_params])]
        
        # sigmoid_inv(beta)

    def create_quad_prox_pep_sdp_layer(self, num_iters):     # def k_step_rate_subdiff(L, mu, step_sizes, proj=True):
        """
        ordering: x*, x_0,...,x_{p-1}, s*,s_0,..,s_{p-1}
        """
        proj = True
        step_sizes_param = cp.Parameter(num_iters)

        k = num_iters #step_sizes.size
        G = cp.Variable((2*(k+1)+2, 2*(k+1)+2), symmetric=True) # gram matrix
        H = cp.Variable(k+2)
        rate = cp.Variable()
        if not proj:
            F = cp.Variable(k+1)

        tol = 0
        constraints = [G >> tol*np.eye(2*(k+1)+2)]

        # denom
        constraints.append(G[0, 0] - 2*G[0,1] + G[1, 1] == 1)

        # num
        x_k_ind = get_x_index_subdiff(k, k)
        print('x_k_ind', x_k_ind)
        
        # subgradient inequalities: ∂h(x_i)^T (x_i - x_j) >= 0
        # (y_{i-1} - x_i)^T (x_i - x_j) >= 0 -- based on triples (x_i, y_{i-1}, x_j)
        x_list_subgrad, s_list_subgrad = collect_subgrad_subdiff_indices(k, 1)
        print('x_list', x_list_subgrad)
        print('s_list', s_list_subgrad)
        for i in range(len(x_list_subgrad)):
            x_i_ind = x_list_subgrad[i]
            s_i_ind = s_list_subgrad[i]
            if x_i_ind == 1:
                x_i_prev_ind = x_i_ind
                alpha_i = 1
            elif x_i_ind == 3:
                x_i_prev_ind = 0
                alpha_i = step_sizes_param[0]
            else:
                x_i_prev_ind = x_i_ind - 1
                alpha_i = step_sizes_param[i-1]
            for j in range(len(x_list_subgrad)):
                x_j_ind = x_list_subgrad[j]
                if x_j_ind != x_i_ind:
                    if proj:
                        # constraints.append(G[s_i_ind, x_i_ind] - G[s_i_ind, x_j_ind] >= 0)
                        constraints.append(G[x_i_prev_ind, x_i_ind] - G[x_i_prev_ind, x_j_ind] -G[x_i_ind, x_i_ind] +  G[x_i_ind, x_j_ind] -alpha_i*G[s_i_ind, x_i_ind] +alpha_i* G[s_i_ind, x_j_ind] >= 0)
                    else:
                        constraints.append((F[j] - F[i])  + G[s_i_ind, x_i_ind] - G[s_i_ind, x_j_ind] >= 0)

                    
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

        prob = cp.Problem(cp.Maximize(H[-1] - H[0]), constraints)
        cvxpylayer = CvxpyLayer(prob, parameters=[step_sizes_param], variables=[G, H])
        return cvxpylayer
    
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
            # import pdb
            # pdb.set_trace()

        # constraints.append(G <= 10)
        # constraints.append(G >= -10)
        prob = cp.Problem(cp.Maximize(H[-1] - H[0]), constraints)
        # step_sizes_param.value = np.ones(num_iters) / np.array(self.smooth_param)
        
        # t_vals = np.ones(num_iters+1)
        # t = 1
        # for i in range(1, num_iters+1):
        #     t = .5 * (1 + np.sqrt(1 + 4 * t ** 2))
        #     t_vals[i] = t
        # beta_vals = jnp.array(convert_t_to_beta(t_vals))[:10]
        
        beta_param.value = beta_vals #np.ones(num_iters)
        beta_param_sq.value = (beta_param.value)**2 #np.ones(num_iters)
        step_sizes_param.value = step_sizes
        
        try:
            prob.solve(verbose=True, solver=cp.CLARABEL)
            return prob.value
        except Exception as e:
            print('exception clarabel', e)
            return 0
            
        # import pdb
        # pdb.set_trace()
        # cvxpylayer = CvxpyLayer(prob, parameters=[step_sizes_param, beta_param, beta_param_sq], variables=[G, H])
        
        # step_sizes = jnp.ones(num_iters) / jnp.array(self.smooth_param)
        # G, H = cvxpylayer(step_sizes, beta_vals, beta_vals ** 2, solver_args={"solve_method": "CLARABEL", "verbose": True})
        # import pdb
        # pdb.set_trace()
        
    
    def create_nesterov_pep_sdp_layer(self, num_iters):     # def k_step_rate_subdiff(L, mu, step_sizes, proj=True):
        """
        ordering: x*, x_0,...,x_{p-1}, s*,s_0,..,s_{p-1}
        """
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
            # import pdb
            # pdb.set_trace()

        # constraints.append(G <= 1)
        # constraints.append(G >= -1)
        prob = cp.Problem(cp.Maximize(H[-1] - H[0]), constraints)
        step_sizes_param.value = np.ones(num_iters) / np.array(self.smooth_param)
        
        t_vals = np.ones(num_iters+1)
        t = 1
        for i in range(1, num_iters+1):
            t = .5 * (1 + np.sqrt(1 + 4 * t ** 2))
            t_vals[i] = t
        beta_vals = jnp.array(convert_t_to_beta(t_vals))[:10]
        
        # beta_param.value = beta_vals #np.ones(num_iters)
        # beta_param_sq.value = (beta_param.value)**2 #np.ones(num_iters)
        # prob.solve(verbose=True, solver=cp.CLARABEL)
        # import pdb
        # pdb.set_trace()
        cvxpylayer = CvxpyLayer(prob, parameters=[step_sizes_param, beta_param, beta_param_sq], variables=[G, H])
        
        # step_sizes = jnp.ones(num_iters) / jnp.array(self.smooth_param)
        # G, H = cvxpylayer(step_sizes, beta_vals, beta_vals ** 2, solver_args={"solve_method": "CLARABEL", "verbose": True})
        # import pdb
        # pdb.set_trace()
        return cvxpylayer


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
                    stochastic_params = stochastic_params.at[-1,0].set(1 / self.smooth_param)
                    stochastic_params = stochastic_params.at[-1,1].set(.9)
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
                    params = stochastic_params[:,:1]
                if diff_required:
                    if self.accel:
                        z_final, iter_losses = train_fn(k=iters,
                                                        z0=z0,
                                                        y0=z0,
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
        num_iters = params[:,0].size
        step_sizes = params[:,0]
        beta_vals = params[:,1]
        
        # Instantiate PEP
        problem = PEP()
        mu = 0
        L = self.smooth_param._value

        # Declare a strongly convex smooth function and a convex function
        f = problem.declare_function(SmoothStronglyConvexFunction, mu=mu, L=L)
        # f = problem.declare_function(SymmetricLinearOperator, mu=mu, L=L)
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

    def calculate_total_penalty(self, N_train, params, c, b, delta):
        return 0
        # priors are already rounded
        rounded_priors = params[2]

        # second: calculate the penalties
        num_groups = len(rounded_priors)
        pi_pen = jnp.log(jnp.pi ** 2 * num_groups * N_train / (6 * delta))
        log_pen = 0
        for i in range(num_groups):
            curr_lambd = jnp.clip(jnp.exp(rounded_priors[i]), a_max=c)
            log_pen += 2 * jnp.log(b * jnp.log((c+1e-6) / curr_lambd))

        # calculate the KL penalty
        penalty_loss = self.compute_all_params_KL(params[0], params[1],
                                                  rounded_priors) + pi_pen + log_pen
        return penalty_loss / N_train

    def compute_all_params_KL(self, mean_params, sigma_params, lambd):
        return 0
        # step size
        total_pen = compute_single_param_KL(
            mean_params, jnp.exp(sigma_params), jnp.exp(lambd[0]))

        # # threshold
        # total_pen += compute_single_param_KL(mean_params, jnp.exp(sigma_params), jnp.exp(lambd[1]))
        return total_pen

    def compute_weight_norm_squared(self, nn_params):
        return jnp.linalg.norm(nn_params) ** 2, nn_params.size

    def calculate_avg_posterior_var(self, params):
        return 0, 0
        sigma_params = params[1]
        flattened_params = jnp.concatenate([jnp.ravel(weight_matrix) for weight_matrix, _ in sigma_params] +
                                           [jnp.ravel(bias_vector) for _, bias_vector in sigma_params])
        variances = jnp.exp(flattened_params)
        avg_posterior_var = variances.mean()
        stddev_posterior_var = variances.std()
        return avg_posterior_var, stddev_posterior_var

def generate_yz_sequences(kappa, t):
    # intended to be t = log_2(K)
    y_vals = {1: 1 / kappa}
    z_vals = {1: 1 / kappa}
    print(y_vals, z_vals)

    # first generate z sequences
    for i in range(1, t+1):
        K = 2 ** i
        z_ihalf = z_vals[int(K / 2)]
        xi = 1 - z_ihalf
        z_i = z_ihalf * (xi + np.sqrt(1 + xi ** 2))
        z_vals[K] = z_i

    for i in range(1, t+1):
        K = 2 ** i
        # z_ihalf = z_vals[int(K / 2)]
        # xi = 1 - z_ihalf
        # yi = z_ihalf / (xi + np.sqrt(1 + xi ** 2))
        # y_vals[K] = yi
        zK = z_vals[K]
        zKhalf = z_vals[int(K // 2)]
        yK = zK - 2 * (zKhalf - zKhalf ** 2)
        y_vals[K] = yK

    # print(y_vals, z_vals)

    # print(z_vals[1], z_vals[2])
    # print((1 / kappa ** 2) / z_vals[2])
    return y_vals, z_vals

def compute_silver_steps(kappa, K):
    # assume K is a power of 2
    idx_vals = compute_silver_idx(kappa, K)
    y_vals, z_vals = generate_yz_sequences(kappa, int(np.log2(K)))

    def psi(t):
        return (1 + kappa * t) / (1 + t)

    # print(y_vals, z_vals)
    silver_steps = []
    for i in range(idx_vals.shape[0] - 1):
        idx = idx_vals[i]
        silver_steps.append(psi(y_vals[idx]))
    silver_steps.append(psi(z_vals[idx_vals[-1]]))
    print(silver_steps)

    return np.array(silver_steps)

def compute_silver_idx(kappa, K):
    two_adics = compute_shifted_2adics(K)
    # print(two_adics)
    idx_vals = np.power(2, two_adics)

    # if np.ceil(np.log2(K)) == np.floor(np.log2(K)):
    last_pow2 = int(np.floor(np.log2(K)))
    # print(last_pow2)
    idx_vals[(2 ** last_pow2) - 1] /= 2
    print('a_idx:', idx_vals)
    return idx_vals

def compute_shifted_2adics(K):
    return np.array([(k & -k).bit_length() for k in range(1, K+1)])

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

