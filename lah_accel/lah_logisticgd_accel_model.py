from functools import partial

import jax.numpy as jnp
from jax import random, vmap

import numpy as np

from lah_accel.algo_steps_logistic import k_steps_eval_lah_nesterov_gd, k_steps_train_lah_nesterov_gd, k_steps_eval_nesterov_logisticgd, compute_gradient, k_steps_eval_adam_logistic, k_steps_eval_adagrad_logistic, k_steps_eval_logistic_backtracking, k_steps_eval_logistic_bb
from lah_accel.l2o_model import L2Omodel
from PEPit.functions import SmoothStronglyConvexFunction
import cvxpy as cp
from cvxpylayers.jax import CvxpyLayer
from jax import lax
from lah_accel.pep import create_nesterov_pep_sdp_layer, build_A_matrix_with_xstar, pepit_nesterov


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
        # self.adam_eval_fn = partial(k_steps_eval_adam_logistic, num_points=num_points, step_size=0.01, beta1=0.9, beta2=0.99, epsilon=1e-8,
        #                                jit=self.jit)
        self.adam_eval_fn = partial(k_steps_eval_adagrad_logistic, num_points=num_points, step_size=0.1,
                                       jit=self.jit)
        # self.adam_eval_fn = partial(k_steps_eval_logistic_backtracking, num_points=num_points, eta0=1.0,
        #                                jit=self.jit)
        # self.bb_eval_fn = partial(k_steps_eval_logistic_bb, num_points=num_points,
        #                           eta0=1.0 / self.smooth_param, jit=self.jit)
        self.bb_eval_fn = partial(k_steps_eval_logistic_bb, num_points=num_points,
                                  eta0=0.001, jit=self.jit)

        self.out_axes_length = 5

        N = self.q_mat_train.shape[0]
        self.lah_train_inputs = jnp.zeros((N, num_weights))

        e2e_loss_fn = self.create_end2end_loss_fn
        
        self.num_pep_iters = self.train_unrolls

        # end-to-end loss fn for silver evaluation
        self.loss_fn_eval_silver = e2e_loss_fn(bypass_nn=False, diff_required=False, 
                                               special_algo='silver')
        self.loss_fn_eval_adam = e2e_loss_fn(bypass_nn=False, diff_required=False,
                                               special_algo='adam')
        self.loss_fn_eval_bb = e2e_loss_fn(bypass_nn=False, diff_required=False,
                                            special_algo='bb')

        self.num_const_steps = input_dict.get('num_const_steps', 1)

        self.num_points = num_points
        
        self.pep_layer = create_nesterov_pep_sdp_layer(self.smooth_param, self.num_pep_iters)
        
        
    def pepit_nesterov_check(self, params):
        return pepit_nesterov(0, self.smooth_param._value, params)


    def nesterov_base_params(self, num_iters):
        """theta_base: standard (non strongly convex) Nesterov's method, step size 1/L,
        with a certified PEP guarantee (Proposition 1)."""
        step_sizes = (1.0 / self.smooth_param) * jnp.ones(num_iters)

        t_vals = [1.0]
        t = 1.0
        for _ in range(num_iters):
            t = (1.0 + np.sqrt(1.0 + 4.0 * t ** 2)) / 2.0
            t_vals.append(t)
        beta_vals = convert_t_to_beta(jnp.array(t_vals))[:num_iters]

        return jnp.stack([step_sizes, beta_vals], axis=1)


    def robustness_polish(self, num_iters=None, gamma_target=None, tol=1e-3, verbose=True):
        """
        Robustness polishing step (Section 4.3): bisect tau in theta(tau) =
        tau * theta_learned + (1 - tau) * theta_base until the PEP-certified
        worst-case gamma(theta(tau)) <= gamma_target, then set theta_final =
        theta(tau_lo). Since tau_lo is only ever advanced to certified values,
        starting from the certified theta(0) = theta_base, the result satisfies
        gamma(theta_final) <= gamma_target regardless of gamma(theta_learned).
        Mutates self.params[0] in place (log-parameterized) and returns
        (tau_final, gamma_final).
        """
        if num_iters is None:
            num_iters = self.num_pep_iters
        if gamma_target is None:
            gamma_target = self.pep_target
        if gamma_target is None:
            raise ValueError("robustness_polish requires pep_target to be set")

        theta_base = self.nesterov_base_params(num_iters)
        theta_learned = jnp.exp(self.params[0][:num_iters, :])

        gamma_learned = self.pepit_nesterov_check(np.array(theta_learned))
        if verbose:
            print(f"robustness polish: gamma(theta_learned)={gamma_learned:.6g}, "
                  f"target={gamma_target:.6g}")
        if gamma_learned <= gamma_target:
            # already certified; leave theta_learned untouched
            return 1.0, gamma_learned

        gamma_base = self.pepit_nesterov_check(np.array(theta_base))
        if gamma_base > gamma_target:
            raise RuntimeError(
                f"theta_base does not certify at target={gamma_target}: "
                f"gamma(theta_base)={gamma_base}")

        tau_lo, tau_hi = 0.0, 1.0
        gamma_lo = gamma_base
        num_bisect = int(np.ceil(np.log2(1.0 / tol)))
        for _ in range(num_bisect):
            tau_mid = 0.5 * (tau_lo + tau_hi)
            theta_mid = tau_mid * theta_learned + (1 - tau_mid) * theta_base
            gamma_mid = self.pepit_nesterov_check(np.array(theta_mid))
            if verbose:
                print(f"robustness polish: tau={tau_mid:.6f} gamma={gamma_mid:.6g}")
            if gamma_mid <= gamma_target:
                tau_lo, gamma_lo = tau_mid, gamma_mid
            else:
                tau_hi = tau_mid

        theta_final = tau_lo * theta_learned + (1 - tau_lo) * theta_base
        self.params[0] = self.params[0].at[:num_iters, :].set(
            jnp.log(jnp.clip(theta_final, a_min=1e-12)))

        if verbose:
            print(f"robustness polish: tau_final={tau_lo:.6f} gamma_final={gamma_lo:.6g} "
                  f"(target={gamma_target:.6g})")

        return tau_lo, gamma_lo


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
        # transformed_params = jnp.zeros((n_iters, 2))
        # transformed_params = jnp.clip(transformed_params.at[:, :].set(jnp.exp(params[0][:, :])), a_max=50000)
        # # transformed_params = transformed_params.at[n_iters - 1, :].set(2 / self.smooth_param * sigmoid(params[0][n_iters - 1, :]))
        # # transformed_params = transformed_params.at[n_iters - 1, 0].set(jnp.exp(params[0][n_iters - 1, 0]))
        # return transformed_params
        return jnp.exp(params[0][:, :])

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
        k = self.eval_unrolls
        
        # step_varying_params = jnp.log(1 / (self.smooth_param)) * jnp.ones((self.step_varying_num, 2))
        step_varying_params = jnp.log(1 / (self.smooth_param)) * jnp.ones((k, 2))
        
        t_params = jnp.ones(k)
        t = 1
        for i in range(0, k):
            t = i
            t_params = t_params.at[i].set(t/(t+3)) #(jnp.log(t))

        step_varying_params = step_varying_params.at[:, 1].set(jnp.log(t_params))

        # init steady_state_params
        steady_state_params = 0 * jnp.ones((1, 2)) #sigmoid_inv(0) * jnp.ones((1, 1))
        self.params = [jnp.vstack([step_varying_params, steady_state_params])]
        
        
    def pep_cvxpylayer(self, params):
        step_sizes = params[:self.num_pep_iters,0]
        momentum_sizes = params[:self.num_pep_iters,1]
        
        A_param = build_A_matrix_with_xstar(step_sizes, momentum_sizes)
        G, H = self.pep_layer(A_param, solver_args={"solve_method": "SCS", "verbose": True})

        return H[-2] - H[-1]


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
                    
                    # stochastic_params = stochastic_params.at[:20,0].set(1/self.smooth_param)
                    # beta = jnp.array([t / (t+3) for t in range(20)])
                    # stochastic_params = stochastic_params.at[:20,1].set(beta)
                    # stochastic_params = stochastic_params.at[-1,0].set(1 / self.smooth_param)
                    # stochastic_params = stochastic_params.at[-1,1].set(.9)
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
            elif special_algo == 'adam':

                eval_out = self.adam_eval_fn(k=iters,
                                   z0=z0,
                                   q=q,
                                   supervised=supervised,
                                   z_star=z_star)
                z_final, iter_losses, z_all_plus_1 = eval_out[0], eval_out[1], eval_out[2]
                angles = None
            elif special_algo == 'bb':
                eval_out = self.bb_eval_fn(k=iters,
                                           z0=z0,
                                           q=q,
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
