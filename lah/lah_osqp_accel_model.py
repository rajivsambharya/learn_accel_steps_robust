from functools import partial

import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import osqp
from scipy.sparse import csc_matrix

from lah.algo_steps_osqp import k_steps_eval_lah_osqp, k_steps_train_lah_osqp, unvec_symm, k_steps_eval_osqp, k_steps_train_osqp
from lah.l2o_model import L2Omodel


class LAHAccelOSQPmodel(L2Omodel):
    def __init__(self, **kwargs):
        super(LAHAccelOSQPmodel, self).__init__(**kwargs)

    def initialize_algo(self, input_dict):
        # self.m, self.n = self.A.shape
        self.algo = 'lah_osqp'
        self.m, self.n = input_dict['m'], input_dict['n']
        self.q_mat_train, self.q_mat_test = input_dict['q_mat_train'], input_dict['q_mat_test']

        self.rho = input_dict['rho']
        self.sigma = input_dict.get('sigma', 1)
        self.alpha = input_dict.get('alpha', 1)
        self.output_size = self.n + self.m

        # custom_loss
        custom_loss = input_dict.get('custom_loss')

        self.num_const_steps = input_dict.get('num_const_steps')
        
        self.num_const_steps = 1
        self.idx_mapping = jnp.arange(self.eval_unrolls) // self.num_const_steps


        """
        break into the 2 cases
        1. factors are the same for each problem (i.e. matrices A and P don't change)
        2. factors change for each problem
        """
        self.factors_required = True
        self.factor_static_bool = input_dict.get('factor_static_bool', True)

        self.k_steps_train_fn = partial(
            k_steps_train_lah_osqp, idx_mapping=self.idx_mapping, jit=self.jit)
        self.k_steps_eval_fn = partial(k_steps_eval_lah_osqp,
                                        idx_mapping=self.idx_mapping,
                                        jit=self.jit, 
                                        custom_loss=custom_loss)
        self.lah = True
        m, n = self.m, self.n
        l0 = self.q_mat_train[0, n: n + m]
        u0 = self.q_mat_train[0, n + m: n + 2 * m]
        self.eq_ind = l0 == u0
        # if self.factor_static_bool:
        #     self.A = input_dict['A']
        #     self.P = input_dict['P']
        #     # self.M = self.P + sigma * jnp.eye(n) + A.T @ jnp.diag(rho_vec) @ A
        #     self.factor_static = input_dict['factor']
        #     self.k_steps_train_fn = partial(
        #         k_steps_train_lah_osqp, A=self.A, idx_mapping=self.idx_mapping, jit=self.jit)
        #     self.k_steps_eval_fn = partial(k_steps_eval_lah_osqp, P=self.P,
        #                                    A=self.A, 
        #                                    idx_mapping=self.idx_mapping,
        #                                    jit=self.jit, 
        #                                    custom_loss=custom_loss)
        # else:
        #     self.k_steps_train_fn = self.create_k_steps_train_fn_dynamic()
        #     self.k_steps_eval_fn = self.create_k_steps_eval_fn_dynamic()
            # self.k_steps_eval_fn = partial(k_steps_eval_osqp, rho=rho, sigma=sigma, jit=self.jit)

            # self.factors_train = input_dict['factors_train']
            # self.factors_test = input_dict['factors_test']

            

        # self.k_steps_train_fn = partial(k_steps_train_osqp, factor=factor, A=self.A, rho=rho, 
        #                                 sigma=sigma, jit=self.jit)
        # self.k_steps_eval_fn = partial(k_steps_eval_osqp, factor=factor, P=self.P, A=self.A, 
        #                                rho=rho, sigma=sigma, jit=self.jit)
        self.out_axes_length = 6
        N = self.q_mat_train.shape[0]
        self.lah_train_inputs = jnp.zeros((N, self.n + self.m))


    def init_params(self):
        # init step-varying params
        step_varying_params = jnp.zeros((self.step_varying_num, 5))
        # step_varying_params = step_varying_params.at[:,1].set(-2)
        step_varying_params = step_varying_params.at[:,3].set(-2)
        step_varying_params = step_varying_params.at[:,4].set(5)
        # step_varying_params = step_varying_params.at[:,0].set(-8)

        # init steady_state_params
        steady_state_params = jnp.zeros((1, 5))
        steady_state_params = steady_state_params.at[:,3].set(-10)
        steady_state_params = steady_state_params.at[:,4].set(5)
        # steady_state_params = steady_state_params.at[:,1].set(-2)

        self.params = [jnp.vstack([step_varying_params, steady_state_params])]
        # self.mean_params = jnp.ones((self.train_unrolls, 3))

        # self.sigma_params = -jnp.ones((self.train_unrolls, 3)) * 10

        # # initialize the prior
        # self.prior_param = jnp.log(self.init_var) * jnp.ones(2)

        # self.params = [self.mean_params, self.sigma_params, self.prior_param]


    def create_end2end_loss_fn(self, bypass_nn, diff_required):
        supervised = False # self.supervised and diff_required
        loss_method = self.loss_method

        def predict(params, input, q, iters, z_star, key, factor):
            
            # z0 = jnp.zeros(self.m + self.n) #self.predict_warm_start(params, input, key, bypass_nn)
            z0 = input

            if diff_required:
                n_iters = key #self.train_unrolls if key else 1
            else:
                n_iters = min(iters, 51)

            if self.train_fn is not None:
                train_fn = self.train_fn
            else:
                train_fn = self.k_steps_train_fn
            if self.eval_fn is not None:
                eval_fn = self.eval_fn
            else:
                eval_fn = self.k_steps_eval_fn

            m, n = self.m, self.n
            nc2 = int(n * (n + 1) / 2)
            P = unvec_symm(q[2 * m + n: 2 * m + n + nc2], n)
            A = jnp.reshape(q[2 * m + n + nc2:], (m, n))
            q_bar = q[:2 * m + n]

            # do all of the factorizations
            factors1 = jnp.zeros((n_iters, self.n, self.n))
            factors2 = jnp.zeros((n_iters, self.n), dtype=jnp.int32)
            rhos, sigmas, rho_eqs = params[0][:, 0], params[0][:, 1], params[0][:, 4]
            rho, sigma, rho_eq = jnp.exp(rhos[0]), jnp.exp(sigmas[0]), jnp.exp(rho_eqs[0])
            rho_vec = rho * jnp.ones(self.m)
            
            rho_vec = rho_vec.at[self.eq_ind].set(rho_eq)
            M = P + sigma * jnp.eye(self.n) + A.T @ jnp.diag(rho_vec) @ A
            factor = jsp.linalg.lu_factor(M)
            # for i in range(n_iters):
            #     factors1 = factors1.at[i, :, :].set(factor[0])
            #     factors2 = factors2.at[i, :].set(factor[1])
            factors1 = factors1.at[0, :, :].set(factor[0])
            factors2 = factors2.at[0, :].set(factor[1])
                
            all_factors = factors1, factors2
            params[0] = params[0].at[-1,3].set(-10)
            params[0] = params[0].at[-1,2].set(0)
            
            osqp_params = (params[0], all_factors, rho_vec)


            if diff_required:
                z_final, iter_losses = train_fn(k=iters,
                                                z0=z0,
                                                q=q_bar,
                                                A=A,
                                                P=P,
                                                params=osqp_params,
                                                supervised=supervised,
                                                z_star=z_star) #,
                                                #factor=factor)
            else:
                eval_out = eval_fn(k=iters,
                                    z0=z0,
                                    q=q_bar,
                                    A=A,
                                    P=P,
                                    params=osqp_params,
                                    supervised=supervised,
                                    z_star=z_star)
                z_final, iter_losses, z_all_plus_1 = eval_out[0], eval_out[1], eval_out[2]

                angles = None

            loss = self.final_loss(loss_method, z_final, iter_losses, supervised, z0, z_star)

            # penalty_loss = calculate_total_penalty(self.N_train, params, self.b, self.c, self.delta)
            # penalty_loss = calculate_pinsker_penalty(self.N_train, params, self.b, self.c, self.delta)
            loss = loss #+ self.penalty_coeff * penalty_loss

            if diff_required:
                return loss
            else:
                return_out = (loss, iter_losses, z_all_plus_1, angles) + eval_out[3:]
                return return_out
        loss_fn = self.predict_2_loss(predict, diff_required)
        return loss_fn


    def create_k_steps_train_fn_dynamic(self):
        """
        creates the self.k_steps_train_fn function for the dynamic case
        acts as a wrapper around the k_steps_train_osqp functino from algo_steps.py

        we want to maintain the argument inputs as (k, z0, q_bar, factor, supervised, z_star)
        """
        m, n = self.m, self.n

        def k_steps_train_osqp_dynamic(k, z0, q, factor, supervised, z_star):
            nc2 = int(n * (n + 1) / 2)
            q_bar = q[:2 * m + n]
            unvec_symm(q[2 * m + n: 2 * m + n + nc2], n)
            A = jnp.reshape(q[2 * m + n + nc2:], (m, n))
            return k_steps_train_osqp(k=k, z0=z0, q=q_bar,
                                      factor=factor, A=A, rho=self.rho, sigma=self.sigma,
                                      supervised=supervised, z_star=z_star, jit=self.jit)
        return k_steps_train_osqp_dynamic

    def create_k_steps_eval_fn_dynamic(self):
        """
        creates the self.k_steps_train_fn function for the dynamic case
        acts as a wrapper around the k_steps_train_osqp functino from algo_steps.py

        we want to maintain the argument inputs as (k, z0, q_bar, factor, supervised, z_star)
        """
        m, n = self.m, self.n

        def k_steps_eval_osqp_dynamic(k, z0, q, factor, supervised, z_star):
            nc2 = int(n * (n + 1) / 2)
            q_bar = q[:2 * m + n]
            P = unvec_symm(q[2 * m + n: 2 * m + n + nc2], n)
            A = jnp.reshape(q[2 * m + n + nc2:], (m, n))
            return k_steps_eval_osqp(k=k, z0=z0, q=q_bar,
                                     factor=factor, P=P, A=A, rho=self.rho, sigma=self.sigma,
                                     supervised=supervised, z_star=z_star, jit=self.jit)
        return k_steps_eval_osqp_dynamic
