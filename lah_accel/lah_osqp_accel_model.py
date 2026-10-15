from functools import partial

import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import osqp
from scipy.sparse import csc_matrix

from lah_accel.algo_steps_osqp import k_steps_eval_lah_osqp, k_steps_train_lah_osqp, unvec_symm, k_steps_eval_osqp, k_steps_train_osqp
from lah_accel.l2o_model import L2Omodel
from lah_accel.pep import create_admm_pep_sdp_layer, pepit_fixed_point, build_A_matrix_op_with_xstar, PEP_SOLVE_TIMEOUT, PEP_POLISH_TIMEOUT


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
        # predict builds the factorization itself from each problem's P and A
        # (its rho is learned), so the model needs no precomputed factors and
        # must stay on L2Omodel's static-factor path (factor=None) -- the
        # dynamic path expects factors_train, which this model never has
        self.factors_required = False
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
        self.num_pep_iters = self.train_unrolls
        self.pep_layer = create_admm_pep_sdp_layer(self.num_pep_iters)


    def init_params(self):
        # init step-varying params
        step_varying_params = jnp.zeros((self.step_varying_num, 5))
        # step_varying_params = step_varying_params.at[:,3].set(-20)
        step_varying_params = step_varying_params.at[:,2].set(1)
        step_varying_params = step_varying_params.at[:,3].set(0)
        # step_varying_params = step_varying_params.at[:,4].set(0)
        # step_varying_params = step_varying_params.at[:,4].set(jnp.log(100))
        # step_varying_params = step_varying_params.at[:,0].set(jnp.log(1e-6))

        # init steady_state_params
        steady_state_params = jnp.zeros((1, 5))
        # steady_state_params = steady_state_params.at[:,3].set(-20)
        steady_state_params = steady_state_params.at[:,2].set(1)
        steady_state_params = steady_state_params.at[:,3].set(0)


        self.params = [jnp.vstack([step_varying_params, steady_state_params])]
        # self.mean_params = jnp.ones((self.train_unrolls, 3))

        # self.sigma_params = -jnp.ones((self.train_unrolls, 3)) * 10

        # # initialize the prior
        # self.prior_param = jnp.log(self.init_var) * jnp.ones(2)

        # self.params = [self.mean_params, self.sigma_params, self.prior_param]

    def pep_cvxpylayer(self, params):
        step_sizes = jnp.log(params[:self.num_pep_iters,2])
        momentum_sizes = jnp.log(params[:self.num_pep_iters,3])
        
        A_param = build_A_matrix_op_with_xstar(step_sizes, momentum_sizes)
        G, = self.pep_layer(A_param, solver_args={"solve_method": "SCS", "verbose": True})
        k = self.num_pep_iters


        return G[2*k+2, 2*k+2] - 2*G[2*k+2, k+2] + G[k+2, k+2]
    
    def pepit_nesterov_check(self, params, timeout_seconds=PEP_SOLVE_TIMEOUT):
        # params is exp(self.params[0]); columns 2 (alpha) and 3 (beta) are
        # stored raw, so take the log to undo the exp
        fp_params = np.vstack([jnp.log(params[:, 2]), jnp.log(params[:, 3])]).T
        return pepit_fixed_point(fp_params, timeout_seconds=timeout_seconds)

    def drs_base_params(self, num_iters):
        """
        theta_base: plain admm (relaxation alpha = 1, i.e. averaged step 1/2 on
        the nonexpansive reflection) with no momentum. a
        krasnoselskii-mann iteration, so its fixed-point residual is certified
        to decrease from the first step. columns are (alpha, beta).
        """
        return jnp.stack([jnp.ones(num_iters), jnp.zeros(num_iters)], axis=1)

    def robustness_polish(self, num_iters=None, gamma_target=None, tol=1e-3, verbose=True):
        """
        Robustness polishing step (Section 4.3): bisect tau in theta(tau) =
        tau * theta_learned + (1 - tau) * theta_base until the PEP-certified
        worst-case gamma(theta(tau)) <= gamma_target, then set theta_final =
        theta(tau_lo). tau_lo only ever advances to certified values, starting
        from theta(0) = theta_base, so the result satisfies the target whatever
        gamma(theta_learned) is. Mutates self.params[0] in place and returns
        (tau_final, gamma_final).

        theta = (alpha, beta) are columns 2 and 3 of self.params[0], which are
        stored raw (not log-parameterized). the rho / sigma columns set the
        single factorization, i.e. the operator itself, so the polish leaves
        them alone.

        If theta_base itself misses the target this does not raise -- it warns
        and keeps whichever endpoint (theta_learned or theta_base) has the
        smaller gamma. Raising would throw away a completed training run at the
        very last step.
        """
        if num_iters is None:
            num_iters = self.num_pep_iters
        if gamma_target is None:
            gamma_target = self.pep_target
        if gamma_target is None:
            raise ValueError("robustness_polish requires pep_target to be set")

        theta_cols = jnp.array([2, 3])
        theta_base = self.drs_base_params(num_iters)
        theta_learned = self.params[0][:num_iters, theta_cols]

        def gamma(theta):
            # a failed or timed-out solve comes back as inf and is scored as
            # uncertified, which is safe but shrinks the bracket below the
            # target -- so the polish solves get a long cap
            return pepit_fixed_point(np.array(theta), timeout_seconds=PEP_POLISH_TIMEOUT)

        def commit(theta):
            self.params[0] = self.params[0].at[:num_iters, theta_cols].set(theta)

        gamma_learned = gamma(theta_learned)
        if verbose:
            print(f"robustness polish: gamma(theta_learned)={gamma_learned:.6g}, "
                  f"target={gamma_target:.6g}")
        if gamma_learned <= gamma_target:
            # already certified; leave theta_learned untouched
            return 1.0, gamma_learned

        gamma_base = gamma(theta_base)
        if gamma_base > gamma_target:
            # neither endpoint certifies, so bisection has nothing to anchor on;
            # keep whichever endpoint has the smaller worst-case bound
            if gamma_learned <= gamma_base:
                tau_final, gamma_final = 1.0, gamma_learned
            else:
                tau_final, gamma_final = 0.0, gamma_base
                commit(theta_base)
            print(f"robustness polish: WARNING theta_base does not certify at "
                  f"target={gamma_target:.6g} (gamma(theta_base)={gamma_base:.6g}); "
                  f"keeping {'theta_learned' if tau_final == 1.0 else 'theta_base'} "
                  f"(gamma={gamma_final:.6g}), the more robust endpoint. "
                  f"raise pep_target to at least {min(gamma_learned, gamma_base):.6g} "
                  f"to make the target reachable.")
            return tau_final, gamma_final

        tau_lo, tau_hi = 0.0, 1.0
        gamma_lo = gamma_base
        for _ in range(int(np.ceil(np.log2(1.0 / tol)))):
            tau_mid = 0.5 * (tau_lo + tau_hi)
            theta_mid = tau_mid * theta_learned + (1 - tau_mid) * theta_base
            gamma_mid = gamma(theta_mid)
            if verbose:
                print(f"robustness polish: tau={tau_mid:.6f} gamma={gamma_mid:.6g}")
            if gamma_mid <= gamma_target:
                tau_lo, gamma_lo = tau_mid, gamma_mid
            else:
                tau_hi = tau_mid

        commit(tau_lo * theta_learned + (1 - tau_lo) * theta_base)
        if verbose:
            print(f"robustness polish: tau_final={tau_lo:.6f} gamma_final={gamma_lo:.6g} "
                  f"(target={gamma_target:.6g})")
        return tau_lo, gamma_lo

    def transform_params(self, params, n_iters):
        transformed_params = jnp.exp(params[0][:n_iters, :])
        transformed_params = transformed_params.at[:,2].set(jnp.log(transformed_params[:,2]))
        transformed_params = transformed_params.at[:,3].set(jnp.log(transformed_params[:,3]))
        return transformed_params
    
    def perturb_params(self):
        step_varying_params = self.params[0][:-1, :]
        steady_state_params = self.params[0][-1:, :]
        step_varying_params = step_varying_params.at[:,3].set(-2)
        self.params = [jnp.vstack([step_varying_params, steady_state_params])]


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
            # params[0] = params[0].at[-1,3].set(-10)
            # params[0] = params[0].at[-1,2].set(0)
            
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
