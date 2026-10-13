from functools import partial

import cvxpy as cp
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import scs
from scipy.sparse import csc_matrix
import jax

from lah_accel.algo_steps import (
    create_M,
    get_scaled_vec_and_factor,
    k_steps_eval_lah_scs,
    k_steps_train_lah_scs,
    lin_sys_solve,
)
from lah_accel.algo_steps_scs import k_steps_eval_lah_accel_scs, k_steps_train_lah_accel_scs
from lah_accel.l2o_model import L2Omodel
from lah_accel.pep import create_admm_pep_sdp_layer, pepit_fixed_point, build_A_matrix_op_with_xstar, PEP_SOLVE_TIMEOUT, PEP_POLISH_TIMEOUT


class LAHAccelSCSmodel(L2Omodel):
    def __init__(self, **kwargs):
        super(LAHAccelSCSmodel, self).__init__(**kwargs)

    def initialize_algo(self, input_dict):
        """
        the input_dict is required to contain these keys
        otherwise there is an error
        """
        self.factors_required = True
        self.factor_static_bool = input_dict.get('factor_static_bool', True)
        self.algo = 'lah_scs'
        self.factors_required = True
        self.hsde = input_dict.get('hsde', True)
        self.m, self.n = input_dict['m'], input_dict['n']
        self.cones = input_dict['cones']
        self.proj, self.static_flag = input_dict['proj'], input_dict.get('static_flag', True)
        self.q_mat_train, self.q_mat_test = input_dict['q_mat_train'], input_dict['q_mat_test']

        M = input_dict['static_M']
        self.P = M[:self.n, :self.n]
        self.A = -M[self.n:, :self.n]
        self.M = M

        # not a hyperparameter, but used for scale knob
        self.zero_cone_size = self.cones['z'] #input_dict['zero_cone_size']
        lightweight = input_dict.get('lightweight', False)

        self.output_size = self.n + self.m
        self.out_axes_length = 9

        self.num_const_steps = input_dict.get('num_const_steps')
        self.idx_mapping = jnp.arange(self.eval_unrolls) // self.num_const_steps

        # custom_loss
        custom_loss = input_dict.get('custom_loss')

        self.k_steps_train_fn = partial(k_steps_train_lah_accel_scs, proj=self.proj,
                                        jit=self.jit,
                                        P=self.P,
                                        A=self.A,
                                        idx_mapping=self.idx_mapping,
                                        hsde=True)
        self.k_steps_train_fn2 = partial(k_steps_train_lah_accel_scs, proj=self.proj,
                                        jit=self.jit,
                                        P=self.P,
                                        A=self.A,
                                        idx_mapping=self.idx_mapping,
                                        hsde=False)
        self.k_steps_eval_fn = partial(k_steps_eval_lah_accel_scs, proj=self.proj,
                                       P=self.P, A=self.A,
                                       idx_mapping=self.idx_mapping,
                                       zero_cone_size=self.zero_cone_size,
                                       jit=self.jit,
                                       hsde=True,
                                       custom_loss=custom_loss,
                                       lightweight=lightweight)
        self.num_pep_iters = self.train_unrolls
        self.pep_layer = create_admm_pep_sdp_layer(self.num_pep_iters)
    
    def transform_params(self, params, n_iters):
        transformed_params = jnp.exp(params[0][:n_iters, :])
        transformed_params = transformed_params.at[:,4].set(-jnp.log(transformed_params[:,4]))
        transformed_params = transformed_params.at[:,2].set(jnp.log(transformed_params[:,2]))
        return transformed_params
        

    def pep_cvxpylayer(self, params):
        step_sizes = jnp.log(params[:self.num_pep_iters,2])
        momentum_sizes = jnp.log(params[:self.num_pep_iters,4])
        
        
        A_param = build_A_matrix_op_with_xstar(step_sizes, momentum_sizes)
        G, = self.pep_layer(A_param, solver_args={"solve_method": "SCS", "verbose": True})
        k = self.num_pep_iters
        # import pdb
        # pdb.set_trace()

        return G[2*k+2, 2*k+2] - 2*G[2*k+2, k+2] + G[k+2, k+2]

    def pepit_nesterov_check(self, params, timeout_seconds=PEP_SOLVE_TIMEOUT):
        # params is exp(self.params[0]); columns 2 (alpha) and 4 (beta) are
        # stored raw, so take the log to undo the exp
        fp_params = np.vstack([jnp.log(params[:, 2]), jnp.log(params[:, 4])]).T
        return pepit_fixed_point(fp_params, timeout_seconds=timeout_seconds)

    def drs_base_params(self, num_iters):
        """
        theta_base: plain douglas-rachford (relaxation alpha = 1, i.e. averaged
        step 1/2 on the nonexpansive reflection) with no momentum. a
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

        theta = (alpha, beta) are columns 2 and 4 of self.params[0], which are
        stored raw (not log-parameterized). the rho columns set the single
        factorization, i.e. the operator itself, so the polish leaves them alone.

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

        theta_cols = jnp.array([2, 4])
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
        
        
    def init_params(self):
        self.mean_params = 0*jnp.ones((self.eval_unrolls, 6))
        # self.mean_params = self.mean_params.at[:, 2].set(jnp.log(1.0))
        # self.mean_params = self.mean_params.at[:, 4].set(-2.0)
        
        # do alpha
        alpha = 1 #jnp.log(1) #jnp.log(.0001)
        self.mean_params = self.mean_params.at[:, 2].set(alpha)
        
        # do beta
        beta = 0.0 * jnp.ones(self.eval_unrolls)
        # beta = jnp.array(np.random.normal(size=(self.eval_unrolls)))
        # t = 1
        # for i in range(0, self.eval_unrolls):
        #     beta = beta.at[i].set(1/(i+2)) #(jnp.log(t))

        # beta = beta.at[:, 1].set(jnp.log(t_params))
        self.mean_params = self.mean_params.at[:, 4].set(beta)
        # self.mean_params = self.mean_params.at[:, 4].set(jnp.log(beta))
        # self.mean_params = self.mean_params.at[-1, 4].set(0)
        self.params = [self.mean_params]

    # def setup_optimal_solutions(self, dict):
    def setup_optimal_solutions(self, 
                                z_stars_train, 
                                z_stars_test, 
                                x_stars_train=None, 
                                x_stars_test=None, 
                                y_stars_train=None, 
                                y_stars_test=None):
        if x_stars_train is not None:
            self.z_stars_train = jnp.array(z_stars_train)
            self.z_stars_test = jnp.array(z_stars_test)
            self.x_stars_train = jnp.array(x_stars_train)
            self.x_stars_test = jnp.array(x_stars_test)
            self.y_stars_train = jnp.array(y_stars_train)
            self.y_stars_test = jnp.array(y_stars_test)
            self.u_stars_train = jnp.hstack([self.x_stars_train, self.y_stars_train])
            self.u_stars_test = jnp.hstack([self.x_stars_test, self.y_stars_test])
        if z_stars_train is not None:
            self.z_stars_train = z_stars_train
            self.z_stars_test = z_stars_test
        else:
            self.z_stars_train, self.z_stars_test = None, None
        self.lah_train_inputs = jnp.hstack([0 * self.z_stars_train, jnp.ones((self.z_stars_train.shape[0], 1))])


    def create_end2end_loss_fn(self, bypass_nn, diff_required):
        supervised = self.supervised and diff_required
        loss_method = self.loss_method

        def predict(params, input, q, iters, z_star, key, factor):
            if diff_required:
                n_iters = key #self.train_unrolls if key else 1
            else:
                n_iters = min(iters, self.step_varying_num + 1)
            # if diff_required:
            #     z0 = input
            # else:
            #     z0 = input + 100
            z0 = input
            # z0 = z0.at[:200].set(100)

            if self.train_fn is not None:
                train_fn = self.train_fn
            else:
                train_fn = self.k_steps_train_fn
            if self.eval_fn is not None:
                eval_fn = self.eval_fn
            else:
                eval_fn = self.k_steps_eval_fn

            # do all of the factorizations
            factors1 = jnp.zeros((n_iters, self.m + self.n, self.m + self.n))
            factors2 = jnp.zeros((n_iters, self.m + self.n), dtype=jnp.int32)
            scaled_vecs = jnp.zeros((n_iters, self.m + self.n))

            rho_xs, rho_ys, rho_ys_zero = params[0][:, 0], params[0][:, 1], params[0][:, 3]
            
            rho_x, rho_y = jnp.exp(rho_xs[0]), jnp.exp(rho_ys[0])
            rho_y_zero = jnp.exp(rho_ys_zero[0])
            factor, scale_vec = get_scaled_vec_and_factor(self.M, rho_x, rho_y, 
                                                              rho_x,
                                                              self.m, self.n, 
                                                              zero_cone_size=self.zero_cone_size, 
                                                              hsde=False)

            
            # for i in range(n_iters):
            #     # rho_x = jnp.exp(rho_xs[0])
            #     # rho_y = jnp.exp(rho_ys[0])
            #     rho_x, rho_y = jnp.exp(rho_xs[0]), jnp.exp(rho_ys[0])
            #     rho_y_zero = jnp.exp(rho_ys_zero[0])
                
            #     factor, scale_vec = get_scaled_vec_and_factor(self.M, rho_x, rho_y, 
            #                                                   rho_x,
            #                                                   self.m, self.n, 
            #                                                   zero_cone_size=self.zero_cone_size, 
            #                                                   hsde=False)
                
            #     factors1 = factors1.at[i, :, :].set(factor[0])
            #     factors2 = factors2.at[i, :].set(factor[1])
            #     scaled_vecs = scaled_vecs.at[i, :].set(scale_vec)
            #     print('factor', scale_vec)

            # params[0] = params[0].at[-1, 2].set(0)
            # params[0] = params[0].at[:, 2].set(0)
            # params[0] = params[0].at[-1, 4].set(-10)
            # all_factors = factors1, factors2
            # scs_params = (params[0][:n_iters, :], all_factors, scaled_vecs)
            # scs_params = (params[0][:, :], all_factors, scaled_vecs)
            scs_params = (params[0][:, :], factor, scale_vec)

            if diff_required:
                z_final, iter_losses = train_fn(k=iters,
                                                z0=z0,
                                                q=q,
                                                params=scs_params,
                                                supervised=supervised,
                                                z_star=z_star) #,
                                                #factor=factor)
                print(jnp.linalg.norm(z_final[:-1] - z_star))
                print('z_final', z_final) #[:3])
            else:
                eval_out = eval_fn(k=iters,
                                    z0=z0,
                                    q=q,
                                    params=scs_params,
                                    supervised=supervised,
                                    z_star=z_star)
                z_final, iter_losses, z_all_plus_1 = eval_out[0], eval_out[1], eval_out[2]
                angles = None

            loss = self.final_loss(loss_method, z_final, iter_losses, supervised, z0, z_star)
            print('loss', loss)

            if diff_required:
                return loss
            else:
                return_out = (loss, iter_losses, z_all_plus_1, angles) + eval_out[3:]
                return return_out
        loss_fn = self.predict_2_loss(predict, diff_required)
        return loss_fn

    
    def get_xys_from_z(self, z_init, m, n):
        """
        z = (x, y + s, 1)
        we always set the last entry of z to be 1
        we allow s to be zero (we just need z[n:m + n] = y + s)
        """
        # m, n = self.l2ws_model.m, self.l2ws_model.n
        x = z_init[:n]
        y = z_init[n:n + m]
        s = jnp.zeros(m)
        return x, y, s
