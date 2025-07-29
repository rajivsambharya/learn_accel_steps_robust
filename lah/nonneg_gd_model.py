from functools import partial
import jax.numpy as jnp

from lah.algo_steps_nonneg_gd import (
    k_steps_eval_nonneg_gd_accel_l2ws,
    k_steps_train_nonneg_gd_accel_l2ws,
)
from lah.l2o_model import L2Omodel


class NONNEGGDmodel(L2Omodel):
    def __init__(self, **kwargs):
        super(NONNEGGDmodel, self).__init__(**kwargs)

    def initialize_algo(self, input_dict):
        self.factor_static = None
        self.algo = 'nonneg_gd'
        self.factors_required = False
        self.q_mat_train, self.q_mat_test = input_dict['b_mat_train'], input_dict['b_mat_test']
        A = input_dict['A']

        # ista_step = input_dict['ista_step']
        m, n = A.shape
        self.output_size = n

        evals, evecs = jnp.linalg.eigh(A.T @ A)
        self.evals = evals

        self.str_cvx_param = jnp.min(evals)
        self.smooth_param = jnp.max(evals)
        kappa = self.smooth_param / self.str_cvx_param
        nonneg_gd_step = 1 / self.smooth_param
        momentum_step = (kappa**.5-1) / (kappa**.5+1)

        self.k_steps_train_fn = partial(k_steps_train_nonneg_gd_accel_l2ws, A=A, 
                                        nonneg_gd_step=nonneg_gd_step, momentum_step=momentum_step, jit=self.jit)
        self.k_steps_eval_fn = partial(k_steps_eval_nonneg_gd_accel_l2ws, A=A, 
                                       nonneg_gd_step=nonneg_gd_step, momentum_step=momentum_step, jit=self.jit)
        self.out_axes_length = 5
        self.lah = False
