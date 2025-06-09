from functools import partial

from lah.algo_steps import k_steps_eval_gd, k_steps_train_gd, k_steps_eval_nesterov_gd, k_steps_train_nesterov_gd
# from lah.algo_steps_gd import k_steps_ev
from lah.l2o_model import L2Omodel
import jax.numpy as jnp

class GDmodel(L2Omodel):
    def __init__(self, **kwargs):
        super(GDmodel, self).__init__(**kwargs)

    def initialize_algo(self, input_dict):
        self.factor_static = None
        self.algo = 'gd'
        self.factors_required = False
        
        self.q_mat_train, self.q_mat_test = input_dict['c_mat_train'], input_dict['c_mat_test']
        P = input_dict['P']
        gd_step = input_dict['gd_step']
        n = P.shape[0]
        self.output_size = n

        evals, evecs = jnp.linalg.eigh(P)
        cond_num = jnp.max(evals) / jnp.min(evals)
        L = jnp.max(evals)

        self.k_steps_train_fn = partial(k_steps_train_nesterov_gd, P=P, gd_step=1/L, cond_num=cond_num, jit=self.jit)
        self.k_steps_eval_fn = partial(k_steps_eval_nesterov_gd, P=P, cond_num=cond_num, params=[1 / L], jit=self.jit)
        self.out_axes_length = 5
        self.lah = False
