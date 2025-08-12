from functools import partial

from lah_accel.algo_steps_ista import (
    k_steps_eval_fista_l2ws,
    k_steps_train_fista_l2ws,
)
from lah_accel.l2o_model import L2Omodel


class ISTAmodel(L2Omodel):
    def __init__(self, **kwargs):
        super(ISTAmodel, self).__init__(**kwargs)

    def initialize_algo(self, input_dict):
        self.factor_static = None
        self.algo = 'ista'
        self.factors_required = False
        self.q_mat_train, self.q_mat_test = input_dict['b_mat_train'], input_dict['b_mat_test']
        A = input_dict['A']
        lambd = input_dict['lambd']
        ista_step = input_dict['ista_step']
        m, n = A.shape
        self.output_size = n

        self.k_steps_train_fn = partial(k_steps_train_fista_l2ws, A=A, lambd=lambd, 
                                        ista_step=ista_step, jit=self.jit)
        self.k_steps_eval_fn = partial(k_steps_eval_fista_l2ws, A=A, lambd=lambd, 
                                       ista_step=ista_step, jit=self.jit)
        self.out_axes_length = 5
        self.lah = False
