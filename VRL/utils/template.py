# -*- coding: utf-8 -*-
# -------------------------------
# Author: Zikang Xiong
# Email: zikangxiong@gmail.com
# Date:   2019-12-16 16:10:11
# Last Modified by:   Zikang Xiong
# Last Modified time: 2019-12-16 17:26:51
# -------------------------------

from main import *
from shield import *
from Environment import Environment


def template(A, B, Q, R,
             initial_space, 
             safe_space,
             action_space,  
             continuous, 
             model_path,
             rollouts, 
             simu_step, 
             nn_controller):

    s_min, s_max = initial_space
    x_min, x_max = safe_space
    u_max, u_min = action_space
    env = Environment(A, B, u_min, u_max, s_min, s_max, x_min, x_max,
                      Q, R, continuous=continuous)

    shield = Shield(env, nn_controller, model_path,
                    debug=False, force_learning=True)
    shield.train_shield("random_search", rollouts,
                        simu_step, eq_err=0, explore_mag=1.0, step_size=1.0)
