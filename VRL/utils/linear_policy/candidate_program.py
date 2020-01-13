# -*- coding: utf-8 -*-
# -------------------------------
# Author: He Zhu
# Date:   2020-01-10 22:16:44
# Last Modified by:   Zikang Xiong
# Last Modified time: 2020-01-13 15:07:14
# -------------------------------
from lqr import dlqr
from random_search import random_search_linear_policy
from gradient_descent import policy_gradient_adam_linear_policy

import numpy as np

def generator(env, x0, eq_err, learning_method, number_of_rollouts, simulation_steps, actor, x_min, x_max, 
    rewardf=None, continuous=False, timestep=.005, explore_mag=.04, step_size=.05, 
    coffset=None, bias=False, unsafe_flag=False, lqr_start=False, without_nn_guide=False):

    if (learning_method == "lqr"):
        K = dlqr(env.A,env.B,env.Q,env.R)
        #K = lqr_gain(A,B,Q,R)
        print("K = {}".format(K))

    def reward_func(x, Q, u, R):
      """
        the smaller the distance between the ouput of NN and linear controller,
        the higher reward.
        distance is measured by L1 distance, np.abs(actor.predict(x) - u) 
        u, Q, and R are useless here, reserved for the interface design.
      """
      sim_score = 0 if actor is None else -np.matrix([[np.sum(np.abs(actor.predict(np.reshape(x, (-1, actor.s_dim))) - u))]])
      safe_score = 0 if actor is not None or rewardf is None else rewardf(x, Q, u, R)
      return sim_score + safe_score

    if actor is None and rewardf is None:
      shield_reward = None
    elif not without_nn_guide:
      shield_reward = reward_func
    else:
      shield_reward = rewardf

    if (learning_method == "random_search"):
        K = random_search_linear_policy(env,x0,eq_err,number_of_rollouts,simulation_steps,x_min,x_max,continuous,timestep,shield_reward,explore_mag,step_size,coffset=coffset,bias=bias,unsafe_flag=unsafe_flag,lqr_start=lqr_start)
        print("K = {}".format(K))
    elif (learning_method == "policy_gradient"):
        K = policy_gradient_adam_linear_policy(env,x0,eq_err,number_of_rollouts,simulation_steps,x_min,x_max,continuous,timestep,shield_reward,explore_mag,step_size,coffset=coffset)
        print("K = {}".format(K))
    else:
        print("Learning method {} is not found".format(learning_method))
    return np.array(K)