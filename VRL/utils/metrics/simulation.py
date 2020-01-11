# -*- coding: utf-8 -*-
# -------------------------------
# Author: Zikang Xiong
# Email: zikangxiong@gmail.com
# Date:   2020-01-10 22:32:42
# Last Modified by:   Zikang Xiong
# Last Modified time: 2020-01-10 22:33:38
# -------------------------------
import numpy as np

def test_controller (A, B, K, x0, simulation_steps, rewardf, continuous=False, timestep=.01, coffset=None, bias=False):
    def f (x, u):
      return A.dot(x)+B.dot(u) 

    return test_controller_helper(f, K, x0, simulation_steps, rewardf, continuous, timestep, coffset, bias)

def test_controller_helper (f, K, x0, simulation_steps, rewardf, continuous=False, timestep=.01, coffset=None, bias=False):
    time = np.linspace(0, simulation_steps, simulation_steps, endpoint=True)
    xk = x0 #np.matrix(".0 ; 0 ; .0 ; 0.1")
    reward = 0
    for t in time:
        uk = K.dot(np.vstack([xk,[1]])) if bias else K.dot(xk)
        reward += rewardf(xk, uk)
        # Use discrete or continuous semantics based on user's choice
        if continuous:
          xk = xk + timestep*(f(xk, uk)) if coffset is None else xk + timestep*(f(xk, uk)+coffset)
        else:
          xk = f(xk, uk) if coffset is None else f(xk, uk)+coffset
    #print("Score of the trace: {}".format(reward) )
    return reward