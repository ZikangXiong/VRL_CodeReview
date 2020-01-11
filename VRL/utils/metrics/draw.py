# -*- coding: utf-8 -*-
# -------------------------------
# Author: Zikang Xiong
# Email: zikangxiong@gmail.com
# Date:   2020-01-10 22:35:59
# Last Modified by:   Zikang Xiong
# Last Modified time: 2020-01-10 22:36:30
# -------------------------------
import numpy as np
import matplotlib.pyplot as plt

def draw_controller (A, B, K, x0, simulation_steps, names, continuous=False, timestep=.01, rewardf=None, coordination=None, coffset=None, bias=False):
  def f (x, u):
    return A.dot(x)+B.dot(u)

  return draw_controller_helper (f, K, x0, simulation_steps, names, continuous, timestep, rewardf, coordination, coffset, bias)    

def draw_controller_helper (f, K, x0, simulation_steps, names, continuous=False, timestep=.01, rewardf=None, coordination=None, coffset=None, bias=False):
    time = np.linspace(0, simulation_steps, simulation_steps, endpoint=True)
    xk = x0 #np.matrix(".0 ; 0 ; .0 ; 0.1")

    XS = []
    for i in range(len(names)):
        XS.append([])
    reward = 0
    for t in time:
        uk = K.dot(np.vstack([xk,[1]])) if bias else K.dot(xk)
        for i, k in enumerate(sorted(names.keys())):
            if coordination is None:
              val = xk[k,0]
              XS[i].append(val)
            else:
              val = xk[k,0]+coordination[k,0]
              XS[i].append(val)
        if rewardf is not None:
          reward += rewardf(xk, uk)
        # Use discrete or continuous semantics based on user's choice
        if continuous:
          xk = xk + timestep*(f(xk, uk)) if coffset is None else xk + timestep*(f(xk, uk)+coffset)
        else:
          xk = f(xk, uk) if coffset is None else f(xk, uk)+coffset

    if rewardf is not None:
      print("Score of the trace: {}".format(reward) )

    for i, k in enumerate(sorted(names.keys())):
        plt.plot(time, XS[i], label=names[k])

    plt.legend(loc='upper right')
    plt.grid()
    plt.show()
    return xk