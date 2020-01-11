# -*- coding: utf-8 -*-
# -------------------------------
# Author: Zikang Xiong
# Email: zikangxiong@gmail.com
# Date:   2020-01-10 22:07:19
# Last Modified by:   Zikang Xiong
# Last Modified time: 2020-01-10 22:12:03
# -------------------------------
import numpy as np

def policy_gradient_adam_linear_policy(A,B,Q,R,x0,eq_err,N,T,x_min=None,x_max=None,continuous=False,timestep=.01,rewardf=None,
    explore_mag = 0.04, step_size = 0.05, batch_size = 8,
    beta1=0.9, beta2=0.999, epsilon=1.0e-8, coffset=None,bias=False):
  '''
    Arguments:
      state transition matrices (A,B)
      LQR Costs (Q,R)
      Initial State x0
      magnitude of noise in dynamics eq_err
      Number of rollouts N
      Time Horizon T

      hyperparameters
         explore_mag magnitude of the noise to explore
         step_size
         batch_size: number of stochastic gradients per minibatch
         beta1, beta2, epsilon are the additional paramters of Adam

    Outputs:
      Static Control Gain K optimized on LQR cost by Policy Gradient
  '''

  def f (x, u):
    return A.dot(x)+B.dot(u)

  d,p = B.shape

  return policy_gradient_helper(f, d, p, Q, R, x0, eq_err, N, T, x_min, x_max, continuous, timestep, rewardf, 
    explore_mag, step_size, batch_size, 
    beta1, beta2, epsilon, coffset, bias)


def policy_gradient_helper(f,d,p,Q,R,x0,eq_err,N,T,x_min=None,x_max=None,continuous=False,timestep=.01,rewardf=None,
    explore_mag = 0.04, step_size = 0.05, batch_size = 8,
    beta1=0.9, beta2=0.999, epsilon=1.0e-8, coffset=None, bias=False):


  def policy_test(K):
    x = x0
    reward = 0
    for t in range(T):
      u = np.dot(K, x)
      # Use discrete or continuous semantics based on user's choice
      if continuous:
        x = x + timestep*(f(x,u))+eq_err*np.random.randn(d,1) if coffset is None else x + timestep*(f(x,u)+coffset)+eq_err*np.random.randn(d,1) 
      else:
        x = f(x,u)+eq_err*np.random.randn(d,1) if coffset is None else f(x,u)+coffset+eq_err*np.random.randn(d,1)
      if rewardf is None:
        reward += -np.dot(x.T,Q.dot(x))-np.dot(u.T,R.dot(u))
      else:
        reward += rewardf (x, Q, u, R)
      #reward += np.array([[0]])
      # Penality added to states 
      if (x_min is not None):
        for index in range(d):
          if x_min[index, 0] is not None and x[index, 0] < x_min[index, 0]:
            reward[0,0] = reward[0,0]-100
      if (x_max is not None):
        for index in range(d):
          if x_max[index, 0] is not None and x[index, 0] > x_max[index, 0]:
            reward[0,0] = reward[0,0]-100
    return reward

  # initial condition for K
  K0 = 0.0*np.random.randn(p,d)
  ###

  #### ALGORITHM
  K = K0
  best_K = K
  best_reward = -float("inf")

  baseline = 0.0
  Adam_M = np.zeros((p,d))
  Adam_V = np.zeros((p,d))

  for k in range(N):
    mini_batch = np.zeros((p,d))
    mb_store = np.zeros((p,d,batch_size))
    reward = np.zeros((batch_size))

    # Collect policy gradients for the current minibatch
    for j in range(batch_size):
      x = x0
      X_store = np.zeros((d,T))
      V_store = np.zeros((p,T))
      for t in range(T):
        v = explore_mag*np.random.randn(p,1)
        X_store[:,t] = x.flatten()
        V_store[:,t] = v.flatten()
        u = np.dot(K,x)+v

        # Use discrete or continuous semantics based on user's choice
        if continuous:
          x = x + timestep*(f(x,u))+eq_err*np.random.randn(d,1) if coffset is None else x + timestep*(f(x,u)+coffset)+eq_err*np.random.randn(d,1)
        else:
          x = f(x,u)+eq_err*np.random.randn(d,1) if coffset is None else f(x,u)+coffset+eq_err*np.random.randn(d,1)
        if rewardf is None:
          reward[j] += -np.dot(x.T,Q.dot(x))-np.dot(u.T,R.dot(u))
        else:
          reward[j] += rewardf (x, Q, u, R)
        #reward += np.array([[0]])
        # Penality added to states 
        #safe = True
        unsafe = False
        if (x_min is not None):
          for index in range(d):
            if x_min[index, 0] is not None and x[index, 0] < x_min[index, 0]:
              reward[j] = reward[j]-100
              #safe = False
              #print(("unsafe state {}".format(x[index, 0])))
        if (x_max is not None):
          for index in range(d):
            if x_max[index, 0] is not None and x[index, 0] > x_max[index, 0]:
              reward[j] = reward[j]-100
              #safe = False
              #print(("unsafe state {}".format(x[index, 0])))
        #if ((x_min is not None or x_max is not None) and safe):
        #    reward[0, 0] = reward[0,0] + 100
        #if safe is False:
          #print(("unsafe x : {} at time {}".format(x, t)))
          #break
        # Break the closed loop system variables are so large
        for index in range(d):
          if abs(x[index, 0]) > 1e72:
            unsafe = True
            break
        if unsafe:
          print(("unsafe x : {} at time {}".format(x, t)))
          break
      mb_store[:,:,j] = np.dot(V_store,X_store.T)

    # Mean of rewards over a minibatch are subtracted from reward.
    # This is a heuristic for baseline subtraction. 

    #print("reward = {}".format(reward))

    for j in range(batch_size):
      mini_batch += ((reward[j]-baseline)/batch_size)*mb_store[:,:,j]
    baseline = np.mean(reward)

    # Adam Algorithm

    Adam_M = beta1*Adam_M + (1-beta1)*mini_batch
    Adam_V = beta2*Adam_V + (1-beta2)*(mini_batch*mini_batch)

    effective_step_size = step_size*np.sqrt(1-beta2**(k+1))/(1-beta1**(k+1))
    K += effective_step_size*Adam_M/(np.sqrt(Adam_V)+epsilon)
    r = policy_test(K)
    if (r > best_reward):
      best_reward = r
      best_K = K

  return best_K