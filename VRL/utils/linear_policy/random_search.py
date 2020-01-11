# -*- coding: utf-8 -*-
# -------------------------------
# Author: He Zhu
# Date:   2020-01-10 21:58:18
# Last Modified by:   Zikang Xiong
# Last Modified time: 2020-01-10 22:09:14
# -------------------------------
import numpy as np
import scipy.linalg

from lqr import dlqr

def uniform_random_linear_policy(A,B,Q,R,x0,eq_err,N,T,x_min=None,x_max=None,continuous=False,timestep=.01,
  linf_norm=3):
  '''
    Arguments:
      state transition matrices (A,B)
      LQR Costs (Q,R)
      Initial State x0
      magnitude of noise in dynamics eq_err
      Number of rollouts N
      Time Horizon T

      hyperparameters
          linf_norm = maximum absolute value of entries of controller gain

    Outputs:
      Static Control Gain K optimized on LQR cost by uniformly sampling policies
      in bounded region
  '''

  d,p = B.shape

  #### "ALGORITHM"
  best_K = np.empty((p,d))
  best_reward = -float("inf")
  for k in range(N):
    K = np.random.uniform(-linf_norm,linf_norm,(p,d))
    x = x0
    reward = 0
    for t in range(T):
      u = np.dot(K,x)
      if continuous:
        x = x + timestep*(A.dot(x)+B.dot(u))+eq_err*np.random.randn(d,1)
      else:
        x = A.dot(x)+B.dot(u)+eq_err*np.random.randn(d,1)
      reward += -np.dot(x.T,Q.dot(x))-np.dot(u.T,R.dot(u))
      # Penality added to states 
      if (x_min is not None):
        for index in range(d):
          if x[index, 0] < x_min[index, 0]:
            reward = reward-100
      if (x_max is not None):
        for index in range(d):
          if x[index, 0] > x_max[index, 0]:
            reward = reward-100
    if reward>best_reward:
        best_reward = reward
        best_K = K

  return best_K

def random_search_linear_policy(A,B,Q,R,x0,eq_err,N,T,x_min=None,x_max=None,continuous=False,timestep=.01,rewardf=None,
    explore_mag = 0.04, step_size = 0.05, batch_size = 4, coffset=None, bias=False, unsafe_flag=False, lqr_start=False):
  '''
    Arguments:
      state transition matrices (A,B)
      LQR Costs (Q,R)
      Initial State x0
      magnitude of noise in dynamics eq_err
      Number of rollouts N
      Time Horizon T

      hyperparameters:
        explore_mag = magnitude of the noise to explore
        step_size
        batch_size = number of directions per minibatches
        safeguard: maximum absolute value of entries of controller gain

    Outputs:
      Static Control Gain K optimized on LQR cost by random search
  '''
  def f (x, u):
    return A.dot(x)+B.dot(u)

  d,p = B.shape

  return random_search_helper(f, d, p, Q, R, x0, eq_err, N, T, x_min, x_max, continuous, timestep, rewardf, 
          explore_mag, step_size, batch_size, coffset, bias, unsafe_flag, 
          A if lqr_start and not bias else None, 
          B if lqr_start and not bias else None)

def random_search_helper(f,d,p,Q,R,x0,eq_err,N,T,x_min=None,x_max=None,continuous=False,timestep=.01,rewardf=None,
    explore_mag = 0.04, step_size = 0.05, batch_size = 4, coffset=None, bias=False, unsafe_flag=False, A=None, B=None):
  
  def policy_test(K):
    x = x0
    reward = 0
    for t in range(T):
      u = np.dot(K, np.vstack([x,[1]])) if bias else np.dot(K, x)
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
      if unsafe_flag:
        if ((np.array(x) < x_max)*(np.array(x) > x_min)).all(axis=1).any():
          reward[0,0] = reward[0,0]-100
      else:
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
  K0 = 0*np.random.randn(p,d+1) if bias else 0*np.random.randn(p,d)
  if (A is not None and B is not None):
    if (continuous):
      X = np.matrix(scipy.linalg.solve_continuous_are(A, B, Q, R))   
      K0 = np.matrix(scipy.linalg.inv(R)*(B.T*X))
    else:
      K0 = dlqr(A, B, Q, R)
  ###

  #### ALGORITHM
  K = K0
  best_K = K
  best_reward = -float("inf")
  for k in range(N):
    reward_store = []
    mini_batch = np.zeros((p,d+1)) if bias else np.zeros((p,d))
    for j in range(batch_size):
      V = np.random.randn(p,d+1) if bias else np.random.randn(p,d)
      for sign in [-1,1]:
        x = x0
        reward = 0
        for t in range(T):
          u = np.dot(K+sign*explore_mag*V,np.vstack([x,[1]])) if bias else np.dot(K+sign*explore_mag*V, x)
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
          #safe = True
          unsafe = False
          if unsafe_flag:
            if ((np.array(x) < x_max)*(np.array(x) > x_min)).all(axis=1).any():
              reward[0,0] = reward[0,0]-100
          else:
            if (x_min is not None):
              for index in range(d):
                if x_min[index, 0] is not None and x[index, 0] < x_min[index, 0]:
                  reward[0,0] = reward[0,0]-100
                  #safe = False
                  #print(("unsafe state {}".format(x[index, 0])))
            if (x_max is not None):
              for index in range(d):
                if x_max[index, 0] is not None and x[index, 0] > x_max[index, 0]:
                  reward[0,0] = reward[0,0]-100
          # Break the closed loop system variables are so large
          for index in range(d):
            if abs(x[index, 0]) > 1e72:
              unsafe = True
              break
          if unsafe:
            print(("unsafe x : {} at time {}".format(x, t)))
            break
        mini_batch += (reward[0,0]*sign)*V
        reward_store.append(reward)
    #print("reward = {}".format(reward_store) )
    std = np.std(reward_store)
    if (std == 0):
      #More thoughts into this required: K already converged?
      #print(("K seems converged!"))
      #return K
      K = K
    else:
      #print(("K is unconverged!"))
      #if (np.sum(reward_store) > best_reward):
      #  best_reward = np.sum(reward_store)
      #  best_K = K
      K += (step_size/std/batch_size)*mini_batch
      r = policy_test(K)
      if (r > best_reward):
        best_reward = r
        best_K = K

  #return K
  return best_K
