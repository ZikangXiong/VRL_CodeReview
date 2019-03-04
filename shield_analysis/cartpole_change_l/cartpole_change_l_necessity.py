# -*- coding: utf-8 -*-
# -------------------------------
# Author: Zikang Xiong
# Email: zikangxiong@gmail.com
# Date:   2018-10-20 11:54:03
# Last Modified by:   Zikang Xiong
# Last Modified time: 2019-02-13 01:33:48
# -------------------------------
import sys
sys.path.append("../")

import numpy as np
from main import *
from DDPG import *
from Environment import Environment
from shield import Shield

from shield_analysis.log_scan import read_scan
from shield_analysis.shield_necessity import test_necessity

def cartpole(learning_method, number_of_rollouts, simulation_steps, ddpg_learning_eposides, critic_structure, actor_structure, train_dir):
  l = .22+0.15 # rod length is 2l
  m = (2*l)*(.006**2)*(3.14/4)*(7856) # rod 6 mm diameter, 44cm length, 7856 kg/m^3
  M = .4
  dt = .02 # 20 ms
  g = 9.8

  A = np.matrix([[1, dt, 0, 0],[0,1, -(3*m*g*dt)/(7*M+4*m),0],[0,0,1,dt],[0,0,(3*g*(m+M)*dt)/(l*(7*M+4*m)),1]])
  B = np.matrix([[0],[7*dt/(7*M+4*m)],[0],[-3*dt/(l*(7*M+4*m))]])

  # amount of Gaussian noise in dynamics
  # eq_err = 0

   #intial state space
  s_min = np.array([[ -0.1],[ -0.1], [-0.05], [ -0.05]])
  s_max = np.array([[  0.1],[  0.1], [ 0.05], [  0.05]])

  Q = np.matrix("1 0 0 0; 0 1 0 0 ; 0 0 1 0; 0 0 0 1")
  R = np.matrix(".0005")

  x_min = np.array([[-0.3],[-0.5],[-0.3],[-0.5]])
  x_max = np.array([[ .3],[ .5],[.3],[.5]])
  u_min = np.array([[-15.]])
  u_max = np.array([[ 15.]])

  env = Environment(A, B, u_min, u_max, s_min, s_max, x_min, x_max, Q, R)

  args = { 'actor_lr': 0.0001,
           'critic_lr': 0.001,
           'actor_structure': actor_structure,
           'critic_structure': critic_structure, 
           'buffer_size': 1000000,
           'gamma': 0.99,
           'max_episode_len': 1,
           'max_episodes': ddpg_learning_eposides,
           'minibatch_size': 64,
           'random_seed': 6553,
           'tau': 0.005,
           'model_path': train_dir+"model.chkp",
           'enable_test': False, 
           'test_episodes': 1000,
           'test_episodes_len': 5000}
  actor = DDPG(env, args)

  shield_list = read_scan("cartpole_change_l/cartpole_change_l.log_ret.pkl")
  test_necessity(env, actor, shield_list)

  actor.sess.close()

if __name__ == "__main__":
  # ddpg_learning_eposides = int(sys.argv[1])
  # actor_structure = [int(i) for i in list(sys.argv[2].split(','))]
  # critic_structure = [int(i) for i in list(sys.argv[3].split(','))]
  # train_dir = sys.argv[4]

  cartpole("random_search", 100, 200, 0, [1200,900], [1000,900,800], "../ddpg_chkp/perfect_model/cartpole/change_l/")
