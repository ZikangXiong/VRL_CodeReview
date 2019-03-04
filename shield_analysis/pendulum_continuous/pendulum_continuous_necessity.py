# -*- coding: utf-8 -*-
# -------------------------------
# Author: Zikang Xiong
# Email: zikangxiong@gmail.com
# Date:   2018-10-23 17:04:25
# Last Modified by:   Zikang Xiong
# Last Modified time: 2019-02-13 01:40:50
# -------------------------------
import sys
sys.path.append("../")

from main import *
import numpy as np
from DDPG import *
from shield import Shield
from Environment import Environment
from shield_analysis.log_scan import read_scan
from shield_analysis.shield_necessity import test_necessity

def pendulum(learning_method, number_of_rollouts, simulation_steps, learning_eposides, critic_structure, actor_structure, train_dir):
  
  m = 1.
  l = 1.
  g = 10.

  #Dynamics that are continuous
  A = np.matrix([
    [ 0., 1.],
    [g/l, 0.]
    ])
  B = np.matrix([
    [          0.],
    [1./(m*l**2.)]
    ])


  #intial state space
  s_min = np.array([[-0.35],[-0.35]])
  s_max = np.array([[ 0.35],[ 0.35]])

  #reward function
  Q = np.matrix([[1., 0.],[0., 1.]])
  R = np.matrix([[.005]])

  #safety constraint
  x_min = np.array([[-0.5],[-0.5]])
  x_max = np.array([[ 0.5],[ 0.5]])
  u_min = np.array([[-15.]])
  u_max = np.array([[ 15.]])

  env = Environment(A, B, u_min, u_max, s_min, s_max, x_min, x_max, Q, R, continuous=True)

  args = { 'actor_lr': 0.0001,
           'critic_lr': 0.001,
           'actor_structure': actor_structure,
           'critic_structure': critic_structure, 
           'buffer_size': 1000000,
           'gamma': 0.99,
           'max_episode_len': 500,
           'max_episodes': learning_eposides,
           'minibatch_size': 64,
           'random_seed': 6553,
           'tau': 0.005,
           'model_path': train_dir+"model.chkp",
           'enable_test': False, 
           'test_episodes': 1000,
           'test_episodes_len': 5000}

  actor = DDPG(env, args)
  
  shield_list = read_scan("pendulum_continuous/pendulum_continuous.log_ret.pkl")
  test_necessity(env, actor, shield_list)

  actor.sess.close()

if __name__ == "__main__":
  # learning_eposides = int(sys.argv[1])
  # actor_structure = [int(i) for i in list(sys.argv[2].split(','))]
  # critic_structure = [int(i) for i in list(sys.argv[3].split(','))]
  # train_dir = sys.argv[4]

  pendulum("random_search", 100, 200, 0, [240,200], [280,240,200], "../ddpg_chkp/pendulum/continuous/240200280240200/") 
