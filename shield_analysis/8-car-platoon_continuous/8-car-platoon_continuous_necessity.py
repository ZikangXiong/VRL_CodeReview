# -*- coding: utf-8 -*-
# -------------------------------
# Author: Zikang Xiong
# Email: zikangxiong@gmail.com
# Date:   2018-10-27 21:02:27
# Last Modified by:   Zikang Xiong
# Last Modified time: 2019-02-13 01:13:29
# -------------------------------
import sys
sys.path.append("../")

import numpy as np
from DDPG import *
from main import *
from Environment import Environment

from shield_analysis.log_scan import read_scan
from shield_analysis.shield_necessity import test_necessity


def carplatoon(learning_method, number_of_rollouts, simulation_steps, learning_eposides, actor_structure, critic_structure, train_dir):
  A = np.matrix([
  [0, 0,0,   0,0,   0,0,   0,0, 0,0, 0,0, 0,0],
  [0, 0,1,   0,0,   0,0,   0,0, 0,0, 0,0, 0,0],
  [0, 0,0,   0,0,   0,0,   0,0, 0,0, 0,0, 0,0],
  [0, 0,0,   0,1,   0,0,   0,0, 0,0, 0,0, 0,0],
  [0, 0,0,   0,0,   0,0,   0,0, 0,0, 0,0, 0,0],
  [0, 0,0,   0,0,   0,1,   0,0, 0,0, 0,0, 0,0],
  [0, 0,0,   0,0,   0,0,   0,0, 0,0, 0,0, 0,0],
  [0, 0,0,   0,0,   0,0,   0,1, 0,0, 0,0, 0,0],
  [0, 0,0,   0,0,   0,0,   0,0, 0,0, 0,0, 0,0],
  [0, 0,0,   0,0,   0,0,   0,0, 0,1, 0,0, 0,0],
  [0, 0,0,   0,0,   0,0,   0,0, 0,0, 0,0, 0,0],
  [0, 0,0,   0,0,   0,0,   0,0, 0,0, 0,1, 0,0],
  [0, 0,0,   0,0,   0,0,   0,0, 0,0, 0,0, 0,0],
  [0, 0,0,   0,0,   0,0,   0,0, 0,0, 0,0, 0,1],
  [0, 0,0,   0,0,   0,0,   0,0, 0,0, 0,0, 0,0]
  ])
  B = np.matrix([
  [1,   0,   0,   0,   0,   0,   0,   0],
  [0,   0,   0,   0,   0,   0,   0,   0],
  [1,  -1,   0,   0,   0,   0,   0,   0],
  [0,   0,   0,   0,   0,   0,   0,   0],
  [0,   1,  -1,   0,   0,   0,   0,   0],
  [0,   0,   0,   0,   0,   0,   0,   0],
  [0,   0,   1,  -1,   0,   0,   0,   0],
  [0,   0,   0,   0,   0,   0,   0,   0],
  [0,   0,   0,   1,  -1,   0,   0,   0],
  [0,   0,   0,   0,   0,   0,   0,   0],
  [0,   0,   0,   0,   1,  -1,   0,   0],
  [0,   0,   0,   0,   0,   0,   0,   0],
  [0,   0,   0,   0,   0,   1,  -1,   0],
  [0,   0,   0,   0,   0,   0,   0,   0],
  [0,   0,   0,   0,   0,   0,   1,  -1],
  ])

  #intial state space
  s_min = np.array([[ 19.9],[ 0.9], [-0.1], [ 0.9],[-0.1], [ 0.9], [-0.1], [ 0.9], [-0.1], [ 0.9],[-0.1], [ 0.9], [-0.1], [ 0.9], [-0.1]])
  s_max = np.array([[ 20.1],[ 1.1], [ 0.1], [ 1.1],[ 0.1], [ 1.1], [ 0.1], [ 1.1], [ 0.1], [ 1.1],[ 0.1], [ 1.1], [ 0.1], [ 1.1], [ 0.1]])

  x_min = np.array([[18],[0.1],[-1],[0.5],[-1],[0.5],[-1],[0.5],[-1],[0.5],[-1],[0.5],[-1],[0.5],[-1]])
  x_max = np.array([[22],[1.5], [1],[1.5],[ 1],[1.5],[ 1],[1.5], [1],[1.5],[ 1],[1.5],[ 1],[1.5],[ 1]])
  u_min = np.array([[-10.], [-10.], [-10.], [-10.], [-10.], [-10.], [-10.], [-10.]])
  u_max = np.array([[ 10.], [ 10.], [ 10.], [ 10.], [ 10.], [ 10.], [ 10.], [ 10.]])

  target = np.array([[20],[1], [0], [1], [0], [1], [0], [1], [0], [1], [0], [1], [0], [1], [0]])

  s_min -= target
  s_max -= target
  x_min -= target
  x_max -= target

  Q = np.zeros((15, 15), float)
  np.fill_diagonal(Q, 1)

  R = np.zeros((8,8), float)
  np.fill_diagonal(R, 1)

  env = Environment(A, B, u_min, u_max, s_min, s_max, x_min, x_max, Q, R, continuous=True, bad_reward=-1000)
  args = { 'actor_lr': 0.000001,
           'critic_lr': 0.00001,
           'actor_structure': actor_structure,
           'critic_structure': critic_structure, 
           'buffer_size': 1000000,
           'gamma': 0.999,
           'max_episode_len': 400,
           'max_episodes': learning_eposides,
           'minibatch_size': 64,
           'random_seed': 122,
           'tau': 0.005,
           'model_path': train_dir+"model.chkp",
           'enable_test': False, 
           'test_episodes': 1000,
           'test_episodes_len': 5000}
  actor = DDPG(env, args)

  shield_list = read_scan("8-car-platoon_continuous/8-car-platoon_continuous.log_ret.pkl")
  test_necessity(env, actor, shield_list)

  actor.sess.close()

if __name__ == "__main__":
  carplatoon("random_search", 200, 2000, 0, [400, 300, 200], [500, 400, 300, 200], "../ddpg_chkp/car-platoon/continuous/8/400300200500400300200/") 