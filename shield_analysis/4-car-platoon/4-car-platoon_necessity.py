# -*- coding: utf-8 -*-
# -------------------------------
# Author: Zikang Xiong
# Email: zikangxiong@gmail.com
# Date:   2018-10-27 17:10:46
# Last Modified by:   Zikang Xiong
# Last Modified time: 2019-02-13 01:32:36
# -------------------------------
import sys
sys.path.append("../")

import numpy as np
from DDPG import *
from main import *
import os.path
from Environment import Environment
from shield import Shield

from shield_analysis.log_scan import read_scan
from shield_analysis.shield_necessity import test_necessity


def carplatoon(learning_method, number_of_rollouts, simulation_steps, learning_eposides, actor_structure, critic_structure, train_dir):
  
  A = np.matrix([
    [1,0,0,0,0,0,0],
    [0,1,0.1,0,0,0,0],
    [0,0,1,0,0,0,0],
    [0,0,0,1,0.1,0,0],
    [0,0,0,0,1,0,0],
    [0,0,0,0,0,1,0.1],
    [0,0,0,0,0,0,1]
  ])
  B = np.matrix([
    [0.1,0,0,0],
    [0,0,0,0],
    [0.1,-0.1,0,0],
    [0,0,0,0],
    [0,0.1,-0.1,0],
    [0,0,0,0],
    [0,0,0.1,-0.1]
  ])

  #intial state space
  s_min = np.array([[ 19.9],[ 0.9], [-0.1], [ 0.9],[-0.1], [ 0.9], [-0.1]])
  s_max = np.array([[ 20.1],[ 1.1], [ 0.1], [ 1.1],[ 0.1], [ 1.1], [ 0.1]])

  Q = np.matrix("1 0 0 0 0 0 0; 0 1 0 0 0 0 0; 0 0 1 0 0 0 0; 0 0 0 1 0 0 0; 0 0 0 0 1 0 0; 0 0 0 0 0 1 0; 0 0 0 0 0 0 1")
  R = np.matrix(".0005 0 0 0; 0 .0005 0 0; 0 0 .0005 0; 0 0 0 .0005")

  x_min = np.array([[18],[0.5],[-0.35],[0.5],[-1],[0.5],[-1]])
  x_max = np.array([[22],[1.5], [0.35],[1.5],[ 1],[1.5],[ 1]])
  u_min = np.array([[-10.], [-10.], [-10.], [-10.]])
  u_max = np.array([[ 10.], [ 10.], [ 10.], [ 10.]])

  #Coordination transformation
  origin = np.array([[20], [1], [0], [1], [0], [1], [0]])
  s_min -= origin
  s_max -= origin
  x_min -= origin
  x_max -= origin

  env = Environment(A, B, u_min, u_max, s_min, s_max, x_min, x_max, Q, R)

  args = { 'actor_lr': 0.0001,
           'critic_lr': 0.001,
           'actor_structure': actor_structure,
           'critic_structure': critic_structure, 
           'buffer_size': 1000000,
           'gamma': 0.99,
           'max_episode_len': 5,
           'max_episodes': learning_eposides,
           'minibatch_size': 64,
           'random_seed': 6553,
           'tau': 0.005,
           'model_path': train_dir+"model.chkp",
           'enable_test': False, 
           'test_episodes': 1000,
           'test_episodes_len': 5000}
  actor = DDPG(env, args)

  shield_list = read_scan("4-car-platoon/4-car-platoon.log_ret.pkl")
  test_necessity(env, actor, shield_list)

  actor.sess.close()

if __name__ == "__main__":
  carplatoon("random_search", 200, 100, 0, [500, 400, 300], [600, 500, 400, 300], "../ddpg_chkp/car-platoon/discrete/4/500400300600500400300/") 
