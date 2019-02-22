# -*- coding: utf-8 -*-
# -------------------------------
# Author: Zikang Xiong
# Email: zikangxiong@gmail.com
# Date:   2018-10-28 16:38:03
# Last Modified by:   Zikang Xiong
# Last Modified time: 2019-02-22 16:44:32
# -------------------------------
import numpy as np
from DDPG import *
from main import *
import os.path
import argparse

from shield import Shield
from Environment import Environment


def pendulum(learning_eposides, actor_structure, critic_structure, train_dir, learning_method, number_of_rollouts, simulation_steps, \
    nn_test=False, retrain_shield=False, shield_test=False, test_episodes=100):
  
  ############## Train NN Controller ###############
  # State transform matrix
  A = np.matrix([[1.9027, -1],
                      [1, 0]
                      ])

  B = np.matrix([[1],
                      [0]
                      ])

  # initial action space
  u_min = np.array([[-1.]])
  u_max = np.array([[1.]])

  # intial state space
  s_min = np.array([[-0.5],[-0.5]])
  s_max = np.array([[ 0.5],[0.5]])
  x_min = np.array([[-0.6], [-0.6]])
  x_max = np.array([[0.6], [0.6]])

  # coefficient of reward function
  Q = np.matrix("1 0 ; 0 1")
  R = np.matrix(".0005")

  env = Environment(A, B, u_min, u_max, s_min, s_max, x_min, x_max, Q, R)

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
           'enable_test': nn_test, 
           'test_episodes': test_episodes,
           'test_episodes_len': 5000}
  actor = DDPG(env, args=args)

  ################# Shield ######################
  model_path = os.path.split(args['model_path'])[0]+'/'
  linear_func_model_name = 'K.model'
  model_path = model_path+linear_func_model_name+'.npy'

  shield = Shield(env, actor, model_path, force_learning=retrain_shield, debug=False)
  shield.train_shield(learning_method, number_of_rollouts, simulation_steps)
  if shield_test:
    shield.test_shield(test_episodes, 5000, mode="single")

  actor.sess.close()

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Running Options')
  parser.add_argument('--nn_test', action="store_true", dest="nn_test")
  parser.add_argument('--retrain_shield', action="store_true", dest="retrain_shield")
  parser.add_argument('--shield_test', action="store_true", dest="shield_test")
  parser.add_argument('--test_episodes', action="store", dest="test_episodes", type=int)
  parser_res = parser.parse_args()
  nn_test = parser_res.nn_test
  retrain_shield = parser_res.retrain_shield
  shield_test = parser_res.shield_test
  test_episodes = parser_res.test_episodes if parser_res.test_episodes is not None else 100

  pendulum(0, [1200,900], [1000,900,800], "ddpg_chkp/pendulum/discrete/", "random_search", 100, 50, nn_test=nn_test, retrain_shield=retrain_shield, shield_test=shield_test, test_episodes=test_episodes) 