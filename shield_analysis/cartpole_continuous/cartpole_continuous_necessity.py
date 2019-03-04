# -*- coding: utf-8 -*-
# -------------------------------
# Author: Zikang Xiong
# Email: zikangxiong@gmail.com
# Date:   2018-10-23 22:14:33
# Last Modified by:   Zikang Xiong
# Last Modified time: 2019-02-13 01:26:14
# -------------------------------
import sys
sys.path.append('../')

from shield_analysis.shield_necessity import test_necessity
from shield_analysis.log_scan import read_scan
from main import *
from DDPG import *
from Environment import Environment

def cartpole(learning_method, number_of_rollouts, simulation_steps, learning_eposides, critic_structure, actor_structure, train_dir):
  A = np.matrix([
  [0, 1,     0, 0],
  [0, 0, 0.716, 0],
  [0, 0,     0, 1],
  [0, 0, 15.76, 0]
  ])
  B = np.matrix([
  [0],
  [0.9755],
  [0],
  [1.46]
  ])

   #intial state space
  s_min = np.array([[ -0.05],[ -0.1], [-0.05], [ -0.05]])
  s_max = np.array([[  0.05],[  0.1], [ 0.05], [  0.05]])

  Q = np.matrix("1 0 0 0; 0 1 0 0 ; 0 0 1 0; 0 0 0 1")
  R = np.matrix(".0005")

  x_min = np.array([[-0.3],[-0.5],[-0.3],[-0.5]])
  x_max = np.array([[ .3],[ .5],[.3],[.5]])
  u_min = np.array([[-15.]])
  u_max = np.array([[ 15.]])
  env = Environment(A, B, u_min, u_max, s_min, s_max, x_min, x_max, Q, R, continuous=True)

  args = { 'actor_lr': 0.0001,
         'critic_lr': 0.001,
         'actor_structure': actor_structure,
         'critic_structure': critic_structure, 
         'buffer_size': 1000000,
         'gamma': 0.99,
         'max_episode_len': 100,
         'max_episodes': learning_eposides,
         'minibatch_size': 64,
         'random_seed': 6553,
         'tau': 0.005,
         'model_path': train_dir+"model.chkp",
         'enable_test': False, 
         'test_episodes': 10,
         'test_episodes_len': 5000}
  actor =  DDPG(env, args=args)

  shield_list = read_scan("cartpole_continuous/cartpole_continuous.log_ret.pkl")
  test_necessity(env, actor, shield_list)

  actor.sess.close()

if __name__ == "__main__":

  cartpole("random_search", 100, 200, 0, [300, 200], [300, 250, 200], "../ddpg_chkp/cartpole/continuous/300200300250200/")
