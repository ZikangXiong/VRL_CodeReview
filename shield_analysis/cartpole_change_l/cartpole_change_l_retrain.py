# -*- coding: utf-8 -*-
# -------------------------------
# Author: Zikang Xiong
# Email: zikangxiong@gmail.com
# Date:   2018-10-20 11:54:03
# Last Modified by:   Zikang Xiong
# Last Modified time: 2019-02-13 00:34:24
# -------------------------------
import sys
sys.path.append("../")

import numpy as np
from main import *
from DDPG import *
from Environment import Environment
from shield import Shield
from shield_analysis.retrain_nn_and_shield import retrain_nn_and_shield
from shield_analysis.log_scan import read_scan

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
           'max_episode_len': 10,
           'max_episodes': ddpg_learning_eposides,
           'minibatch_size': 64,
           'random_seed': 6553,
           'tau': 0.005,
           'model_path': train_dir+"model.chkp",
           'enable_test': False, 
           'test_episodes': 1000,
           'test_episodes_len': 5000}
  actor = DDPG(env, args)

  #################### Shield #################
  model_path = os.path.split(args['model_path'])[0]+'/'
  linear_func_model_name = 'K.model'
  model_path = model_path+linear_func_model_name+'.npy'

  def rewardf(x, Q, u, R):
    return np.matrix([[env.reward(x, u)]])

  names = {0:"cart position, meters", 1:"cart velocity", 2:"pendulum angle, radians", 3:"pendulum angle velocity"}
  shield = Shield(env, None, model_path)
  shield.train_shield(learning_method, number_of_rollouts, simulation_steps, rewardf=rewardf, names=names, explore_mag = 1.0, step_size = 1.0)
  K = shield.K_list[0]
  shield_state_list = read_scan("cartpole_change_l/cartpole_change_l.log_ret.pkl")
  retrain_nn_and_shield(actor, K, shield_state_list)

  # shield.test_shield(1000, 5000, mode="single")
  # shield.test_shield(1000, 100, mode="all")

  ################# Metrics ######################
  # actor_boundary(env, actor)
  # shield.shield_boundary(2000, 50)
  # terminal_err = 0.004
  # sample_steps = 100
  # sample_ep = 1000
  # print "---\nterminal error: {}\nsample_ep: {}\nsample_steps: {}\n---".format(terminal_err, sample_ep, sample_steps)
  # dist_nn_lf = metrics.distance_between_linear_function_and_neural_network(env, actor, shield.K, terminal_err, sample_ep, sample_steps)
  # print "dist_nn_lf: ", dist_nn_lf
  # nn_perf = metrics.neural_network_performance(env, actor, terminal_err, sample_ep, sample_steps)
  # print "nn_perf", nn_perf
  # shield_perf = metrics.linear_function_performance(env, shield.K, terminal_err, sample_ep, sample_steps)
  # print "shield_perf", shield_perf

  actor.sess.close()

if __name__ == "__main__":
  # ddpg_learning_eposides = int(sys.argv[1])
  # actor_structure = [int(i) for i in list(sys.argv[2].split(','))]
  # critic_structure = [int(i) for i in list(sys.argv[3].split(','))]
  # train_dir = sys.argv[4]

  cartpole("random_search", 200, 500, 0, [1200,900], [1000,900,800], "cartpole_change_l/retrain/1200901000900800/")
