import sys
sys.path.append("../")

from main import *
from shield import Shield
from Environment import PolySysEnvironment
from DDPG import *

import argparse

def biology (learning_method, number_of_rollouts, simulation_steps, learning_eposides, critic_structure, actor_structure, train_dir,\
            nn_test=False, retrain_shield=False, shield_test=False, test_episodes=100):
  # 10-dimension and 1-input system and 1-disturbance system
  ds = 3
  us = 2

  #Dynamics that are defined as a continuous function!
  def f (x, u):
    #random disturbance
    #d = random.uniform(0, 20)
    delta = np.zeros((ds, 1), float)
    delta[0,0] = -0.01*x[0,0] - x[1,0]*(x[0,0]+4.5) + u[0,0]
    delta[1,0] = -0.025*x[1,0] + 0.000013*x[2,0]
    delta[2,0] = -0.093*(x[2,0] + 15) + (1/12)*u[1,0]
    return delta

  #Closed loop system dynamics to text
  def f_to_str(K):
    kstr = K_to_str(K)
    f = []
    f.append("-0.01*x[1] - x[2]*(x[1]+4.5) + {}".format(kstr[0]))
    f.append("-0.025*x[2] + 0.000013*x[3]")
    f.append("-0.093*(x[3] + 15) + (1/12)*{}".format(kstr[1]))
    return f

  h = 0.01

  # amount of Gaussian noise in dynamics
  eq_err = 1e-2

  #intial state space
  s_min = np.array([[-2],[-0],[-0.1]])
  s_max = np.array([[ 2],[ 0],[ 0.1]])

  Q = np.zeros((ds,ds), float)
  R = np.zeros((us,us), float)
  np.fill_diagonal(Q, 1)
  np.fill_diagonal(R, 1)

  #user defined unsafety condition
  def unsafe_eval(x):
    if (x[0,0] >= 5):
      return True
    return False
  def unsafe_string():
    return ["x[1] - 5"]

  def rewardf(x, Q, u, R):
    reward = 0
    u = np.reshape(u, (us, 1))
    reward += -np.dot(x.T,Q.dot(x))-np.dot(u.T,R.dot(u))
    if (unsafe_eval(x)):
      reward -= 100
    return reward

  def testf(x, u):
    if (unsafe_eval(x)):
      print x
      return -1
    return 0 

  u_min = np.array([[-50.], [-50]])
  u_max = np.array([[ 50.], [ 50]])

  env = PolySysEnvironment(f, f_to_str,rewardf, testf, unsafe_string, ds, us, Q, R, s_min, s_max, u_max=u_max, u_min=u_min, timestep=h)

  ############ Train and Test NN model ############
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
       'test_episodes': test_episodes,
       'test_episodes_len': 1000}
  actor =  DDPG(env, args=args)

  #################### Shield #################
  model_path = os.path.split(args['model_path'])[0]+'/'
  linear_func_model_name = 'K.model'
  model_path = model_path+linear_func_model_name+'.npy'

  shield = Shield(env, actor, model_path=model_path, force_learning=retrain_shield)
  shield.train_polysys_shield(learning_method, number_of_rollouts, simulation_steps, eq_err=eq_err, explore_mag = 0.4, step_size = 0.5, aggressive=True, without_nn_guide=True, enable_jit=True)

  ################# Metrics ######################
  actor_boundary(env, actor, 2000, 1000)
  shield.shield_boundary(2000, 1000)
  terminal_err = 1e-1
  sample_steps = 2000
  sample_ep = 1000
  print "---\nterminal error: {}\nsample_ep: {}\nsample_steps: {}\n---".format(terminal_err, sample_ep, sample_steps)
  # dist_nn_lf = metrics.distance_between_linear_function_and_neural_network(env, actor, shield.K, terminal_err, sample_ep, sample_steps)
  # print "dist_nn_lf: ", dist_nn_lf
  nn_perf = metrics.neural_network_performance(env, actor, terminal_err, sample_ep, sample_steps)
  print "nn_perf", nn_perf
  shield_perf = metrics.linear_function_performance(env, shield.K_list[0], terminal_err, sample_ep, sample_steps)
  print "shield_perf", shield_perf

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

  biology ("random_search", 200, 500, 0, [240, 200], [280, 240, 200], "../ddpg_chkp/biology/240200280240200/", nn_test=nn_test, retrain_shield=retrain_shield, shield_test=shield_test, test_episodes=test_episodes)