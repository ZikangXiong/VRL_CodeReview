import sys
sys.path.append(".")

from main import *
from shield import Shield
from Environment import Environment
import argparse
from DDPG import *
from metrics import draw_actor

def navigation(learning_method, number_of_rollouts, simulation_steps, learning_eposides, actor_structure, critic_structure, train_dir,\
                nn_test=False, retrain_shield=False, shield_test=False, test_episodes=100, retrain_nn=False):

  A = np.matrix([
    [ 0., 0., 1., 0., 0., 0., 0., 0.],
    [ 0., 0., 0., 1., 0., 0., 0., 0.],
    [ 0., 0., -1.2, .1, 0., 0., 0., 0.],
    [ 0., 0., .1, -1.2, 0., 0., 0., 0.],
    [ 0., 0., 0., 0., 0., 0., 1., 0.],
    [ 0., 0., 0., 0., 0., 0., 0., 1.],
    [ 0., 0., 0., 0., 0., 0., -1.2, .1],
    [ 0., 0., 0., 0., 0., 0., .1, -1.2]
    ])
  B = np.matrix([
    [0,0,0,0],
    [0,0,0,0],
    [1,0,0,0],
    [0,1,0,0],
    [0,0,0,0],
    [0,0,0,0],
    [0,0,1,0],
    [0,0,0,1]
    ])

  d, p = B.shape

  h = .01

  # amount of Gaussian noise in dynamics
  eq_err = 1e-2

  target = np.array([[0],[5], [0], [0], [0], [6], [0], [0]])

  #intial state space
  s_min = np.array([[-2.1],[-4.1],[0],[0], [1.9], [-4.1], [0], [0]])
  s_max = np.array([[-1.9],[-3.9],[0],[0], [2.1], [-3.9], [0], [0]])
  s_min -= target
  s_max -= target

  coffset = np.dot(A, target)
  print "coffset:\n {}".format(coffset)

  #reward functions
  Q = np.zeros((d,d), float)
  np.fill_diagonal(Q, 0.1)

  R = np.zeros((p,p), float)
  np.fill_diagonal(R, .0005)

  #user defined unsafety condition
  def unsafe_eval(x):
    return 0.25 - (pow((x[0]-x[4]+target[0,0]-target[4,0]), 2) + pow((x[1]-x[5]+target[1,0]-target[5,0]), 2))
  def unsafe_string():
    return ["0.25 - (x[1]-x[5])^2 - (x[2]-x[6]-1)^2"]

  def rewardf(x, u):
    reward = 0
    reward += -np.dot(x.T,Q.dot(x))-np.dot(u.T,R.dot(u))
    # Do not allow two cars crashing with each other
    distance = pow((x[0]-x[4]+target[0,0]-target[4,0]), 2) + pow((x[1]-x[5]+target[1,0]-target[5,0]), 2)
    if (unsafe_eval(x) >= 0):
      reward -= 2000
    return reward

  def testf(x, u):
    unsafe = True
    if (unsafe_eval(x) < 0):
      unsafe = False
    if (unsafe):
      print "unsafe : {}".format(x+target)
      return -1
    return 0  

  u_min = np.array([[-50.], [-50], [-50.], [-50]])
  u_max = np.array([[ 50.], [ 50], [ 50.], [ 50]])

  env = Environment(A, B, u_min, u_max, s_min, s_max, None, None, Q, R, 
            continuous=True, rewardf=rewardf, unsafe=True, unsafe_property=unsafe_string,
            terminal_err=0, bad_reward=-2000)

  ############ Train and Test NN model ############
  if retrain_nn:
    args = { 'actor_lr': 0.0001,
         'critic_lr': 0.001,
         'actor_structure': actor_structure,
         'critic_structure': critic_structure, 
         'buffer_size': 1000000,
         'gamma': 0.99,
         'max_episode_len': 100,
         'max_episodes': 1000,
         'minibatch_size': 64,
         'random_seed': 6553,
         'tau': 0.005,
         'model_path': train_dir+"retrained_model.chkp",
         'enable_test': nn_test, 
         'test_episodes': test_episodes,
         'test_episodes_len': 5000}
  else:
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
             'enable_test': nn_test, 
             'test_episodes': test_episodes,
             'test_episodes_len': 5000}
  actor = DDPG(env, args)
  draw_actor(env, actor, 5000)

  #################### Shield #################
  model_path = os.path.split(args['model_path'])[0]+'/'
  linear_func_model_name = 'K.model'
  model_path = model_path+linear_func_model_name+'.npy'

  shield = Shield(env, actor, model_path, force_learning=retrain_shield)
  shield.train_shield(learning_method, number_of_rollouts, simulation_steps, rewardf=rewardf, testf=testf, eq_err=eq_err, explore_mag = 0.4, step_size = 0.5)
  if shield_test:
    shield.test_shield(test_episodes, 5000)

  actor.sess.close()

K = [[  0.30923238,  -8.21860525,  -3.55703702, -10.34199675,   2.44158251,
    2.89652896,  10.77504469,   5.88230113],
 [  8.6784715,   -7.6048211,    6.16931975, -11.17736349,  -3.08448797,
   -0.52737569,  -1.87705704,   3.7769782 ],
 [-15.82419075,   1.84858296,  -1.84047885,  -3.68344919,   1.04760547,
   10.4921827,   -8.3316463,   11.01894962],
 [ -4.43962048,   0.20532201,  -6.85347123,  -5.4735437,    2.8929692,
  -11.30919106,  -7.42490079,  -8.19019377]]

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Running Options')
  parser.add_argument('--nn_test', action="store_true", dest="nn_test")
  parser.add_argument('--retrain_shield', action="store_true", dest="retrain_shield")
  parser.add_argument('--shield_test', action="store_true", dest="shield_test")
  parser.add_argument('--test_episodes', action="store", dest="test_episodes", type=int)
  parser.add_argument('--retrain_nn', action="store_true", dest="retrain_nn")
  parser_res = parser.parse_args()
  nn_test = parser_res.nn_test
  retrain_shield = parser_res.retrain_shield
  shield_test = parser_res.shield_test
  test_episodes = parser_res.test_episodes if parser_res.test_episodes is not None else 100
  retrain_nn = parser_res.retrain_nn

  navigation ("random_search", 200, 100, 0, [500, 400, 300], [600, 500, 400, 300], "ddpg_chkp/2-car-navigation/500400300600500400300/", \
    nn_test=nn_test, retrain_shield=retrain_shield, shield_test=shield_test, test_episodes=test_episodes, retrain_nn=retrain_nn)
