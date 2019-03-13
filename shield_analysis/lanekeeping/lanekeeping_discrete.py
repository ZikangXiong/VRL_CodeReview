import sys
sys.path.append("../")

from main import *
from shield import Shield
from Environment import Environment
from DDPG import *
from metrics import *

def lanekeep (learning_method, number_of_rollouts, simulation_steps, learning_eposides, actor_structure, critic_structure, train_dir):
  v0 = 27.7
  cf = 133000
  cr = 98800
  M  = 1650
  b  = 1.59
  a  = 1.11
  Iz = 2315.3

  ds = 4
  us = 2

  disturbance_x_min = np.array([[0],[0],[-0.035],[0]])
  disturbance_x_max = np.array([[0],[0],[ 0.035],[0]])

  #Dynamics that are defined as a continuous function!
  def f (x, u):
    rd = random.uniform(-0.6, 0.6)
    delta = np.zeros((ds, 1), float)
    delta[0,0] = 1*x[1,0] + v0*x[2,0] + random.uniform(disturbance_x_min[0], disturbance_x_max[0])                                                                        #lateral displacement
    delta[1,0] = (-1*(cf+cr)/(M*v0))*x[1,0] + ((b*cr-a*cf)/(M*v0)-v0)*x[3,0] + (cf/M)*u[0,0] + random.uniform(disturbance_x_min[1], disturbance_x_max[1])                 #lateral velocity
    delta[2,0] = x[3,0] + random.uniform(disturbance_x_min[2], disturbance_x_max[2])                                                                                      #error yaw angle
    delta[3,0] = ((b*cr-a*cf)/(Iz*v0))*x[1,0] + (-1*(a*a*cf + b*b*cr)/(Iz*v0))*x[3,0] + (a*cf/Iz)*u[1,0]  + random.uniform(disturbance_x_min[3], disturbance_x_max[3])    #yaw rate

    return delta

  #Closed loop system dynamics to text
  def f_to_str(K):
    kstr = K_to_str(K)
    f = []
    f.append("1*x[2] + 27.7*x[3] + d[1]")
    f.append("(-1*(133000+98800)/(1650*27.7))*x[2] + ((1.59*98800-1.11*133000)/(1650*27.7)-27.7)*x[4] + (133000/1650)*{} + d[2]".format(kstr[0]))
    f.append("x[4] + d[3]")
    f.append("((1.59*98800-1.11*133000)/(2315.3*27.7))*x[2] + (-1*(1.11*1.11*133000 + 1.59*1.59*98800)/(2315.3*27.7))*x[4] + (1.11*133000/2315.3)*{} + d[4]".format(kstr[1]))
    return f

  A = [
    [0, 1, 27.7, 0],
    [0, (-1*(133000+98800)/(1650*27.7)), 0, ((1.59*98800-1.11*133000)/(1650*27.7)-27.7)],
    [0, 0, 0, 1],
    [0, ((1.59*98800-1.11*133000)/(2315.3*27.7)), 0, (-1*(1.11*1.11*133000 + 1.59*1.59*98800)/(2315.3*27.7))]]
  B = [
    [0, 0], 
    [(133000/1650), 0],
    [0, 0],
    [0, (1.11*133000/2315.3)]]

  h = 0.01

  dim = len(A[0])

  # Discretize the system using matrix exponential
  A_conti = np.array(A)
  B_conti = np.array(B)

  #A = np.matrix(la.expm(A_conti * h))
  #B = np.matrix(np.dot( np.dot(np.linalg.pinv(A_conti), (A - np.eye(dim))), B_conti))

  A = A_conti*h + np.eye(A_conti.shape[0])
  B = B_conti*h

  # amount of Gaussian noise in dynamics
  eq_err = 1e-2

  #intial state space
  s_min = np.array([[ -0.1],[ -0.1], [-0.1], [ -0.1]])
  s_max = np.array([[  0.1],[  0.1], [ 0.1], [  0.1]])

  Q = np.matrix("1 0 0 0; 0 1 0 0 ; 0 0 1 0; 0 0 0 1")
  R = np.matrix("1 0; 0 1")

  #user defined unsafety condition
  def unsafe_eval(x):
    if (x[0,0] > 0.9 or x[0, 0] < -0.9): # keep a safe distance from the car in front of you
      return True
    return False

  def unsafe_string():
    return ["-(x[1]- -0.9)*(0.9-x[1])"]

  def rewardf(x, Q, u, R):
    reward = 0
    reward += -np.dot(x.T,Q.dot(x))-np.dot(u.T,R.dot(u))

    if (unsafe_eval(x)):
      reward -= 1
    return reward

  def testf(x, u):
    if (unsafe_eval(x)):
      return -1
    return 0 

  u_min = np.array([[-1.], [-1.]])
  u_max = np.array([[ 1.], [ 1.]])

  x_min = np.array([[-0.9],[-2],[-2],[-2]])
  x_max = np.array([[ 0.9],[ 2],[ 2],[ 2]])

  env = Environment(A, B, u_min, u_max, s_min, s_max, x_min, x_max, Q, R, continuous=False)

  args = { 'actor_lr': 0.0001,
           'critic_lr': 0.001,
           'actor_structure': actor_structure,
           'critic_structure': critic_structure, 
           'buffer_size': 1000000,
           'gamma': 0.99,
           'max_episode_len': 10,
           'max_episodes': learning_eposides,
           'minibatch_size': 64,
           'random_seed': 6553,
           'tau': 0.005,
           'model_path': train_dir+"model.chkp",
           'enable_test': False, 
           'test_episodes': 1000,
           'test_episodes_len': 1000}
  actor = DDPG(env, args)
  # draw_actor(env, actor, 2000)

  model_path = os.path.split(args['model_path'])[0]+'/'
  linear_func_model_name = 'K.model'
  model_path = model_path+linear_func_model_name+'.npy'
  shield = Shield(env, actor, model_path=model_path, force_learning=False)
  shield.train_shield(learning_method, number_of_rollouts, simulation_steps, explore_mag = 0.1, step_size = 0.1, without_nn_guide=False)
  # K = np.array([[-4.21528005, -0.55237926, -9.45587692, -0.50038062],
  #               [-2.04298819,  0.50994964,  3.29331539, -1.02047674]])
  # draw_K (env, shield.K_list[0], 2000)

  ################# Metrics ######################
  # actor_boundary(env, actor, 2000, 1000)
  # shield.shield_boundary(2000, 1000)
  terminal_err = 1e-1
  sample_steps = 2000
  sample_ep = 1000
  measure_steps = 100
  print "---\nterminal error: {}\nsample_ep: {}\nsample_steps: {}\nmeasure_steps: {}\n---".format(terminal_err, sample_ep, sample_steps, measure_steps)
  # dist_nn_lf = metrics.distance_between_linear_function_and_neural_network(env, actor, shield.K, terminal_err, sample_ep, sample_steps)
  # print "dist_nn_lf: ", dist_nn_lf
  nn_perf = metrics.neural_network_performance_converge(env, actor, terminal_err, sample_ep, sample_steps, measure_steps)
  print "nn_perf", nn_perf
  shield_perf = metrics.linear_function_performance_converge(env, shield.K_list[0], terminal_err, sample_ep, sample_steps, measure_steps)
  print "shield_perf", shield_perf

# K = np.array([[-4.21528005, -0.55237926, -9.45587692, -0.50038062],
#  [-2.04298819,  0.50994964,  3.29331539, -1.02047674]])

lanekeep("random_search", 10, 500, 0, [240, 200], [280, 240, 200], "../ddpg_chkp/lanekeeping/240200280240200/")