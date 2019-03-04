import sys
sys.path.append("../")

from main import *
from Environment import Environment
from shield import Shield
from DDPG import *

def quadcopter (learning_method, number_of_rollouts, simulation_steps, learning_eposides, actor_structure, critic_structure, train_dir):
    A = np.matrix([[1,1], [0,1]])
    B = np.matrix([[0],[1]])

    #intial state space
    s_min = np.array([[-0.5],[-0.5]])
    s_max = np.array([[ 0.5],[ 0.5]])

    # LQR quadratic cost per state
    Q = np.matrix("1 0; 0 0")
    R = np.matrix("1.0")

    x_min = np.array([[-1.],[-1.]])
    x_max = np.array([[ 1.],[ 1.]])
    u_min = np.array([[-15.]])
    u_max = np.array([[ 15.]])

    env = Environment(A, B, u_min, u_max, s_min, s_max, x_min, x_max, Q, R)

    args = { 'actor_lr': 0.001,
           'critic_lr': 0.01,
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
           'test_episodes': 1000,
           'test_episodes_len': 5000}
    actor = DDPG(env, args=args)
    # actor_boundary(env, actor)

    ################# Shield ######################
    model_path = os.path.split(args['model_path'])[0]+'/'
    linear_func_model_name = 'K.model'
    model_path = model_path+linear_func_model_name+'.npy'

    shield = Shield(env, actor, model_path, force_learning=False, debug=False)
    shield.train_shield(learning_method, number_of_rollouts, simulation_steps)
    # shield.test_shield(1000, 5000, mode="single")
    # shield.test_shield(10, 5000, mode="all")


    ################# Metrics ######################
    # actor_boundary(env, actor, epsoides=500, steps=200)
    # shield.shield_boundary(2000, 50)
    terminal_err = 4e-1
    sample_steps = 100
    sample_ep = 100
    print "---\nterminal error: {}\nsample_ep: {}\nsample_steps: {}\n---".format(terminal_err, sample_ep, sample_steps)
    # dist_nn_lf = metrics.distance_between_linear_function_and_neural_network(env, actor, shield.K, terminal_err, sample_ep, sample_steps)
    # print "dist_nn_lf: ", dist_nn_lf
    nn_perf = metrics.neural_network_performance(env, actor, terminal_err, sample_ep, sample_steps)
    print "nn_perf", nn_perf
    shield_perf = metrics.linear_function_performance(env, shield.K_list[0], terminal_err, sample_ep, sample_steps)
    print "shield_perf", shield_perf

    actor.sess.close()

quadcopter ("random_search", 50, 100, 0, [240,200], [280,240,200], "../ddpg_chkp/quadcopter/240200280240200/") 