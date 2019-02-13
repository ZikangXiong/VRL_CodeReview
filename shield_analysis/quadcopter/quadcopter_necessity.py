import sys
sys.path.append("../")

from main import *
from Environment import Environment
from shield import Shield
from DDPG import *
from shield_analysis.log_scan import read_scan
from shield_analysis.shield_necessity import test_necessity

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
    shield_list = read_scan("quadcopter/quadcopter.log_ret.pkl")
    test_necessity(env, actor, shield_list)

quadcopter ("random_search", 50, 100, 0, [240,200], [280,240,200], "../ddpg_chkp/quadcopter/240200280240200/") 