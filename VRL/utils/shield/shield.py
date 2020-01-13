from VRL.utils.verification.z3verify import verify_controller_z3
from VRL.utils.metrics.simulation import test_controller
from VRL.utils.linear_policy import candidate_program
import VRL.utils.verification.SOS.tools.verify as sos_verify
import VRL.utils.verification.Polyhedron.tools.verify as poly_verify
from VRL.utils.third_party.pympc.geometry.polyhedron import Polyhedron

from utils import state2list, barrier_certificate_str2func

import numpy as np

shield_testing_on_x_ep_len = 10

# TODO: oinf name uniform
class Shield(object):

    def __init__(self, env, neural_policy, K_list, initial_range_list, inv_list, enable_jit=False):
        # enable jit will increase the initializing time,
        # but decrease the running time significantly in high demension case.

        self.env = env
        self.neural_policy = neural_policy

        # linear policy initialize
        self.K = None
        self.K_list = K_list
        self.initial_range_list = initial_range_list
        self.continuous = env.continuous

        # invariant initialize
        if self.env.continuous:
            self.B = None

            self.B_list = []
            for inv in inv_list:
                self.B_list.append(barrier_certificate_str2func(
                    inv, env.state_dim, enable_jit))
        else:
            self.O_inf = None
            self.O_inf_list = inv_list

        # metrics
        self.shield_count = 0

    def select_shield(self):
        i = -1

        if (len(self.initial_range_list) > 1):
            lowboundaries = np.array([item[0]
                                      for item in self.initial_range_list])
            upboundaries = np.array([item[1]
                                     for item in self.initial_range_list])
            select_list = [(self.env.x0 > low).all() * (self.env.x0 < high).all()
                           for low, high in zip(lowboundaries, upboundaries)]
            i = select_list.index(True)
        elif (len(self.initial_range_list) == 1):
            i == 0
        else:
            raise RuntimeError("No shield available!")

        self.K = self.K_list[i]

        if self.continuous:
            self.B = self.B_list[i]
            return self.B
        else:
            self.O_inf = self.O_inf_list[i]
            return self.O_inf

    def detactor(self, x, u, mode="single"):
        """detact if there are dangerous state in furture

        Args:
            x: current state
            u: current action
            mode (str, optional): single(faster, more calls) -> choose one shield according to the initial state.
                                  all(slower, less calls) -> use all shield at run time, if all the B > 0, call shield.

        Returns:
            Bool: True -> call shield
                  False -> call neural network
        """
        mode_tuple = ("single", "all")
        assert mode in mode_tuple

        xk = self.env.simulation(u)
        # single shield model
        if mode == mode_tuple[0]:
            # continuous
            if self.env.continuous:
                if self.B is None:
                    self.select_shield()
                B_value = self.B(*state2list(xk))

                return B_value > 0.0
            # discrete
            else:
                if self.O_inf is None:
                    self.select_shield()
                if self.O_inf.contains(xk):
                    return False
                return True

        # TODO: shield "all" model
        # # all shield model
        # elif mode == mode_tuple[1]:
        #     # continuous
        #     if self.env.continuous:
        #         current_B_result = []
        #         if self.last_B_result == []:
        #             lowboundaries = np.array(
        #                 [i[0] for i in self.initial_range_list])
        #             upboundaries = np.array([i[1]
        #                                      for i in self.initial_range_list])
        #             self.last_B_result = [np.logical_not((self.env.x0 > low).all(
        #             ) * (self.env.x0 < high).all()) for low, high in zip(lowboundaries, upboundaries)]
        #         for B in self.B_list:
        #             B_value = B(*state2list(xk))
        #             res = B_value > 0.0
        #             current_B_result.append(res)

        #         if np.array(current_B_result).all():
        #             # The K will be called latter
        #             self.K = self.K_list[self.last_B_result.index(False)]
        #             return True

        #         self.last_B_result = current_B_result
        #         return False
        #     # discrete
        #     else:
        #         current_O_inf_result = []
        #         if self.last_O_inf_result == []:
        #             lowboundaries = np.array(
        #                 [i[0] for i in self.initial_range_list])
        #             upboundaries = np.array([i[1]
        #                                      for i in self.initial_range_list])
        #             self.last_O_inf_result = [np.logical_not((self.env.x0 > low).all(
        #             ) * (self.env.x0 < high).all()) for low, high in zip(lowboundaries, upboundaries)]

        #         for O_inf in self.O_inf_list:
        #             res = not O_inf.contains(xk)
        #             current_O_inf_result.append(res)

        #         if np.array(current_O_inf_result).all():
        #             # The K will be called latter
        #             self.K = self.K_list[self.last_O_inf_result.index(False)]
        #             return True

        #         self.last_O_inf_result = current_O_inf_result
        #         return False

    def call_shield(self, x, mute=False):
        """call shield

        Args:
            x : current state
            mute (bool, optional): print(!shield or not)

        Returns:  
            shield action
        """
        u = self.K.dot(x)
        if not mute:
            print('Shield! in state: \n', x)
        self.shield_count += 1

        return u


def train_sos_shield(env,
                     neural_policy,
                     learning_method,
                     number_of_rollouts,
                     simulation_steps,
                     explore_mag=.04,
                     step_size=.05,
                     eq_err=1e-2,
                     rewardf=None,
                     testf=None,
                     # names=None,
                     coffset=None,
                     bias=False,
                     # discretization=False,
                     lqr_start=False,
                     degree=4,
                     without_nn_guide=False):

    def verifyf(x, initial_size, Theta, K):
        return sos_verify(env, K, x, initial_size)

    def learningf(x):
        K = candidate_program.generator(env, x, eq_err,
                                        learning_method, number_of_rollouts,
                                        simulation_steps, neural_policy,
                                        env.x_min, env.x_max, rewardf=rewardf,
                                        continuous=env.continuous, timestep=env.timestep,
                                        explore_mag=explore_mag, step_size=step_size,
                                        coffset=coffset, bias=bias,
                                        unsafe_flag=env.unsafe, lqr_start=lqr_start,
                                        without_nn_guide=without_nn_guide)
        return K

    def default_testf(x, u):
        if env.unsafe:
            if ((np.array(x) < env.x_max) * (np.array(x) > env.x_min)).all(axis=1).any():
                return -1
            else:
                return 0
        else:
            if ((x < env.x_max).all() and (x > env.x_min).all()):
                return 0
            else:
                return -1

    def testf(x0, K):
        test_reward = testf if testf is not None else default_testf
        result = test_controller(env, K, x0, simulation_steps * shield_testing_on_x_ep_len, rewardf=test_reward,
                                 continuous=env.continuous, timestep=env.timestep, coffset=coffset, bias=bias)
        return result

    Theta = (env.s_min, env.s_max)
    result, resultList = verify_controller_z3(Theta,
                                              learningf,
                                              testf,
                                              verifyf,
                                              continuous=env.continuous)
    print("Shield synthesis result: {}".format(result))

    if result:
        B_str_list = []
        K_list = []
        initial_range_list = []

        for (x, initial_size, inv, K) in resultList:
            B_str_list.append(inv + "\n")
            K_list.append(K)
            initial_range = np.array(
                [x - initial_size.reshape(len(initial_size), 1), x + initial_size.reshape(len(initial_size), 1)])
            initial_range_list.append(initial_range)

        return B_str_list, K_list, initial_range_list

# TODO: uniform shield training
def train_polyhedro_shield(env,
                           neural_policy,
                           learning_method,
                           number_of_rollouts,
                           simulation_steps,
                           explore_mag=.04,
                           step_size=.05,
                           eq_err=1e-2,
                           rewardf=None,
                           testf=None,
                           # names=None,
                           coffset=None,
                           bias=False,
                           # discretization=False,
                           lqr_start=False,
                           degree=4,
                           without_nn_guide=False):

    def verifyf(x, initial_size, Theta, K):
        O_inf = poly_verify(env, np.asarray(K))

        init_min = np.array([[x[i, 0] - initial_size[i]]
                             for i in range(env.state_dim)])
        init_max = np.array([[x[i, 0] + initial_size[i]]
                             for i in range(env.state_dim)])

        S0 = Polyhedron.from_bounds(env.s_min, env.s_max)
        S = Polyhedron.from_bounds(init_min, init_max)
        S = S.intersection(S0)
        ce = S.is_included_in_with_ce(O_inf)
        return (ce is None), O_inf

    def learningf(x):
        K = candidate_program.generator(env, x, eq_err,
                                        learning_method, number_of_rollouts,
                                        simulation_steps, neural_policy,
                                        env.x_min, env.x_max, rewardf=rewardf,
                                        continuous=env.continuous, timestep=env.timestep,
                                        explore_mag=explore_mag, step_size=step_size,
                                        coffset=coffset, bias=bias,
                                        unsafe_flag=env.unsafe, lqr_start=lqr_start,
                                        without_nn_guide=without_nn_guide)
        return K

    def default_testf(x, u):
        if env.unsafe:
            if ((np.array(x) < env.x_max) * (np.array(x) > env.x_min)).all(axis=1).any():
                return -1
            else:
                return 0
        else:
            if ((x < env.x_max).all() and (x > env.x_min).all()):
                return 0
            else:
                return -1

    def testf(x0, K):
        test_reward = testf if testf is not None else default_testf
        result = test_controller(env, K, x0, simulation_steps * shield_testing_on_x_ep_len, rewardf=test_reward,
                                 continuous=env.continuous, timestep=env.timestep, coffset=coffset, bias=bias)
        return result

    # verify
    Theta = (env.s_min, env.s_max)
    result, resultList = verify_controller_z3(Theta,
                                              learningf,
                                              testf,
                                              verifyf,
                                              continuous=env.continuous)
    print("Shield synthesis result: {}".format(result))

    # verification results
    if result:
        Oinf_list = []
        K_list = []
        initial_range_list = []

        for (x, initial_size, inv, K) in resultList:
            Oinf_list.append(inv + "\n")
            K_list.append(K)
            initial_range = np.array(
                [x - initial_size.reshape(len(initial_size), 1), x + initial_size.reshape(len(initial_size), 1)])
            initial_range_list.append(initial_range)

        return Oinf_list, K_list, initial_range_list
