# -*- coding: utf-8 -*-
# -------------------------------
# Author: Zikang Xiong
# Email: zikangxiong@gmail.com
# Date:   2020-01-10 22:32:42
# Last Modified by:   Zikang Xiong
# Last Modified time: 2020-01-13 15:05:21
# -------------------------------
import numpy as np

def test_controller (env, K, x0, simulation_steps, rewardf, continuous=False, timestep=.01, coffset=None, bias=False):
    # TODO: env type
    if type(env) is "":
        f = lambda x, u: env.A.dot(x)+env.B.dot(u) 
    elif type(env) is "":
        f = env.polyf

    return test_controller_helper(f, K, x0, simulation_steps, rewardf, continuous, timestep, coffset, bias)

def test_controller_helper (f, K, x0, simulation_steps, rewardf, continuous=False, timestep=.01, coffset=None, bias=False):
    time = np.linspace(0, simulation_steps, simulation_steps, endpoint=True)
    xk = x0 #np.matrix(".0 ; 0 ; .0 ; 0.1")
    reward = 0
    for t in time:
        uk = K.dot(np.vstack([xk,[1]])) if bias else K.dot(xk)
        reward += rewardf(xk, uk)
        # Use discrete or continuous semantics based on user's choice
        if continuous:
          xk = xk + timestep*(f(xk, uk)) if coffset is None else xk + timestep*(f(xk, uk)+coffset)
        else:
          xk = f(xk, uk) if coffset is None else f(xk, uk)+coffset
    #print("Score of the trace: {}".format(reward) )
    return reward

def test_shield(self, test_ep=1, test_step=5000, x0=None, mode="single", loss_compensation=0, shield_combo=1, mute=False):
    """test if shield works

    Args:
        test_ep (int, optional): test episodes
        test_step (int, optional): test step in each episode
    """
    assert shield_combo > 0
    assert loss_compensation >= 0

    fail_time = 0
    success_time = 0
    fail_list = []
    self.shield_count = 0
    combo_remain = 0

    for ep in range(test_ep):
        if x0 is not None:
            x = self.env.reset(x0)
        else:
            x = self.env.reset()
        init_x = x
        for i in range(test_step):
            u = np.reshape(self.actor.predict(np.reshape(np.array(x),
                                                         (1, self.actor.s_dim))), (self.actor.a_dim, 1))

            # safe or not
            if self.detactor(x, u, mode=mode, loss_compensation=loss_compensation) or (combo_remain > 0):
                if combo_remain == 0:
                    combo_remain = shield_combo

                u = self.call_shield(x, mute=mute)
                if not mute:
                    print("!shield at step {}".format(i))

                combo_remain -= 1

            # step
            x, _, terminal = self.env.step(u)

            # success or fail
            if terminal:
                if np.sum(np.power(self.env.xk, 2)) < self.env.terminal_err:
                    success_time += 1
                else:
                    fail_time += 1
                    fail_list.append((init_x, x))
                break

            if i == test_step - 1:
                success_time += 1

        print("----epoch: {} ----".format(ep))
        print('initial state:\n', init_x, '\nterminal state:\n', x, '\nlast action:\n', self.env.last_u)
        print("----step: {} ----".format(i))

    print('Success: {}, Fail: {}'.format(success_time, fail_time))
    print('#############Fail List:###############')
    for (i, e) in fail_list:
        print('initial state:\n{}\nend state: \n{}\n----'.format(i, e))

    print('shield times: {}, shield ratio: {}'.format(self.shield_count, float(self.shield_count) / (test_ep * test_step)))