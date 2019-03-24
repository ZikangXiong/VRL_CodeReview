# -*- coding: utf-8 -*-
# -------------------------------
# Author: Zikang Xiong
# Email: zikangxiong@gmail.com
# Date:   2019-02-10 15:40:07
# Last Modified by:   Zikang Xiong
# Last Modified time: 2019-03-22 18:08:35
# -------------------------------
import numpy as np
import random

def test_necessity(env, actor, shield_state_list):
    """Start from a shield states, see if 
        the shield action is necessary.
    
    Args:
        env (Environment): Test Environment
        actor (DDPG.ActorNetwork): agent
        shield_state_list (list): shield state list
    """
    TEST_STEP = 5000
    SAMPLE_SIZE = 5000
    fail_time = 0.0

    now = 0
    total = len(shield_state_list)
    if total >= SAMPLE_SIZE:
        indices = random.sample(range(SAMPLE_SIZE), SAMPLE_SIZE)
        total = SAMPLE_SIZE
    else:
        indices = range(total)

    for i in indices:
        ss = shield_state_list[i]
        x = env.reset(ss)
        for step in range(TEST_STEP):
            a = actor.predict(np.reshape(x, (1, actor.s_dim)))
            x, _, t = env.step(a.reshape(actor.a_dim, 1))
            if t:
                print "{}/{}: start from {}, terminal at step {}\n{}".format(now, total, ss, step, x)
                fail_time += 1
                break
        now += 1

    print "Test step: {}\n, Sample Size: {}\nstarting from shield state, fail time: {}\n, ratio: {}".format(TEST_STEP, total, fail_time, fail_time/total)

def test_necessity_extension(shield, fail_list, success_list, test_step=1000):
    """Test the shield calls of 
    the initial states that will not fail 
    and the initial states that will fail
    
    Args:
        shield (Shield): Shield
    """
    print "#### Shield for Fail case ####"
    shield.test_shield(x0_list=fail_list, test_step=1000)
    print "#### Shield for Success case ####"
    shield.test_shield(x0_list=success_list, test_step=1000)

