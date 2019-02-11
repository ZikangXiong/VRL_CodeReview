# -*- coding: utf-8 -*-
# -------------------------------
# Author: Zikang Xiong
# Email: zikangxiong@gmail.com
# Date:   2019-02-10 15:40:07
# Last Modified by:   Zikang Xiong
# Last Modified time: 2019-02-10 20:06:32
# -------------------------------
import numpy as np


def test_necessity(env, actor, shield_state_list):
    """Start from a shield states, see if 
    	the shield action is necessary.
    
    Args:
        env (Environment): Test Environment
        actor (DDPG.ActorNetwork): agent
        shield_state_list (list): shield state list
    """
    TEST_STEP = 5000
    fail_time = 0.0

    now = 0
    total = len(shield_state_list)
    for ss in shield_state_list:
    	x = env.reset(ss)
    	for step in range(TEST_STEP):
    		a = actor.predict(np.reshape(x, (1, actor.s_dim)))
    		x, _, t = env.step(a)
    		if t:
    			print "{}/{}: terminal at step {}\n{}".format(now, total, step, ss)
    			fail_time += 1
    			break
    	now += 1

    print "Test step: {}\n, \
    		Test state list length: {}\n\
    		starting from shield state, fail time: {}\n, \
    		ratio: {}\
    		".format(TEST_STEP, len(shield_state_list), fail_time, fail_time/ len(shield_state_list))
