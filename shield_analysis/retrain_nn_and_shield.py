# -*- coding: utf-8 -*-
# -------------------------------
# Author: Zikang Xiong
# Email: zikangxiong@gmail.com
# Date:   2019-02-11 11:32:11
# Last Modified by:   Zikang Xiong
# Last Modified time: 2019-02-11 15:08:30
# -------------------------------
import numpy as np

def unit_vector(vector):
    """ Returns the unit vector of the vector. """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v1_u = v1_u.reshape((len(v1_u)))
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))[0]/np.pi*180

def distance(v1, v2):
    return np.linalg.norm(v1 - v2)


def retrain_nn_and_shield(actor, K, shield_state_list):
    ssl_len = len(shield_state_list)
    assert ssl_len > 0
    process = 0.0

    angle_sum = 0.0
    distance_sum = 0.0

    for ss in shield_state_list:
        a_u = actor.predict(np.reshape(ss, (1, actor.s_dim)))
        s_u = K.dot(ss)
        angle_sum += angle_between(a_u, s_u)
        distance_sum += distance(a_u, s_u)
        process += 1
        if process % 1000 == 0:
            print "Process: {}/{}".format(process, ssl_len) 
    print "Process: {}/{}".format(ssl_len, ssl_len) 

    print "average angle difference:{}\naverage distance difference:{}".format(angle_sum/ssl_len, distance_sum/ssl_len)
