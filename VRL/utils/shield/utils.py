# -*- coding: utf-8 -*-
# -------------------------------
# Author: Zikang Xiong
# Email: zikangxiong@gmail.com
# Date:   2020-01-10 23:49:31
# Last Modified by:   Zikang Xiong
# Last Modified time: 2020-01-13 15:22:45
# -------------------------------
import re
import numpy as np

from numba import jit, float64
import pickle

from shield import Shield


def barrier_certificate_str2func(bc_str, vars_num, enable_jit=False):
    """transform julia barrier string to function

    Args:
        bc_str (str): string
        vars_num (int): the dimension number of state
        enable_jit: enable jit, the performance of B will increase, but it takes time to preprocess B
    """
    eval_str = re.sub("\^", r"**", bc_str)
    variables = ["x" + str(i + 1) for i in range(vars_num)]

    var_pattern = re.compile(r"(?P<var>x\d*)")
    eval_str = var_pattern.sub(r'*\g<var>', eval_str)

    args_str = ""
    for arg in variables:
        args_str += (arg + ",")
    args_str = args_str[:-1]
    if enable_jit:
        exec(("@jit" + "(float64 ({}))" + "\ndef B({}): return {}")
             .format(("float64," * vars_num)[:-1], args_str, eval_str)) in locals()
    else:
        exec("""def B({}): return {}""".format(args_str, eval_str))

    return B


def get_values_name(vars_num):
    return ["state[" + str(i) + "][0]" for i in range(vars_num)]


def state2list(state):
    return [x[0] for x in state.tolist()]


def save_shield(shield, model_path):
    if shield.env.continuous:
        with open(model_path + "/shield.model", "w") as f:
            for B_str in shield.B_str_list:
                f.write(B_str)
        np.save(model_path + "/K.model",
                np.array(shield.K_list), allow_pickle=True)
        np.save(model_path + "/initial_range.model",
                np.array(shield.initial_range_list), allow_pickle=True)

    else:
        # TODO: verify this works
        with open(model_path + "/shield.model", "wb") as f:
            pickle.dump(shield.O_inf_list, f)
        np.save(model_path + "/K.model",
                np.array(shield.K_list), allow_pickle=True)
        np.save(model_path + "/initial_range.model",
                np.array(shield.initial_range_list), allow_pickle=True)


def load_shield(env, neural_policy, model_path, enable_jit):
    inv_list = []
    initial_range_list = []
    K_list = []

    if env.continuous:
        K_list = [K for K in np.load(model_path + "/K.model.npy")]
        initial_range_list = [initr for initr in np.load(
            model_path + "/initial_range.model.npy", allow_pickle=True)]
        with open(model_path + "/shield.model", "r") as f:
            for B_str in f:
                inv_list.append(B_str)
    else:
        K_list = [K for K in np.load(
            model_path + "/K.model.npy", allow_pickle=True)]
        initial_range_list = [initr for initr in np.load(
            model_path + "/initial_range.model.npy", allow_pickle=True)]
        # TODO: save oinf with pickle
        inv_list = pickle.load(model_path+"/oinf.model.pkl")

    return Shield(env, neural_policy, K_list, initial_range_list, inv_list, enable_jit)
