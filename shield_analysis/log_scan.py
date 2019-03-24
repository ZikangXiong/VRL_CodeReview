# -*- coding: utf-8 -*-
# -------------------------------
# Author: Zikang Xiong
# Email: zikangxiong@gmail.com
# Date:   2019-02-10 13:45:51
# Last Modified by:   Zikang Xiong
# Last Modified time: 2019-03-22 18:15:12
# -------------------------------
import re
import numpy as np 
import pickle

def log_scan(log_path, scan_type="ss"):
    """naive scan log, return a list of shield state
    
    Args:
        log_path (string): log path
        type (string): ss-shield state, fis-fail initial state, sis-success initial state
    
    Returns:
        list: list of shield state array.
    """
    assert scan_type in ("ss", "fis", "sis")

    with open(log_path, 'r') as log_file:
        log_buffer = log_file.read()

    if scan_type == "ss":
        pattern = r"Shield! in state: \n(.+?\]\])"
        prog = re.compile(pattern, flags=re.DOTALL)
        result = prog.findall(log_buffer)

    elif scan_type == "fis":
        fail_list_parttern = r"###\n(.+?)\ntest run time:"
        fail_list_prog = re.compile(fail_list_parttern, flags=re.DOTALL)
        fail_list = fail_list_prog.findall(log_buffer)[0]
        state_pattern = r"initial state: \n(.+?\]\])"
        state_prog = re.compile(state_pattern, flags=re.DOTALL)
        result = state_prog.findall(fail_list)

    elif scan_type == "sis":
        nn_test_log = log_buffer.split("###")[0]
        state_pattern = r"initial state:\n(.+?\]\])"
        state_prog = re.compile(state_pattern, flags=re.DOTALL)
        result = state_prog.findall(nn_test_log)

        fail_list_parttern = r"#############Fail List:###############\n(.+?)\ntest run time:"
        fail_list_prog = re.compile(fail_list_parttern, flags=re.DOTALL)
        fail_list = fail_list_prog.findall(log_buffer)[0]
        state_pattern = r"initial state: \n(.+?\]\])"
        state_prog = re.compile(state_pattern, flags=re.DOTALL)
        fis_result = state_prog.findall(fail_list)

        fis_list = []
        for i in fis_result:
            result.remove(i)

    ret_list = []
    for i in result:
        s = np.fromstring(i.replace("[", "").replace("]", ""), dtype=float, sep=' ')
        s = s.reshape((len(s), 1))  
        ret_list.append(s)

    with open(log_path+"_{}_ret.pkl".format(scan_type), "wb") as fp:
        pickle.dump(ret_list, fp)

    return ret_list

def read_scan(log_scan_path):
    """read scan results
    
    Args:
        log_scan_path (string): pkl file path
    
    Returns:
        list: list of shield state array.
    """
    with open(log_scan_path, "rb") as fp: 
        ret_list = pickle.load(fp)
    return ret_list

log_scan("oscillator18/oscillator18.log", "sis")
