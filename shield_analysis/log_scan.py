# -*- coding: utf-8 -*-
# -------------------------------
# Author: Zikang Xiong
# Email: zikangxiong@gmail.com
# Date:   2019-02-10 13:45:51
# Last Modified by:   Zikang Xiong
# Last Modified time: 2019-02-10 16:26:15
# -------------------------------
import re
import numpy as np 
import pickle

def log_scan(log_path):
    """naive scan log, return a list of shield state
    
    Args:
        log_path (string): log path
    
    Returns:
        list: list of shield state array.
    """
    with open(log_path, 'r') as log_file:
        log_buffer = log_file.read()

    pattern = r"Shield! in state: \n(.+?\]\])"
    prog = re.compile(pattern, flags=re.DOTALL)
    result = prog.findall(log_buffer)

    ret_list = []
    for i in result:
        s = np.fromstring(i.replace("[", "").replace("]", ""), dtype=float, sep=' ')
        s = s.reshape((len(s), 1))  
        ret_list.append(s)

    with open(log_path+"_ret.pkl", "wb") as fp:
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
