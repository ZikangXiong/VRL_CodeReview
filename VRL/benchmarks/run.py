# -*- coding: utf-8 -*-
# -------------------------------
# Author: Zikang Xiong
# Email: zikangxiong@gmail.com
# Date:   2020-01-13 15:36:52
# Last Modified by:   Zikang Xiong
# Last Modified time: 2020-01-13 15:47:18
# -------------------------------
import argparse

def run(env_name, nn_test, retrain_shield, shield_test, test_episodes, retrain_nn):
	pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Running Options')
    parser.add_argument('--env_name', action="store", dest="env_name", type=str)
    parser.add_argument('--nn_test', action="store_true", dest="nn_test")
    parser.add_argument('--retrain_shield',
                        action="store_true", dest="retrain_shield")
    parser.add_argument(
        '--shield_test', action="store_true", dest="shield_test")
    parser.add_argument('--test_episodes', action="store",
                        dest="test_episodes", type=int)
    parser.add_argument('--retrain_nn', action="store_true", dest="retrain_nn")
    parser_res = parser.parse_args()

    nn_test = parser_res.nn_test
    retrain_shield = parser_res.retrain_shield
    shield_test = parser_res.shield_test
    test_episodes = parser_res.test_episodes if parser_res.test_episodes is not None else 100
    retrain_nn = parser_res.retrain_nn
