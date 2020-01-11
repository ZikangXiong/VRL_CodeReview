# -*- coding: utf-8 -*-
# -------------------------------
# Author: He Zhu
# Date:   2020-01-10 22:44:46
# Last Modified by:   Zikang Xiong
# Last Modified time: 2020-01-10 22:49:10
# -------------------------------
from VRL.utils.third_party.pympc.geometry.polyhedron import Polyhedron
from VRL.utils.third_party.pympc.dynamics.discrete_time_systems import LinearSystem
from VRL.utils.third_party.pympc.dynamics.discrete_time_systems import mcais
import scipy.linalg as la

import time


def verify(A, B, K, x_min, x_max, u_min, u_max, dimensions=[0, 1]):
    S = LinearSystem(A, B)
    X = Polyhedron.from_bounds(x_min, x_max)
    U = Polyhedron.from_bounds(u_min, u_max)
    D = X.cartesian_product(U)

    start = time.time()
    O_inf = S.mcais(K, D)
    end = time.time()
    print(("mcais execution time: {} secs".format(end - start)))

    return O_inf


def verify_via_discretization(Acl, h, x_min, x_max):
    # discretize the system for efficient verification
    X = Polyhedron.from_bounds(x_min, x_max)
    O_inf = mcais(la.expm(Acl * h), X, verbose=False)

    return O_inf
