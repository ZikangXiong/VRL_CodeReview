# -*- coding: utf-8 -*-
# -------------------------------
# Author: He Zhu
# Date:   2020-01-10 22:44:46
# Last Modified by:   Zikang Xiong
# Last Modified time: 2020-01-13 13:52:39
# -------------------------------
from VRL.utils.third_party.pympc.geometry.polyhedron import Polyhedron
from VRL.utils.third_party.pympc.dynamics.discrete_time_systems import LinearSystem
from VRL.utils.third_party.pympc.dynamics.discrete_time_systems import mcais
import scipy.linalg as la

import time


def verify(env, K, dimensions=[0, 1]):
    S = LinearSystem(env.A, env.B)
    X = Polyhedron.from_bounds(env.x_min, env.x_max)
    U = Polyhedron.from_bounds(env.u_min, env.u_max)
    D = X.cartesian_product(U)

    O_inf = S.mcais(K, D)

    return O_inf


def verify_via_discretization(Acl, h, x_min, x_max):
    # discretize the system for efficient verification
    X = Polyhedron.from_bounds(x_min, x_max)
    O_inf = mcais(la.expm(Acl * h), X, verbose=False)

    return O_inf
