# -*- coding: utf-8 -*-
# -------------------------------
# Author: Zikang Xiong
# Email: zikangxiong@gmail.com
# Date:   2020-01-10 22:51:52
# Last Modified by:   Zikang Xiong
# Last Modified time: 2020-01-13 14:54:00
# -------------------------------
import os
import subprocess
import platform
from threading import Timer

import numpy as np

from vcsos import genSOS, genSOSContinuousAsDiscreteMultipleUnsafes, genSOSwithBound, genSOSwithDisturbance


def dxdt(A, coffset=None):
    d, p = A.shape
    X = []
    for i in range(p):
        X.append("x[" + str(i + 1) + "]")

    f = []
    for i in range(len(A)):
        strstr = ""
        for k in range(len(X)):
            if (strstr is ""):
                strstr = str(A[i, k]) + "*" + X[k]
            else:
                strstr = strstr + "+" + str(A[i, k]) + "*" + X[k]
        if coffset is not None:
            strstr += ("+" + str(coffset[i, 0]))
        f.append(strstr)
    return f


def K_to_str(K):
    # Control policy K to text
    nvars = len(K[0])
    X = []
    for i in range(nvars):
        X.append("x[" + str(i + 1) + "]")

    ks = []
    for i in range(len(K)):
        strstr = ""
        for k in range(len(X)):
            if (strstr is ""):
                strstr = str(K[i, k]) + "*" + X[k]
            else:
                strstr = strstr + "+" + str(K[i, k]) + "*" + X[k]
        ks.append(strstr)
    return ks


def write_to_tmp(fname, sostext):
    fname = "/tmp/" + fname
    file = open(fname, "w")
    file.write(sostext)
    file.close()
    return fname


def get_julia_path():
    if platform.system() == "Linux":
        return "julia"
    else:
        return "/Applications/Julia-0.6.app/Contents/Resources/julia/bin/julia"


def generate_sos_linear_env(env, K, x, initial_size, degree=4):
    init = []
    initSOSPoly = []
    init_cnstr = []
    for i in range(env.state_dim):
        init.append("init" + str(i + 1) + " = (x[" + str(i + 1) + "] - " + str(
            env.s_min[i, 0]) + ")*(" + str(env.s_max[i, 0]) + "-x[" + str(i + 1) + "])")
    for i in range(env.state_dim):
        initSOSPoly.append("@variable m Zinit" +
                           str(i + 1) + " SOSPoly(Z)")
    for i in range(env.state_dim):
        init_cnstr.append(
            " - Zinit" + str(i + 1) + "*init" + str(i + 1))
    # Specs for initial conditions subject to intial_size
    for i in range(env.state_dim):
        l = x[i, 0] - initial_size[i]
        h = x[i, 0] + initial_size[i]
        init.append("init" + str(env.state_dim + i + 1) + " = (x[" + str(
            i + 1) + "] - (" + str(l) + "))*((" + str(h) + ")-x[" + str(i + 1) + "])")
    for i in range(env.state_dim):
        initSOSPoly.append(
            "@variable m Zinit" + str(env.state_dim + i + 1) + " SOSPoly(Z)")
    for i in range(env.state_dim):
        init_cnstr.append(
            " - Zinit" + str(env.state_dim + i + 1) + "*init" + str(env.state_dim + i + 1))

    # Specs for unsafe condions depends on env.unsafe
    unsafe = []
    unsafeSOSPoly = []
    unsafe_cnstr = []
    if (env.unsafe):
        # unsafe is given either via unsafe regions or unsfe properties in the
        # env
        if (env.unsafe_property is not None):
            unsafes = env.unsafe_property()
            unsafe = []
            unsafeSOSPoly = []
            unsafe_cnstr = []
            for i in range(len(unsafes)):
                unsafe.append("unsafe" + str(i + 1) + " = " + unsafes[i])
                unsafeSOSPoly.append(
                    "@variable m Zunsafe" + str(i + 1) + " SOSPoly(Z)")
                unsafe_cnstr.append(
                    " - Zunsafe" + str(i + 1) + "*unsafe" + str(i + 1))
        if (env.x_min is not None):
            for j in range(len(env.x_min)):
                unsafe_query = ""
                unsafe_x_min = env.x_min[j]
                unsafe_x_max = env.x_max[j]
                for i in range(env.state_dim):
                    if unsafe_x_min[i, 0] != np.NINF and unsafe_x_max[i, 0] != np.inf:
                        unsafe.append("unsafe" + str(i + 1) + " = (x[" + str(i + 1) + "] - " + str(
                            unsafe_x_min[i, 0]) + ")*(" + str(unsafe_x_max[i, 0]) + "-x[" + str(i + 1) + "])")
                        unsafeSOSPoly.append(
                            "@variable m Zunsafe" + str(i + 1) + " SOSPoly(Z)")
                        unsafe_query += " - Zunsafe" + \
                            str(i + 1) + "*unsafe" + str(i + 1)
                    elif unsafe_x_min[i, 0] != np.NINF:
                        unsafe.append("unsafe" + str(i + 1) + " = (x[" + str(i + 1) + "] - " + str(
                            unsafe_x_min[i, 0]) + ")*(" + str(unsafe_x_max[i, 0]) + "-x[" + str(i + 1) + "])")
                        unsafeSOSPoly.append(
                            "@variable m Zunsafe" + str(i + 1) + " SOSPoly(Z)")
                        unsafe_query += " - Zunsafe" + \
                            str(i + 1) + "*unsafe" + str(i + 1)
                    elif unsafe_x_max[i, 0] != np.inf:
                        unsafe.append("unsafe" + str(i + 1) + " = (x[" + str(i + 1) + "] - " + str(
                            unsafe_x_min[i, 0]) + ")*(" + str(unsafe_x_max[i, 0]) + "-x[" + str(i + 1) + "])")
                        unsafeSOSPoly.append(
                            "@variable m Zunsafe" + str(i + 1) + " SOSPoly(Z)")
                        unsafe_query += " - Zunsafe" + \
                            str(i + 1) + "*unsafe" + str(i + 1)
                if unsafe_query != "":
                    unsafe_cnstr.append(unsafe_query)
    else:
        for i in range(env.state_dim):
            mid = (env.x_min[i, 0] + env.x_max[i, 0]) / 2
            radium = env.x_max[i, 0] - mid
            unsafe.append("unsafe" + str(i + 1) +
                          " = (x[" + str(i + 1) + "] - " + str(mid) + ")^2 - " + str(pow(radium, 2)))
            unsafeSOSPoly.append("@variable m Zunsafe" +
                                 str(i + 1) + " SOSPoly(Z)")
            unsafe_cnstr.append(
                " - Zunsafe" + str(i + 1) + "*unsafe" + str(i + 1))
         # Now we have init, unsafe and sysdynamics for verification
    Acl = env.A + env.B.dot(K)
    sos = genSOSContinuousAsDiscreteMultipleUnsafes(
        env.timestep, env.state_dim, ",".join(
            dxdt(Acl)), "\n".join(init), "\n".join(unsafe),
        "\n".join(initSOSPoly), "\n".join(unsafeSOSPoly), "".join(init_cnstr), unsafe_cnstr, degree=degree)

    return sos


def generate_sos_poly_env(env, K, x, initial_size, degree=4):
    init = []
    initSOSPoly = []
    init_cnstr = []
    for i in range(env.state_dim):
        init.append("init" + str(i + 1) + " = (x[" + str(i + 1) + "] - " + str(
            env.s_min[i, 0]) + ")*(" + str(env.s_max[i, 0]) + "-x[" + str(i + 1) + "])")
    for i in range(env.state_dim):
        initSOSPoly.append("@variable m Zinit" +
                           str(i + 1) + " SOSPoly(Z)")
    for i in range(env.state_dim):
        init_cnstr.append(
            " - Zinit" + str(i + 1) + "*init" + str(i + 1))
    # Specs for initial conditions subject to initial_size
    for i in range(env.state_dim):
        l = x[i, 0] - initial_size[i]
        h = x[i, 0] + initial_size[i]
        init.append("init" + str(env.state_dim + i + 1) + " = (x[" + str(
            i + 1) + "] - (" + str(l) + "))*((" + str(h) + ")-x[" + str(i + 1) + "])")
    for i in range(env.state_dim):
        initSOSPoly.append(
            "@variable m Zinit" + str(env.state_dim + i + 1) + " SOSPoly(Z)")
    for i in range(env.state_dim):
        init_cnstr.append(
            " - Zinit" + str(env.state_dim + i + 1) + "*init" + str(env.state_dim + i + 1))

    # Specs for unsafe condions
    unsafes = env.unsafe_property()
    unsafe = []
    unsafeSOSPoly = []
    unsafe_cnstr = []
    for i in range(len(unsafes)):
        unsafe.append("unsafe" + str(i + 1) + " = " + unsafes[i])
    for i in range(len(unsafes)):
        unsafeSOSPoly.append(
            "@variable m Zunsafe" + str(i + 1) + " SOSPoly(Z)")
    for i in range(len(unsafes)):
        unsafe_cnstr.append(
            " - Zunsafe" + str(i + 1) + "*unsafe" + str(i + 1))

    # Specs for bounded state space
    bound = []
    boundSOSPoly = []
    bound_cnstr = []
    if (env.bound_x_min is not None and env.bound_x_max is not None):
        for i in range(env.state_dim):
            if (env.bound_x_min[i, 0] is not None and env.bound_x_max[i, 0] is not None):
                bound.append("bound" + str(i + 1) + " = (x[" + str(i + 1) + "] - " + str(env.bound_x_min[
                             i, 0]) + ")*(" + str(env.bound_x_max[i, 0]) + "-x[" + str(i + 1) + "])")
        for i in range(env.state_dim):
            if (env.bound_x_min[i, 0] is not None and env.bound_x_max[i, 0] is not None):
                boundSOSPoly.append(
                    "@variable m Zbound" + str(i + 1) + " SOSPoly(Z)")
        for i in range(env.state_dim):
            if (env.bound_x_min[i, 0] is not None and env.bound_x_max[i, 0] is not None):
                bound_cnstr.append(
                    " - Zbound" + str(i + 1) + "*bound" + str(i + 1))

    # Specs for bounded environment disturbance
    disturbance = []
    disturbanceSOSPoly = []
    disturbance_cnstr = []
    if (env.disturbance_x_min is not None and env.disturbance_x_max is not None):
        for i in range(env.state_dim):
            if (env.disturbance_x_min[i, 0] is not None and env.disturbance_x_max[i, 0] is not None):
                disturbance.append("disturbance" + str(i + 1) + " = (d[" + str(i + 1) + "] - " + str(
                    env.disturbance_x_min[i, 0]) + ")*(" + str(env.disturbance_x_max[i, 0]) + "-d[" + str(i + 1) + "])")
        for i in range(env.state_dim):
            if (env.disturbance_x_min[i, 0] is not None and env.disturbance_x_max[i, 0] is not None):
                disturbanceSOSPoly.append(
                    "@variable m Zdisturbance" + str(i + 1) + " SOSPoly(D)")
        for i in range(env.state_dim):
            if (env.disturbance_x_min[i, 0] is not None and env.disturbance_x_max[i, 0] is not None):
                disturbance_cnstr.append(
                    " - Zdisturbance" + str(i + 1) + "*disturbance" + str(i + 1))

    # Now we have init, unsafe and sysdynamics for verification
    sos = None
    if (env.bound_x_min is not None and env.bound_x_max is not None):
        sos = genSOSwithBound(env.state_dim, ",".join(env.polyf_to_str(K)), "\n".join(init), "\n".join(unsafe), "\n".join(bound),
                              "\n".join(initSOSPoly), "\n".join(
                                  unsafeSOSPoly), "\n".join(boundSOSPoly),
                              "".join(init_cnstr), "".join(unsafe_cnstr), "".join(bound_cnstr), degree=degree)
    elif (env.disturbance_x_min is not None and env.disturbance_x_max is not None):
        sos = genSOSwithDisturbance(env.state_dim, ",".join(env.polyf_to_str(K)), "\n".join(init), "\n".join(unsafe), "\n".join(disturbance),
                                    "\n".join(initSOSPoly), "\n".join(
                                        unsafeSOSPoly), "\n".join(disturbanceSOSPoly),
                                    "".join(init_cnstr), "".join(unsafe_cnstr), "".join(disturbance_cnstr), degree=degree)
    else:
        sos = genSOS(env.state_dim, ",".join(env.polyf_to_str(K)), "\n".join(init), "\n".join(unsafe),
                     "\n".join(initSOSPoly), "\n".join(
                         unsafeSOSPoly),
                     "".join(init_cnstr), "".join(unsafe_cnstr), degree=degree)
    return sos


def run_sos(sosfile, quite, timeout, aggressive=False):
    def logged_sys_call(args, quiet, timeout):
        print("exec: " + " ".join(args))
        kill = lambda process: process.kill()
        julia = subprocess.Popen(
            args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        timer = Timer(timeout, kill, [julia])

        bcresult = None
        try:
            timer.start()
            bcresult = julia.communicate()
            if (aggressive):
                if (bcresult[0].find("Solution status : OPTIMAL") >= 0 and bcresult[1].split("#")[0] != "Optimal"):
                    bcresult = "Optimal" + "#" + bcresult[1].split("#")[1]
                else:
                    bcresult = bcresult[1]
            else:
                bcresult = bcresult[1]
        finally:
            timer.cancel()
            poll = julia.poll()
            if poll < 0:
                print("------------ Time-outs! ------------ ")
                os.system("killall -9 julia")
                child = subprocess.Popen(
                    ["pgrep julia"], stdout=subprocess.PIPE, shell=True)
                while True:
                    result = child.communicate()[0]
                    if result == "":
                        break
        return bcresult
    # call /Applications/Julia-0.6.app/Contents/Resources/julia/bin/julia
    # ./sos.jl
    juliapath = get_julia_path()
    return logged_sys_call(juliapath + " " + sosfile, quite, timeout)


def verify(env, K, x, initial_size, quite=True, timeout=900, aggressive=False):
    # TODO: env type
    if type(env) is "":
        sos = generate_sos_linear_env(env, K, x, initial_size)
    elif type(env) is "": 
        sos = generate_sos_poly_env(env, K, x, initial_size)

    verified = run_sos(write_to_tmp("SOS.jl", sos), quite,
                       timeout, aggressive=aggressive)

    if verified.split("#")[0].find("Optimal") >= 0:
        return True, verified.split("#")[1]
    else:
        return False, None
