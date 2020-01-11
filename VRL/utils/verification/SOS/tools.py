# -*- coding: utf-8 -*-
# -------------------------------
# Author: Zikang Xiong
# Email: zikangxiong@gmail.com
# Date:   2020-01-10 22:51:52
# Last Modified by:   Zikang Xiong
# Last Modified time: 2020-01-10 23:12:10
# -------------------------------
import os
import subprocess
import platform
from threading import Timer

def dxdt(A, coffset=None): 
  d, p = A.shape
  X = []
  for i in range(p):
    X.append("x[" + str(i+1) + "]")

  f = []
  for i in range(len(A)):
    strstr = ""
    for k in range(len(X)):
      if (strstr is ""):
        strstr = str(A[i,k]) + "*" + X[k]
      else:
        strstr = strstr + "+" + str(A[i,k]) + "*" + X[k]
    if coffset is not None:
      strstr += ("+" + str(coffset[i,0]))
    f.append(strstr)
  return f  

def K_to_str (K):
  #Control policy K to text 
  nvars = len(K[0])
  X = []
  for i in range(nvars):
    X.append("x[" + str(i+1) + "]")

  ks = []
  for i in range(len(K)):
    strstr = ""
    for k in range(len(X)):
      if (strstr is ""):
        strstr = str(K[i,k]) + "*" + X[k]
      else:
        strstr = strstr + "+" + str(K[i,k]) + "*" + X[k]
    ks.append(strstr)
  return ks

def writeSOS(fname, sostext):
  file = open("/tmp/"+fname,"w") 
  file.write(sostext) 
  file.close()
  return fname

def get_julia_path():
  if platform.system() == "Linux":
    return "julia"
  else:
    return "/Applications/Julia-0.6.app/Contents/Resources/julia/bin/julia"

def verify(sosfile, quite, timeout, aggressive=False):
  def logged_sys_call(args, quiet, timeout):
    print("exec: " + " ".join(args))
    kill = lambda process: process.kill()  
    julia = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
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
        os.system("killall -9 julia");
        child = subprocess.Popen(["pgrep julia"], stdout=subprocess.PIPE, shell=True)
        while True:
          result = child.communicate()[0]
          if result == "":
            break
    return bcresult
  #call /Applications/Julia-0.6.app/Contents/Resources/julia/bin/julia ./sos.jl
  juliapath = get_julia_path()
  return logged_sys_call([juliapath] + [("%s" % sosfile)], quite, timeout)
