# -*- coding: utf-8 -*-
# -------------------------------
# Author: Zikang Xiong
# Email: zikangxiong@gmail.com
# Date:   2020-01-11 00:13:27
# Last Modified by:   Zikang Xiong
# Last Modified time: 2020-01-11 00:13:49
# -------------------------------
import numpy as np 
import os

# Self-defined Neural Netwrok
class SelfDefinedNN:

  def __init__(self, para, activate, scale=None):
    self.para = para
    self.activate = activate
    self.scale = scale

  def predict(self, obs):
    h = obs
    for p, a in zip(self.para, self.activate):
      h = self.linear(p[0], p[1], h)

      if a == "relu":
        h = self.relu(h)
      elif a == "tanh":
        h = self.tanh(h)
      elif a == "linear":
        h = h
      else:
        raise NotImplementedError

    return h if self.scale is None else self.scale*h, None
    
  def relu(self, x):
    return np.maximum(0,x)

  def tanh(self, x):
    return np.tanh(x)

  def linear(self, W, b, x):
    return x.dot(W) + b

  @classmethod
  def load(cls, model_path):
    para, activate = np.load(f"{model_path}/NN.npy", allow_pickle=True)
    scale = None
    if os.path.isfile(f"{model_path}/u_max.npy"):
        scale = np.load(f"{model_path}/u_max.npy")

    return cls(para, activate, scale)