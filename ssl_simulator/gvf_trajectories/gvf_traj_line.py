"""
"""

__all__ = ["GvfLine"]

import numpy as np

from .gvf_traj import GvfTrajectory

#######################################################################################

class GvfLine(GvfTrajectory):
  def __init__(self, m, a, len=30):
    super().__init__()

    # Line parameters
    self.m = m
    self.a = a

    self.len = len

  def gen_param_points(self, pts = 100):
    x = np.linspace(-self.len, self.len, pts)
    y = self.m * x + self.a
    return np.array([x, y])

  def phi(self, p):
    return p[1] - (self.m * p[0] + self.a)

  def grad_phi(self, p):
    return np.ones_like(p) * np.array([-self.m, 1])

  def hess_phi(self, p):
    H = np.zeros((2,2))
    return H

#######################################################################################