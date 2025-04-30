"""
"""

__all__ = ["GvfLine"]

import numpy as np

from ._gvf_traj import GvfTrajectory

#######################################################################################

class GvfLine(GvfTrajectory):
    def __init__(self, m, b, line_length=30):
        super().__init__()

        # Line parameters
        self.m = m
        self.b = b

        self.line_length = line_length

    def gen_param_points(self, pts=100):
        x = np.linspace(-self.line_length, self.line_length, pts)
        y = self.m * x + self.b
        return np.array([x, y])

    def phi(self, p):
        return p[1] - (self.m * p[0] + self.b)

    def grad_phi(self, p):
        return np.ones_like(p) * np.array([-self.m, 1])

    def hess_phi(self, p):
        H = np.zeros((2, 2))
        return H

#######################################################################################
