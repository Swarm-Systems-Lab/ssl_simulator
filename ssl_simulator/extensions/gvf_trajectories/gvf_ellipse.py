"""
"""

__all__ = ["GvfEllipse"]

import numpy as np

from ._gvf_traj import GvfTrajectory

#######################################################################################


class GvfEllipse(GvfTrajectory):
    def __init__(self, XYoff, alpha, a, b):
        super().__init__()

        self.vec_hess = False

        # Ellipse parameters
        self.XYoff = XYoff
        self.alpha = alpha
        self.a, self.b = a, b

        self.cosa, self.sina = np.cos(alpha), np.sin(alpha)
        self.R = np.array([[self.cosa, self.sina], [-self.sina, self.cosa]])

    def gen_param_points(self, pts=100):
        t = np.linspace(0, 2 * np.pi, pts)
        x = (
            self.XYoff[0]
            + self.a * np.cos(-self.alpha) * np.cos(t)
            - self.b * np.sin(-self.alpha) * np.sin(t)
        )
        y = (
            self.XYoff[1]
            + self.a * np.sin(-self.alpha) * np.cos(t)
            + self.b * np.cos(-self.alpha) * np.sin(t)
        )
        return [x, y]

    def phi(self, p):
        w = self.XYoff
        pel = (p - w) @ self.R
        return (pel[0] / self.a) ** 2 + (pel[1] / self.b) ** 2 - 1

    def grad_phi(self, p):
        w = self.XYoff
        pel = (p - w) @ self.R
        return 2 * pel / [self.a**2, self.b**2] @ self.R.T

    def hess_phi(self, p):
        H = np.zeros((2, 2))
        H[0, 0] = 2 * (self.cosa**2 / self.a**2 + self.sina**2 / self.b**2)
        H[0, 1] = 2 * self.sina * self.cosa * (1 / self.b**2 - 1 / self.a**2)
        H[1, 0] = H[0, 1]
        H[1, 1] = 2 * (self.sina**2 / self.a**2 + self.cosa**2 / self.b**2)
        return H


#######################################################################################
