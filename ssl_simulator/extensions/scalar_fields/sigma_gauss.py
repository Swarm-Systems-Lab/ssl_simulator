"""
Gaussian function
"""

__all__ = [
    "SigmaGauss"
]

import numpy as np
from numpy import linalg as la

from ._scalar_field import ScalarField
from ssl_simulator.math.basics import adapt_to_nd, exp

#######################################################################################

class SigmaGauss(ScalarField):
    """
    Gaussian scalar field function

    Attributes
    ----------
        mu: list
            center of the Gaussian.
        max_intensity: float
            scalar field value at the source
        dev: float
            models the width of the Gaussian
        S: numpy array
            2x2 matrix, rotation matrix applied to the scalar field
        R: numpy array
            2x2 matrix, scaling matrix applied to the scalar field
    """

    def __init__(self, mu=[0, 0], max_intensity=100, dev=10):
        self.x0 = mu
        self.max_intensity = max_intensity
        self.dev = dev
        self.Q = -np.eye(2) / (2 * self.dev**2)

        self._mu = mu
        self._mu = self.find_max(mu)

    @property
    def mu(self):
        return self._mu

    def eval_value(self, X):
        X = adapt_to_nd(X, target_ndim=2)
        sigma = (
            self.max_intensity
            * exp(X, self.Q, self.x0)
            / np.sqrt(2 * np.pi * self.dev**2)
        )
        return sigma

    # sigma(p) \prop exp(g(p)), where g(p) = (X-x0)^T @ Q @ (X-x0)

    def eval_grad(self, X):
        X = adapt_to_nd(X, target_ndim=2)
        x, y = X[0,0], X[0,1]
        x0, y0 = self.x0[0], self.x0[1] 
        q11,q12 = self.Q[0,0], self.Q[0,1]
        q21,q22 = self.Q[1,0], self.Q[1,1]

        # Compute the gradient
        g_dx = 2 * q11 * (x-x0) + (q12 + q21) * (y-y0)
        g_dy = 2 * q22 * (y-y0) + (q12 + q21) * (x-x0)

        return np.array([[g_dx, g_dy]]) * self.value(X)
    
    def eval_hessian(self, X):
        X = adapt_to_nd(X, target_ndim=2)
        x, y = X[0,0], X[0,1]
        x0, y0 = self.x0[0], self.x0[1]
        q11,q12 = self.Q[0,0], self.Q[0,1]
        q21,q22 = self.Q[1,0], self.Q[1,1]

        # Compute the hessian
        g_dx = 2 * q11 * (x-x0) + (q12 + q21) * (y-y0)
        g_dy = 2 * q22 * (y-y0) + (q12 + q21) * (x-x0)

        g_dxx = 2 * q11
        g_dxy = (q12 + q21)
        g_dyx = (q12 + q21)
        g_dyy = 2 * q22

        H = np.zeros((2,2))
        H[0,0] = self.value(X) * (g_dx * g_dx + g_dxx) 
        H[0,1] = self.value(X) * (g_dx * g_dy + g_dyx)
        H[1,0] = self.value(X) * (g_dy * g_dx + g_dxy)
        H[1,1] = self.value(X) * (g_dy * g_dy + g_dyy)

        return H

#######################################################################################