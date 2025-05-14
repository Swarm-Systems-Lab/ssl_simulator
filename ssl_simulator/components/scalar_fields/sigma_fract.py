"""
Fractal function (two Gaussians + contant * norm)
"""

__all__ = [
    "SigmaFract"
]

import numpy as np
from numpy import linalg as la

from ._scalar_field import ScalarField
from ssl_simulator.math.basics import adapt_to_nd, Q_prod_xi, exp

#######################################################################################

# Helper functions for clearer initialization of Qa and Qb
def create_Qa():
    """
    Create the quadratic transformation matrix Qa for the first Gaussian.
    """
    S1 = 0.9 * np.array([[1 / np.sqrt(30), 0], [0, 1]])
    return -S1

def create_Qb():
    """
    Create the quadratic transformation matrix Qb for the second Gaussian.
    """
    S2 = 0.9 * np.array([[1, 0], [0, 1 / np.sqrt(15)]])
    A = (1 / np.sqrt(2)) * np.array([[1, -1], [1, 1]])
    return -A.T @ S2 @ A

# Default centers for the Gaussians
default_a = np.array([1, 0])
default_b = np.array([0, -1.5])

class SigmaFract(ScalarField):
    """
    Fractal scalar field

    Attributes
    ----------
        k: float
            norm factor
        mu: list
            center of the Gaussian.
        dev: float
            models the scale of the distribution while maintaining its properties
        a: np.ndarray
            center of the first Gaussian
        b: np.ndarray
            center of the second Gaussian
        Qa: numpy array
            2x2 matrix, quadratic transformation of the first Gaussian input
        Qb: numpy array
            2x2 matrix, quadratic transformation of the second Gaussian input
    """

    def __init__(self, k, mu=[0, 0], dev=[1, 1], a=default_a, b=default_b, Qa=None, Qb=None):
        super().__init__()
        
        self.x0 = mu
        self.k = k
        self.dev = dev
        
        # Set default Qa and Qb if not provided
        self.Qa = Qa if Qa is not None else create_Qa()
        self.Qb = Qa if Qa is not None else create_Qb()
    
        # Convert a and b to numpy arrays if they are lists
        self.a = np.array(a) if isinstance(a, list) else a
        self.b = np.array(b) if isinstance(b, list) else b

        self.mu = mu
        self.mu = self._find_max(mu)

    def eval_value(self, X):
        X = adapt_to_nd(X, target_ndim=2)
        X = X - self.x0
        c1 = -exp(X / self.dev[0], self.Qa, self.a) - exp(
            X / self.dev[0], self.Qb, self.b
        )
        c2 = -exp(X / self.dev[1], self.Qa, self.a) - exp(
            X / self.dev[1], self.Qb, self.b
        )
        x_dist = la.norm(X, axis=1)
        sigma = -2 + 2 * c1 + c2 + self.k * x_dist
        return -sigma

    def eval_grad(self, X):
        X = adapt_to_nd(X, target_ndim=2)
        X = (X - self.x0) / self.dev
        alfa = 0.0001  # Trick to avoid x/0
        sigma_grad = (
            -Q_prod_xi(self.Qa, X - self.a) * exp(X, self.Qa, self.a)[:, None]
            - Q_prod_xi(self.Qb, X - self.b) * exp(X, self.Qb, self.b)[:, None]
            + self.k * X / (la.norm(X, axis=1)[:, None] + alfa)
        )
        return -sigma_grad

    def eval_hessian(self, X):
        return None

#######################################################################################