"""
"""

from abc import ABC, abstractmethod
import numpy as np

from ssl_simulator import parse_kwargs
from ssl_simulator.visualization import Plotter, vector2d

E = np.array([[0, 1],[-1, 0]]) # -90 degree 2D rotation matrix

#######################################################################################

class GvfTrajectory(ABC):
    """
    Abstract base class for generating and visualizing a GVF trajectory.
    
    This class provides methods for:
    - Generating parametric trajectory points.
    - Computing the trajectory's parametric equation (`phi`), igs gradient and its Hessian.
    - Creating and visualizing the corresponding vector field.
    """
    def __init__(self):
        self.vec_phi = True
        self.vec_grad = True
        self.vec_hess = True

    # ------------------------------------------------------------------------
    # These operations have to be implemented in the subclass ################
    @abstractmethod
    def gen_param_points(self, pts: int) -> np.ndarray:
        """
        Generate a set of parametric points along the trajectory.

        Args:
            pts (int): Number of points to generate.

        Returns:
            np.ndarray: A 2D array of shape (2, pts), where each column represents 
                        a point [x, y] along the trajectory.
        """
        pass
    
    @abstractmethod
    def phi(self, p: np.ndarray) -> float:
        """
        Evaluate the scalar field function (phi) at a given position.

        Args:
            p (np.ndarray): A 1D NumPy array of shape (2,), representing a position [x, y].

        Returns:
            float: The scalar field value at the given position.
        """
        pass

    @abstractmethod
    def grad_phi(self, p: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of the scalar field (phi) at a given position.

        Args:
            p (np.ndarray): A 1D NumPy array of shape (2,), representing a position [x, y].

        Returns:
            np.ndarray: A 1D array of shape (2,) representing the gradient vector 
                        [∂phi/∂x, ∂phi/∂y].
        """
        pass

    @abstractmethod
    def hess_phi(self, p: np.ndarray) -> np.ndarray:
        """
        Compute the Hessian matrix of the scalar field (phi) at a given position.

        Args:
            p (np.ndarray): A 1D NumPy array of shape (2,), representing a position [x, y].

        Returns:
            np.ndarray: A 2D array of shape (2, 2), representing the Hessian matrix:
                        [[∂²phi/∂x², ∂²phi/∂x∂y],
                        [∂²phi/∂y∂x, ∂²phi/∂y²]].
        """
        pass

    def get_config(self):
        """Returns the key parameters used for reinitialization."""
        return {}

    # ------------------------------------------------------------------------
    # Evaluation #############################################################
    
    def phi_vec(self, p):
        return self._vectorial_comp(p, self.phi, self.vec_phi)

    def grad_vec(self, p):
        return self._vectorial_comp(p, self.grad_phi, self.vec_grad)

    def hess_vec(self, p):
        return self._vectorial_comp(p, self.hess_phi, self.vec_hess)

    # --------------------------------------------------------------------------
    # Internal helper methods
    # --------------------------------------------------------------------------

    def _vectorial_comp(self, p, func, vec_flag):
        N = p.shape[0]

        if len(p.shape) == 1:
            return func(p)
        
        # If the Hessian is the same for every p just repeat it
        if not vec_flag:
            X = self.hess_phi(p)
            if  N > 1:
                X = np.array([X])
                X = np.repeat(X, N, axis=0)
            return X

        # Otherwise, compute it for every p
        X = []
        for i in range(N):
            X.append(func(p[i]))
        X = np.array(X)

        return X
    
#######################################################################################