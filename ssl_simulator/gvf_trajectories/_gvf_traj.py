"""
"""

__all__ = ["GvfTrajectory"]

from abc import abstractmethod
import numpy as np
import matplotlib.pylab as plt

E = np.array([[0, 1],[-1, 0]]) # -90 degree 2D rotation matrix

#######################################################################################

class GvfTrajectory:
    """
    Abstract base class for generating and visualizing a GVF trajectory.
    
    This class provides methods for:
    - Generating parametric trajectory points.
    - Computing the trajectory's parametric equation (`phi`), igs gradient and its Hessian.
    - Creating and visualizing the corresponding vector field.
    """
    def __init__(self):
        self.XYoff = [0,0]
        self.traj_points = []
        self.mapgrad_pos = np.empty(1)
        self.mapgrad_vec = np.empty(1)

        self.vec_phi = True
        self.vec_grad = True
        self.vec_hess = True

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

    # ---------------------------------------------------

    def vectorial_comp(self, p, func, vec_flag):
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
    
    def phi_vec(self, p):
        return self.vectorial_comp(p, self.phi, self.vec_phi)

    def grad_vec(self, p):
        return self.vectorial_comp(p, self.grad_phi, self.vec_grad)

    def hess_vec(self, p):
        return self.vectorial_comp(p, self.hess_phi, self.vec_hess)

    # ---------------------------------------------------

    def gen_vector_field(self, area, s, ke, pts = 30, xy_offset = None):
        """
        Generate the vector field to be plotted.
        """
        if xy_offset is None:
            xy_offset = self.XYoff
        
        # Create the XY mesh
        x_lin = np.linspace(xy_offset[0] - 0.5*np.sqrt(area), \
                            xy_offset[0] + 0.5*np.sqrt(area), pts)
        y_lin = np.linspace(xy_offset[1] - 0.5*np.sqrt(area), \
                            xy_offset[1] + 0.5*np.sqrt(area), pts)
        mapgrad_X, mapgrad_Y = np.meshgrid(x_lin, y_lin)
        mapgrad_X = np.reshape(mapgrad_X, -1)
        mapgrad_Y = np.reshape(mapgrad_Y, -1)
        self.mapgrad_pos = np.array([mapgrad_X, mapgrad_Y]).T

        # Compute the GVF on every point of the mesh
        n = self.grad_vec(self.mapgrad_pos)
        t = s*n @ E.T

        e = self.phi_vec(self.mapgrad_pos)[:,None]

        self.mapgrad_vec = t - ke*e*n

        norm = np.sqrt(self.mapgrad_vec[:,0]**2 + self.mapgrad_vec[:,1]**2)[:,None]
        self.mapgrad_vec = self.mapgrad_vec / norm

    def draw(self, fig=None, ax=None, xlim=None, ylim=None, draw_field=True, 
             alpha=0.2, ls="--", lw=1, width=0.0025, color="k"):
        """
        Plot the trajectory and the vector field.
        """
        if fig == None:
            fig = plt.figure(dpi=100)
            ax = fig.subplots()
        elif ax == None:
            ax = fig.subplots()

        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)

        # Get the trajectory points
        self.traj_points = self.gen_param_points()

        # Plot the trajectory
        ax.plot(self.XYoff[0], self.XYoff[1], "+k", zorder=0)
        traj, = ax.plot(self.traj_points[0], self.traj_points[1], 
                        c=color, ls=ls, lw=lw, zorder=0)

        # Plot the vector field
        if draw_field:
            if self.mapgrad_vec is not None:
                ax.quiver(self.mapgrad_pos[:,0], self.mapgrad_pos[:,1],
                          self.mapgrad_vec[:,0], self.mapgrad_vec[:,1],
                          alpha=alpha, width=width)
            else:
                print("Please run gen_vector_field() first.")
            
        return traj
    
#######################################################################################