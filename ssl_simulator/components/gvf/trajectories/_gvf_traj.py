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
    
# -------------------------------------------------------------------------------------

class GvfTrajectoryPlotter(Plotter):
    def __init__(self, gvf_traj, fig = None, ax = None, **kwargs):
        super().__init__(fig = fig, **kwargs)
        self.gvf_traj = gvf_traj

        self.XYoff = None
        self.traj_points = None
        self.mapgrad_pos = None
        self.mapgrad_vec = None

        if ax is None:
            self.ax = self.fig.subplots()
        else:
            self.ax = ax

    def _compute_vector_field(self, s, ke, pts, xlim=None, ylim=None, **kwargs):
        """
        Generate the vector field to be plotted.
        """
        # Handle xlim
        if xlim is None:
            xmin, xmax = self.ax.get_xlim()
        elif isinstance(xlim, (list, tuple)) and len(xlim) == 2:
            xmin, xmax = xlim
        else:  # assume scalar
            xmin, xmax = self.xy_offset[0] - xlim, self.xy_offset[0] + xlim

        # Handle ylim
        if ylim is None:
            ymin, ymax = self.ax.get_ylim()
        elif isinstance(ylim, (list, tuple)) and len(ylim) == 2:
            ymin, ymax = ylim
        else:  # assume scalar
            ymin, ymax = self.xy_offset[1] - ylim, self.xy_offset[1] + ylim

        x = np.linspace(xmin, xmax, pts)
        y = np.linspace(ymin, ymax, pts)
        xx, yy = np.meshgrid(x, y)

        # Compute the GVF on every point of the mesh
        pos = np.column_stack([xx.flatten(), yy.flatten()])
        n = self.gvf_traj.grad_vec(pos)
        t = s * (n @ E.T)
        e = self.gvf_traj.phi_vec(pos)[:, None]
        vec = t - ke * e * n

        norm = np.linalg.norm(vec, axis=1, keepdims=True)
        vec_normalized = vec / np.clip(norm, a_min=1e-8, a_max=None)

        return pos, vec_normalized

    def draw(self, draw_field=True, s=1, ke=1.0, pts=30, xy_offset=[0,0], 
             **kwargs):
        """
        Plot the trajectory and the vector field.
        """
        self.xy_offset = xy_offset

        # Config
        kw_line = parse_kwargs(kwargs, dict(ls="--", lw=1, c="k"))
        kw_vect = parse_kwargs(kwargs, dict(alpha=0.2, width=0.0025))

        # Plot trajectory
        traj = self.gvf_traj.gen_param_points()
        self.ax.plot(*xy_offset, "+k", zorder=0)
        traj_plot, = self.ax.plot(traj[0], traj[1], zorder=0, **kw_line)

        # Plot vector field
        if draw_field:
            self.mapgrad_pos, self.mapgrad_vec = self._compute_vector_field(
                s, ke, pts, xy_offset, **kwargs
            )
            vector2d(self.ax, 
                     [self.mapgrad_pos[:, 0], self.mapgrad_pos[:, 1]], 
                     [self.mapgrad_vec[:, 0], self.mapgrad_vec[:, 1]], **kw_vect)
            # self.ax.quiver(self.mapgrad_pos[:, 0], self.mapgrad_pos[:, 1],
            #           self.mapgrad_vec[:, 0], self.mapgrad_vec[:, 1],
            #           alpha=alpha, width=width)

        return traj_plot
    
#######################################################################################