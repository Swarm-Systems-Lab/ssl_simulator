"""
"""

import numpy as np
from ssl_simulator import parse_kwargs
from ssl_simulator.visualization import PlotterVF

E = np.array([[0, 1],[-1, 0]]) # -90 degree 2D rotation matrix

#######################################################################################

class GvfTrajectoryPlotter(PlotterVF):
    def __init__(self, gvf_traj, fig = None, ax = None, **kwargs):
        super().__init__(fig = fig, **kwargs)
        self.gvf_traj = gvf_traj

        self.xy_offset = None
        self.mapgrad_pos = None
        self.mapgrad_vec = None

        self.quivers = None

        if ax is None:
            self.ax = self.fig.subplots()
        else:
            self.ax = ax

    def draw(self, draw_field=True, s=1, ke=1.0, pts=30, xy_offset=[0,0], 
             **kwargs):
        """
        Plot the trajectory and the vector field.
        """
        self.s = s
        self.ke = ke
        self.xy_offset = xy_offset

        # Config
        self.kw_line = parse_kwargs(kwargs, dict(ls="--", lw=1, c="k"))
        self.kw_vect.update(dict(lw=1, zorder=4, alpha=0.2, color="black"))
        self.kw_vect = parse_kwargs(kwargs, self.kw_vect)

        # Plot trajectory
        traj = self.gvf_traj.gen_param_points()
        self.ax.plot(*xy_offset, "+k", zorder=0)
        self.ax.plot(traj[0], traj[1], zorder=0, **self.kw_line)

        # Plot vector field
        if draw_field:
            self.mapgrad_pos = self._compute_xymesh(pts, **kwargs)
            self.mapgrad_vec = self._compute_vector_field(**kwargs)
            self.quivers = self._draw_field(self.ax)

    # ---------------------------------------------------------------------------------
    
    def _compute_vector_field(self, **kwargs):
        """
        Generate the vector field to be plotted.
        """
        pos = self.mapgrad_pos

        # Compute the GVF on every point of the mesh
        n = self.gvf_traj.grad_vec(pos)
        t = self.s * (n @ E.T)
        e = self.gvf_traj.phi_vec(pos)[:, None]
        vec = t - self.ke * e * n

        norm = np.linalg.norm(vec, axis=1, keepdims=True)
        vec_normalized = vec / np.clip(norm, a_min=1e-8, a_max=None)

        return vec_normalized
    
#######################################################################################