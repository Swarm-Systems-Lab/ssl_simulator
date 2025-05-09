"""
"""

import numpy as np
from ssl_simulator import parse_kwargs
from ssl_simulator.visualization import Plotter

#######################################################################################

class PlotterVF(Plotter):
    traj_points = None
    mapgrad_pos = None
    mapgrad_vec = None

    kw_vect = dict(angles="xy", scale_units="xy", scale=None, width=None)
    arrow_scale_factor = 1.5
    arrow_width_factor = 0.0002

    def __init__(self, fig = None, **kwargs):
        super().__init__(fig = fig, **kwargs)
    
    # ---------------------------------------------------------------------------------

    def _compute_xymesh(self, pts, xlim=None, ylim=None, return_mesh=False, **kwargs):
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

        if return_mesh:
            return xx, yy
        else:
            pos = np.column_stack([xx.flatten(), yy.flatten()])
            return pos
    
    def _draw_field(self, ax):
        self._adjust_vec_scale()
        quivers = ax.quiver(
            self.mapgrad_pos[:,0], self.mapgrad_pos[:,1], 
            self.mapgrad_vec[:,0], self.mapgrad_vec[:,1], **self.kw_vect
        )
        return quivers

    def _adjust_vec_scale(self):
        """
        Dynamically adjust vector field scale and width
        """
        x_spacing = np.mean(np.diff(np.unique(self.mapgrad_pos[:, 0])))
        y_spacing = np.mean(np.diff(np.unique(self.mapgrad_pos[:, 1])))
        avg_spacing = np.sqrt(x_spacing**2 + y_spacing**2)

        # Set scale so that arrows fit the mesh
        self.kw_vect["scale"] = 1.0 / (avg_spacing / self.arrow_scale_factor)

        # Set width in axis-relative units (smaller spacing -> thinner arrows)
        self.kw_vect["width"] = self.arrow_width_factor * avg_spacing
    
#######################################################################################