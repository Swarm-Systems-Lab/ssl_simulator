__all__ = ["SO3Plot"]

import numpy as np

from ssl_simulator.math import check_and_parse_dimensions
from ssl_simulator.visualization.matplotlib.methods.plot_3d import plot_so3_attitude_vectors
from ssl_simulator.visualization.matplotlib.plots._plot import PlotBase
from ssl_simulator.visualization.matplotlib.utils.anim import update_quivers, update_scatters


class SO3Plot(PlotBase):
    def __init__(self, R_data, p_data=None, sphere=True, **kwargs):
        super().__init__(**kwargs)
        self.sphere = sphere

        self.R_data = check_and_parse_dimensions(R_data, (None, None, 3, 3), "R_data")
        self.p_data = (
            check_and_parse_dimensions(p_data, (None, None, 3), "p_data")
            if p_data is not None
            else None
        )
        self.n_frames = self.R_data.shape[0]

        # Define axes layout
        # position: [left, bottom, width, height]
        # self.axes_config = {
        #     "attitude": {"position":[0,0,0.4,1], "projection":"3d"},
        #     "omega": {"position":[0.5,0,0.4,1], "projection":"3d"},
        # }
        self.axes_config = {
            "attitude": {"position": [0, 0, 1, 1], "projection": "3d"},
        }

    def init_artists(self):
        ax1 = self.axes["attitude"]
        # ax2 = self.axes["omega"]

        # Create quivers for first frame
        artists_attitude = plot_so3_attitude_vectors(
            self.R_data, self.p_data, ax=ax1, arr_len=1.0, quiver_thickness=2, sphere=self.sphere
        )
        self.artists["attitude"].update(artists_attitude)

        # artists_omega = plot_so3_omega_traj(self.R_data, ax=ax2)
        # self.artists["omega"].update(artists_omega)

    def update_artists(self, frame_data):
        frame_idx = int(frame_data)
        R_frame = self.R_data[frame_idx, ...]
        p_frame = self.p_data[frame_idx, ...] if self.p_data is not None else np.zeros((1, 3))

        quivers_array = self.artists["attitude"]["quivers"][0, ...]  # shape (n_agents, 3)
        update_quivers(quivers_array=quivers_array, p_frame=p_frame, R_frame=R_frame)

        if self.p_data is not None:
            scatters_array = self.artists["attitude"]["scatters"]  # shape (n_agents,)
            update_scatters(scatter_list=scatters_array, p_frame=p_frame)
