"""Unicycle robot plotter for Swarm Systems Lab Simulator."""

from typing import Any

import numpy as np

# Import visualization tools from the Swarm Systems Lab Simulator
from ssl_simulator.utils.dict_ops import parse_kwargs
from ssl_simulator.visualization import unicycle_patch

#######################################################################################


class PlotterUnySS:
    """Plotter for unicycle robots with status and vector fields."""

    def __init__(self, ax, data, tail_len=10, **kwargs):
        """Initialize the plotter.

        Args:
            ax: Matplotlib axis.
            data: Simulation data dict.
            tail_len: Length of robot tail.
            **kwargs: Additional visual kwargs.
        """
        self.ax = ax
        self.data = data

        self.tail_len = tail_len
        self.tails = []
        self.icons = []
        self.centroid = None
        self.vector_grad: Any = None
        self.vector_mu: Any = None

        # Default visual properties
        kw_patch = {"size": 2, "fc": "royalblue", "ec": "k", "lw": 0.5, "zorder": 3}
        kw_patch_dead = {"c": "darkred", "linewidths": 0}
        kw_arr = {
            "angles": "xy",
            "scale_units": "xy",
            "lw": 1,
            "width": 0.003,
            "scale": 1,
            "zorder": 4,
        }

        # Update defaults with user-specified values
        self.kw_patch = kw_patch
        self.kw_patch_dead = parse_kwargs(kw_patch_dead, self.kw_patch)
        self.kw_lines = {"c": "royalblue", "lw": 0.2, "zorder": 2}
        self.kw_lines_dead = parse_kwargs({"c": "darkred"}, self.kw_lines)
        self.kw_arr = parse_kwargs(kwargs, kw_arr)

    def draw(self, **kwargs):
        """Draw robots, tails, and vectors on the axis."""
        x = self.data["robot.p"][:, :, 0]
        y = self.data["robot.p"][:, :, 1]
        theta = self.data["robot.theta"]
        status = self.data["status"]

        sigma_grad = self.data["sigma_grad"][:, :]
        mu_centralized = self.data["mu_centralized"][:, :]
        pc_centralized = self.data["pc_centralized"][:, :]

        n_robots = x.shape[1]

        for i in range(n_robots):
            if status[-1, i]:
                kw_patch = self.kw_patch
                kw_lines = self.kw_lines
            else:
                kw_patch = self.kw_patch_dead
                kw_lines = self.kw_lines_dead

            icon = unicycle_patch([x[-1, i], y[-1, i]], theta[-1, i], **kw_patch)
            self.ax.add_patch(icon)
            self.icons.append(icon)

            (line,) = self.ax.plot(x[0, i], y[0, i], **kw_lines)
            self.tails.append(line)

        self.vector_grad = self.ax.quiver(
            pc_centralized[0, 0],
            pc_centralized[0, 1],
            sigma_grad[0, 0],
            sigma_grad[0, 1],
            color="black",
            **self.kw_arr,
        )
        self.vector_mu = self.ax.quiver(
            pc_centralized[0, 0],
            pc_centralized[0, 1],
            mu_centralized[0, 0],
            mu_centralized[0, 1],
            color="red",
            **self.kw_arr,
        )

    def update(self):
        """Update robot icons, tails, and vectors for animation."""
        x = self.data["robot.p"][:, :, 0]
        y = self.data["robot.p"][:, :, 1]
        theta = self.data["robot.theta"]
        status = self.data["status"]

        sigma_grad = self.data["sigma_grad"][:, :]
        mu_centralized = self.data["mu_centralized"][:, :]
        pc_centralized = self.data["pc_centralized"][:, :]

        n_robots = x.shape[1]
        n_steps = x.shape[0]

        tail_start = -self.tail_len if self.tail_len is not None and self.tail_len < n_steps else 0

        for i in range(n_robots):
            self.icons[i].remove()
            if status[-1, i]:
                kw_patch = self.kw_patch
                kw_lines = self.kw_lines
            else:
                kw_patch = self.kw_patch_dead
                kw_lines = self.kw_lines_dead
            icon = unicycle_patch([x[-1, i], y[-1, i]], theta[-1, i], **kw_patch)
            self.ax.add_patch(icon)
            self.icons[i] = icon
            self.tails[i].set_data(x[tail_start:, i], y[tail_start:, i])
            self.tails[i].update(kw_lines)

        dx = sigma_grad[-1, 0] / np.linalg.norm(sigma_grad[-1, :]) * 10
        dy = sigma_grad[-1, 1] / np.linalg.norm(sigma_grad[-1, :]) * 10
        self.vector_grad.set_offsets([[pc_centralized[-1, 0], pc_centralized[-1, 1]]])
        self.vector_grad.set_UVC(dx, dy)

        dx = mu_centralized[-1, 0] / np.linalg.norm(mu_centralized[-1, :]) * 10
        dy = mu_centralized[-1, 1] / np.linalg.norm(mu_centralized[-1, :]) * 10
        self.vector_mu.set_offsets([[pc_centralized[-1, 0], pc_centralized[-1, 1]]])
        self.vector_mu.set_UVC(dx, dy)


#######################################################################################
