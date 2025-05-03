"""
"""

__all__ = ["PlotterUny"]

import numpy as np
from collections.abc import Iterable

# Graphic tools
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D

# Import visualization tools and GVF trajectory from the Swarm Systems Lab Simulator
from ssl_simulator import parse_kwargs
from ssl_simulator.visualization.patches import unicycle_patch

#######################################################################################

class PlotterUny:
    def __init__(self, ax, logger, tail_len = 10, **kwargs):
        self.ax = ax
        self.logger = logger
        
        # if not isinstance(logger, RealTimeLogger): # TODO: check other way
        #     raise ValueError("logger must be an instance of RealTimeLogger")
        if tail_len > self.logger.log_size:
            raise ValueError(f"tail_len ({tail_len}) cannot exceed logger.log_size ({self.logger.log_size})")

        self.tail_len = 10
        self.tails = []
        self.icons = []

        # Default visual properties 
        kw_patch = {
            "size": 2,
            "fc": "royalblue",
            "ec": "k",
            "lw": 0.5,
            "zorder": 3,
        }

        kw_patch_dead = {
            "c": "darkred",
            "linewidths": 0,
        }

        # Update defaults with user-specified values
        self.kw_patch = parse_kwargs(kwargs, kw_patch)
        self.kw_patch_dead = parse_kwargs(kw_patch_dead, self.kw_patch)

    def draw(self, **kwargs):

        # Lines visual properties
        kw_lines = {
            "c": "royalblue",
            "lw": 0.2,
            "zorder": 2
        }

        kw_lines_dead = {
            "c": "darkred",
        }

        self.kw_lines = parse_kwargs(kwargs, kw_lines)
        self.kw_lines_dead = parse_kwargs(kw_lines_dead, kw_lines)

        # Extract derired data
        x = self.logger.data["p"][:,:,0]
        y = self.logger.data["p"][:,:,1]
        theta = self.logger.data["theta"]
        status = self.logger.data["status"]

        # Plot the robots
        for i in range(x.shape[1]):
            if status[-1,i]:
                kw_patch = self.kw_patch
            else:
                kw_patch = self.kw_patch_dead

            icon = unicycle_patch(
                [x[-1,i], y[-1,i]], theta[-1,i], 
                **kw_patch)

            self.ax.add_patch(icon)
            self.icons.append(icon)

        # for i in range(x.shape[1]):
        #     line, = self.ax.plot(x, y, **self.kw_lines)
        #     self.tails.append(line)
        # self.ax.plot(x[:,status[-1,:]], y[:,status[-1,:]], **self.kw_lines)

    def update(self):
        # Extract derired data
        x = self.logger.data["p"][:,:,0]
        y = self.logger.data["p"][:,:,1]
        theta = self.logger.data["theta"]
        status = self.logger.data["status"]
        status_not = np.logical_not(status)

        # ------------------------------------------------
        # Plot the robots
        for i in range(x.shape[1]):
            self.icons[i].remove()

            if status[-1,i]:
                kw_patch = self.kw_patch
            else:
                kw_patch = self.kw_patch_dead
            
            icon = unicycle_patch(
                [x[-1,i], y[-1,i]], theta[-1,i], 
                **kw_patch)

            self.ax.add_patch(icon)
            self.icons[i] = icon
         
    def _update_icon(self, icon, pos, angle):
        transform = Affine2D().rotate_around(0, 0, angle).translate(*pos)
        icon.set_transform(transform + self.ax.transData)
#######################################################################################