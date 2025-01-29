"""
"""

__all__ = ["PlotBasic"]

import numpy as np
import matplotlib.pyplot as plt

#######################################################################################

class PlotBasic:
    def __init__(self, data):
        self.data = data
        self.fig, self.ax = plt.subplots()

    # ---------------------------------------------------------------------------------
    def config_axes(self):
        self.ax.set_xlabel(r"$X$ [m]")
        self.ax.set_ylabel(r"$Y$ [m]")
        self.ax.grid(True)

    def plot(self):
        self.config_axes()
        
        # Extract derired data
        x = np.array(self.data["p"].tolist())[:,:,0]
        y = np.array(self.data["p"].tolist())[:,:,1]

        # Create the plot and show it
        self.ax.plot(x, y, "b")
        self.ax.scatter(x[0,:], y[0,:], edgecolors="r", marker="s", facecolors="None")
        self.ax.scatter(x[-1,:], y[-1,:],edgecolors="b", facecolors="None")

        plt.show()
    
    def save(self, filename, dpi=100):
        plt.savefig(filename, dpi=dpi)

#######################################################################################