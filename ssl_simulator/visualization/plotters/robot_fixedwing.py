"""
"""

# Import visualization tools from the Swarm Systems Lab Simulator
from ssl_simulator import parse_kwargs
from ssl_simulator.visualization import Plotter, fixedwing_patch

#######################################################################################

class PlotterFixedwing(Plotter):
    def __init__(self, ax, data, tail_len = 10, **kwargs):
        self.ax = ax
        self.data = data
        
        self.tail_len = tail_len
        self.tails = []
        self.icons = []

        # Default visual properties
        kw_patch = dict(fc="royalblue", ec="black", size=15, lw=1, zorder=3)
        kw_patch_dead = dict(fc="darkred", linewidths=0)

        # Update defaults with user-specified values
        self.kw_patch = parse_kwargs(kwargs, kw_patch)
        self.kw_patch_dead = parse_kwargs(kw_patch_dead, self.kw_patch)
        self.kw_lines = dict(c="royalblue", lw=0.2, zorder=2)
        self.kw_lines_dead = parse_kwargs(dict(c="darkred"), self.kw_lines)

    def draw(self, init=True, **kwargs):

        self.kw_lines = parse_kwargs(kwargs, self.kw_lines)
        self.kw_lines_dead = parse_kwargs(dict(c="darkred"), self.kw_lines)

        # Extract derired data
        x = self.data["p"][:,:,0]
        y = self.data["p"][:,:,1]
        theta = self.data["theta"]
        
        n_robots = x.shape[1]
        
        # ------------------------------------------------
        # Plot each robot's icon and tail
        for i in range(n_robots):
            if init:
                patch_i = fixedwing_patch([x[0,i], y[0,i]], theta[0,i], **self.kw_patch)
                patch_i.set_alpha(0.5)
                self.ax.add_artist(patch_i)

            patch_f = fixedwing_patch([x[-1,i], y[-1,i]], theta[-1,i], **self.kw_patch)
            self.ax.add_artist(patch_f)
            self.icons.append(patch_f)

            line, = self.ax.plot(x[0,i], y[0,i], **self.kw_lines)
            self.tails.append(line)

    def update(self):

        # Extract derired data
        x = self.data["p"][:,:,0]
        y = self.data["p"][:,:,1]
        theta = self.data["theta"]

        n_robots = x.shape[1]
        n_steps = x.shape[0]

        # Compute start index for tail
        tail_start = -self.tail_len if self.tail_len is not None and self.tail_len < n_steps else 0

        # ------------------------------------------------
        # Update each robot's icon and tail
        for i in range(n_robots):
            self.icons[i].remove()
            patch_f = fixedwing_patch([x[-1,i], y[-1,i]], theta[-1,i], **self.kw_patch)
            self.ax.add_artist(patch_f)
            self.icons[i] = patch_f

            self.tails[i].set_data(x[tail_start:, i], y[tail_start:, i])

#######################################################################################