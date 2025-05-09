"""
"""

# Import visualization tools from the Swarm Systems Lab Simulator
from ssl_simulator import parse_kwargs
from ssl_simulator.visualization import unicycle_patch

#######################################################################################

class PlotterUny:
    def __init__(self, ax, data, tail_len = 10, **kwargs):
        self.ax = ax
        self.data = data
        
        self.tail_len = tail_len
        self.tails = []
        self.icons = []

        # Default visual properties
        kw_patch = dict(size=2, fc="royalblue", ec="k", lw=0.5, zorder=3)
        kw_patch_dead = dict(c="darkred", linewidths=0)

        # Update defaults with user-specified values
        self.kw_patch = parse_kwargs(kwargs, kw_patch)
        self.kw_patch_dead = parse_kwargs(kw_patch_dead, self.kw_patch)

    def draw(self, **kwargs):

        # Lines visual properties
        kw_lines = dict(c="royalblue", lw=0.2, zorder=2)
        kw_lines_dead = dict(c="darkred")

        self.kw_lines = parse_kwargs(kwargs, kw_lines)
        self.kw_lines_dead = parse_kwargs(kw_lines_dead, kw_lines)

        # Extract derired data
        x = self.data["p"][:,:,0]
        y = self.data["p"][:,:,1]
        theta = self.data["theta"]
        status = self.data["status"]
        
        n_robots = x.shape[1]
        
        # Plot the robots
        for i in range(n_robots):
            
            if status[-1,i]:
                kw_patch = self.kw_patch
                kw_lines = self.kw_lines
            else:
                kw_patch = self.kw_patch_dead
                kw_lines = self.kw_lines_dead

            icon = unicycle_patch([x[-1,i], y[-1,i]], theta[-1,i], **kw_patch)
            self.ax.add_patch(icon)
            self.icons.append(icon)

            line, = self.ax.plot(x[0,i], y[0,i], **kw_lines)
            self.tails.append(line)

    def update(self):
        # Extract derired data
        x = self.data["p"][:,:,0]
        y = self.data["p"][:,:,1]
        theta = self.data["theta"]
        status = self.data["status"]

        n_robots = x.shape[1]
        n_steps = x.shape[0]

        # Compute start index for tail
        tail_start = -self.tail_len if self.tail_len is not None and self.tail_len < n_steps else 0

        # ------------------------------------------------
        # Update each robot's icon and tail
        for i in range(n_robots):
            self.icons[i].remove()

            if status[-1,i]:
                kw_patch = self.kw_patch
                kw_lines = self.kw_lines
            else:
                kw_patch = self.kw_patch_dead
                kw_lines = self.kw_lines_dead
            
            icon = unicycle_patch(
                [x[-1,i], y[-1,i]], theta[-1,i], 
                **kw_patch)
            self.ax.add_patch(icon)
            self.icons[i] = icon

            self.tails[i].set_data(x[tail_start:, i], y[tail_start:, i])
            self.tails[i].update(kw_lines)

    # def _update_icon(self, icon, pos, angle):
    #     transform = Affine2D().rotate_around(0, 0, angle).translate(*pos)
    #     icon.set_transform(transform + self.ax.transData)
        
#######################################################################################