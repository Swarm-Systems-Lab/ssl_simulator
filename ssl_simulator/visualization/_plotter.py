"""
    PLOTTER ABSTRACT BASE CLASS
"""

# Graphic tools
import matplotlib.pyplot as plt

# Import visualization tools from the Swarm Systems Lab Simulator
from ssl_simulator import parse_kwargs

#######################################################################################

class Plotter:
    def __init__(self, fig=None, title=None, **kwargs):
        self.fig = fig
        self.title = title

        # Default visual properties
        kw_fig = {  
            "dpi": 100,
            "figsize": (8, 8)
        }

        # Update defaults with user-specified values
        self.kw_fig = parse_kwargs(kwargs, kw_fig)
        
        # Create subplots
        if self.fig is None:
            self.fig = plt.figure(**self.kw_fig)

    # ------------------------------------------------------------------------
    # These methods have to be implemented in the subclass ################

    def draw(self, **kwargs):
        pass

    # ------------------------------------------------------------------------
    # These methods are optional, can be overridden by subclasses ################

    def update(self, **kwargs):
        pass
    
    # ------------------------------------------------------------------------

    def save(self, filename, dpi=100, **kwargs):
        plt.savefig(filename, dpi=dpi, **kwargs)

#######################################################################################