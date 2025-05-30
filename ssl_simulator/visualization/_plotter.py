"""
    PLOTTER ABSTRACT BASE CLASS
"""

from ssl_simulator import parse_kwargs
from matplotlib.pyplot import figure

#######################################################################################

class Plotter:
    fig = None

    def __init__(self, **kwargs):
        kw_fig = dict(dpi=None, figsize=None)
        self.kw_fig = parse_kwargs(kwargs, kw_fig)
        self.fig = figure(**self.kw_fig)

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
        self.fig.savefig(filename, dpi=dpi, **kwargs)

#######################################################################################