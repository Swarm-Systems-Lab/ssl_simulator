"""PLOTTER ABSTRACT BASE CLASS."""

from matplotlib.pyplot import figure

from ssl_simulator.utils.dict_ops import parse_kwargs


class Plotter:
    fig = None

    def __init__(self, **kwargs):
        kw_fig = {"dpi": None, "figsize": None}
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
        if self.fig is None:
            raise RuntimeError("Figure not initialized; cannot save.")
        self.fig.savefig(filename, dpi=dpi, **kwargs)
