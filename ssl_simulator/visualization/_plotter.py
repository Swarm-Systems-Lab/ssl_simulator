"""
    PLOTTER ABSTRACT BASE CLASS
"""

#######################################################################################

class Plotter:
    fig = None

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