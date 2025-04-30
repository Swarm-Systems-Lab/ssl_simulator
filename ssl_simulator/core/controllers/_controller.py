"""
"""
__all__ = ["Controller"]

#######################################################################################

class Controller:
    def __init__(self):
        self.control_vars = {} # Controller output variables (go to the dynamics)
        self.tracked_vars = {} # Controller variables to be tracked by logger

    # Data ----------------------------------------------------------------------------
    def init_data(self):
        self.data = self.control_vars.copy()
        self.data.update(self.tracked_vars.copy())
    
    def update_data(self):
        for key,value in self.control_vars.items():
            self.data[key] = value
        for key,value in self.tracked_vars.items():
            self.data[key] = value
    
    def get_labels(self):
        return self.data.keys()

    def get_data(self):
        self.update_data()
        return self.data.copy()
    
    # Control law ---------------------------------------------------------------------
    def compute_control(self, time, state):
        raise NotImplementedError("The controller have not been implemented.")
    


#######################################################################################