"""
"""

#######################################################################################

class RobotModel:
    def __init__(self):
        self.state = {}
        self.state_dot = {}

    # Data ----------------------------------------------------------------------------
    def init_data(self):
        self.data = self.state.copy()
        self.data.update(self.state_dot.copy())
    
    def update_data(self):
        for key,value in self.state.items():
            self.data[key] = value
        for key,value in self.state_dot.items():
            self.data[key] = value

    def get_labels(self):
        return self.data.keys() 

    def get_data(self):
        self.update_data()
        return self.data.copy()
    
    # State ---------------------------------------------------------------------------
    def get_state(self):
        return self.state

    def set_state(self, new_state):
        self.state = new_state
    
    # Dynamics  -----------------------------------------------------------------------
    def dynamics(self, state, dynamics_input):
        raise NotImplementedError(
            "The dynamics have not been implemented."
            )

#######################################################################################