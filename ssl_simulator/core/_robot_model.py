"""
"""

#######################################################################################

class RobotModel:
    state = {}
    state_dot = {}
    tracked_vars = {} # Robot model variables to be tracked by logger
    tracked_settings = {} # Robot model settings to be tracked by logger

    # Data ----------------------------------------------------------------------------
    def init_data(self):
        self.data = self.state.copy()
        self.data.update(self.state_dot.copy())
        self.data.update(self.tracked_vars.copy())
        self.settings = self.tracked_settings.copy()

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

    def get_settings(self):
        return self.settings.copy()
    
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