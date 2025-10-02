#######################################################################################

class RobotModel:
    def __init__(self, context, *args, **kwargs):
        """
        Base class for all robot models.

        Every subclass must accept a SimulationContext as its first argument
        so that it can access shared simulation data.
        """
        if not hasattr(context, "add_controller") or not hasattr(context, "robot_model"):
            raise TypeError(
                f"Invalid context type: expected a SimulationContext (provided by the "
                f"SimulationEngine when using set_robot_model or add_controller), "
                f"but got {type(context).__name__} instead. "
                f"RobotModel and Controller subclasses should not be instantiated "
                f"directly — use SimulationContext.set_robot_model() and "
                f"SimulationContext.add_controller() to ensure proper initialization."
            )
        self.context = context
        self.state = {}
        self.state_dot = {}
        self.tracked_vars = {} # Robot model variables to be tracked by logger
        self.tracked_settings = {} # Robot model settings to be tracked by logger

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