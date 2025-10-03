from ssl_simulator import safe_update, safe_assign

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
        self.control_inputs = {} # Robot control inputs (from the controllers)
        self.tracked_vars = {} # Robot model variables to be tracked by logger
        self.tracked_settings = {} # Robot model settings to be tracked by logger

    # Data ----------------------------------------------------------------------------
    def init_data(self):
        self.data = {}
        resolved_state     = {k: v() if callable(v) else v for k, v in self.state.items()}
        resolved_state_dot = {k: v() if callable(v) else v for k, v in self.state_dot.items()}
        resolved_control_inputs = {k: v() if callable(v) else v for k, v in self.control_inputs.items()}
        resolved_tracked_vars   = {k: v() if callable(v) else v for k, v in self.tracked_vars.items()}
        safe_update(self.data, resolved_state, "state")
        safe_update(self.data, resolved_state_dot, "state_dot")
        safe_update(self.data, resolved_control_inputs, "control_inputs")
        safe_update(self.data, resolved_tracked_vars, "tracked_vars")
        self.settings = self.tracked_settings.copy()

    def update_data(self):
        resolved_state     = {k: v() if callable(v) else v for k, v in self.state.items()}
        resolved_state_dot = {k: v() if callable(v) else v for k, v in self.state_dot.items()}
        resolved_control_inputs = {k: v() if callable(v) else v for k, v in self.control_inputs.items()}
        resolved_tracked_vars   = {k: v() if callable(v) else v for k, v in self.tracked_vars.items()}
        safe_assign(self.data, resolved_state, "state")
        safe_assign(self.data, resolved_state_dot, "state_dot")
        safe_assign(self.data, resolved_control_inputs, "control_inputs")
        safe_assign(self.data, resolved_tracked_vars, "tracked_vars")

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
    
    def get_state_dot(self):
        return self.state_dot
    
    def set_state(self, new_state):
        self.state = new_state
    
    # Dynamics  -----------------------------------------------------------------------
    def dynamics(self, time):
        raise NotImplementedError(
            "The dynamics have not been implemented."
            )

#######################################################################################