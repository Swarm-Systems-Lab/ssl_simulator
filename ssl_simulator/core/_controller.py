#######################################################################################

class Controller:
    def __init__(self, context, *args, **kwargs):
        """
        Base controller.
        Every subclass must accept `context` as its first argument.
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
        self.control_vars = {} # Controller output variables (go to the dynamics)
        self.tracked_vars = {} # Controller variables to be tracked by logger
        self.tracked_settings = {} # Controller settings to be tracked by logger

    # Data ----------------------------------------------------------------------------
    def init_data(self):
        self.data = self.control_vars.copy()
        self.data.update(self.tracked_vars.copy())
        self.settings = self.tracked_settings.copy()
    
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

    def get_settings(self):
        return self.settings.copy()
    
    # Control law ---------------------------------------------------------------------
    def compute_control(self, time, state, input_control_vars):
        raise NotImplementedError("The controller have not been implemented.")

#######################################################################################