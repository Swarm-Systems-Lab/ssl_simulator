from ssl_simulator.utils.dict_ops import safe_assign, safe_update, validate_dict_attributes

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
        self.control_vars = {}  # Controller OURPUT variables (go to the dynamics)
        self.tracked_vars = {}  # Controller variables to be tracked by logger
        self.tracked_settings = {}  # Controller settings to be tracked by logger
        self.control_interface = {}  # Interface for other controllers to interact

    # Data ----------------------------------------------------------------------------
    def init_data(self):
        # Validate that all required attributes are dictionaries
        validate_dict_attributes(self, ["control_vars", "tracked_vars"])

        self.data = {}
        resolved_control_vars = {k: v() if callable(v) else v for k, v in self.control_vars.items()}
        resolved_tracked_vars = {k: v() if callable(v) else v for k, v in self.tracked_vars.items()}
        safe_update(self.data, resolved_control_vars, "tracked_vars")
        safe_update(self.data, resolved_tracked_vars, "tracked_vars")
        self.settings = self.tracked_settings.copy()

    def update_data(self):
        resolved_control_vars = {k: v() if callable(v) else v for k, v in self.control_vars.items()}
        resolved_tracked_vars = {k: v() if callable(v) else v for k, v in self.tracked_vars.items()}
        safe_assign(self.data, resolved_control_vars, "control_vars")
        safe_assign(self.data, resolved_tracked_vars, "tracked_vars")

    def get_labels(self):
        return self.data.keys()

    def get_data(self):
        self.update_data()
        return self.data.copy()

    def get_settings(self):
        return self.settings.copy()

    def get_control_vars(self):
        resolved_control_vars = {k: v() if callable(v) else v for k, v in self.control_vars.items()}
        return resolved_control_vars

    def get_interface(self):
        return self.control_interface.copy()

    def register_interface(self, *methods):
        new_interfaces = {m.__name__: m for m in methods}
        safe_update(self.control_interface, new_interfaces, "new_interfaces")

    # Control law ---------------------------------------------------------------------
    def compute_control(self, time, dt):
        raise NotImplementedError("The controller have not been implemented.")


#######################################################################################
