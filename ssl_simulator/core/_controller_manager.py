
#######################################################################################

class ControllerManager:
    def __init__(self):
        self.controllers = []

    def add_controller(self, controller):
        """Add a controller to the manager."""
        self.controllers.append(controller)

    def compute_controls(self, time, state):
        """
        Compute controls for all controllers.
        - `time`: Current simulation time.
        - `state`: Current state of the robot.
        Returns the final control input.
        """
        # Shared data structure for inter-controller & robot model communication
        control_vars = {}
        for controller in self.controllers:
            controller.compute_control(time, state, control_vars)
            control_vars.update(controller.control_vars.copy())
        return control_vars

    def get_labels(self):
        """Aggregate labels from all controllers."""
        labels = []
        for controller in self.controllers:
            labels.extend(controller.get_labels())
        return labels
    
    def get_data(self):
        """Aggregate data from all controllers."""
        data = {}
        for controller in self.controllers:
            data.update(controller.get_data())
        return data

    def get_settings(self):
        """Aggregate settings from all controllers."""
        settings = {}
        for controller in self.controllers:
            settings.update(controller.get_settings())
        return settings
    
#######################################################################################