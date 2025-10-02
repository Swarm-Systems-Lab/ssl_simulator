from ._robot_model import RobotModel
from ._controller import Controller

#######################################################################################

class SimulationContext:
    """
    Central context for managing a robot model and its controllers during a simulation.
    """

    def __init__(self):
        self.robot_model: RobotModel = None
        self.controllers: list[Controller] = []

    # -------------------------------------------------------------------------
    # Robot model
    # -------------------------------------------------------------------------
    def set_robot_model(self, robot_model_class, *args, **kwargs):
        """
        Set the robot model for this simulation context.

        The robot model is automatically provided with a reference 
        to this context during initialization.
        """
        self.robot_model = robot_model_class(self, *args, **kwargs)
        return self.robot_model

    def get_robot_state(self):
        """Return the current state of the robot model."""
        return self.robot_model.get_state()
    
    def get_robot_dynamics(self):
        """Return the robot model dynamics method."""
        return self.robot_model.dynamics
    
    def set_robot_state(self, state):
        """Set the state of the robot model."""
        self.robot_model.set_state(state)
        
    # -------------------------------------------------------------------------
    # Controllers
    # -------------------------------------------------------------------------
    def add_controller(self, controller_class, *args, **kwargs):
        """
        Add a controller to this simulation context.
        
        The controller is automatically provided with a reference 
        to this context during initialization.
        """
        controller = controller_class(self, *args, **kwargs)
        self.controllers.append(controller)
        return controller

    def compute_controls(self, time):
        """
        Compute control signals from all controllers.

        Parameters
        ----------
        time : float
            Current simulation time.

        Returns
        -------
        dict
            Final aggregated control variables from all controllers.
        """
        state = self.get_robot_state()
        control_vars = {}
        for controller in self.controllers:
            controller.compute_control(time, state, control_vars)
            control_vars.update(controller.control_vars.copy())
        return control_vars
    
    # -------------------------------------------------------------------------
    # Logging helpers
    # -------------------------------------------------------------------------
    def get_labels(self):
        """Aggregate variable labels from the robot model and all controllers."""
        labels = list(self.robot_model.get_labels())
        for controller in self.controllers:
            labels.extend(controller.get_labels())
        return labels
    
    def get_data(self):
        """Aggregate variable data from the robot model and all controllers."""
        data = self.robot_model.get_data()
        for controller in self.controllers:
            data.update(controller.get_data())
        return data

    def get_settings(self):
        """Aggregate settings from the robot model and all controllers."""
        settings = self.robot_model.get_settings()
        for controller in self.controllers:
            settings.update(controller.get_settings())
        return settings
    
#######################################################################################