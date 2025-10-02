import inspect

from ._robot_model import RobotModel
from ._controller import Controller

from ssl_simulator import safe_update

#######################################################################################

class SimulationContext:
    """
    Central context for managing a robot model and its controllers during a simulation.
    """

    def __init__(self):
        self.robot_model: RobotModel = None
        self.controllers: dict[str, Controller] = {}   # keyed controllers
        self.connections: list[tuple[str, dict]] = []  # controller_key -> mapping to robot inputs

        self.ctrl_interfaces: dict[str, dict[str]] = {}
        self.initialized = False

    # -------------------------------------------------------------------------
    # Robot model
    # -------------------------------------------------------------------------
    def set_robot_model(self, robot_model_class: RobotModel, *args, **kwargs):
        """
        Set the robot model for this simulation context.

        The robot model is automatically provided with a reference 
        to this context during initialization.
        """
        self.robot_model = robot_model_class(self, *args, **kwargs)
        self.robot_model.init_data()
        return self.robot_model

    def compute_robot_dynamics(self, time):
        """Compute the robot model dynamics."""
        return self.robot_model.dynamics(time)

    def get_robot_state(self):
        """Return the current state of the robot model."""
        return self.robot_model.get_state()
  
    def get_robot_state_dot(self):
        """Return the current state variation from the robot model dynamics."""
        return self.robot_model.get_state_dot()  

    def set_robot_state(self, state):
        """Set the state of the robot model."""
        self.robot_model.set_state(state)
        
    # -------------------------------------------------------------------------
    # Controllers
    # -------------------------------------------------------------------------
    def add_controller(self, key: str, controller_class: Controller, *args, **kwargs):
        """
        Add a controller to this simulation context with a unique key.
        """
        if key in self.controllers:
            raise KeyError(f"A controller with key '{key}' already exists.")
        controller = controller_class(self, *args, **kwargs)
        controller.init_data()
        self.controllers[key] = controller
        self.ctrl_interfaces[key] = controller.control_interface

    def connect_controller_to_robot(self, controller_key: str, mapping: dict = None):
        """
        Manually connect controller outputs to robot inputs using the controller key.
        """
        if controller_key not in self.controllers:
            raise KeyError(f"Controller with key '{controller_key}' not found.")
        controller = self.controllers[controller_key]

        if mapping is None:
            mapping = {var: var for var in controller.control_vars.keys()}

        self.connections.append((controller_key, mapping))

    def compute_controls(self, time):
        """
        Compute control signals from all controllers and propagate manual connections.
        """
        control_vars = {}
        for key, controller in reversed(list(self.controllers.items())):
            controller.compute_control(time)
            control_vars.update(controller.control_vars.copy())

        if self.robot_model is not None:
            for controller_key, mapping in self.connections:
                controller = self.controllers[controller_key]
                for ctrl_var, robot_input in mapping.items():
                    if ctrl_var in controller.control_vars and robot_input in self.robot_model.control_inputs:
                        self.robot_model.control_inputs[robot_input] = controller.control_vars[ctrl_var]

        self.control_vars = control_vars

    # Interfaces
    def call_interface(self, ctrl_key, method, *args, **kwargs):
        """
        Call an exposed control interface method from a specific controller.

        Automatically detects the calling controller to check execution order.
        """
        if not self.initialized:
            if ctrl_key not in self.ctrl_interfaces:
                raise KeyError(f"Controller '{ctrl_key}' not found in ctrl_interfaces.")

            if method not in self.ctrl_interfaces[ctrl_key]:
                raise KeyError(f"Method '{method}' not found in controller '{ctrl_key}'.")
            
            # Attempt to automatically detect the caller object
            stack = inspect.stack()
            caller_self = None
            for frame_info in stack:
                # Look for a frame that has 'self' in local variables
                if 'self' in frame_info.frame.f_locals:
                    candidate = frame_info.frame.f_locals['self']
                    # Check if the candidate object is one of the controllers
                    if candidate in self.controllers.values():
                        caller_self = candidate
                        break

            if caller_self is None:
                raise RuntimeError("Cannot detect the calling controller automatically.")

            # Identify the key of the calling controller
            caller_key = None
            for key, ctrl in self.controllers.items():
                if ctrl is caller_self:
                    caller_key = key
                    break

            # Check execution order
            controller_keys = list(self.controllers.keys())[::-1]  # reversed execution order
            caller_index = controller_keys.index(caller_key)
            target_index = controller_keys.index(ctrl_key)

            if caller_index > target_index:
                raise RuntimeError(
                    f"Controller '{caller_key}' is attempting to modify controller '{ctrl_key}' "
                    f"during the same simulation step, but '{ctrl_key}' executes **before** '{caller_key}'.\n"
                    f"As a result, the change will only take effect in the next simulation step.\n"
                    f"Potential issues:\n"
                    f"  - You may have cyclic or unintended interface calls.\n"
                    f"  - The execution order of controllers matters; the last added controller executes first.\n"
                    f"Suggestions to fix:\n"
                    f"  -> Add '{caller_key}' before '{ctrl_key}' in the simulation context to execute it first.\n"
                    )

        return self.ctrl_interfaces[ctrl_key][method](*args, **kwargs)
        
    def list_interfaces(self, controller_key=None):
        if controller_key is not None:
            if controller_key not in self.controllers:
                raise KeyError(f"No controller with key '{controller_key}' found.")
            return list(self.controllers[controller_key].control_interface.keys())
        else:
            return self.ctrl_interfaces

    # -------------------------------------------------------------------------
    # Logging helpers
    # -------------------------------------------------------------------------
    def get_labels(self):
        """
        Aggregate variable labels from the robot model and all controllers,
        prefixing them to avoid conflicts.
        """
        labels = []
        # Robot variables
        for var in self.robot_model.get_labels():
            labels.append(f"robot.{var}")

        # Controller variables
        for key, controller in self.controllers.items():
            for var in controller.get_labels():
                labels.append(f"{key}.{var}")

        return labels
    
    def get_data(self):
        """
        Aggregate variable data from the robot model and all controllers,
        prefixing keys to avoid conflicts.
        """
        data = {}
        # Robot variables
        robot_data = self.robot_model.get_data()
        for var, value in robot_data.items():
            data[f"robot.{var}"] = value

        # Controller variables
        for key, controller in self.controllers.items():
            ctrl_data = controller.get_data()
            for var, value in ctrl_data.items():
                data[f"{key}.{var}"] = value

        return data

    def get_settings(self):
        """
        Aggregate settings from the robot model and all controllers,
        prefixing keys to avoid conflicts.
        """
        settings = {}
        # Robot settings
        robot_settings = self.robot_model.get_settings()
        for var, value in robot_settings.items():
            settings[f"robot.{var}"] = value

        # Controller settings
        for key, controller in self.controllers.items():
            ctrl_settings = controller.get_settings()
            for var, value in ctrl_settings.items():
                settings[f"{key}.{var}"] = value

        return settings

    # -------------------------------------------------------------------------
    # Debug / Print helpers
    # -------------------------------------------------------------------------
    def print_robot_inputs(self, show_values: bool = False):
        """
        Print the control inputs of the robot model and optionally their shapes/values.
        
        Parameters
        ----------
        show_values : bool
            If True, prints the current value and shape of each input.
        """
        if self.robot_model is None:
            print("No robot model set in the simulation context.")
            return

        if not hasattr(self.robot_model, "control_inputs") or not self.robot_model.control_inputs:
            print("Robot model has no control inputs defined.")
            return

        print(f"Robot model '{self.robot_model.__class__.__name__}' control inputs:")
        for input_name, value in self.robot_model.control_inputs.items():
            if value is None:
                shape = "None"
            else:
                try:
                    shape = value.shape
                except AttributeError:
                    shape = "scalar"
            if show_values:
                print(f"  - Input: '{input_name}', Value: {value}, Shape: {shape}")
            else:
                print(f"  - Input: '{input_name}', Shape: {shape}")

    def print_controllers(self, show_outputs: bool = True):
        """
        Print all controllers in the context and optionally their control outputs with shapes.
        
        Parameters
        ----------
        show_outputs : bool
            If True, prints the control variables and their shapes.
        """
        if not self.controllers:
            print("No controllers added to the simulation context.")
            return

        self.compute_controls(0) # Run all controllers one step to update shape

        print("Controllers in the simulation context:")
        for key, controller in self.controllers.items():
            print(f"  - Key: '{key}', Controller: {controller.__class__.__name__}")
            if show_outputs:
                if not controller.control_vars:
                    print("      Control outputs: None")
                else:
                    for var_name, value in controller.control_vars.items():
                        if value is None:
                            shape = None
                        else:
                            try:
                                shape = value.shape
                            except AttributeError:
                                shape = "scalar"
                        print(f"      Output: '{var_name}', Shape: {shape}")

    def print_control_interfaces(self, key=None):
        """
        Print control interfaces in the simulation context.

        Parameters
        ----------
        key : str, optional
            If provided, only print the interfaces of the specified controller.
        """
        if not self.ctrl_interfaces:
            print("No control interfaces registered in the simulation context.")
            return

        def _print_methods(methods, pre=""):
            for method_name, method in methods.items():
                try:
                    sig = inspect.signature(method)
                    print(pre + f"  - {method_name}{sig}")
                except (TypeError, ValueError):
                    # Some callables may not have a retrievable signature
                    print(pre + f"  - {method_name}(...)")

        if key is not None:
            if key not in self.ctrl_interfaces:
                print(f"Controller '{key}' not found in ctrl_interfaces.")
                return
            print(f"Control interfaces for controller '{key}':")
            _print_methods(self.ctrl_interfaces[key])
        else:
            print("Control interfaces in the simulation context:")
            for ctrl_key, methods in self.ctrl_interfaces.items():
                print(f"  * Controller '{ctrl_key}':")
                _print_methods(methods, pre="    ")

#######################################################################################