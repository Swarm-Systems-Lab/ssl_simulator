import contextlib
import inspect
from collections.abc import Callable, MutableMapping

from ssl_simulator.exceptions import InitializationError

from ._controller import Controller
from ._robot_model import RobotModel
from .types import ControllerProtocol, ControlMap, MutableStateMap, RobotModelProtocol

#######################################################################################


class SimulationContext:
    """Central context for managing a robot model and its controllers during a simulation."""

    def __init__(self):
        self.robot_model: RobotModel | None = None
        self.controllers: dict[str, Controller] = {}  # keyed controllers
        self.connections: list[tuple[str, dict]] = []  # controller_key -> mapping to robot inputs

        # Last computed control vars aggregated from controllers
        self.control_vars: dict[str, object] = {}

        self.ctrl_interfaces: dict[str, dict[str, Callable[..., object]]] = {}
        self.initialized = False

    # -------------------------------------------------------------------------
    # Robot model
    # -------------------------------------------------------------------------
    def set_robot_model(self, robot_model_class: type[RobotModel], *args, **kwargs) -> RobotModel:
        """
        Set the robot model for this simulation context.

        The robot model is automatically provided with a reference
        to this context during initialization.
        """
        self.robot_model = robot_model_class(self, *args, **kwargs)
        try:
            self.robot_model.init_data()
            return self.robot_model
        except Exception as e:
            raise InitializationError(str(e)) from None

    def compute_robot_dynamics(self, time: float) -> MutableStateMap:
        """Compute the robot model dynamics."""
        if self.robot_model is None:
            raise RuntimeError("Robot model not set. Call `set_robot_model()` first.")
        return self.robot_model.dynamics(time)

    def get_robot_state(self) -> MutableStateMap:
        """Return the current state of the robot model."""
        if self.robot_model is None:
            raise RuntimeError("Robot model not set. Call `set_robot_model()` first.")
        return self.robot_model.get_state()

    def get_robot_state_dot(self) -> MutableStateMap:
        """Return the current state variation from the robot model dynamics."""
        if self.robot_model is None:
            raise RuntimeError("Robot model not set. Call `set_robot_model()` first.")
        return self.robot_model.get_state_dot()

    def set_robot_state(self, state: MutableStateMap) -> None:
        """Set the state of the robot model."""
        if self.robot_model is None:
            raise RuntimeError("Robot model not set. Call `set_robot_model()` first.")
        self.robot_model.set_state(state)

    # -------------------------------------------------------------------------
    # Controllers
    # -------------------------------------------------------------------------
    def add_controller(
        self, key: str, controller_class: type[Controller], *args, **kwargs
    ) -> Controller:
        """Add a controller to this simulation context with a unique key."""
        if key in self.controllers:
            raise KeyError(f"A controller with key '{key}' already exists.")
        controller = controller_class(self, *args, **kwargs)
        try:
            controller.init_data()
            self.controllers[key] = controller
            self.ctrl_interfaces[key] = controller.control_interface
            return controller
        except Exception as e:
            raise InitializationError(str(e)) from None

    def connect_controller_to_robot(
        self, controller_key: str, mapping: dict[str, str] | None = None
    ) -> None:
        """
        Manually connect controller outputs to robot inputs using the controller key.

        Args:
            controller_key (str): The key identifying the controller in self.controllers.
            mapping (dict, optional): A dictionary mapping controller variables to robot inputs.
                Example: {"controller_var": "robot_input"}

        Raises
        ------
            KeyError: If the controller key, controller variable, or robot input does not exist.
        """
        if controller_key not in self.controllers:
            raise KeyError(f"Controller with key '{controller_key}' not found.")
        if self.robot_model is None:
            raise RuntimeError("Robot model not set. Call `set_robot_model()` first.")

        controller = self.controllers[controller_key]
        controller_vars = set(controller.get_control_vars().keys())
        robot_inputs = set(self.robot_model.control_inputs.keys())

        if mapping is None:
            # Default: identity mapping for all control variables
            mapping = {var: var for var in controller_vars}
        else:
            # Validate mapping
            for c_var, r_in in mapping.items():
                if c_var not in controller_vars:
                    raise KeyError(
                        f"Controller var '{c_var}' not found in controller '{controller_key}'. "
                        f"Available: {list(controller_vars)}"
                    )
                if r_in not in robot_inputs:
                    raise KeyError(
                        f"Robot input '{r_in}' not found in robot model. "
                        f"Available: {list(robot_inputs)}"
                    )

        self.connections.append((controller_key, mapping))

    def compute_controls(self, time: float, dt: float) -> None:
        """Compute control signals from all controllers and propagate manual connections."""
        control_vars = {}
        for _key, controller in reversed(list(self.controllers.items())):
            controller.compute_control(time, dt)
            control_vars.update(controller.get_control_vars().copy())

        if self.robot_model is not None:
            for controller_key, mapping in self.connections:
                controller = self.controllers[controller_key]
                for ctrl_var, robot_input in mapping.items():
                    if (
                        ctrl_var in controller.get_control_vars()
                        and robot_input in self.robot_model.control_inputs
                    ):
                        self.robot_model.control_inputs[robot_input] = (
                            controller.get_control_vars()[ctrl_var]
                        )

        self.control_vars = control_vars

    # Interfaces
    def call_interface(self, ctrl_key: str, method: str, *args, **kwargs) -> object:
        """
        Call an exposed control interface method from a specific controller.

        Automatically detects the calling controller to check execution order.
        """
        if ctrl_key not in self.ctrl_interfaces:
            raise KeyError(f"Controller '{ctrl_key}' not found in ctrl_interfaces.")

        if method not in self.ctrl_interfaces[ctrl_key]:
            raise KeyError(f"Method '{method}' not found in controller '{ctrl_key}'.")

        # Attempt to automatically detect the caller object. This may fail for
        # non-controller callers (e.g. user code), which is a valid use case.
        stack = inspect.stack()
        caller_self = None
        with contextlib.suppress(Exception):
            for frame_info in stack:
                if "self" in frame_info.frame.f_locals:
                    candidate = frame_info.frame.f_locals["self"]
                    if candidate in self.controllers.values():
                        caller_self = candidate
                        break

        # If called from within a controller, enforce execution-order safety.
        if caller_self is not None:
            caller_key = None
            for key, ctrl in self.controllers.items():
                if ctrl is caller_self:
                    caller_key = key
                    break

            if caller_key is not None:
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

    def list_interfaces(
        self, controller_key: str | None = None
    ) -> list[str] | dict[str, dict[str, Callable[..., object]]]:
        if controller_key is not None:
            if controller_key not in self.controllers:
                raise KeyError(f"No controller with key '{controller_key}' found.")
            return list(self.controllers[controller_key].control_interface.keys())
        else:
            return self.ctrl_interfaces

    # -------------------------------------------------------------------------
    # Logging helpers
    # -------------------------------------------------------------------------
    def get_labels(self) -> list[str]:
        """
        Aggregate variable labels from the robot model and all controllers,
        prefixing them to avoid conflicts.
        """
        labels = []
        # Robot variables
        if self.robot_model is None:
            raise RuntimeError("Robot model not set. Call `set_robot_model()` first.")
        for var in self.robot_model.get_labels():
            labels.append(f"robot.{var}")

        # Controller variables
        for key, controller in self.controllers.items():
            for var in controller.get_labels():
                labels.append(f"{key}.{var}")

        return labels

    def get_data(self) -> dict[str, object]:
        """
        Aggregate variable data from the robot model and all controllers,
        prefixing keys to avoid conflicts.
        """
        data = {}
        # Robot variables
        if self.robot_model is None:
            raise RuntimeError("Robot model not set. Call `set_robot_model()` first.")
        robot_data = self.robot_model.get_data()
        for var, value in robot_data.items():
            data[f"robot.{var}"] = value

        # Controller variables
        for key, controller in self.controllers.items():
            ctrl_data = controller.get_data()
            for var, value in ctrl_data.items():
                data[f"{key}.{var}"] = value

        return data

    def get_settings(self) -> dict[str, object]:
        """
        Aggregate settings from the robot model and all controllers,
        prefixing keys to avoid conflicts.
        """
        settings = {}
        # Robot settings
        if self.robot_model is None:
            raise RuntimeError("Robot model not set. Call `set_robot_model()` first.")
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
            return

        if not hasattr(self.robot_model, "control_inputs") or not self.robot_model.control_inputs:
            return

        for _input_name, value in self.robot_model.control_inputs.items():
            if value is None:
                pass
            else:
                with contextlib.suppress(AttributeError):
                    pass
            if show_values:
                pass
            else:
                pass

    def print_controllers(self, show_outputs: bool = True):
        """
        Print all controllers in the context and optionally their control outputs with shapes.

        Parameters
        ----------
        show_outputs : bool
            If True, prints the control variables and their shapes.
        """
        if not self.controllers:
            return

        self.compute_controls(0, 0)  # Run all controllers one step to update shape

        for _key, controller in self.controllers.items():
            if show_outputs:
                if not controller.get_control_vars():
                    pass
                else:
                    for _var_name, value in controller.get_control_vars().items():
                        if value is None:
                            pass
                        else:
                            with contextlib.suppress(AttributeError):
                                pass

    def print_control_interfaces(self, key=None):
        """
        Print control interfaces in the simulation context.

        Parameters
        ----------
        key : str, optional
            If provided, only print the interfaces of the specified controller.
        """
        if not self.ctrl_interfaces:
            return

        def _print_methods(methods, pre=""):
            for _method_name, method in methods.items():
                with contextlib.suppress(TypeError, ValueError):
                    inspect.signature(method)

        if key is not None:
            if key not in self.ctrl_interfaces:
                return
            _print_methods(self.ctrl_interfaces[key])
        else:
            for _ctrl_key, methods in self.ctrl_interfaces.items():
                _print_methods(methods, pre="    ")


#######################################################################################
