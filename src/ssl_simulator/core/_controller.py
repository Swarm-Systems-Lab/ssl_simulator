from collections.abc import Callable, MutableMapping

from ssl_simulator.utils.dict_ops import safe_assign, safe_update, validate_dict_attributes

from .types import ControlMap


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
        self.control_vars: dict[
            str, object
        ] = {}  # Controller OUTPUT variables (go to the dynamics)
        self.tracked_vars: dict[str, object] = {}  # Controller variables to be tracked by logger
        self.tracked_settings: dict[str, object] = {}  # Controller settings to be tracked by logger
        self.control_interface: dict[
            str, Callable[..., object]
        ] = {}  # Interface for other controllers to interact

    # Data ----------------------------------------------------------------------------
    def init_data(self) -> None:
        # Validate that all required attributes are dictionaries
        validate_dict_attributes(self, ["control_vars", "tracked_vars"])

        # Pre-compute whether any dicts contain callables (avoids per-step callable checks)
        self._has_callable_control_vars = any(callable(v) for v in self.control_vars.values())
        self._has_callable_tracked_vars = any(callable(v) for v in self.tracked_vars.values())
        self._dirty = True

        self.data = {}
        resolved_control_vars = {k: v() if callable(v) else v for k, v in self.control_vars.items()}  # type: ignore[misc]
        resolved_tracked_vars = {k: v() if callable(v) else v for k, v in self.tracked_vars.items()}  # type: ignore[misc]
        safe_update(self.data, resolved_control_vars, "control_vars")
        safe_update(self.data, resolved_tracked_vars, "tracked_vars")
        self.settings = self.tracked_settings.copy()

    def update_data(self) -> None:
        if not self._dirty:
            return
        if self._has_callable_control_vars:
            resolved_control_vars = {
                k: v() if callable(v) else v for k, v in self.control_vars.items()
            }  # type: ignore[misc]
        else:
            resolved_control_vars = dict(self.control_vars)
        if self._has_callable_tracked_vars:
            resolved_tracked_vars = {
                k: v() if callable(v) else v for k, v in self.tracked_vars.items()
            }  # type: ignore[misc]
        else:
            resolved_tracked_vars = dict(self.tracked_vars)
        safe_assign(self.data, resolved_control_vars, "control_vars")
        safe_assign(self.data, resolved_tracked_vars, "tracked_vars")
        self._dirty = False

    def get_labels(self) -> list[str]:
        return list(self.data.keys())

    def get_data(self) -> dict[str, object]:
        self.update_data()
        return self.data.copy()

    def get_settings(self) -> dict[str, object]:
        return self.settings.copy()

    def get_control_vars(self) -> dict[str, object]:
        if self._has_callable_control_vars:
            return {k: v() if callable(v) else v for k, v in self.control_vars.items()}  # type: ignore[misc]
        return dict(self.control_vars)

    def get_interface(self) -> dict[str, Callable[..., object]]:
        return self.control_interface.copy()

    def register_interface(self, *methods) -> None:
        new_interfaces = {m.__name__: m for m in methods}
        safe_update(self.control_interface, new_interfaces, "new_interfaces")

    # Control law ---------------------------------------------------------------------
    def compute_control(self, time: float, dt: float) -> None:
        raise NotImplementedError("The controller has not been implemented.")
