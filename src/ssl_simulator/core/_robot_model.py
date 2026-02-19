from ssl_simulator.utils.dict_ops import safe_assign, safe_update, validate_dict_attributes

from .types import ControlMap, MutableStateMap

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
        self.state: MutableStateMap = {}
        self.state_dot: MutableStateMap = {}
        self.control_inputs: ControlMap = {}  # Robot control inputs (from the controllers)
        self.tracked_vars: dict[str, object] = {}  # Robot model variables to be tracked by logger
        self.tracked_settings: dict[
            str, object
        ] = {}  # Robot model settings to be tracked by logger

    # Data ----------------------------------------------------------------------------
    def init_data(self) -> None:
        # Validate that all required attributes are dictionaries
        validate_dict_attributes(self, ["state", "state_dot", "control_inputs", "tracked_vars"])

        # Pre-compute whether any dicts contain callables (avoids per-step callable checks)
        self._has_callable_state = any(callable(v) for v in self.state.values())
        self._has_callable_state_dot = any(callable(v) for v in self.state_dot.values())
        self._has_callable_control_inputs = any(callable(v) for v in self.control_inputs.values())
        self._has_callable_tracked_vars = any(callable(v) for v in self.tracked_vars.values())
        self._dirty = True

        self.data = {}
        resolved_state = {k: v() if callable(v) else v for k, v in self.state.items()}
        resolved_state_dot = {k: v() if callable(v) else v for k, v in self.state_dot.items()}
        resolved_control_inputs = {
            k: v() if callable(v) else v for k, v in self.control_inputs.items()
        }
        resolved_tracked_vars = {k: v() if callable(v) else v for k, v in self.tracked_vars.items()}  # type: ignore[misc]
        safe_update(self.data, resolved_state, "state")
        safe_update(self.data, resolved_state_dot, "state_dot")
        safe_update(self.data, resolved_control_inputs, "control_inputs")
        safe_update(self.data, resolved_tracked_vars, "tracked_vars")
        self.settings = self.tracked_settings.copy()

    def update_data(self) -> None:
        if not self._dirty:
            return
        resolved_state = (
            {k: v() if callable(v) else v for k, v in self.state.items()}
            if self._has_callable_state
            else dict(self.state)
        )
        resolved_state_dot = (
            {k: v() if callable(v) else v for k, v in self.state_dot.items()}
            if self._has_callable_state_dot
            else dict(self.state_dot)
        )
        resolved_control_inputs = (
            {k: v() if callable(v) else v for k, v in self.control_inputs.items()}
            if self._has_callable_control_inputs
            else dict(self.control_inputs)
        )
        resolved_tracked_vars = (
            {k: v() if callable(v) else v for k, v in self.tracked_vars.items()}  # type: ignore[misc]
            if self._has_callable_tracked_vars
            else dict(self.tracked_vars)
        )
        safe_assign(self.data, resolved_state, "state")
        safe_assign(self.data, resolved_state_dot, "state_dot")
        safe_assign(self.data, resolved_control_inputs, "control_inputs")
        safe_assign(self.data, resolved_tracked_vars, "tracked_vars")
        self._dirty = False

    def get_labels(self) -> list[str]:
        return list(self.data.keys())

    def get_data(self) -> dict[str, object]:
        self.update_data()
        return self.data.copy()

    def get_settings(self) -> dict[str, object]:
        return self.settings.copy()

    # State ---------------------------------------------------------------------------
    def get_state(self) -> MutableStateMap:
        return self.state

    def get_state_dot(self) -> MutableStateMap:
        return self.state_dot

    def set_state(self, new_state: MutableStateMap) -> None:
        self.state = new_state

    # Dynamics  -----------------------------------------------------------------------
    def dynamics(self, time: float) -> MutableStateMap:
        raise NotImplementedError("The dynamics have not been implemented.")


#######################################################################################
