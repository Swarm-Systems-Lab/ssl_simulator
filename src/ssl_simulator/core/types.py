from collections.abc import Callable, Mapping, MutableMapping
from typing import Any, Protocol, runtime_checkable

import numpy as np

# Core array type alias
Array = np.ndarray

# Mappings used across the core
StateMap = Mapping[str, Array]
MutableStateMap = MutableMapping[str, Array]
ControlMap = MutableMapping[str, Any]  # Can hold arrays or other objects


@runtime_checkable
class RobotModelProtocol(Protocol):
    state: MutableStateMap
    state_dot: MutableStateMap
    control_inputs: MutableStateMap

    def init_data(self) -> None: ...

    def get_state(self) -> MutableStateMap: ...

    def get_state_dot(self) -> MutableStateMap: ...

    def set_state(self, state: MutableStateMap) -> None: ...

    def dynamics(self, time: float) -> MutableStateMap: ...


@runtime_checkable
class ControllerProtocol(Protocol):
    control_vars: dict[str, object]
    tracked_vars: dict[str, object]
    control_interface: dict[str, Callable[..., object]]

    def init_data(self) -> None: ...

    def compute_control(self, time: float, dt: float) -> None: ...
