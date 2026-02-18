import importlib

# Package public API: import subpackages and key symbols explicitly to avoid
# polluting the top-level namespace with wildcards and to reduce eager imports.
# Exceptions and utils modules (expose modules, not wildcard symbols)
from . import components, controllers, exceptions, math, robot_models, utils

# Configuration
from .config import CONFIG

# Core classes
from .core._controller import Controller
from .core._robot_model import RobotModel
from .core.simulation_context import SimulationContext
from .core.simulation_engine import INTEGRATORS, SimulationEngine

__all__ = [
    "CONFIG",
    "INTEGRATORS",
    "Controller",
    "RobotModel",
    "SimulationContext",
    "SimulationEngine",
    "components",
    "controllers",
    "exceptions",
    "math",
    "robot_models",
    "utils",
]


def __getattr__(name: str):
    if name == "visualization":
        return importlib.import_module("ssl_simulator.visualization")
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
