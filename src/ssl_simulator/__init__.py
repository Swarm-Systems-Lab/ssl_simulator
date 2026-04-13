"""
Top-level public API for ssl_simulator.
"""

from typing import Final

# Package public API: import subpackages and key symbols explicitly to avoid
# polluting the top-level namespace with wildcards and to reduce eager imports.
from . import components, controllers, exceptions, logging, math, robot_models, utils
from .config import CONFIG
from .core._controller import Controller
from .core._robot_model import RobotModel
from .core.simulation_context import SimulationContext
from .core.simulation_engine import INTEGRATORS, SimulationEngine
from .logging import __all__ as _LOGGING_PUBLIC_API
from .utils import __all__ as _UTILS_PUBLIC_API

# Configure default logging for the package
logging.setup_logging()

_BASE_PUBLIC_API: Final[list[str]] = [
    # Configuration
    "CONFIG",
    "INTEGRATORS",
    # Core classes
    "Controller",
    "RobotModel",
    "SimulationContext",
    "SimulationEngine",
    # Submodules
    "components",
    "controllers",
    "exceptions",
    "math",
    "robot_models",
    "utils",
]

__all__: Final[list[str]] = [*_BASE_PUBLIC_API, *_UTILS_PUBLIC_API, *_LOGGING_PUBLIC_API]


def __getattr__(name: str):
    if name in _LOGGING_PUBLIC_API:
        return getattr(logging, name)
    if name in _UTILS_PUBLIC_API:
        return getattr(utils, name)
    if name == "visualization":
        import importlib

        return importlib.import_module("ssl_simulator.visualization")
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__():
    return sorted(set(__all__) | {"visualization"})
