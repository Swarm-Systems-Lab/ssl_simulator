"""
"""
import sys

# Expose core modules at the top level
from .core.simulator_engine.simulator_engine import SimulationEngine

from .core import controllers, data_manager, math, robot_models
sys.modules["ssl_simulator.controllers"] = controllers
sys.modules["ssl_simulator.data_manager"] = data_manager
sys.modules["ssl_simulator.math"] = math
sys.modules["ssl_simulator.robot_models"] = robot_models

from .core.utils import *

# Expose extensions optionally
from .extensions import scalar_fields, network, gvf_trajectories
sys.modules["ssl_simulator.scalar_fields"] = scalar_fields
sys.modules["ssl_simulator.network"] = network
sys.modules["ssl_simulator.gvf_trajectories"] = gvf_trajectories

# Expose visualization separately
from . import visualization

# Expose testing functions
from .testing.debug import *

del sys, core, extensions, testing