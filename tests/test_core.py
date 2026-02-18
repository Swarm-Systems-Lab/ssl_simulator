import sys

import numpy as np
import pytest


def test_controller_import():
    from ssl_simulator.core import _controller

    assert hasattr(_controller, "Controller")


def test_robot_model_import():
    from ssl_simulator.core import _robot_model

    assert hasattr(_robot_model, "RobotModel")


def test_simulation_context_import():
    from ssl_simulator.core import simulation_context

    assert hasattr(simulation_context, "SimulationContext")


def test_simulation_engine_import():
    from ssl_simulator.core import simulation_engine

    assert hasattr(simulation_engine, "SimulationEngine")


def test_config_import():
    from ssl_simulator import config

    assert hasattr(config, "CONFIG")


def test_oscillator_control_vars_propagate_to_robot_input():
    from ssl_simulator.controllers import Oscillator
    from ssl_simulator.core import SimulationEngine
    from ssl_simulator.robot_models import SingleIntegrator

    sim = SimulationEngine(time_step=0.1)
    sim.set_robot_model(SingleIntegrator, [np.zeros((3, 2))])
    sim.add_controller("osc", Oscillator, A=np.zeros(3), omega=np.zeros(3), speed=np.ones(3))
    sim.connect_controller_to_robot("osc", {"u": "u"})

    sim.compute_controls(0.0, sim.time_step)

    assert sim.robot_model.control_inputs["u"].shape == (3, 2)


def test_call_interface_execution_order_enforced_after_initialization():
    from ssl_simulator.core import Controller, SimulationContext
    from ssl_simulator.robot_models import SingleIntegrator

    class TargetController(Controller):
        def __init__(self, context):
            super().__init__(context)
            self.value = 0
            self.register_interface(self._set_value)

        def _set_value(self, value):
            self.value = value

        def compute_control(self, time, dt):
            return None

    class CallerController(Controller):
        def __init__(self, context):
            super().__init__(context)

        def compute_control(self, time, dt):
            self.context.call_interface("target", "_set_value", 1)
            return None

    context = SimulationContext()
    context.set_robot_model(SingleIntegrator, [np.zeros((1, 2))])
    context.add_controller("caller", CallerController)
    context.add_controller("target", TargetController)
    context.initialized = True

    with pytest.raises(RuntimeError, match="attempting to modify controller"):
        context.compute_controls(0.0, 0.1)


def test_root_package_import_does_not_eager_import_visualization():
    import ssl_simulator

    assert "ssl_simulator.visualization" not in sys.modules
    assert hasattr(ssl_simulator, "SimulationEngine")
