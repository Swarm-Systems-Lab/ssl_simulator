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
