"""Shared fixtures and factory helpers for performance benchmark tests."""

import numpy as np
import pytest

from ssl_simulator.controllers.constant_signal import ConstantSignal
from ssl_simulator.core import SimulationEngine
from ssl_simulator.robot_models import SingleIntegrator
from ssl_simulator.robot_models.unicycle_2d import Unicycle2D

# ---------------------------------------------------------------------------
# Factory helpers — used both by fixtures and by benchmark.pedantic setup fns
# ---------------------------------------------------------------------------


def make_engine_si(n_robots: int = 10) -> SimulationEngine:
    """
    Single-integrator engine with a ConstantSignal controller, fully initialised.

    The engine is warmed up (one _step_test call) so that
    ``context.initialized = True`` and no dimension-check overhead appears in
    subsequent hot-path benchmarks.
    """
    positions = np.zeros((n_robots, 2))
    signal = np.ones((n_robots, 2)) * 0.1
    sim = SimulationEngine(time_step=0.01)
    sim.set_robot_model(SingleIntegrator, [positions])
    sim.add_controller("ctrl", ConstantSignal, signal=signal)
    sim.connect_controller_to_robot("ctrl", {"u": "u"})
    # Mirror the initialisation block inside SimulationEngine.run()
    sim._step_test()
    sim.context.initialized = True
    return sim


def make_engine_si_uninit(n_robots: int = 10) -> SimulationEngine:
    """
    Single-integrator engine *not yet initialised* — for use in run() benchmarks
    where SimulationEngine.run() must perform its own first-call initialisation.
    """
    positions = np.zeros((n_robots, 2))
    signal = np.ones((n_robots, 2)) * 0.1
    sim = SimulationEngine(time_step=0.01)
    sim.set_robot_model(SingleIntegrator, [positions])
    sim.add_controller("ctrl", ConstantSignal, signal=signal)
    sim.connect_controller_to_robot("ctrl", {"u": "u"})
    return sim


def make_engine_unicycle(n_robots: int = 10) -> SimulationEngine:
    """Unicycle-2D engine fully initialised (no controller)."""
    positions = np.zeros((n_robots, 2))
    speeds = np.ones(n_robots) * 0.5
    thetas = np.zeros(n_robots)
    sim = SimulationEngine(time_step=0.01)
    sim.set_robot_model(Unicycle2D, [positions, speeds, thetas])
    sim._step_test()
    sim.context.initialized = True
    return sim


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine_si_small():
    return make_engine_si(n_robots=10)


@pytest.fixture
def engine_si_large():
    return make_engine_si(n_robots=100)


@pytest.fixture
def engine_unicycle_small():
    return make_engine_unicycle(n_robots=10)


@pytest.fixture
def engine_unicycle_large():
    return make_engine_unicycle(n_robots=100)
