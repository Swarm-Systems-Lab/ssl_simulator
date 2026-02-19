"""
Benchmarks for robot model dynamics called in isolation.

By bypassing the engine and context dispatch overhead, regressions in the
dynamics implementation itself (numpy ops, broadcasting, clipping) surface
clearly without interference from the simulation loop.
"""

import numpy as np
import pytest

from ssl_simulator.core import SimulationContext
from ssl_simulator.robot_models import SingleIntegrator
from ssl_simulator.robot_models.unicycle_2d import Unicycle2D

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_si_robot(n: int):
    ctx = SimulationContext()
    ctx.set_robot_model(SingleIntegrator, [np.zeros((n, 2))])
    ctx.robot_model.control_inputs["u"] = np.ones((n, 2)) * 0.1
    return ctx.robot_model


def _make_unicycle_robot(n: int):
    ctx = SimulationContext()
    ctx.set_robot_model(Unicycle2D, [np.zeros((n, 2)), np.ones(n) * 0.5, np.zeros(n)])
    ctx.robot_model.control_inputs["omega"] = np.ones(n) * 0.05
    return ctx.robot_model


# ---------------------------------------------------------------------------
# SingleIntegrator
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
def test_bench_si_dynamics_small(benchmark):
    """SingleIntegrator.dynamics, 10 robots."""
    robot = _make_si_robot(10)
    benchmark(robot.dynamics, 0.0)


@pytest.mark.benchmark
def test_bench_si_dynamics_large(benchmark):
    """SingleIntegrator.dynamics, 1 000 robots — exercises np.broadcast_to scaling."""
    robot = _make_si_robot(1000)
    benchmark(robot.dynamics, 0.0)


# ---------------------------------------------------------------------------
# Unicycle2D
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
def test_bench_unicycle_dynamics_small(benchmark):
    """Unicycle2D.dynamics, 10 robots."""
    robot = _make_unicycle_robot(10)
    benchmark(robot.dynamics, 0.0)


@pytest.mark.benchmark
def test_bench_unicycle_dynamics_large(benchmark):
    """Unicycle2D.dynamics, 1 000 robots — exercises per-agent trig ops scaling."""
    robot = _make_unicycle_robot(1000)
    benchmark(robot.dynamics, 0.0)


@pytest.mark.benchmark
def test_bench_unicycle_dynamics_with_clipping(benchmark):
    """Unicycle2D.dynamics with omega_lims — exercises the np.clip branch."""
    ctx = SimulationContext()
    n = 100
    ctx.set_robot_model(
        Unicycle2D,
        [np.zeros((n, 2)), np.ones(n) * 0.5, np.zeros(n)],
        omega_lims=(-1.0, 1.0),
    )
    ctx.robot_model.control_inputs["omega"] = np.ones(n) * 2.0  # will be clipped
    benchmark(ctx.robot_model.dynamics, 0.0)
