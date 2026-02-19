"""
Benchmarks for RobotModel.update_data and Controller.update_data.

Two sub-paths are exercised for each class:
  - plain values only  (``_has_callable_*`` flags are False → dict copy fast-path)
  - callable values    (lambda-resolution path, as used by ConstantSignal)

The _dirty flag is reset before each call via benchmark.pedantic so every
invocation exercises a real update rather than the early-return guard.
"""

import numpy as np
import pytest

from ssl_simulator.controllers.constant_signal import ConstantSignal
from ssl_simulator.core import SimulationContext
from ssl_simulator.robot_models import SingleIntegrator

# ---------------------------------------------------------------------------
# RobotModel.update_data
# ---------------------------------------------------------------------------


def _make_robot_plain(n: int):
    """SingleIntegrator — all state values are plain ndarrays (no callables)."""
    ctx = SimulationContext()
    ctx.set_robot_model(SingleIntegrator, [np.zeros((n, 2))])
    return ctx.robot_model


@pytest.mark.benchmark
def test_bench_robot_update_data_plain_small(benchmark):
    """RobotModel.update_data, plain values, 10 robots."""
    robot = _make_robot_plain(10)

    def setup():
        robot._dirty = True

    benchmark.pedantic(robot.update_data, setup=setup, rounds=200, warmup_rounds=5)


@pytest.mark.benchmark
def test_bench_robot_update_data_plain_large(benchmark):
    """RobotModel.update_data, plain values, 1 000 robots."""
    robot = _make_robot_plain(1000)

    def setup():
        robot._dirty = True

    benchmark.pedantic(robot.update_data, setup=setup, rounds=200, warmup_rounds=5)


# ---------------------------------------------------------------------------
# Controller.update_data — callable path (ConstantSignal uses lambdas)
# ---------------------------------------------------------------------------


def _make_controller_with_callables(n: int):
    ctx = SimulationContext()
    ctx.set_robot_model(SingleIntegrator, [np.zeros((n, 2))])
    ctx.add_controller("ctrl", ConstantSignal, signal=np.ones((n, 2)) * 0.1)
    ctrl = ctx.controllers["ctrl"]
    # Ensure ctrl_u is populated so the lambda returns an ndarray, not None
    ctrl.compute_control(0.0, 0.01)
    return ctrl


@pytest.mark.benchmark
def test_bench_controller_update_data_callables_small(benchmark):
    """Controller.update_data, callable control_vars path, 10 robots."""
    ctrl = _make_controller_with_callables(10)

    def setup():
        ctrl._dirty = True

    benchmark.pedantic(ctrl.update_data, setup=setup, rounds=200, warmup_rounds=5)


@pytest.mark.benchmark
def test_bench_controller_update_data_callables_large(benchmark):
    """Controller.update_data, callable control_vars path, 1 000 robots."""
    ctrl = _make_controller_with_callables(1000)

    def setup():
        ctrl._dirty = True

    benchmark.pedantic(ctrl.update_data, setup=setup, rounds=200, warmup_rounds=5)
