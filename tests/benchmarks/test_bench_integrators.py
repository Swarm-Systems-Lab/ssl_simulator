"""
Benchmarks for EulerIntegrator.integrate.

The integrator is exercised in isolation (no controller dispatch) to measure
the inner state-update loop cost independently from dynamics computation.
"""

import numpy as np
import pytest

from ssl_simulator.core import SimulationContext
from ssl_simulator.core.integrators import EulerIntegrator
from ssl_simulator.robot_models import SingleIntegrator

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _make_initialized_context(n: int) -> SimulationContext:
    """Context with pre-computed state_dot, marked as initialised."""
    ctx = SimulationContext()
    ctx.set_robot_model(SingleIntegrator, [np.zeros((n, 2))])
    ctx.robot_model.control_inputs["u"] = np.ones((n, 2)) * 0.1
    # Pre-populate state_dot so integrate() has a valid target
    ctx.robot_model.dynamics(0.0)
    ctx.initialized = True
    return ctx


# ---------------------------------------------------------------------------
# EulerIntegrator
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
def test_bench_euler_integrate_small(benchmark):
    """EulerIntegrator.integrate, 10 robots — baseline overhead."""
    ctx = _make_initialized_context(10)
    integrator = EulerIntegrator()
    benchmark(integrator.integrate, ctx, 0.01)


@pytest.mark.benchmark
def test_bench_euler_integrate_medium(benchmark):
    """EulerIntegrator.integrate, 100 robots."""
    ctx = _make_initialized_context(100)
    integrator = EulerIntegrator()
    benchmark(integrator.integrate, ctx, 0.01)


@pytest.mark.benchmark
def test_bench_euler_integrate_large(benchmark):
    """EulerIntegrator.integrate, 1 000 robots — exercises state-key loop at scale."""
    ctx = _make_initialized_context(1000)
    integrator = EulerIntegrator()
    benchmark(integrator.integrate, ctx, 0.01)
