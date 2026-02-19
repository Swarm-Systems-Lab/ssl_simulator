"""
Benchmarks for SimulationEngine._step_test (hot dynamics loop) and run().

_step_test  = compute_controls + compute_robot_dynamics + integrate
              (no logging overhead — isolates the numerical core)

run()       = full pipeline including DataLogger initialisation and logging;
              measured with benchmark.pedantic so a fresh engine is created
              for every round.
"""

import pytest

from .conftest import make_engine_si_uninit, make_engine_unicycle

# ---------------------------------------------------------------------------
# _step_test — core dynamics loop (no I/O, no logging)
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
def test_bench_step_si_small(benchmark, engine_si_small):
    """SingleIntegrator, 10 robots — hot-path step cost."""
    benchmark(engine_si_small._step_test)


@pytest.mark.benchmark
def test_bench_step_si_large(benchmark, engine_si_large):
    """SingleIntegrator, 100 robots — hot-path step cost."""
    benchmark(engine_si_large._step_test)


@pytest.mark.benchmark
def test_bench_step_unicycle_small(benchmark, engine_unicycle_small):
    """Unicycle2D, 10 robots — hot-path step cost."""
    benchmark(engine_unicycle_small._step_test)


@pytest.mark.benchmark
def test_bench_step_unicycle_large(benchmark, engine_unicycle_large):
    """Unicycle2D, 100 robots — hot-path step cost."""
    benchmark(engine_unicycle_large._step_test)


# ---------------------------------------------------------------------------
# run() — end-to-end pipeline including logger
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
def test_bench_run_si_small_50steps(benchmark):
    """Full run() over 50 steps, 10 robots — includes DataLogger overhead."""

    def setup():
        return (make_engine_si_uninit(n_robots=10),), {}

    def run_sim(sim):
        sim.run(duration=0.5, eta=False)  # 50 steps at dt=0.01

    benchmark.pedantic(run_sim, setup=setup, rounds=5, warmup_rounds=1)


@pytest.mark.benchmark
def test_bench_run_si_large_50steps(benchmark):
    """Full run() over 50 steps, 100 robots — includes DataLogger overhead."""

    def setup():
        return (make_engine_si_uninit(n_robots=100),), {}

    def run_sim(sim):
        sim.run(duration=0.5, eta=False)

    benchmark.pedantic(run_sim, setup=setup, rounds=5, warmup_rounds=1)


@pytest.mark.benchmark
def test_bench_run_unicycle_small_50steps(benchmark):
    """Full run() over 50 steps, 10 unicycle robots."""

    def setup():
        # Unicycle needs an uninitialised engine created fresh
        import numpy as np

        from ssl_simulator.core import SimulationEngine
        from ssl_simulator.robot_models.unicycle_2d import Unicycle2D

        n = 10
        sim = SimulationEngine(time_step=0.01)
        sim.set_robot_model(Unicycle2D, [np.zeros((n, 2)), np.ones(n) * 0.5, np.zeros(n)])
        return (sim,), {}

    def run_sim(sim):
        sim.run(duration=0.5, eta=False)

    benchmark.pedantic(run_sim, setup=setup, rounds=5, warmup_rounds=1)
