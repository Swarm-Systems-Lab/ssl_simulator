"""
Benchmarks for math utility functions.

Covers the functions most likely to appear in hot loops or at initialisation:
  - unit_vec         : per-step normalisation of large vector batches
  - cov_matrix       : batched covariance, uses matmul
  - norm_2           : spectral norm via eigvals
  - check_and_parse_dimensions : init-time shape validation
"""

import numpy as np
import pytest

from ssl_simulator.math.algebra import cov_matrix, norm_2
from ssl_simulator.math.basics import check_and_parse_dimensions, unit_vec

# ---------------------------------------------------------------------------
# unit_vec — vector normalisation
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
def test_bench_unit_vec_small(benchmark):
    """unit_vec on a (10, 2) array — minimal fleet."""
    V = np.random.default_rng(0).standard_normal((10, 2))
    benchmark(unit_vec, V)


@pytest.mark.benchmark
def test_bench_unit_vec_large(benchmark):
    """unit_vec on a (1 000, 100, 2) array — T=1000 time-steps, N=100 agents."""
    V = np.random.default_rng(0).standard_normal((1000, 100, 2))
    benchmark(unit_vec, V)


# ---------------------------------------------------------------------------
# cov_matrix — batched covariance
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
def test_bench_cov_matrix_small(benchmark):
    """cov_matrix on (10, 50, 2) — 10 datasets, 50 samples, 2-D."""
    X = np.random.default_rng(0).standard_normal((10, 50, 2))
    benchmark(cov_matrix, X)


@pytest.mark.benchmark
def test_bench_cov_matrix_large(benchmark):
    """cov_matrix on (100, 200, 4) — 100 datasets, 200 samples, 4-D."""
    X = np.random.default_rng(0).standard_normal((100, 200, 4))
    benchmark(cov_matrix, X)


# ---------------------------------------------------------------------------
# norm_2 — spectral norm (eigvals of AᵀA)
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
def test_bench_norm_2_small(benchmark):
    """norm_2 on a (10, 10) matrix."""
    A = np.random.default_rng(0).standard_normal((10, 10))
    benchmark(norm_2, A)


@pytest.mark.benchmark
def test_bench_norm_2_large(benchmark):
    """norm_2 on a (100, 100) matrix — exercises eigvals cost at scale."""
    A = np.random.default_rng(0).standard_normal((100, 100))
    benchmark(norm_2, A)


# ---------------------------------------------------------------------------
# check_and_parse_dimensions — init-time shape validation
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
def test_bench_check_and_parse_dimensions(benchmark):
    """Shape-validation cost for a (100, 2) positions array."""
    arr = np.random.default_rng(0).standard_normal((100, 2))
    benchmark(check_and_parse_dimensions, arr, (None, 2))


@pytest.mark.benchmark
def test_bench_check_and_parse_dimensions_3d(benchmark):
    """Shape-validation cost for a 3-D (100, 50, 4) array with two free axes."""
    arr = np.random.default_rng(0).standard_normal((100, 50, 4))
    benchmark(check_and_parse_dimensions, arr, (None, None, 4))
