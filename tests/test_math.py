import numpy as np
import pytest


def test_unit_vec():
    from ssl_simulator.math import basics

    arr = np.array([[3, 4, 0]])
    unit = basics.unit_vec(arr)
    assert np.allclose(np.linalg.norm(unit, axis=-1), 1)


def test_so3_hat_and_vee():
    from ssl_simulator.math import lie

    v = np.array([1.0, 2.0, 3.0])
    hat = lie.so3_hat(v)
    v2 = lie.so3_vee(hat)
    assert np.allclose(v, v2)
