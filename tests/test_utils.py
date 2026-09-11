from math import factorial as factorial_math

import numpy as np
import pytest

from natto.utils import (
    double_factorial,
    factorial,
    get_trace,
    is_traceless,
)


def test_factorial():
    for i in range(10):
        assert factorial(i) == factorial_math(i)


def test_double_factorial():
    assert double_factorial(0) == 1
    assert double_factorial(1) == 1
    assert double_factorial(2) == 2
    assert double_factorial(3) == 3
    assert double_factorial(4) == 8
    assert double_factorial(5) == 15
    assert double_factorial(6) == 48
    assert double_factorial(7) == 105
    assert double_factorial(8) == 384

    assert double_factorial(7, lower_bound=3) == 105
    assert double_factorial(8, lower_bound=4) == 192

    for i in range(5, 10):
        assert double_factorial(i) // double_factorial(i - 4) == double_factorial(
            i, lower_bound=i - 4 + 2
        )


def test_get_trace():
    T2 = np.arange(9).reshape(3, 3).astype(np.float64)
    T3 = np.arange(27).reshape(3, 3, 3).astype(np.float64)

    trace = get_trace(T2, i=0, j=1)
    assert np.allclose(trace, np.array([12.0]))

    trace = get_trace(T3, i=0, j=1)
    assert np.allclose(trace, np.array([36.0, 39.0, 42.0]))

    trace = get_trace(T3, i=1, j=2)
    assert np.allclose(trace, np.array([12.0, 39.0, 66.0]))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_traceless_check_accepts_any_float_dtype(dtype):
    """`np.allclose` raises on a dtype mismatch, so the zero must follow the input.

    A zero built in the default dtype made `is_traceless` raise for every tensor
    in another one, rather than report on its trace.
    """
    traceless = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -2.0]], dtype=dtype
    )

    assert is_traceless(traceless)
    assert not is_traceless(np.eye(3, dtype=dtype))
