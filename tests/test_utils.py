from math import factorial as factorial_math

import pytest
import torch

from natto.utils import (
    double_factorial,
    double_index,
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


def test_multi_double_index():
    assert double_index(2) == ["ab", "cd"]
    assert double_index(3, start=1) == ["bc", "de", "fg"]


def test_get_trace():
    T2 = torch.arange(9).reshape(3, 3).to(torch.float)
    T3 = torch.arange(27).reshape(3, 3, 3).to(torch.float)

    trace = get_trace(T2, i=0, j=1)
    assert torch.allclose(trace, torch.tensor([12.0]))

    trace = get_trace(T3, i=0, j=1)
    assert torch.allclose(trace, torch.tensor([36.0, 39.0, 42.0]))

    trace = get_trace(T3, i=1, j=2)
    assert torch.allclose(trace, torch.tensor([12.0, 39.0, 66.0]))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_traceless_check_accepts_any_float_dtype(dtype):
    """`torch.allclose` raises on a dtype mismatch, so the zero must follow the input.

    A zero built in the default dtype made `is_traceless` raise for every tensor
    in another one, rather than report on its trace.
    """
    traceless = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -2.0]]).to(
        dtype
    )

    assert is_traceless(traceless)
    assert not is_traceless(torch.eye(3, dtype=dtype))
