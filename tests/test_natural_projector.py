"""The natural projector.

Its defining properties are checked directly: applying it twice is applying it once,
exactly, and its trace counts the 2 * ell + 1 components of an ICT.
"""

import pytest

from natto.algebra import contract
from natto.natural_projector import get_natural_projector


@pytest.mark.parametrize("ell", range(5))
def test_idempotent(ell: int):
    """The projector composed with itself is the projector, term for term."""
    projector = get_natural_projector(ell)
    middle = [("middle", k) for k in range(ell)]
    factors = [
        (projector, list(range(ell)) + middle),
        (projector, middle + [ell + k for k in range(ell)]),
    ]

    assert contract(factors, projector.signature) == projector


@pytest.mark.parametrize("ell", range(5))
def test_trace(ell: int):
    """The trace of the projector is the number of components of an ICT."""
    array = get_natural_projector(ell).evaluate()

    assert array.reshape(3**ell, 3**ell).trace() == pytest.approx(2 * ell + 1)
