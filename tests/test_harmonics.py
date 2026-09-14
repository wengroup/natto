"""Tests for the Cartesian harmonics.

The harmonic operator is the natural projector under another name and another
scale, so what needs asserting is not its shape but the two things the scale is
chosen for: that contracting the harmonic of one direction with another
direction gives the Legendre polynomial of the angle between them, and that the
result is symmetric and traceless, which is what makes it a natural tensor.
Without normalization it must also be what the general reduction extracts from the
polyadic, which checks the closed form against the reduction itself.

`scipy.special.eval_legendre` is the independent reference; nothing in
the package is used to compute the expected values.
"""

import functools
import math

import numpy as np
import pytest
import scipy.special

from natto.harmonics import coeff_harmonic, get_harmonic_operator
from natto.reduction import get_reduction
from natto.utils import is_symmetric_traceless

#: Weights checked. The operator is a closed form, so this is cheap; the ceiling
#: keeps the rank-2n arrays small.
MAX_WEIGHT = 5


def unit_vector(seed: int) -> np.ndarray:
    """A reproducible unit vector."""
    vector = np.random.default_rng(seed).standard_normal(3)

    return vector / np.linalg.norm(vector)


def harmonic(direction: np.ndarray, weight: int, normalize: str = "legendre"):
    """The Cartesian harmonic of `direction`, through the operator."""
    operator, rule = get_harmonic_operator(weight, normalize)

    return np.einsum(rule, operator, *[direction] * weight)


def outer_power(vector: np.ndarray, power: int) -> np.ndarray:
    """The polyadic `vector^(otimes power)`."""
    result = np.ones((), dtype=vector.dtype)
    for _ in range(power):
        result = np.tensordot(result, vector, axes=0)

    return result


@functools.lru_cache(maxsize=None)
def reduction(rank: int) -> dict:
    """The general reduction of a generic tensor of this rank, built once."""
    return get_reduction(rank)


@pytest.mark.parametrize("weight", range(MAX_WEIGHT + 1))
def test_generates_the_legendre_polynomial(weight: int):
    """The condition that fixes the normalization, Eq. 47."""
    a, b = unit_vector(0), unit_vector(1)

    value = np.tensordot(harmonic(a, weight), outer_power(b, weight), axes=weight)
    expected = scipy.special.eval_legendre(weight, a @ b)

    np.testing.assert_allclose(value, expected, atol=1e-12)


@pytest.mark.parametrize("weight", range(MAX_WEIGHT + 1))
def test_contracting_with_its_own_direction_gives_one(weight: int):
    """The same condition at zero angle, where the Legendre polynomial is 1."""
    a = unit_vector(0)

    value = np.tensordot(harmonic(a, weight), outer_power(a, weight), axes=weight)

    np.testing.assert_allclose(value, np.ones((), dtype=value.dtype), atol=1e-12)


@pytest.mark.parametrize("weight", range(2, MAX_WEIGHT + 1))
def test_the_harmonic_is_a_natural_tensor(weight: int):
    """Symmetric and traceless, which is what the natural projector is for."""
    assert is_symmetric_traceless(harmonic(unit_vector(0), weight), atol=1e-10)


@pytest.mark.parametrize("weight", range(MAX_WEIGHT + 1))
def test_without_normalization_the_scale_is_the_only_difference(weight: int):
    """`normalize="none"` differs from `legendre` by `coeff_harmonic` and nothing else."""
    a = unit_vector(0)

    # `coeff_harmonic` is exact, and a float array times a Fraction would be an
    # object array, so the conversion happens here as it does in the package.
    scaled = harmonic(a, weight, normalize="none") * float(coeff_harmonic(weight))

    np.testing.assert_allclose(scaled, harmonic(a, weight), atol=1e-12)


@pytest.mark.parametrize("weight", range(MAX_WEIGHT + 1))
def test_coefficient_matches_the_paper(weight: int):
    """`(2n - 1)!! / n!`, written out."""
    expected = math.prod(range(2 * weight - 1, 0, -2)) / math.factorial(weight)

    assert coeff_harmonic(weight) == pytest.approx(expected)


@pytest.mark.parametrize("weight", [1, 2, 3, pytest.param(4, marks=pytest.mark.slow)])
def test_matches_reduction(weight: int):
    """Without normalization, the harmonic is what the general reduction gives.

    A polyadic of rank n has a single weight-n channel, whose mapping tensor and dual
    are both the natural projector, so each of them applied to the polyadic must equal
    the unnormalized harmonic. The rank-four reduction takes seconds, hence the mark.
    """
    a = unit_vector(0)
    polyadic = outer_power(a, weight)
    (embedding,) = reduction(weight)[weight]["embedding"]
    (extraction,) = reduction(weight)[weight]["extraction"]

    expected = harmonic(a, weight, normalize="none")
    via_embedding = np.tensordot(
        np.asarray(embedding["numerical"], dtype=np.float64),
        polyadic,
        axes=(list(range(weight)), list(range(weight))),
    )
    via_extraction = np.einsum(
        extraction["rule"],
        np.asarray(extraction["numerical"], dtype=np.float64),
        polyadic,
    )

    np.testing.assert_allclose(via_embedding, expected, atol=1e-12)
    np.testing.assert_allclose(via_extraction, expected, atol=1e-12)


def test_a_batch_of_directions_works_like_one():
    """Every rule carries a leading ellipsis, so batches need no special case."""
    directions = np.stack([unit_vector(seed) for seed in range(4)])

    batched = harmonic(directions, 3)

    for index, direction in enumerate(directions):
        np.testing.assert_allclose(batched[index], harmonic(direction, 3), atol=1e-12)


def test_a_negative_weight_is_rejected():
    """There is no harmonic of negative weight."""
    with pytest.raises(ValueError):
        get_harmonic_operator(-1)


def test_an_unknown_normalization_is_rejected():
    """A typo in `normalize` must not silently return the unscaled operator."""
    with pytest.raises(ValueError):
        get_harmonic_operator(2, normalize="unit")
