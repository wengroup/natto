"""Tests for the Cartesian coupling operator.

`natto.coupling` builds the operator that couples two irreducible Cartesian
tensors into a third, the Cartesian counterpart of the Clebsch-Gordan coefficients. Its shape
follows from the natural projector, but its overall scale does not: the
construction leaves one factor free per weight triple, and the paper fixes it
with two conditions, one for each parity of `L = l1 + l2 + l3`.

Those conditions are what these tests assert. They are the only thing that pins
`coeff_C_even` and `coeff_C_odd`, so without them the coupling operator is checked only
up to a scalar.

The unnormalized operator is checked separately, against the general reduction:
applied to two ICTs, it must equal what the mapping tensors of the reduction of their
product give. That pins the coefficients of Eq. 49, which the two conditions alone
would leave free up to one factor per weight triple.

The conditions are stated on the Cartesian harmonics, which the package does not
build, so `cartesian_harmonic` below constructs them from the natural projector
and the rank-dependent factor the paper gives. That reference is itself pinned
here, against the Legendre polynomials it must reproduce, so it cannot drift
into agreeing with a wrong operator.
"""

import functools
import math

import numpy as np
import pytest
import scipy.special

from natto.coupling import get_coupling_operator
from natto.natural_projector import (
    get_random_natural_tensor,
    get_symmetric_traceless_part,
)
from natto.reduction import get_reduction

#: Weights the conditions are checked at. The operator is closed form, so this
#: is cheap; the ceiling is only to keep the parametrization readable.
MAX_WEIGHT = 3

#: Separation at which the odd-parity limit is evaluated. The condition is a
#: limit as one direction approaches the other, so it is approached rather than
#: reached, and `LIMIT_RTOL` reflects that rather than float precision.
LIMIT_SEPARATION = 1e-4
LIMIT_RTOL = 1e-3

#: Everything here is float64, and the worst triple -- (3, 3, 6), whose operator
#: carries the most terms -- agrees to about 1e-13 relative. This leaves two
#: orders of margin on that. A wrong normalization constant would be off by a
#: factor of order one, so it stays far from anything the check needs to catch.
OPERATOR_RTOL = 1e-11

#: Largest product rank, l1 + l2, compared against the general reduction in the default
#: run. Reducing a rank-four tensor takes seconds, so those triples are marked slow.
FAST_PRODUCT_RANK = 3


def double_factorial(n: int) -> float:
    """Double factorial `n!!`, with `(-1)!! = 1`."""
    result = 1.0
    while n > 1:
        result *= n
        n -= 2

    return result


def outer_power(vector: np.ndarray, power: int) -> np.ndarray:
    """The polyadic `vector^(otimes power)`."""
    result = np.ones((), dtype=vector.dtype)
    for _ in range(power):
        result = np.tensordot(result, vector, axes=0)

    return result


def cartesian_harmonic(direction: np.ndarray, weight: int) -> np.ndarray:
    """The Cartesian harmonic of `weight` built from a unit vector.

    The natural projector applied to the polyadic returns its traceless part,
    and the paper rescales that by `(2n - 1)!! / n!` so the result generates the
    Legendre polynomial. The traceless part is taken numerically, which stays cheap
    at weight six where evaluating the projector does not.

    Args:
        direction: A unit vector.
        weight: Weight of the harmonic, at least zero.

    Returns:
        The rank-`weight` harmonic.
    """
    if weight == 0:
        return np.ones((), dtype=direction.dtype)

    scale = double_factorial(2 * weight - 1) / math.factorial(weight)

    traceless = get_symmetric_traceless_part(outer_power(direction, weight))

    return scale * traceless


def contract_all(operator: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Contract every index of `operator` with `vector`."""
    if operator.ndim == 0:
        return operator

    return np.tensordot(
        operator, outer_power(vector, operator.ndim), axes=operator.ndim
    )


def random_unit_vector(seed: int) -> np.ndarray:
    """A reproducible unit vector."""
    vector = np.random.default_rng(seed).standard_normal(3)

    return vector / np.linalg.norm(vector)


def weight_triples(parity: int) -> list[tuple[int, int, int]]:
    """Every admissible weight triple of the given parity of `L`.

    Args:
        parity: 0 for even `L`, 1 for odd.

    Returns:
        Triples satisfying the triangle rule. Odd `L` additionally needs a
        nonzero output weight, the Levi-Civita symbol having to leave one index
        free.
    """
    triples = []
    for l1 in range(MAX_WEIGHT + 1):
        for l2 in range(MAX_WEIGHT + 1):
            for l3 in range(abs(l1 - l2), l1 + l2 + 1):
                if (l1 + l2 + l3) % 2 != parity:
                    continue
                if parity == 1 and l3 == 0:
                    continue
                triples.append((l1, l2, l3))

    return triples


@functools.lru_cache(maxsize=None)
def reduction(rank: int) -> dict:
    """The general reduction of a generic tensor of this rank, built once."""
    return get_reduction(rank)


def product_triples() -> list:
    """Weight triples with nonzero input weights and a product of rank at most four.

    Triples whose product has rank above `FAST_PRODUCT_RANK` are marked slow.
    """
    params = []
    for l1 in range(1, 4):
        for l2 in range(1, 5 - l1):
            for l3 in range(abs(l1 - l2), l1 + l2 + 1):
                marks = pytest.mark.slow if l1 + l2 > FAST_PRODUCT_RANK else ()
                params.append(pytest.param(l1, l2, l3, marks=marks))

    return params


def couple_unnormalized(l1: int, l2: int, l3: int, X: np.ndarray, Y: np.ndarray):
    """The coupling of `X` and `Y` to weight `l3`, without its normalization."""
    operator, rule = get_coupling_operator(l1, l2, l3, normalize="none")

    return np.einsum(rule, operator.astype(np.float64), X, Y)


@pytest.mark.parametrize("weight", range(MAX_WEIGHT + 1))
def test_cartesian_harmonic_generates_legendre(weight: int):
    """The reference harmonic reproduces the Legendre polynomial it must.

    This pins the construction the two conditions below are stated on, so that
    they cannot pass by a compensating error in the reference.
    """
    a = random_unit_vector(0)
    b = random_unit_vector(1)

    value = contract_all(cartesian_harmonic(a, weight), b)
    expected = scipy.special.eval_legendre(weight, a @ b)

    np.testing.assert_allclose(value, expected, atol=1e-12)


@pytest.mark.parametrize("l1,l2,l3", weight_triples(parity=0))
def test_even_parity_normalization(l1: int, l2: int, l3: int):
    """Two harmonics of one direction couple to the harmonic of that direction.

    This is the condition that fixes `coeff_C_even`.
    """
    a = random_unit_vector(0)
    operator, rule = get_coupling_operator(l1, l2, l3, normalize="legendre")

    coupled = np.einsum(
        rule,
        operator.astype(np.float64),
        cartesian_harmonic(a, l1),
        cartesian_harmonic(a, l2),
    )

    np.testing.assert_allclose(
        coupled, cartesian_harmonic(a, l3), rtol=OPERATOR_RTOL, atol=0
    )


@pytest.mark.parametrize("l1,l2,l3", weight_triples(parity=1))
def test_odd_parity_normalization(l1: int, l2: int, l3: int):
    """The coupling vanishes as the directions merge, and does so at unit rate.

    For odd `L` the operator carries a Levi-Civita symbol, so contracting the
    output with a single direction vanishes identically and cannot fix the
    scale. The paper's second condition takes the rate of that vanishing
    instead. This is what fixes `coeff_C_odd`.
    """
    a = random_unit_vector(0)
    perpendicular = np.cross(a, random_unit_vector(1))
    b = a + LIMIT_SEPARATION * perpendicular / np.linalg.norm(perpendicular)
    b = b / np.linalg.norm(b)

    operator, rule = get_coupling_operator(l1, l2, l3, normalize="legendre")
    coupled = np.einsum(
        rule,
        operator.astype(np.float64),
        cartesian_harmonic(a, l1),
        cartesian_harmonic(b, l2),
    )

    # Contract all but one index, leaving a vector, and compare its length with
    # the separation of the two directions.
    if l3 == 1:
        residual = coupled
    else:
        residual = np.tensordot(coupled, outer_power(a, l3 - 1), axes=l3 - 1)

    rate = np.linalg.norm(residual) / np.linalg.norm(np.cross(a, b))

    np.testing.assert_allclose(
        rate, np.ones((), dtype=rate.dtype), rtol=LIMIT_RTOL, atol=0
    )


@pytest.mark.parametrize("l1,l2,l3", product_triples())
def test_matches_mapping_tensors(l1: int, l2: int, l3: int):
    """Without normalization, the coupling is the general reduction's mapping tensor.

    Applied to the product of two ICTs, every weight-l3 mapping tensor of the
    rank-(l1 + l2) reduction gives either the coupling's output or zero. A candidate
    whose rank lowering joins indices of X only to indices of Y is the coupling's own
    construction, and any other takes a trace of X or of Y. At least one kept
    candidate must be of the first kind, since the kept ones span them all.
    """
    X = get_random_natural_tensor(l1, seed=3)
    Y = get_random_natural_tensor(l2, seed=7)
    coupled = couple_unnormalized(l1, l2, l3, X, Y)

    rank = l1 + l2
    product = np.multiply.outer(X, Y)
    outcomes = []
    for entry in reduction(rank)[l3]["embedding"]:
        mapped = np.tensordot(
            np.asarray(entry["numerical"], dtype=np.float64),
            product,
            axes=(list(range(rank)), list(range(rank))),
        )
        if np.allclose(mapped, coupled, rtol=OPERATOR_RTOL, atol=1e-12):
            outcomes.append("coupling")
        elif np.allclose(mapped, 0.0, atol=1e-12):
            outcomes.append("zero")
        else:
            outcomes.append("other")

    assert "other" not in outcomes, outcomes
    assert "coupling" in outcomes, outcomes


@pytest.mark.parametrize(
    "l1,l2",
    [
        (1, 1),
        (2, 1),
        (1, 2),
        pytest.param(2, 2, marks=pytest.mark.slow),
        pytest.param(3, 1, marks=pytest.mark.slow),
    ],
)
def test_top_weight_matches_extraction(l1: int, l2: int):
    """At l3 = l1 + l2 the single mapping tensor is its own dual.

    The Gram entry is one there, so the extraction operator, which below the top
    weight mixes the channels, is the mapping tensor itself and must also give the
    coupling's output.
    """
    l3 = l1 + l2
    X = get_random_natural_tensor(l1, seed=3)
    Y = get_random_natural_tensor(l2, seed=7)

    (extraction,) = reduction(l3)[l3]["extraction"]
    extracted = np.einsum(
        extraction["rule"],
        np.asarray(extraction["numerical"], dtype=np.float64),
        np.multiply.outer(X, Y),
    )

    np.testing.assert_allclose(
        extracted, couple_unnormalized(l1, l2, l3, X, Y), rtol=OPERATOR_RTOL, atol=1e-12
    )
