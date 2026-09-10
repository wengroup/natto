"""Tests for the Cartesian coupling operator.

`natto.coupling` builds the operator that couples two irreducible Cartesian
tensors into a third, the Cartesian counterpart of the Clebsch-Gordan coefficients. Its shape
follows from the natural projector, but its overall scale does not: the
construction leaves one factor free per weight triple, and the paper fixes it
with two conditions, one for each parity of `L = l1 + l2 + l3`.

Those conditions are what these tests assert. They are the only thing that pins
`coeff_C_even` and `coeff_C_odd`, so without them the coupling operator is checked only
up to a scalar.

The conditions are stated on the Cartesian harmonics, which the package does not
build, so `cartesian_harmonic` below constructs them from the natural projector
and the rank-dependent factor the paper gives. That reference is itself pinned
here, against the Legendre polynomials it must reproduce, so it cannot drift
into agreeing with a wrong operator.
"""

import math

import pytest
import torch

from natto.coupling import get_coupling_operator
from natto.symmetrize import remove_trace

#: Weights the conditions are checked at. The operator is closed form, so this
#: is cheap; the ceiling is only to keep the parametrization readable.
MAX_WEIGHT = 3

#: Separation at which the odd-parity limit is evaluated. The condition is a
#: limit as one direction approaches the other, so it is approached rather than
#: reached, and `LIMIT_RTOL` reflects that rather than float precision.
LIMIT_SEPARATION = 1e-4
LIMIT_RTOL = 1e-3

#: The package evaluates the coupling operator in float32, so a triple like
#: (3, 3, 6), whose operator carries many terms, agrees to about 1e-6 relative
#: rather than to float64 precision. A wrong normalization constant would be off
#: by a factor of order one, so this stays far from anything it needs to catch.
OPERATOR_RTOL = 1e-5


def double_factorial(n: int) -> float:
    """Double factorial `n!!`, with `(-1)!! = 1`."""
    result = 1.0
    while n > 1:
        result *= n
        n -= 2

    return result


def outer_power(vector: torch.Tensor, power: int) -> torch.Tensor:
    """The polyadic `vector^(otimes power)`."""
    result = torch.ones((), dtype=vector.dtype)
    for _ in range(power):
        result = torch.tensordot(result, vector, dims=0)

    return result


def cartesian_harmonic(direction: torch.Tensor, weight: int) -> torch.Tensor:
    """The Cartesian harmonic of `weight` built from a unit vector.

    The natural projector applied to the polyadic returns its traceless part,
    and the paper rescales that by `(2n - 1)!! / n!` so the result generates the
    Legendre polynomial. `remove_trace` performs the projection, the polyadic
    being symmetric already.

    Args:
        direction: A unit vector.
        weight: Weight of the harmonic, at least zero.

    Returns:
        The rank-`weight` harmonic.
    """
    if weight == 0:
        return torch.ones((), dtype=direction.dtype)

    scale = double_factorial(2 * weight - 1) / math.factorial(weight)

    return scale * remove_trace(outer_power(direction, weight))


def contract_all(operator: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Contract every index of `operator` with `vector`."""
    if operator.ndim == 0:
        return operator

    return torch.tensordot(
        operator, outer_power(vector, operator.ndim), dims=operator.ndim
    )


def random_unit_vector(seed: int) -> torch.Tensor:
    """A reproducible unit vector."""
    generator = torch.Generator().manual_seed(seed)
    vector = torch.randn(3, generator=generator, dtype=torch.float64)

    return vector / vector.norm()


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


@pytest.mark.parametrize("weight", range(MAX_WEIGHT + 1))
def test_cartesian_harmonic_generates_legendre(weight: int):
    """The reference harmonic reproduces the Legendre polynomial it must.

    This pins the construction the two conditions below are stated on, so that
    they cannot pass by a compensating error in the reference.
    """
    a = random_unit_vector(0)
    b = random_unit_vector(1)

    value = contract_all(cartesian_harmonic(a, weight), b)
    expected = torch.special.legendre_polynomial_p(a @ b, weight)

    torch.testing.assert_close(value, expected)


@pytest.mark.parametrize("l1,l2,l3", weight_triples(parity=0))
def test_even_parity_normalization(l1: int, l2: int, l3: int):
    """Two harmonics of one direction couple to the harmonic of that direction.

    This is the condition that fixes `coeff_C_even`.
    """
    a = random_unit_vector(0)
    operator, rule = get_coupling_operator(l1, l2, l3, normalize="unity")

    coupled = torch.einsum(
        rule,
        operator.to(torch.float64),
        cartesian_harmonic(a, l1),
        cartesian_harmonic(a, l2),
    )

    torch.testing.assert_close(
        coupled, cartesian_harmonic(a, l3), rtol=OPERATOR_RTOL, atol=1e-7
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
    perpendicular = torch.linalg.cross(a, random_unit_vector(1))
    b = a + LIMIT_SEPARATION * perpendicular / perpendicular.norm()
    b = b / b.norm()

    operator, rule = get_coupling_operator(l1, l2, l3, normalize="unity")
    coupled = torch.einsum(
        rule,
        operator.to(torch.float64),
        cartesian_harmonic(a, l1),
        cartesian_harmonic(b, l2),
    )

    # Contract all but one index, leaving a vector, and compare its length with
    # the separation of the two directions.
    if l3 == 1:
        residual = coupled
    else:
        residual = torch.tensordot(coupled, outer_power(a, l3 - 1), dims=l3 - 1)

    rate = residual.norm() / torch.linalg.cross(a, b).norm()

    torch.testing.assert_close(
        rate, torch.ones((), dtype=rate.dtype), rtol=LIMIT_RTOL, atol=0
    )
