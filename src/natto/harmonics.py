"""Cartesian harmonics of a direction.

The harmonic operator takes the polyadic of a unit vector -- that vector tensored
with itself once per rank -- to the Cartesian harmonic of that direction, the
Cartesian counterpart of a spherical harmonic.

It is the natural projector in the form it takes on a fully symmetric tensor,
rescaled. On a polyadic the average over the Cartesian indices is redundant, so only
the average over the harmonic's own indices remains, and the constant (2n - 1)!! / n!
is what makes the result the one that generates the Legendre polynomials:
contracting the harmonic of one direction with the polyadic of another gives the
Legendre polynomial of the angle between them. That is the condition
`tests/test_harmonics.py` asserts, against `scipy.special.eval_legendre`.

Building the operator once and contracting is what makes this worth having as an
operator at all: the harmonic of any direction is then a single `einsum` rather than
a sum over terms.
"""

import itertools
import math
from fractions import Fraction
from typing import Literal

from natto.indices import get_slot_partitions
from natto.natural_projector import get_projector_coefficients
from natto.symbolic import IndexGroup, Operator, Signature, Term
from natto.utils import double_factorial

#: How the harmonic operator is scaled. `legendre` applies `coeff_harmonic`, the
#: constant of Eq. 46 that makes the harmonic generate the Legendre polynomials;
#: `none` leaves the operator of Eq. 45 unscaled.
Normalization = Literal["legendre", "none"]


def get_harmonic_operator(n: int, normalize: Normalization = "legendre") -> Operator:
    """Build the harmonic operator of one rank.

    This is the natural projector as it acts on a fully symmetric tensor. The
    Cartesian indices are fixed: the first n - 2t meet a harmonic index each and the
    last 2t are paired in order. Only the harmonic's own indices are averaged over, so
    the term of each t is c_t times the average of its distinct products. Contracted
    with a polyadic it gives what the full projector gives, with far fewer terms; with
    a tensor that is not fully symmetric it does not.

    The `ict` group carries the harmonic's indices. Each Cartesian index is a group of
    its own, `vector_1` to `vector_n`, and an input, so that `act(H, *[a] * n)` gives
    the harmonic of a unit vector `a`, or of a batch of them.

    Args:
        n: Rank of the harmonic, which for a harmonic is also its weight. At least
            zero.
        normalize: `legendre` applies the normalization constant `coeff_harmonic`,
            under which the n-fold contraction of the harmonic with a unit vector is
            the Legendre polynomial of the angle between the two, and is 1 when the
            two coincide. `none` leaves the operator unscaled.

    Returns:
        The exact operator.

    Raises:
        ValueError: If `n` is negative, or `normalize` is not recognized.

    References:
        Eq. 45 of [Wen2026], with the coefficients c_t of Eq. 8, and the
        normalization of Eq. 46 fixed by Eq. 47.
    """
    if normalize not in ("legendre", "none"):
        supported = ["legendre", "none"]
        raise ValueError(
            f"Unknown normalization method: {normalize}. Supported are: {supported}."
        )
    if n < 0:
        raise ValueError(f"rank (n) must be at least zero, got n={n}")

    vectors = [
        IndexGroup(f"vector_{k + 1}", 1, upper=True, input=True) for k in range(n)
    ]
    signature = Signature((IndexGroup("ict", n, upper=False), *vectors))

    terms = []
    for t, c in enumerate(get_projector_coefficients(n)):
        free = n - 2 * t
        sigma_pairs = [(n + free + 2 * k, n + free + 2 * k + 1) for k in range(t)]

        # Every distinct product: which harmonic indices are paired, and which of the
        # rest meets each of the first `free` Cartesian indices.
        products = []
        for (unpaired,), weight_pairs in get_slot_partitions([free], t):
            for order in itertools.permutations(unpaired):
                crossing = list(zip(order, range(n, n + free)))
                products.append(crossing + list(weight_pairs) + sigma_pairs)

        for deltas in products:
            sign, term = Term.from_blocks(deltas)
            terms.append((sign * c / len(products), term))

    H = Operator(signature, terms)

    if normalize == "legendre":
        H = H * coeff_harmonic(n)

    return H


def coeff_harmonic(n: int) -> Fraction:
    """Normalization constant of the Cartesian harmonic.

    Fixed by requiring that the n-fold contraction of the harmonic with the direction
    it was built from is 1.

    The value is a ratio of factorials, so it is returned exactly, and scales the
    exact operator without leaving the rationals.

    Args:
        n: Rank of the harmonic, which for a harmonic is also its weight. At least
            zero.

    Returns:
        The normalization constant.

    References:
        Eq. 46 of [Wen2026].
    """
    return Fraction(double_factorial(2 * n - 1), math.factorial(n))
