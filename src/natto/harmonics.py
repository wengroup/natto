"""Cartesian harmonics of a direction.

The harmonic operator takes the polyadic of a unit vector -- that vector tensored
with itself once per rank -- to the Cartesian harmonic of that direction, the
Cartesian counterpart of a spherical harmonic.

It is the natural projector rescaled. Projecting the polyadic onto its symmetric
traceless part is what makes a harmonic, and the constant (2n - 1)!! / n! is what
makes it the one that generates the Legendre polynomials:
contracting the harmonic of one direction with the polyadic of another gives the
Legendre polynomial of the angle between them. That is the condition
`tests/test_harmonics.py` asserts, against `scipy.special.eval_legendre`.

Building the operator once and contracting is what makes this worth having as an
operator at all: the harmonic of any direction is then a single `einsum` rather than
a sum over terms.

References:
    Eq. 46 of [Wen2026] for the operator, Eq. 47 for the Legendre condition, and
    Eq. 7 for the natural projector behind both.
"""

import math
from fractions import Fraction

import numpy as np

from natto.algebra import simplify_linear_combination
from natto.evaluate import evaluate_tensors
from natto.indices import letter_index
from natto.natural_projector import get_natural_projector
from natto.symbolic import LinearCombination
from natto.utils import double_factorial


def get_harmonic_operator(n: int, normalize: str = "unity") -> tuple[np.ndarray, str]:
    """Build the harmonic operator of one rank, evaluated.

    Args:
        n: Rank of the harmonic, which for a harmonic is also its weight. At least
            zero.
        normalize: `unity` applies the normalization constant, under which the
            n-fold contraction of the harmonic with a unit vector is the Legendre
            polynomial of the angle between the two, and is 1 when the two coincide.
            `none` leaves the natural projector unscaled.

    Returns:
        The evaluated operator, and the einsum rule that applies it, so that
        `V = numpy.einsum(rule, H, *[a] * n)` for a unit vector `a`.

    Raises:
        ValueError: If `n` is negative, or `normalize` is not recognized.

    References:
        Eq. 46 of [Wen2026], with the normalization of Eq. 47.
    """
    if n < 0:
        raise ValueError(f"rank (n) must be at least zero, got n={n}")

    H, upper, lower = get_harmonic_symbolic(n)
    H_numerical = evaluate_tensors(H, mode="extraction")

    if normalize == "unity":
        H_numerical = H_numerical * float(coeff_harmonic(n))
    elif normalize != "none":
        supported = ["none", "unity"]
        raise ValueError(
            f"Unknown normalization method: {normalize}. Supported are: {supported}."
        )

    # one operand per index, each of them the same vector
    operands = ",".join(f"...{letter}" for letter in upper)
    rule = f"{lower}{upper},{operands}->...{lower}" if n else "->..."

    return H_numerical, rule


def get_harmonic_symbolic(n: int) -> tuple[LinearCombination, str, str]:
    """Build the harmonic operator of one rank, symbolically.

    The terms are exact, with rational coefficients, and unnormalized; the
    normalization constant is `coeff_harmonic`.

    Args:
        n: Rank of the harmonic, which for a harmonic is also its weight. At least
            zero.

    Returns:
        The symbolic operator, the letters carrying the polyadic's indices, and the
        letters carrying the harmonic's.

    References:
        Eq. 46 of [Wen2026].
    """
    H = simplify_linear_combination(get_natural_projector(n))

    return H, letter_index(n, upper_case=True), letter_index(n)


def coeff_harmonic(n: int) -> Fraction:
    """Normalization constant of the Cartesian harmonic.

    Fixed by requiring that the n-fold contraction of the harmonic with the direction
    it was built from is 1.

    The value is a ratio of factorials, so it is returned exactly. Multiplying a float
    array by a Fraction would silently make it an object array, so the conversion
    happens at that boundary rather than here.

    Args:
        n: Rank of the harmonic, which for a harmonic is also its weight. At least
            zero.

    Returns:
        The normalization constant.

    References:
        Eq. 47 of [Wen2026].
    """
    return Fraction(double_factorial(2 * n - 1), math.factorial(n))
