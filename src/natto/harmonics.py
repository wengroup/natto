r"""Cartesian harmonics of a direction.

The harmonic operator $\mathbf{H}_{(n\mid n)}$ of Eq. (40) takes the polyadic
$\hat{\mathbf{a}}^{\otimes n}$ of a unit vector to the Cartesian harmonic
$\mathbf{V}_n$ of Eq. (41),

$$\mathbf{V}_n(\hat{\mathbf{a}}) = \mathbf{H}_{(n\mid n)} \odot^n
\hat{\mathbf{a}}^{\otimes n},$$

the Cartesian counterpart of a spherical harmonic. It is the natural projector
of Eq. (12) rescaled: projecting the polyadic onto the symmetric traceless part
is what makes a harmonic, and the constant $(2n-1)!!/n!$ is what makes it the
one that generates the Legendre polynomials,

$$\mathbf{V}_n(\hat{\mathbf{a}}) \odot^n \hat{\mathbf{b}}^{\otimes n}
= P_n(\hat{\mathbf{a}} \cdot \hat{\mathbf{b}}),$$

which is the condition `tests/test_harmonics.py` asserts.

Building the operator once and contracting is what makes this worth having as
an operator at all: the harmonic of any direction is then a single `einsum`
rather than a sum over the terms of Eq. (41).
"""

import math

import numpy as np
from numpy.typing import DTypeLike

from natto.algebra import simplify_linear_combination
from natto.evaluate import evaluate_tensors
from natto.operators import get_natural_projector
from natto.symbolic import LinearCombination
from natto.utils import double_factorial, letter_index


def get_harmonic_operator(
    weight: int, normalize: str = "unity", dtype: DTypeLike = None
) -> tuple[np.ndarray, str]:
    """Build the harmonic operator `H` of one weight, evaluated.

    Args:
        weight: Weight of the harmonic, at least zero.
        normalize: `unity` applies the constant of Eq. (41), under which the
            weight-fold contraction of the harmonic with a unit vector is the
            Legendre polynomial of the angle between the two, and is 1 when the
            two coincide. `none` leaves the natural projector unscaled.
        dtype: Floating-point dtype of the evaluated operator, double precision
            if not given.

    Returns:
        The evaluated operator, and the einsum rule that applies it, so that
        `V = numpy.einsum(rule, H, *[a] * weight)` for a unit vector `a`.

    Raises:
        ValueError: If `weight` is negative, or `normalize` is not recognized.
    """
    if weight < 0:
        raise ValueError(f"Weight must be at least zero, got {weight}")

    H, upper, lower = get_harmonic_symbolic(weight)
    H_numerical = evaluate_tensors(H, mode="extraction", dtype=dtype)

    if normalize == "unity":
        H_numerical = H_numerical * coeff_harmonic(weight)
    elif normalize != "none":
        supported = ["none", "unity"]
        raise ValueError(
            f"Unknown normalization method: {normalize}. Supported are: {supported}."
        )

    # one operand per index, each of them the same vector
    operands = ",".join(f"...{letter}" for letter in upper)
    rule = f"{lower}{upper},{operands}->...{lower}" if weight else "->..."

    return H_numerical, rule


def get_harmonic_symbolic(weight: int) -> tuple[LinearCombination, str, str]:
    """Build the harmonic operator `H` of one weight, symbolically.

    The terms are exact, with rational coefficients, and unnormalized; the
    normalization constant is `coeff_harmonic`.

    Args:
        weight: Weight of the harmonic, at least zero.

    Returns:
        The symbolic operator, the letters carrying the polyadic's indices, and
        the letters carrying the harmonic's.
    """
    H = simplify_linear_combination(get_natural_projector(weight))

    return H, letter_index(weight, upper_case=True), letter_index(weight)


def coeff_harmonic(weight: int) -> float:
    """Normalization constant of the Cartesian harmonic, Eq. (41).

    Fixed by requiring that the weight-fold contraction of the harmonic with the
    direction it was built from is 1.

    Args:
        weight: Weight of the harmonic, at least zero.

    Returns:
        The normalization constant.
    """
    return double_factorial(2 * weight - 1) / math.factorial(weight)
