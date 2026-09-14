"""Coupling of two irreducible Cartesian tensors into a third.

The coupling operator takes an ICT of weight l1 and one of weight l2 to the
weight-l3 part of their product, the Cartesian counterpart of a Clebsch-Gordan
coefficient. Two closed forms build it, one for each parity of l1 + l2 - l3; the odd
one carries a Levi-Civita symbol. `get_coupling_operator` dispatches on that parity,
so a caller supplies only the three weights.

The scale is not fixed by the construction. One factor per weight triple is free,
and the normalization constants fix it, one per parity; `tests/test_coupling.py`
asserts the two conditions they come from.

## Why this is not a call into the reduction

This is the only route to the coupling, and it is a closed form rather than a call
into the general symmetry machinery. That is worth explaining, since the product of
the two ICTs is symmetric within its first l1 indices and within its last l2, which
is an intrinsic symmetry in the ordinary sense, and the symmetry route does apply to
it.

Applying it is not enough on its own. What reduces the weight-l3 mapping space to a
single dimension is a second property: both inputs are traceless, so a contraction
taken inside either vanishes and only contractions between the two survive.
Tracelessness is not an index permutation, so no symmetry string expresses it and
the symmetry-adapted construction cannot use it. The derivation narrows the
candidates with it separately, and the result is the single operator built here.

References:
    Eq. 50 of [Wen2026] for even l1 + l2 + l3 and Eq. 51 for odd, both with the
    coefficient of Eq. 49, and with the normalization constant of Eq. 53 for even
    and Eq. 54 for odd.

    [Wen2026] M. Wen, Reusable Operators for Irreducible Cartesian Tensor
    Decomposition and Coupling, arXiv:2609.05971 (2026).
"""

from fractions import Fraction
from typing import Literal

import numpy as np

from natto.indices import get_slot_partitions
from natto.symbolic import IndexGroup, Operator, Signature, Term
from natto.utils import (
    double_factorial,
    factorial,
)

#: How the coupling operator is scaled. `legendre` applies the normalization
#: constant `C` of Eq. 53 and Eq. 54; `none` leaves the operator unscaled.
Normalization = Literal["legendre", "none"]


def get_coupling_operator(
    l1: int, l2: int, l3: int, normalize: Normalization = "legendre"
) -> tuple[np.ndarray, str]:
    """Build the coupling operator `K` of one weight triple, evaluated.

    Args:
        l1: Weight of the first natural tensor X.
        l2: Weight of the second natural tensor Y.
        l3: Weight of the output natural tensor Z.
        normalize: `legendre` applies the normalization constant `C` of the paper,
            fixing the scale by the condition for the parity of
            `L = l1 + l2 + l3`; `none` leaves the operator unscaled.

    Returns:
        The evaluated operator, and the einsum rule that applies it, so that
        `Z = numpy.einsum(rule, K, X, Y)`.

    Raises:
        ValueError: If `normalize` is not recognized.
    """
    if normalize not in ("legendre", "none"):
        supported = ["legendre", "none"]
        raise ValueError(
            f"Unknown normalization method: {normalize}. Supported are: {supported}."
        )

    K = get_coupling_symbolic(l1, l2, l3)

    # The signature orders the axes as Z, X, Y, which is what the rule contracts.
    K_numerical = K.evaluate()

    if normalize == "legendre":
        if (l1 + l2 - l3) % 2 == 0:
            constant = coeff_C_even(l1, l2, l3)
        else:
            constant = coeff_C_odd(l1, l2, l3)
        K_numerical = K_numerical * float(constant)

    X_idx = K.signature.letters_of("x")
    Y_idx = K.signature.letters_of("y")
    Z_idx = K.signature.letters_of("z")
    rule = f"{Z_idx}{X_idx}{Y_idx},...{X_idx},...{Y_idx}->...{Z_idx}"

    return K_numerical, rule


def get_coupling_symbolic(l1: int, l2: int, l3: int) -> Operator:
    """Build the coupling operator `K` of one weight triple, symbolically.

    The terms are exact, with rational coefficients, and unnormalized; the
    normalization constants are `coeff_C_even` and `coeff_C_odd`.

    Args:
        l1: Weight of the first natural tensor X.
        l2: Weight of the second natural tensor Y.
        l3: Weight of the output natural tensor Z.

    Returns:
        The symbolic operator, whose groups `z`, `x` and `y` carry the Z, X and Y
        indices.
    """
    if (l1 + l2 - l3) % 2 == 0:
        return _get_coupling_symbolic_even(l1, l2, l3)

    return _get_coupling_symbolic_odd(l1, l2, l3)


def triangle_numbers(l1: int, l2: int, l3: int) -> tuple[int, int, int]:
    """The triangle numbers of a weight triple.

    Each counts the contractions between the two groups other than its own: L1 those
    between Y and Z, L2 those between X and Z, and L3 those between X and Y. They
    bound the operator's sum over t, and appear in the factorials of both its
    coefficient and its normalization constant.

    Args:
        l1: Weight of the first natural tensor X.
        l2: Weight of the second natural tensor Y.
        l3: Weight of the output natural tensor Z.

    Returns:
        L1, L2 and L3.

    References:
        Defined below Eq. 48 of [Wen2026] as L_i = floor(L / 2) - l_i, with
        L = l1 + l2 + l3.
    """
    half = (l1 + l2 + l3) // 2

    return half - l1, half - l2, half - l3


def coeff_k(l1: int, l2: int, l3: int, t: int) -> Fraction:
    """The coefficient k_t of one term of the coupling operator.

    It is the counterpart of the natural projector's c_t, and what composing the
    projector with the rank lowering leaves in front of the average of Eq. 50 and
    Eq. 51. Taking that average is the caller's job; this returns the coefficient
    alone.

    Args:
        l1: Weight of the first natural tensor X.
        l2: Weight of the second natural tensor Y.
        l3: Weight of the output natural tensor Z.
        t: Index of the term, counting the traces taken within the output group.

    Returns:
        The exact coefficient k_t.

    References:
        Eq. 49 of [Wen2026].
    """
    L1, L2, _ = triangle_numbers(l1, l2, l3)

    # The equation's (2*l3 - 2*t - 1)!! / (2*l3 - 1)!! is the reciprocal of the t
    # factors separating the two, which is the bounded double factorial below.
    numerator = (-1) ** t * factorial(L2) * factorial(L1)
    denominator = (
        double_factorial(2 * l3 - 1, 2 * l3 - 2 * t + 1)
        * factorial(L2 - t)
        * factorial(L1 - t)
        * factorial(t)
    )

    return Fraction(numerator, denominator)


def coeff_C_even(l1: int, l2: int, l3: int) -> Fraction:
    """Normalization constant `C` for even `L = l1 + l2 + l3`.

    The constant is fixed by requiring that the l3-fold contraction of the output
    tensor with a unit vector yields 1.

    The value is a ratio of factorials, so it is returned exactly. Multiplying a
    float array by a `Fraction` would silently make it an object array, so the
    conversion happens at that boundary rather than here.

    Args:
        l1: Weight of the first natural tensor X.
        l2: Weight of the second natural tensor Y.
        l3: Weight of the output natural tensor Z.

    Returns:
        The normalization constant.

    References:
        Eq. 53 of [Wen2026].
    """
    L = l1 + l2 + l3
    L1, L2, L3 = triangle_numbers(l1, l2, l3)

    numerator = factorial(l1) * factorial(l2) * double_factorial(2 * l3 - 1)
    denominator = (
        double_factorial(2 * L2 - 1)
        * double_factorial(2 * L1 - 1)
        * double_factorial(2 * L3 - 1)
        * factorial(L // 2)
    )

    return Fraction(numerator, denominator)


def coeff_C_odd(l1: int, l2: int, l3: int) -> Fraction:
    """Normalization constant `C` for odd `L = l1 + l2 + l3`.

    The condition differs from the one behind `coeff_C_even`. For odd `L` the
    coupling operator carries a Levi-Civita symbol, so contracting the output l3
    times with a single unit vector vanishes identically by antisymmetry and cannot
    fix the scale. The constant is fixed instead by the rate of that vanishing: the
    output's contraction with l3 - 1 copies of one direction, divided by the norm of
    the cross product of the two input directions, tends to one as the directions
    merge.

    The value is a ratio of factorials, so it is returned exactly. Multiplying a
    float array by a Fraction would silently make it an object array, so the
    conversion happens at that boundary rather than here.

    Args:
        l1: Weight of the first ICT, X.
        l2: Weight of the second ICT, Y.
        l3: Weight of the output ICT, Z.

    Returns:
        The normalization constant.

    References:
        Eq. 54 of [Wen2026].
    """
    L = l1 + l2 + l3
    L1, L2, L3 = triangle_numbers(l1, l2, l3)

    numerator = 2 * l3 * factorial(l1) * factorial(l2) * double_factorial(2 * l3 - 1)
    denominator = (
        double_factorial(2 * L2 + 1)
        * double_factorial(2 * L1 + 1)
        * double_factorial(2 * L3 + 1)
        * factorial((L + 1) // 2)
    )

    return Fraction(numerator, denominator)


def _get_coupling_symbolic_even(l1: int, l2: int, l3: int) -> Operator:
    """Build `K` symbolically for even `l1 + l2 - l3`; see `get_coupling_symbolic`."""
    if (l1 + l2 - l3) % 2 != 0:
        raise ValueError(
            f"the weight sum (l1 + l2 - l3) must be even, got l1={l1}, l2={l2}, l3={l3}"
        )

    signature = _coupling_signature(l1, l2, l3)
    L1, L2, _ = triangle_numbers(l1, l2, l3)

    terms = []
    for t in range(min(L2, L1) + 1):
        all_blocks = _get_coupling_blocks_even(l1, l2, l3, t)

        # Total factor: the coefficient of Eq. 49 divided by len(all_blocks), which
        # averages over the terms as the angle brackets of Eq. 50 ask. The blocks are
        # the distinct terms of that average.
        factor = coeff_k(l1, l2, l3, t) / len(all_blocks)

        for deltas in all_blocks:
            sign, term = Term.from_blocks(deltas)
            terms.append((sign * factor, term))

    return Operator(signature, terms)


def _get_coupling_symbolic_odd(l1: int, l2: int, l3: int) -> Operator:
    """Build `K` symbolically for odd `l1 + l2 - l3`; see `get_coupling_symbolic`."""
    if (l1 + l2 - l3) % 2 != 1:
        raise ValueError(
            f"the weight sum (l1 + l2 - l3) must be odd, got l1={l1}, l2={l2}, l3={l3}"
        )

    signature = _coupling_signature(l1, l2, l3)
    L1, L2, _ = triangle_numbers(l1, l2, l3)

    terms = []
    for t in range(min(L2, L1) + 1):
        all_blocks = _get_coupling_blocks_odd(l1, l2, l3, t)

        # Total factor: as in the even case, the coefficient of Eq. 49 averaged over
        # the terms, here those of Eq. 51.
        factor = coeff_k(l1, l2, l3, t) / len(all_blocks)

        for deltas, epsilon in all_blocks:
            sign, term = Term.from_blocks(deltas, [epsilon])
            terms.append((sign * factor, term))

    return Operator(signature, terms)


def _coupling_signature(l1: int, l2: int, l3: int) -> Signature:
    """The Z indices, printing as a, b, ..., then those of X and Y, as A, B, ...

    Z takes slots 0 to l3 - 1, X the next l1 and Y the last l2; these are the slots
    `_get_coupling_blocks_even` and `_get_coupling_blocks_odd` write the terms in.
    """
    return Signature(
        (
            IndexGroup("z", l3, upper=False),
            IndexGroup("x", l1, upper=True),
            IndexGroup("y", l2, upper=True),
        )
    )


def _get_coupling_blocks_even(
    l1: int, l2: int, l3: int, t: int
) -> list[list[tuple[int, int]]]:
    """The delta pairs of each term of { d_ra^(l1-k-t) d_sa^(l2-k-t) d_aa^t } d_rs^(k+t).

    The r indices of X, s of Y and a of Z are the slots of `_coupling_signature`.

    Args:
        l1: Weight of the first ICT, X.
        l2: Weight of the second ICT, Y.
        l3: Weight of the output ICT, Z.
        t: Number of deltas pairing two a indices.

    Returns:
        The delta pairs of each term, in the order the operator lists them.

    Raises:
        ValueError: If l1 + l2 - l3 is odd.

    References:
        Eq. 50 of [Wen2026].
    """
    if (l1 + l2 - l3) % 2 != 0:
        raise ValueError(
            f"the weight sum (l1 + l2 - l3) must be even, got l1={l1}, l2={l2}, l3={l3}"
        )

    k = (l1 + l2 - l3) // 2

    r_idx = list(range(l3, l3 + l1))
    s_idx = list(range(l3 + l1, l3 + l1 + l2))

    n_ra = l1 - k - t
    n_sa = l2 - k - t
    rs_pairs = list(zip(r_idx[n_ra:], s_idx[n_sa:]))

    # Only the a indices, the slots of Z, are placed in every distinct way, to
    # symmetrize the output as the curly braces ask; the r and s indices come from
    # ICTs and are symmetric already.
    all_blocks = []
    for (ra_slots, sa_slots), aa_pairs in get_slot_partitions([n_ra, n_sa], t):
        ra_pairs = list(zip(r_idx[:n_ra], ra_slots))
        sa_pairs = list(zip(s_idx[:n_sa], sa_slots))
        all_blocks.append(ra_pairs + sa_pairs + list(aa_pairs) + rs_pairs)

    return all_blocks


def _get_coupling_blocks_odd(
    l1: int, l2: int, l3: int, t: int
) -> list[tuple[list[tuple[int, int]], tuple[int, int, int]]]:
    """The blocks of each term of { eps_rsa d_ra^(l1-k-t) d_sa^(l2-k-t) d_aa^t } d_rs^(k+t).

    The r indices of X, s of Y and a of Z are the slots of `_coupling_signature`.

    Args:
        l1: Weight of the first ICT, X.
        l2: Weight of the second ICT, Y.
        l3: Weight of the output ICT, Z.
        t: Number of deltas pairing two a indices.

    Returns:
        The delta pairs and the Levi-Civita triple, in its r, s, a order, of each term,
        in the order the operator lists them.

    Raises:
        ValueError: If l1 + l2 - l3 is even.

    References:
        Eq. 51 of [Wen2026].
    """
    if (l1 + l2 - l3) % 2 != 1:
        raise ValueError(
            f"the weight sum (l1 + l2 - l3) must be odd, got l1={l1}, l2={l2}, l3={l3}"
        )

    k = (l1 + l2 - l3 - 1) // 2

    r_idx = list(range(l3, l3 + l1))
    s_idx = list(range(l3 + l1, l3 + l1 + l2))

    # The first r and s indices go to the Levi-Civita symbol
    n_ra = l1 - k - t - 1
    n_sa = l2 - k - t - 1
    rs_pairs = list(zip(r_idx[n_ra + 1 :], s_idx[n_sa + 1 :]))

    # As in the even case, only the a indices are placed; one goes to the symbol.
    all_blocks = []
    partitions = get_slot_partitions([1, n_ra, n_sa], t)
    for ((a_epsilon,), ra_slots, sa_slots), aa_pairs in partitions:
        epsilon = (r_idx[0], s_idx[0], a_epsilon)
        ra_pairs = list(zip(r_idx[1 : n_ra + 1], ra_slots))
        sa_pairs = list(zip(s_idx[1 : n_sa + 1], sa_slots))
        all_blocks.append((ra_pairs + sa_pairs + list(aa_pairs) + rs_pairs, epsilon))

    return all_blocks
