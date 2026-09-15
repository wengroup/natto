"""The natural projector.

A rank-lowered tensor has the right rank but is in general neither symmetric nor
traceless. The natural projector is what makes it so: acting on any tensor of the
weight's rank it returns the symmetric traceless part, and acting on one already
symmetric and traceless it returns it unchanged.

It is built here symbolically and exactly, as a sum of averaged products of
Kronecker deltas with rational coefficients, one product per way of pairing its
indices. Being isotropic, it is the only building block the reduction needs beyond
the Levi-Civita symbol.
"""

import itertools
from fractions import Fraction

import numpy as np

from natto.indices import get_slot_partitions
from natto.symbolic import IndexGroup, Operator, Signature, Term


def get_natural_projector(ell: int) -> Operator:
    """The natural projector of one weight, as an operator over index slots.

    Its signature is the `ict` group, printing as a, b, c, ..., followed by the `gct`
    group, printing as A, B, C, ..., which takes the tensor it projects.

    Args:
        ell: Weight of the ICT the projector belongs to.

    Returns:
        The projector, with exact coefficients.

    References:
        Eq. 7 of [Wen2026], with the coefficients of Eq. 8 built by the recursion of
        Eq. S27. Definition 3 (C.3) of [Wen2026Refactor] for the operator.
    """
    signature = Signature(
        (
            IndexGroup("ict", ell, upper=False),
            IndexGroup("gct", ell, upper=True, input=True),
        )
    )

    terms = []
    for t, c in enumerate(get_projector_coefficients(ell)):
        matchings = _projector_matchings(ell, t)

        # The coefficient of Eq. 8 is shared by the matchings of one t, and averaged
        # over them.
        factor = c / len(matchings)

        for pairs in matchings:
            sign, term = Term.from_blocks(pairs)
            terms.append((sign * factor, term))

    return Operator(signature, terms)


def get_symmetric_traceless_part(t: np.ndarray) -> np.ndarray:
    """The symmetric traceless part of an array, computed numerically.

    The array is averaged over the permutations of its axes, and its traces are then
    subtracted as a signed sum over the number d of delta pairs taken out, each term
    summed over the distinct placements of those deltas among the axes. The result is
    the natural projector applied to `t`, but it works on the rank-n array rather than
    on the rank-2n projector, so it stays cheap at high rank.

    Args:
        t: An array of shape `(3,) * n`.

    Returns:
        Its symmetric traceless part, of the same shape.

    References:
        Eq. 10 of [JCB78] J. Jerphagnon, D. Chemla, and R. Bonneville, Advances in
        Physics 27, 609 (1978).
    """
    n = t.ndim
    permuted = [np.transpose(t, perm) for perm in itertools.permutations(range(n))]
    symmetric = np.mean(permuted, axis=0)

    delta = np.eye(3)
    traceless = symmetric
    coefficient = 1.0
    for d in range(1, n // 2 + 1):
        coefficient = -coefficient / (2 * n - 2 * d + 1)

        # The array is symmetric, so taking d traces over its first 2d axes is as good
        # as over any.
        remaining = list(range(d, n - d))
        traced = np.einsum(
            symmetric, [k // 2 for k in range(2 * d)] + remaining, remaining
        )

        for (free,), pairs in get_slot_partitions([n - 2 * d], d):
            operands = [traced, list(free)]
            for pair in pairs:
                operands += [delta, list(pair)]
            traceless = traceless + coefficient * np.einsum(*operands, list(range(n)))

    return traceless


def get_random_natural_tensor(n: int, seed: int = 35) -> np.ndarray:
    """A random ICT: the symmetric traceless part of a random array.

    Args:
        n: Weight of the ICT, at least zero.
        seed: Seed of the random array.

    Returns:
        A symmetric traceless array of shape `(3,) * n`.
    """
    X = np.random.default_rng(seed).standard_normal((3,) * n)
    natural = get_symmetric_traceless_part(X)

    return natural


def get_projector_coefficients(ell: int) -> list[Fraction]:
    """The coefficients c_t of the natural projector, for t = 0, ..., ell // 2.

    Built by the recursion of Eq. S27, which keeps every step exact.

    Args:
        ell: Weight of the projector.

    Returns:
        The exact coefficients, starting from c_0 = 1.

    References:
        Eq. 8 of [Wen2026], by the recursion of Eq. S27.
    """
    coefficients = [Fraction(1)]
    for t in range(1, ell // 2 + 1):
        ratio = Fraction(
            (ell - 2 * t + 2) * (ell - 2 * t + 1), 2 * t * (2 * ell - 2 * t + 1)
        )
        coefficients.append(-ratio * coefficients[-1])

    return coefficients


def _projector_matchings(ell: int, t: int) -> list[list[tuple[int, int]]]:
    """The delta pairs of the terms of the natural projector with a given t.

    Slots 0 to ell - 1 are the ICT indices and ell to 2 * ell - 1 the tensor indices.
    Each term pairs ell - 2t ICT slots with tensor slots, t ICT slots among themselves
    and t tensor slots among themselves.

    Args:
        ell: Weight of the ICT the projector belongs to.
        t: Number of deltas inside each group.

    Returns:
        The delta pairs of each term, in the order the operator lists them.

    Raises:
        ValueError: If `ell` is less than `2 * t`.

    References:
        Eq. 7 of [Wen2026].
    """
    if ell < 2 * t:
        raise ValueError(f"weight (ell) must be at least 2*t, got ell={ell}, t={t}")

    partitions = get_slot_partitions([ell - 2 * t], t)

    matchings = []
    for (r_free,), rr_pairs in partitions:
        r_free_perms = list(itertools.permutations(r_free))

        for (s_free,), s_pairs in partitions:
            s_free = [ell + slot for slot in s_free]
            ss_pairs = [(ell + a, ell + b) for a, b in s_pairs]

            # The free s slots are not permuted: the r slots they meet already are.
            for r_perm in r_free_perms:
                rs_pairs = sorted(zip(r_perm, s_free))
                matchings.append(rs_pairs + list(rr_pairs) + ss_pairs)

    return matchings
