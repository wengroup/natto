"""The natural projector.

A rank-lowered tensor has the right rank but is in general neither symmetric nor
traceless. The natural projector is what makes it so: acting on any tensor of the
weight's rank it returns the symmetric traceless part, and acting on one already
symmetric and traceless it returns it unchanged.

It is built here symbolically and exactly, as a sum of averaged products of
Kronecker deltas with rational coefficients, one product per way of pairing its
indices. Being isotropic, it is the only building block the reduction needs beyond
the Levi-Civita symbol.

References:
    Eq. 7 of [Wen2026], with the coefficients of Eq. 8 and the recursion that
    builds them in Eq. S27.

    [Wen2026] M. Wen, Reusable Operators for Irreducible Cartesian Tensor
    Decomposition and Coupling, arXiv:2609.05971 (2026).
"""

import itertools
from fractions import Fraction

from natto.indices import get_permutations_2
from natto.symbolic import IndexGroup, Operator, Signature, Term


def get_natural_projector(ell: int) -> Operator:
    """The natural projector of one weight, as an operator over index slots.

    Its signature is the `weight` group, printing as a, b, c, ..., followed by the
    `sigma` group, printing as A, B, C, ....

    Args:
        ell: Weight of the ICT the projector belongs to.

    Returns:
        The projector, with exact coefficients.

    References:
        Eq. 7 of [Wen2026], with the coefficients of Eq. 8 built by the recursion of
        Eq. S27. Definition 3 (Sec. 6.3) of [Wen2026Refactor] for the operator.
    """
    signature = Signature(
        (IndexGroup("weight", ell, upper=False), IndexGroup("sigma", ell, upper=True))
    )

    terms = []
    for t, c in enumerate(_projector_coefficients(ell)):
        matchings = _projector_matchings(ell, t)

        # The coefficient of Eq. 8 is shared by the matchings of one t, and averaged
        # over them.
        factor = c / len(matchings)

        for pairs in matchings:
            sign, term = Term.from_blocks(pairs)
            terms.append((sign * factor, term))

    return Operator(signature, terms)


def _projector_matchings(ell: int, t: int) -> list[list[tuple[int, int]]]:
    """The delta pairs of the terms of the natural projector with a given t.

    Slots 0 to ell - 1 are the weight indices and ell to 2 * ell - 1 the sigma indices.
    Each term pairs ell - 2t weight slots with sigma slots, t weight slots among
    themselves and t sigma slots among themselves.

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

    perms = get_permutations_2(ell, num_delta=t)

    # `get_permutations_2` lists the slots left unpaired first and the paired ones
    # after them, two by two.
    start = ell - 2 * t

    matchings = []
    for p_r in perms:
        r_slots = [p_r.index(i) for i in range(ell)]
        rr_pairs = [(r_slots[i], r_slots[i + 1]) for i in range(start, ell, 2)]
        r_remaining_perms = list(itertools.permutations(r_slots[:start]))

        for p_s in perms:
            s_slots = [ell + p_s.index(i) for i in range(ell)]
            ss_pairs = [(s_slots[i], s_slots[i + 1]) for i in range(start, ell, 2)]

            # The remaining s slots are not permuted: the r slots they meet already are.
            for r_remaining in r_remaining_perms:
                rs_pairs = sorted(zip(r_remaining, s_slots[:start]))
                matchings.append(rs_pairs + rr_pairs + ss_pairs)

    return matchings


def _projector_coefficients(ell: int) -> list[Fraction]:
    """The coefficients c_t of Eq. 8 of [Wen2026], for t = 0, ..., ell // 2.

    Built by the recursion of Eq. S27, which keeps every step exact.
    """
    coefficients = [Fraction(1)]
    for t in range(1, ell // 2 + 1):
        ratio = Fraction(
            (ell - 2 * t + 2) * (ell - 2 * t + 1), 2 * t * (2 * ell - 2 * t + 1)
        )
        coefficients.append(-ratio * coefficients[-1])

    return coefficients
