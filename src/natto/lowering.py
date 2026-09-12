"""Rank-lowering tensors.

To reach an ICT of some weight from a generic tensor of higher rank, the surplus
indices must first be contracted away in pairs against Kronecker deltas and, when
the parity of n - ell requires it, one Levi-Civita symbol. Which indices
are paired is a choice, and each choice leaves different information behind -- which
is exactly why a single tensor can carry several ICTs of the same weight.

Each function returns the tensors themselves, paired with the letters that tensor
leaves unused. Those letters are not part of the rank-lowering tensor; they are what
is left of the rank once it has taken the indices it contracts, and `mapping_tensors`
is where they are put to work.

References:
    Eq. 2 of [Wen2026] for the rank lowering, Eq. 3 for even n - ell and Eq. 5 for
    odd. Sec. III A for the discussion.

    [Wen2026] M. Wen, Reusable Operators for Irreducible Cartesian Tensor
    Decomposition and Coupling, arXiv:2609.05971 (2026).
"""

import itertools

from natto.indices import get_permutations_2, letter_index
from natto.symbolic import IsotropicProduct, create_delta_epsilon_tensors


def get_lowering_tensors(ell: int, n: int) -> tuple[list[IsotropicProduct], list[str]]:
    """The rank-lowering tensors of a weight, whichever parity applies.

    Which form the rank-lowering tensor takes depends on the parity of n - ell:
    deltas alone when it is even, and one Levi-Civita symbol alongside them when it
    is odd. This dispatches on that, so a caller supplies only the weight and rank.

    Args:
        ell: Weight of the ICT.
        n: Rank of the Cartesian tensor.

    Returns:
        The rank-lowering tensors, one per choice of contracted indices, and the
        letters each one leaves unused.

    References:
        Eq. 2 of [Wen2026], with Eq. 3 for even n - ell and Eq. 5 for odd.
    """
    if (n - ell) % 2 == 0:
        return get_lowering_tensors_even(ell, n)

    return get_lowering_tensors_odd(ell, n)


def get_lowering_tensors_even(
    ell: int, n: int
) -> tuple[list[IsotropicProduct], list[str]]:
    """The rank-lowering tensors of even n - ell.

    With the parity even the rank-lowering tensor is built from Kronecker deltas alone,
    (n - ell) / 2 of them, pairing off the surplus indices.

    Args:
        ell: Weight of the ICT.
        n: Rank of the Cartesian tensor.

    Returns:
        The rank-lowering tensors, one per choice of contracted indices, and the
        letters each one leaves unused.

    References:
        Eq. 3 of [Wen2026].
    """
    letters = letter_index(n, upper_case=True)

    all_perms = get_permutations_2(n, num_delta=(n - ell) // 2)

    # TODO, this depends on the order of the indices get_permutations_2 returns, where
    #   we put the remaining indices of t at the front, and the contracted indices at
    #   the end.
    start = ell

    tensors = []
    remaining_letters = []
    for perm in all_perms:  # each perm for a choice of contracted indices
        indices = [letters[perm.index(i)] for i in range(n)]

        # the delta pairs of F^p, the rank-lowering tensor
        delta_pairs = [indices[i] + indices[i + 1] for i in range(start, n, 2)]
        tensors.append(create_delta_epsilon_tensors(delta_pairs))

        # the letters this tensor does not use
        remaining_letters.append("".join(indices[:start]))

    return tensors, remaining_letters


def get_lowering_tensors_odd(
    ell: int, n: int
) -> tuple[list[IsotropicProduct], list[str]]:
    """The rank-lowering tensors of odd n - ell.

    With the parity odd, one Levi-Civita symbol is needed alongside the Kronecker deltas.
    Upper-case letter n + 1 is the index of that symbol which contracts with the
    natural projector -- the tau index. For n = 3, that is the letter D.

    Args:
        ell: Weight of the ICT.
        n: Rank of the Cartesian tensor.

    Returns:
        The rank-lowering tensors, one per choice of contracted indices, and the
        letters each one leaves unused.

    References:
        Eq. 5 of [Wen2026].
    """
    if ell == 0:
        return get_lowering_tensors_odd_weight_zero(ell, n)

    # All s letters
    letters = letter_index(n, upper_case=True)

    # The epsilon index that is not contracted with the Cartesian tensor -- the tau
    # index, which the natural projector takes instead. The n Cartesian indices have
    # used the first n upper-case letters, so this takes the next free one.
    tau_letter = letter_index(1, start=n, upper_case=True)

    all_perms = get_permutations_2(n, num_delta=(n - ell - 1) // 2)

    # TODO, this depends on the order of the indices get_permutations_2 returns, where
    #   we put the remaining indices of t at the front, and the contracted indices at
    #   the end.
    start = ell + 1

    tensors = []
    remaining_letters = []
    for perm in all_perms:  # each perm for a choice of contracted indices
        indices = [letters[perm.index(i)] for i in range(n)]

        # the delta pairs of F^p, the rank-lowering tensor
        delta_pairs = [indices[i] + indices[i + 1] for i in range(start, n, 2)]

        # remaining indices, shared out between epsilon and the natural projector
        s_remaining = indices[:start]
        s_remaining_set = set(s_remaining)

        for comb in itertools.combinations(s_remaining, 2):
            # two of them go to epsilon, along with tau
            epsilon = tau_letter + "".join(sorted(comb))
            tensors.append(create_delta_epsilon_tensors(delta_pairs, epsilon=epsilon))

            # the rest, plus tau again, are left unused
            remaining_letters.append(
                "".join(sorted(s_remaining_set - set(comb))) + tau_letter
            )

    return tensors, remaining_letters


def get_lowering_tensors_odd_weight_zero(
    ell: int, n: int
) -> tuple[list[IsotropicProduct], list[str]]:
    """
    For j = 0, and odd n, the rules for G(n|0) are different from the general case.

    Here we do a trivial contraction with epsilon tensor, instead of a double
    contraction in the general case.
    """
    if ell != 0:
        raise ValueError(f"weight (ell) must be 0, got ell={ell}")
    if n % 2 != 1:
        raise ValueError(f"rank (n) must be odd, got n={n}")
    if n < 3:
        raise ValueError(f"rank (n) must be at least 3, got n={n}")

    # All s letters
    letters = letter_index(n, upper_case=True)

    all_perms = get_permutations_2(n, num_delta=(n - 3) // 2)

    # TODO, this depends on the order of the indices get_permutations_2 returns, where
    #   we put the remaining indices of t at the front, and the contracted indices at
    #   the end.
    start = 3

    tensors = []
    remaining_letters = []
    for perm in all_perms:  # each perm for a choice of contracted indices
        indices = [letters[perm.index(i)] for i in range(n)]

        # the delta pairs of F^p, the rank-lowering tensor
        delta_pairs = [indices[i] + indices[i + 1] for i in range(start, n, 2)]

        # every remaining index goes to epsilon, so nothing is left unused
        epsilon = "".join(sorted(indices[:start]))
        tensors.append(create_delta_epsilon_tensors(delta_pairs, epsilon=epsilon))
        remaining_letters.append("")

    return tensors, remaining_letters
