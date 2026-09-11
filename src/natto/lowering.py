"""Rank-lowering tensors.

To reach an ICT of some weight from a generic tensor of higher rank, the surplus
indices must first be contracted away in pairs against Kronecker deltas and, when
the parity of n - ell requires it, one Levi-Civita symbol. Which indices
are paired is a choice, and each choice leaves different information behind -- which
is exactly why a single tensor can carry several ICTs of the same weight.

What this module returns is not the tensor itself but the index assignments that
define it: the delta pairs, the epsilon triple when there is one, and the letters
left over for the natural projector to act on. `mapping_tensors` composes the two.

References:
    Eq. 2 of [Wen2026] for the rank lowering, Eq. 3 for even n - ell and Eq. 5 for
    odd. Sec. III A for the discussion.

    [Wen2026] M. Wen, Reusable Operators for Irreducible Cartesian Tensor
    Decomposition and Coupling, arXiv:2609.05971 (2026).
"""

import itertools

from natto.indices import get_permutations_2, letter_index


def get_lowering_rules_even(ell: int, n: int) -> tuple[list[str], list[list[str]]]:
    """Index rules for the rank-lowering tensors of even n - ell.

    With the parity even the rank-lowering tensor is built from Kronecker deltas alone,
    (n - ell) / 2 of them, pairing off the surplus indices.

    Args:
        ell: Weight of the ICT.
        n: Rank of the Cartesian tensor.

    Returns:
        The letters left over for the natural projector to act on, and the delta index
        pairs of each rank-lowering tensor, one entry per choice.

    References:
        Eq. 3 of [Wen2026].
    """
    if (n - ell) % 2 != 0:
        raise ValueError(
            f"rank minus weight (n - ell) must be even, got n={n}, ell={ell}"
        )

    letters = letter_index(n, upper_case=True)

    all_perms = get_permutations_2(n, num_delta=(n - ell) // 2)

    # TODO, this depends on the order of the indices get_permutations_2 returns, where
    #   we put the remaining indices of t at the front, and the contracted indices at
    #   the end.
    start = ell

    f_rules = []
    E_s_letters = []
    for perm in all_perms:  # each perm for a q in f_q
        indices = [letters[perm.index(i)] for i in range(n)]

        # indices for F^p, the rank-lowering tensor
        delta_pairs = [indices[i] + indices[i + 1] for i in range(start, n, 2)]
        f_rules.append(delta_pairs)

        # s indices for the natural projector
        s_remaining = "".join(indices[:start])
        E_s_letters.append(s_remaining)

    return E_s_letters, f_rules


def get_lowering_rules_odd(
    ell: int, n: int
) -> tuple[list[str], list[str], list[list[str]]]:
    """Index rules for the rank-lowering tensors of odd n - ell.

    With the parity odd, one Levi-Civita symbol is needed alongside the Kronecker deltas.
    Upper-case letter n + 1 is the index of that symbol which contracts with the
    natural projector -- the tau index. For n = 3, that is the letter D.

    Args:
        ell: Weight of the ICT.
        n: Rank of the Cartesian tensor.

    Returns:
        The letters left over for the natural projector, the epsilon index triple, and
        the delta index pairs, one entry each per choice.

    References:
        Eq. 5 of [Wen2026].
    """
    if (n - ell) % 2 != 1:
        raise ValueError(
            f"rank minus weight (n - ell) must be odd, got n={n}, ell={ell}"
        )

    if ell == 0:
        return get_lowering_rules_odd_weight_zero(ell, n)

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

    f_delta_rules = []
    f_epsilon_rules = []
    E_s_letters = []
    for perm in all_perms:  # each perm for a q in f_q
        indices = [letters[perm.index(i)] for i in range(n)]

        # delta indices for F^p, the rank-lowering tensor
        delta_pairs = [indices[i] + indices[i + 1] for i in range(start, n, 2)]

        # remaining indices for epsilon and the natural projector
        s_remaining = indices[:start]
        s_remaining_set = set(s_remaining)

        for comb in itertools.combinations(s_remaining, 2):
            f_delta_rules.append(delta_pairs)

            # choose two indices for epsilon
            f_epsilon_rules.append(tau_letter + "".join(sorted(comb)))

            # the remaining indices and also tau for the natural projector
            E_s_letters.append(
                "".join(sorted(s_remaining_set - set(comb))) + tau_letter
            )

    return E_s_letters, f_epsilon_rules, f_delta_rules


def get_lowering_rules_odd_weight_zero(ell, n):
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

    f_delta_rules = []
    f_epsilon_rules = []
    E_s_letters = []

    for perm in all_perms:  # each perm for a q in f_q
        indices = [letters[perm.index(i)] for i in range(n)]

        # delta indices for F^p, the rank-lowering tensor
        delta_pairs = [indices[i] + indices[i + 1] for i in range(start, n, 2)]
        f_delta_rules.append(delta_pairs)

        # remaining indices for epsilon (the natural projector gets none)
        s_remaining = indices[:start]
        f_epsilon_rules.append("".join(sorted(s_remaining)))
        E_s_letters.append("")

    return E_s_letters, f_epsilon_rules, f_delta_rules
