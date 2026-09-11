r"""Rank-lowering tensors $\mathbf{F}^p_{n \to \ell}$, Sec. III A of the paper.

To reach a weight-$\ell$ ICT from a generic tensor $\mathbf{T}_n$ of higher rank
$n \geq \ell$, the surplus indices must first be contracted away in pairs against
$\bm\delta$ and, when the parity of $n - \ell$ requires it, one $\bm\epsilon$.
Which indices are paired is a choice, and each choice $\mathcal{D}_p$ leaves
different information behind -- which is exactly why a single rank-$n$ tensor can
carry several ICTs of the same weight.

What this module returns is not the tensor itself but the index assignments that
define it: the $\bm\delta$ pairs, the $\bm\epsilon$ triple when there is one, and
the letters left over for the natural projector to act on. `mappings` composes the
two into $\mathbf{G}^p_{(\ell|n)}$.

References:
1. [AG82] Irreducible fourth-rank Cartesian tensors, https://doi.org/10.1103/PhysRevA.25.2647
"""

import itertools

from natto.indices import letter_index
from natto.symmetric_traceless import get_permutations_2


def get_lowering_rules_even(j: int, n: int) -> tuple[list[str], list[list[str]]]:
    """
    Rules for G(n|j) for even n-j.

    Args:
        j:
        n:

    Returns:
        E_s_indices: s letters to use for E_j
        f_rules: rules to create deltas for for f_{n-j}^q
    """
    assert (n - j) % 2 == 0, f"n-j must be even, got n={n}, j={j}"

    letters = letter_index(n, upper_case=True)

    all_perms = get_permutations_2(n, num_delta=(n - j) // 2)

    # TODO, this depends on the order of the indices get_permutations_2 returns, where
    #   we put the remaining indices of t at the front, and the contracted indices at
    #   the end.
    start = j

    f_rules = []
    E_s_letters = []
    for perm in all_perms:  # each perm for a q in f_q
        indices = [letters[perm.index(i)] for i in range(n)]

        # indices for f_{n-j}^q
        delta_pairs = [indices[i] + indices[i + 1] for i in range(start, n, 2)]
        f_rules.append(delta_pairs)

        # s indices for E_j
        s_remaining = "".join(indices[:start])
        E_s_letters.append(s_remaining)

    return E_s_letters, f_rules


def get_lowering_rules_odd(
    j: int, n: int
) -> tuple[list[str], list[str], list[list[str]]]:
    """
    Rules for G(n|j) for odd n-j.


    # NOTE,
    Upper case letter n+1 will be used as the index in epsilon to contract with E(j|j).
    In other words, it is the tau index.
    For example, if n = 3, then the letter D will be used as the tau index.

    Args:
        j:
        n:

    Returns:
        E_s_indices: s letters to use for E_j
        f_epsilon_rules: rules to create epsilons for f_{n-j}^q
        f_delta_rules: rules to create deltas for f_{n-j}^q
    """
    assert (n - j) % 2 == 1, f"n-j must be odd, got n={n}, j={j}"

    if j == 0:
        return get_lowering_rules_odd_weight_zero(j, n)

    # All s letters
    letters = letter_index(n, upper_case=True)

    # Extra letter used in epsilon. See Table I of [AG82]
    tau_letter = letter_index(1, start=n, upper_case=True)

    all_perms = get_permutations_2(n, num_delta=(n - j - 1) // 2)

    # TODO, this depends on the order of the indices get_permutations_2 returns, where
    #   we put the remaining indices of t at the front, and the contracted indices at
    #   the end.
    start = j + 1

    f_delta_rules = []
    f_epsilon_rules = []
    E_s_letters = []
    for perm in all_perms:  # each perm for a q in f_q
        indices = [letters[perm.index(i)] for i in range(n)]

        # delta indices for f_{n-j}^q
        delta_pairs = [indices[i] + indices[i + 1] for i in range(start, n, 2)]

        # remaining indices for epsilon and E_j
        s_remaining = indices[:start]
        s_remaining_set = set(s_remaining)

        for comb in itertools.combinations(s_remaining, 2):
            f_delta_rules.append(delta_pairs)

            # choose two indices for epsilon
            f_epsilon_rules.append(tau_letter + "".join(sorted(comb)))

            # the remaining indices and also tau for E_j
            E_s_letters.append(
                "".join(sorted(s_remaining_set - set(comb))) + tau_letter
            )

    return E_s_letters, f_epsilon_rules, f_delta_rules


def get_lowering_rules_odd_weight_zero(j, n):
    """
    For j = 0, and odd n, the rules for G(n|0) are different from the general case.

    Here we do a trivial contraction with epsilon tensor, instead of a double
    contraction in the general case.
    """
    assert j == 0, f"j must be 0, got {j}"
    assert n % 2 == 1, f"n must be odd, got {n}"
    assert n >= 3, f"n must be greater than or equal to 3, got {n}"

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

        # delta indices for f_{n-j}^q
        delta_pairs = [indices[i] + indices[i + 1] for i in range(start, n, 2)]
        f_delta_rules.append(delta_pairs)

        # remaining indices for epsilon (indices for E_j is empty)
        s_remaining = indices[:start]
        f_epsilon_rules.append("".join(sorted(s_remaining)))
        E_s_letters.append("")

    return E_s_letters, f_epsilon_rules, f_delta_rules
