r"""The natural projector $\mathbf{E}_{(\ell|\ell)}$, Eq. (12) of the paper.

A rank-lowered tensor has the right rank but is in general neither symmetric nor
traceless. The natural projector is what makes it so: acting on any rank-$\ell$
tensor it returns the symmetric traceless part, and acting on one already
symmetric and traceless it returns it unchanged.

It is built here symbolically and exactly, as a sum over $t$ of averaged products
of $\bm\delta$'s with rational coefficients; `get_projector_rules` enumerates the
index patterns of each term. Being isotropic, it is the only building block the
reduction needs beyond $\bm\epsilon$.

References:
1. [CS70] Irreducible Cartesian Tensors. II. General Formulation, http://dx.doi.org/10.1063/1.1665190
"""

import itertools
from fractions import Fraction

from natto.indices import letter_index
from natto.symbolic import LinearCombination, create_delta_epsilon_tensors
from natto.symmetric_traceless import get_permutations_2


def get_natural_projector(
    j: int, s_letters: str = None, verbose: int = 0
) -> LinearCombination:
    """
    Invariant tensors of rank j: E(j | j).

    References:
        Eq 19 and Eq 21 of [CS70].

    Args:
        j: rank of the projection operator
        s_letters: letters for the upper case indices, if None, use the default:
            A, B, C, etc.
        verbose: verbosity level for debugging

    Returns:
        Tensor operations with delta tensors. We use lower case letters `a`, `b`, `c`,
        etc. for indices `r`, and upper case letters `A`, `B`, `C`, etc. for indices `s`.
    """
    k = j // 2

    out = []
    c = Fraction(1, 1)  # c for t = 0
    for t in range(k + 1):
        if t > 0:
            c *= -Fraction(
                (j - 2 * t + 2) * (j - 2 * t + 1), 2 * t * (2 * j - 2 * t + 1)
            )

        # get all rules
        all_rules = get_projector_rules(j, t, s_letters)

        # Total factor: c / len(all_rules), where len(all_rules) averages over all
        # the rules.
        factor = c / len(all_rules)

        # create tensor products of deltas for each rule
        delta_tensors = [
            create_delta_epsilon_tensors(
                rule["d_rs"] + rule["d_rr"] + rule["d_ss"], factor=factor
            )
            for rule in all_rules
        ]

        out.extend(delta_tensors)

        if verbose > 0:
            tmp = []
            tmp.extend(delta_tensors)
            print(
                f"@ debug E_j: j={j}, t={t}, c={c}, "
                f"num terms: {len(tmp)}, terms: {LinearCombination(*tmp)}"
            )

    return LinearCombination(*out)


def get_projector_rules(
    j: int, t: int, s_letters: str = None
) -> list[dict[str, list[str]]]:
    """
    Rules for E(j|j): d_{rs}^{j-2t} d_{rr}^t d_{ss}^t.

    This is in Eq. 19 of the paper.

    Args:
        j: rank of the projection operator
        t: number of d_rr and d_ss
        s_letters: letters for the upper case indices, if None, use the default:
            A, B, C, etc.

    Examples:
        get_projector_rules(3, 1)
        [{'d_rs': ['cC'], 'd_rr': ['ab'], 'd_ss': ['AB']},
         {'d_rs': ['cB'], 'd_rr': ['ab'], 'd_ss': ['AC']},
         {'d_rs': ['cA'], 'd_rr': ['ab'], 'd_ss': ['BC']},
         {'d_rs': ['bC'], 'd_rr': ['ac'], 'd_ss': ['AB']},
         {'d_rs': ['bB'], 'd_rr': ['ac'], 'd_ss': ['AC']},
         {'d_rs': ['bA'], 'd_rr': ['ac'], 'd_ss': ['BC']},
         {'d_rs': ['aC'], 'd_rr': ['bc'], 'd_ss': ['AB']},
         {'d_rs': ['aB'], 'd_rr': ['bc'], 'd_ss': ['AC']},
         {'d_rs': ['aA'], 'd_rr': ['bc'], 'd_ss': ['BC']},
        ]

    Returns:
        Each dict {'d_rs': list_rs, 'd_rr': list_rr, 'd_ss': list_ss} contains the
        indices for constructing the deltas.

    """
    assert j >= 2 * t, f"j must be greater than or equal to 2*t, got j={j}, t={t}"

    r_letters = letter_index(j, upper_case=False)

    if s_letters is None:
        s_letters = letter_index(j, upper_case=True)

    perms = get_permutations_2(j, num_delta=t)

    # TODO, this depends on the order of the indices get_permutations_2 returns, where
    #   we put the remaining indices of t at the front, and the contracted indices at
    #   the end.
    start = j - 2 * t

    all_indices = []
    for p_r in perms:
        r_indices = [r_letters[p_r.index(i)] for i in range(j)]

        # indices for d_{rr}^t
        rr_pairs = [r_indices[i] + r_indices[i + 1] for i in range(start, j, 2)]

        # permute the remaining r indices that will be used for d_{rs}^{j-2t}
        r_remaining = r_indices[:start]
        r_remaining_perms = list(itertools.permutations(r_remaining))

        for p_s in perms:
            s_indices = [s_letters[p_s.index(i)] for i in range(j)]

            # indices for d_{ss}^t
            ss_pairs = [s_indices[i] + s_indices[i + 1] for i in range(start, j, 2)]

            # get the remaining s indices that will be used for d_{rs}^{j-2t}
            # no need to permute it, since it is to be combined with r_remaining
            s_remaining = s_indices[:start]

            # Create indices permutations for d_{rs}^{j-2t}
            for r_remaining_p in r_remaining_perms:
                rs_pairs = [f"{r}{s}" for r, s in zip(r_remaining_p, s_remaining)]

                # Sort it here to make it ordered according to the indices of r.
                # For example, ['bA', 'aB'] -> ['aB', 'bA']
                # But sort it or not does not matter for the creation of the delta
                # tensors.
                rs_pairs = sorted(rs_pairs)

                all_indices.append(
                    {"d_rs": rs_pairs, "d_rr": rr_pairs, "d_ss": ss_pairs}
                )

    return all_indices
