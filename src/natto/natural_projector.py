"""The natural projector.

A rank-lowered tensor has the right rank but is in general neither symmetric nor
traceless. The natural projector is what makes it so: acting on any tensor of the
weight's rank it returns the symmetric traceless part, and acting on one already
symmetric and traceless it returns it unchanged.

It is built here symbolically and exactly, as a sum of averaged products of
Kronecker deltas with rational coefficients; `get_projector_rules` enumerates the
index patterns of each term. Being isotropic, it is the only building block the
reduction needs beyond the Levi-Civita symbol.

References:
    Eq. 7 of [Wen2026], with the coefficients of Eq. 8 and the recursion that
    builds them in Eq. S27.

    [Wen2026] M. Wen, Reusable Operators for Irreducible Cartesian Tensor
    Decomposition and Coupling, arXiv:2609.05971 (2026).
"""

import itertools
from fractions import Fraction

from natto.indices import get_permutations_2, letter_index
from natto.symbolic import LinearCombination, create_delta_epsilon_tensors


def get_natural_projector(
    ell: int, s_letters: str = None, verbose: int = 0
) -> LinearCombination:
    """The natural projector of one weight.

    Args:
        ell: Weight of the ICT the projector belongs to.
        s_letters: Letters for the upper case indices. If None, use the default:
            A, B, C, etc.
        verbose: Verbosity level for debugging.

    Returns:
        A linear combination of delta products. Lower case letters a, b, c, etc. carry
        the r indices and upper case A, B, C, etc. the s indices.

    References:
        Eq. 7 of [Wen2026], with the coefficients built by the recursion of Eq. S27.
    """
    k = ell // 2

    out = []
    c = Fraction(1, 1)  # c for t = 0
    for t in range(k + 1):
        if t > 0:
            c *= -Fraction(
                (ell - 2 * t + 2) * (ell - 2 * t + 1),
                2 * t * (2 * ell - 2 * t + 1),
            )

        # get all rules
        all_rules = get_projector_rules(ell, t, s_letters)

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
                f"@ debug E: ell={ell}, t={t}, c={c}, "
                f"num terms: {len(tmp)}, terms: {LinearCombination(*tmp)}"
            )

    return LinearCombination(*out)


def get_projector_rules(
    ell: int, t: int, s_letters: str = None
) -> list[dict[str, list[str]]]:
    """Index rules for one term of the natural projector.

    The term is a product of Kronecker deltas: l - 2t of them pairing an r index with
    an s index, t pairing r with r, and t pairing s with s.

    Args:
        ell: Weight of the ICT the projector belongs to.
        t: Number of d_rr and d_ss.
        s_letters: Letters for the upper case indices. If None, use the default:
            A, B, C, etc.

    Returns:
        One dict per term, {'d_rs': list_rs, 'd_rr': list_rr, 'd_ss': list_ss}, holding
        the indices for constructing the deltas.

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

    References:
        Eq. 7 of [Wen2026].
    """
    assert ell >= 2 * t, f"weight (ell) must be at least 2*t, got ell={ell}, t={t}"

    r_letters = letter_index(ell, upper_case=False)

    if s_letters is None:
        s_letters = letter_index(ell, upper_case=True)

    perms = get_permutations_2(ell, num_delta=t)

    # TODO, this depends on the order of the indices get_permutations_2 returns, where
    #   we put the remaining indices of t at the front, and the contracted indices at
    #   the end.
    start = ell - 2 * t

    all_indices = []
    for p_r in perms:
        r_indices = [r_letters[p_r.index(i)] for i in range(ell)]

        # indices for d_{rr}^t
        rr_pairs = [r_indices[i] + r_indices[i + 1] for i in range(start, ell, 2)]

        # permute the remaining r indices that will be used for d_{rs}^{ell-2t}
        r_remaining = r_indices[:start]
        r_remaining_perms = list(itertools.permutations(r_remaining))

        for p_s in perms:
            s_indices = [s_letters[p_s.index(i)] for i in range(ell)]

            # indices for d_{ss}^t
            ss_pairs = [s_indices[i] + s_indices[i + 1] for i in range(start, ell, 2)]

            # get the remaining s indices that will be used for d_{rs}^{ell-2t}
            # no need to permute it, since it is to be combined with r_remaining
            s_remaining = s_indices[:start]

            # Create indices permutations for d_{rs}^{ell-2t}
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
