r"""Mapping tensors between a Cartesian tensor and its irreducible parts.

The candidate mapping tensors $\mathbf{G}^p_{(\ell|n)}$ of Eq. (19), the extraction
duals $\widetilde{\mathbf{G}}^p_{(\ell|n)}$ of Eq. (22), and the decomposition
operators $\mathbf{S}$ of Eq. (25) that compose the two. Everything here is
symbolic and exact.

A mapping tensor is a rank-lowering tensor followed by the natural projector, so the
candidates of a weight are assembled out of `lowering` and `natural_projector`.
Which of them are independent is settled in `qr`, against the exact Gram matrix of
`gram`; `reduction` is what runs the whole pipeline and publishes the result.

References:
1. [CS70] Irreducible Cartesian Tensors. II. General Formulation, http://dx.doi.org/10.1063/1.1665190
2. [AG82] Irreducible fourth-rank Cartesian tensors, https://doi.org/10.1103/PhysRevA.25.2647
"""

from fractions import Fraction

from natto.algebra import multiply_2, simplify_linear_combination
from natto.indices import letter_index, shift_index_2
from natto.lowering import get_lowering_rules_even, get_lowering_rules_odd
from natto.natural_projector import get_natural_projector
from natto.symbolic import (
    Epsilon,
    LinearCombination,
    Scalar,
    create_delta_epsilon_tensors,
)


def get_mappings_even(j: int, n: int) -> list[LinearCombination]:
    r"""
    Mapping operator G to map minimal rank tensor subspaces j onto the space n.

    G(n|j)^q = E_j \otimes^{n-j} f_{n-j}^q.

    This is for even n-j.

    Reference: Eq. 2.4 of [AG82].

    Args:
        j: the minimal tensor subspace
        n: the space to map to

    Returns:
        A list of Tensors objects, each corresponding to a q in f_{n-j}^q.
    """

    assert (n - j) % 2 == 0, f"n-j must be even, got n={n}, j={j}"

    E_s_letters, delta_rules = get_lowering_rules_even(j, n)

    all_G = []
    for si, rule in zip(E_s_letters, delta_rules):
        E_j = get_natural_projector(j, s_letters=si)
        f_q = create_delta_epsilon_tensors(rule)
        G = multiply_2(E_j, f_q)
        all_G.append(G)

    return all_G


def get_mappings_odd(j: int, n: int) -> list[LinearCombination]:
    r"""
    Mapping operator G to map minimal rank tensor subspaces j onto the space n.


    G(n|j)^q = E_j \otimes^{n-j} f_{n-j}^q.

    This is for odd n-j.

    Reference: Eq. 2.5 of [AG82].

    Args:
        j: the minimal tensor subspace
        n: the space to map to

    Returns:
        A list of Tensors objects, each corresponding to a q in f_{n-j}^q.
    """
    assert (n - j) % 2 == 1, f"n-j must be odd, got n={n}, j={j}"

    E_s_letters, f_epsilon_rules, f_delta_rules = get_lowering_rules_odd(j, n)

    all_G = []
    for si, e_rule, d_rule in zip(E_s_letters, f_epsilon_rules, f_delta_rules):
        E_j = get_natural_projector(j, s_letters=si)
        f_q_epsilon = Epsilon(e_rule)
        f_q_delta = create_delta_epsilon_tensors(d_rule)
        G = multiply_2(E_j, f_q_epsilon, f_q_delta)
        all_G.append(G)

    return all_G


def get_extraction_operators(
    gram_inverse: list[list[Fraction]], embedding: list[LinearCombination]
) -> list[LinearCombination]:
    r"""Build the extraction operators dual to a set of embedding operators.

    The paper's Eq. (22),

    $$
    \widetilde{\mathbf{G}}^p_{(\ell|n)}
        = \sum_q (\mathbf{g}^{-1})_{pq} \mathbf{G}^q_{(\ell|n)}
    $$

    Contracted with a Cartesian tensor, each returns the natural tensor of its
    weight and channel.

    Args:
        gram_inverse: Exact inverse of the embedding operators' Gram matrix.
        embedding: The embedding operators, in the order the matrix indexes.

    Returns:
        One extraction operator per row of `gram_inverse`.
    """
    extraction = []
    for row in gram_inverse:
        terms = []
        for c, G in zip(row, embedding):
            if c:
                terms.extend(multiply_2(Scalar(c), G))
        extraction.append(LinearCombination(*terms))

    return extraction


def get_decomposition_operators(
    G: list[LinearCombination], G_tilde: list[LinearCombination], n: int
) -> list[LinearCombination]:
    r"""
    Get the decomposition operators of a mapping and its dual.

    S = G \odot^j G~

    Args:
        G: mapping tensors
        G_tilde: the duals, in the order of the mappings they correspond to.
        n: rank of the Cartesian tensor.

    Returns:
        S: one decomposition operator per channel
    """
    S = []
    for G_i, dual_i in zip(G, G_tilde):
        # Shift upper letters of the dual to distinguish them from those of G
        dual_i = shift_index_2(dual_i, n, letter_index(24, upper_case=True))

        S_i = multiply_2(G_i, dual_i)
        S_i = simplify_linear_combination(S_i)

        S.append(S_i)

    return S
