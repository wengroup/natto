"""Mapping tensors between a Cartesian tensor and its irreducible parts.

The candidate mapping tensors, the extraction duals, and the decomposition operators
that compose the two. Everything here is symbolic and exact.

A mapping tensor is a rank-lowering tensor followed by the natural projector, so the
candidates of a weight are assembled out of `lowering` and `natural_projector`.
Which of them are independent is settled in `independence`, against the exact Gram
matrix of `gram`; `reduction` is what runs the whole pipeline and publishes the
result.

References:
    Eq. 13 of [Wen2026] for the mapping tensors, Eq. 16 for the duals, and Eq. 19
    for the decomposition operators.

    [Wen2026] M. Wen, Reusable Operators for Irreducible Cartesian Tensor
    Decomposition and Coupling, arXiv:2609.05971 (2026).
"""

from fractions import Fraction

from natto.algebra import multiply_2, simplify_linear_combination
from natto.indices import letter_index, shift_index_2
from natto.lowering import get_lowering_tensors
from natto.natural_projector import get_natural_projector
from natto.symbolic import (
    LinearCombination,
    Scalar,
)


def get_mappings(ell: int, n: int) -> list[LinearCombination]:
    """The candidate mapping tensors of a weight.

    A mapping tensor is a rank-lowering tensor followed by the natural projector, so
    there is one candidate per choice of which indices the rank lowering contracts
    away. The projector takes the letters that choice leaves unused.

    The parity of n - ell decides what the rank lowering looks like -- deltas alone,
    or deltas with one Levi-Civita symbol -- but that is `lowering`'s concern, and a
    caller supplies only the weight and rank.

    Args:
        ell: Weight of the ICT.
        n: Rank of the Cartesian tensor to map onto.

    Returns:
        One mapping tensor per choice of contracted indices.

    References:
        Eq. 13 of [Wen2026].
    """
    tensors, remaining_letters = get_lowering_tensors(ell, n)

    return [
        multiply_2(get_natural_projector(ell, s_letters=letters), F)
        for F, letters in zip(tensors, remaining_letters)
    ]


def get_extraction_operators(
    gram_inverse: list[list[Fraction]], embedding: list[LinearCombination]
) -> list[LinearCombination]:
    """Build the extraction operators dual to a set of embedding operators.

    Each dual is the combination of the embedding operators whose coefficients are the
    corresponding row of the inverse Gram matrix. Contracted with a Cartesian tensor,
    each returns the ICT of its weight and channel.

    Args:
        gram_inverse: Exact inverse of the embedding operators' Gram matrix.
        embedding: The embedding operators, in the order the matrix indexes.

    Returns:
        One extraction operator per row of `gram_inverse`.

    References:
        Eq. 16 of [Wen2026].
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
    """Get the decomposition operators of a mapping and its dual.

    Each is a mapping composed with its own dual, so contracting one with a Cartesian
    tensor gives that tensor's part of this weight and channel directly, without
    forming the ICT on the way.

    Careful with the paper's S: there it is a rank-n *tensor*, one weight's part of a
    particular T, while here it is the rank-2n *operator* that produces it. Same
    letter, one the map and one its output.

    Args:
        G: Mapping tensors.
        G_tilde: The duals, in the order of the mappings they correspond to.
        n: Rank of the Cartesian tensor.

    Returns:
        One decomposition operator per channel.

    References:
        Eq. 19 of [Wen2026].
    """
    S = []
    for G_i, dual_i in zip(G, G_tilde):
        # Shift upper letters of the dual to distinguish them from those of G
        dual_i = shift_index_2(dual_i, n, letter_index(24, upper_case=True))

        S_i = multiply_2(G_i, dual_i)
        S_i = simplify_linear_combination(S_i)

        S.append(S_i)

    return S
