"""The Gram matrix of the mapping tensors.

Contracting two mapping tensors of the same weight over all the rank indices leaves
an isotropic tensor that is symmetric and traceless within each group of weight
indices. Up to a scalar the only such tensor is the natural projector, so the
contraction collapses to a number, and that number is the Gram matrix entry.

Its rank counts the independent channels of the weight, and its inverse turns the
embedding mappings into the extraction duals of Eq. 16. Everything here is computed
with Fraction, so the entries are exact and the channel count never depends on a
tolerance.

References:
    Eq. 15 of [Wen2026] for the Gram matrix, Eq. 16 for the duals it gives.
"""

from collections import Counter
from fractions import Fraction

from natto.algebra import multiply_2, simplify_linear_combination
from natto.indices import shift_index_2
from natto.symbolic import Delta, LinearCombination, TensorProduct


def contract_mappings(
    G1: LinearCombination, G2: LinearCombination, G1_indices: str, G2_indices: str
) -> LinearCombination:
    """
    Contract two mapping tensors over the index pairs named by the callers.

    Args:
        G1: The first mapping tensor.
        G2: The second mapping tensor.
        G1_indices: Free indices of `G1`, in the order they are to be paired.
        G2_indices: Free indices of `G2`, paired against `G1_indices` in order.

    Returns:
        The contracted tensor.
    """
    contraction_delta = [Delta(i + j) for i, j in zip(G1_indices, G2_indices)]
    contraction_delta = TensorProduct(*contraction_delta)
    prod = multiply_2(G1, G2, contraction_delta)
    simplified = simplify_linear_combination(prod)

    return simplified


def get_gram_entry(
    ell: int, n: int, G_p: LinearCombination, G_q: LinearCombination
) -> Fraction:
    """Compute one entry of the exact Gram matrix of the mapping tensors.

    The two mappings are contracted over all their indices and the result divided by
    2*ell + 1. Every free index of the first is paired with the corresponding free index
    of the second; indices repeated within a single term are internal dummies and are
    contracted independently.

    Args:
        ell: Weight of the ICT space.
        n: Rank of the Cartesian tensor space.
        G_p: First mapping tensor, of rank n + ell.
        G_q: Second mapping tensor, of rank n + ell.

    Returns:
        The exact Gram matrix entry, as a Fraction.

    References:
        Eq. 15 of [Wen2026].
    """

    def get_free_indices(tensor: LinearCombination) -> str:
        free_indices = None
        for term in tensor:
            if term.factor == 0:
                continue
            counts = Counter(term.indices)
            term_free_indices = "".join(
                sorted(index for index, count in counts.items() if count == 1)
            )
            if free_indices is None:
                free_indices = term_free_indices
            elif term_free_indices != free_indices:
                raise ValueError("All terms must have the same free indices")

        return free_indices or ""

    # Give the second tensor a disjoint index set before pairing corresponding free
    # indices. Repeated indices within either tensor are internal contraction indices.
    G_q = shift_index_2(G_q, n + ell + 1)
    p_indices = get_free_indices(G_p)
    q_indices = get_free_indices(G_q)
    expected_rank = n + ell
    if len(p_indices) != expected_rank or len(q_indices) != expected_rank:
        raise ValueError(
            f"mapping tensors must have rank plus weight (n + ell) = {expected_rank}, got "
            f"{len(p_indices)} and {len(q_indices)}"
        )

    contracted = contract_mappings(G_p, G_q, p_indices, q_indices)
    if any(term.indices for term in contracted):
        raise ValueError("Full contraction left unpaired indices")

    full_contraction = sum((term.factor for term in contracted), Fraction())

    return full_contraction / (2 * ell + 1)


def get_gram_matrix(
    ell: int, n: int, all_G: list[LinearCombination]
) -> list[list[Fraction]]:
    """Compute the exact Gram matrix of the mapping tensors.

    Each entry comes from `get_gram_entry`.

    Args:
        ell: Weight of the ICT space.
        n: Rank of the Cartesian tensor space.
        all_G: Mapping tensors spanning this weight, each of rank n + ell.

    Only the lower triangle is contracted; the rest is mirrored. The Gram matrix of
    real tensors is symmetric, and each entry costs a symbolic contraction, so
    computing both halves would double the work for nothing.

    Returns:
        Symmetric matrix whose entries are exact Fraction values.

    References:
        Eq. 15 of [Wen2026].
    """
    num = len(all_G)

    matrix = [[None] * num for _ in range(num)]
    for p in range(num):
        for q in range(p + 1):
            entry = get_gram_entry(ell, n, all_G[p], all_G[q])
            matrix[p][q] = entry
            matrix[q][p] = entry

    return matrix
