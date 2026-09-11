r"""The Gram matrix $g_{pq}$ of the mapping tensors, Eq. (21) of the paper.

Contracting two mapping tensors of the same weight over all $n$ Roman indices
leaves a rank-$2\ell$ isotropic tensor that is symmetric and traceless within each
group of Greek indices. Up to a scalar the only such tensor is the natural
projector, so the contraction collapses to a number:

$$
\mathbf{G}^p_{(\ell|n)} \odot^n \mathbf{G}^q_{(\ell|n)} = g_{pq}\,
\mathbf{E}_{(\ell|\ell)}.
$$

That number is $g_{pq}$. Its rank counts the independent channels of the weight,
and its inverse turns the embedding mappings into the extraction duals of
Eq. (22). Everything here is computed with `Fraction`, so $g_{pq}$ is exact and
the channel count never depends on a tolerance.
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
    j: int, n: int, G_p: LinearCombination, G_q: LinearCombination
) -> Fraction:
    r"""
    Compute one entry of the exact Gram matrix of symbolic mapping tensors.

    For rank-``n``, weight-``j`` mapping tensors, this evaluates

    ``g_pq = (G_p \odot^(n+j) G_q) / (2*j + 1)``.

    Every free index of ``G_p`` is paired with the corresponding free index of
    ``G_q``. Indices repeated within an individual symbolic term are internal dummy
    indices and are contracted independently. The resulting scalar is evaluated with
    :class:`fractions.Fraction`, so the result is exact.

    Args:
        j: Weight of the natural-tensor space.
        n: Rank of the Cartesian tensor space.
        G_p: First rank-``n + j`` symbolic mapping tensor.
        G_q: Second rank-``n + j`` symbolic mapping tensor.

    Returns:
        The exact Gram-matrix entry ``g_pq``.
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
    G_q = shift_index_2(G_q, n + j + 1)
    p_indices = get_free_indices(G_p)
    q_indices = get_free_indices(G_q)
    expected_rank = n + j
    if len(p_indices) != expected_rank or len(q_indices) != expected_rank:
        raise ValueError(
            f"Mapping tensors must have rank {expected_rank}, got "
            f"{len(p_indices)} and {len(q_indices)}"
        )

    contracted = contract_mappings(G_p, G_q, p_indices, q_indices)
    if any(term.indices for term in contracted):
        raise ValueError("Full contraction left unpaired indices")

    full_contraction = sum((term.factor for term in contracted), Fraction())

    return full_contraction / (2 * j + 1)


def get_gram_matrix(
    j: int, n: int, all_G: list[LinearCombination]
) -> list[list[Fraction]]:
    r"""Compute the exact Gram matrix of symbolic mapping tensors.

    Each entry is evaluated as

    ``g_pq = (G_p \odot^(n+j) G_q) / (2*j + 1)``

    using :func:`get_gram_entry`.

    Args:
        j: Weight of the natural-tensor space.
        n: Rank of the Cartesian tensor space.
        all_G: Rank-``n + j`` symbolic mapping tensors spanning the weight-``j``
            sector.

    Only the lower triangle is contracted; the rest is mirrored. The Gram matrix of
    real tensors is symmetric, and each entry costs a symbolic contraction, so
    computing both halves would double the work for nothing.

    Returns:
        Symmetric matrix whose entries are exact :class:`fractions.Fraction` values.
    """
    num = len(all_G)

    matrix = [[None] * num for _ in range(num)]
    for p in range(num):
        for q in range(p + 1):
            entry = get_gram_entry(j, n, all_G[p], all_G[q])
            matrix[p][q] = entry
            matrix[q][p] = entry

    return matrix
