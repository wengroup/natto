"""The Gram matrix of the mapping tensors.

Contracting two mapping tensors of the same weight over all the rank indices leaves
an isotropic tensor that is symmetric and traceless within each group of weight
indices. Up to a scalar the only such tensor is the natural projector, so the
contraction collapses to a number, and that number is the Gram matrix entry.

Its rank counts the independent channels of the weight, and its inverse turns the
embedding mappings into the extraction duals of Eq. 16. Everything here is computed
with Fraction, so the entries are exact and the channel count never depends on a
tolerance.
"""

from fractions import Fraction

from natto.mapping_tensors import Mapping


def get_gram_entry(G_p: Mapping, G_q: Mapping) -> Fraction:
    """Compute one entry of the exact Gram matrix of the mapping tensors.

    The two mappings are contracted over all their indices and the result divided by
    2*ell + 1. Both are combinations of the same candidates, so by bilinearity the
    contraction is the sector's label table weighted by their coefficients.

    Args:
        G_p: First mapping.
        G_q: Second mapping, of the same sector.

    Returns:
        The exact Gram matrix entry, as a Fraction.

    Raises:
        ValueError: If the mappings belong to different sectors.

    References:
        Eq. 15 of [Wen2026]. Computed as in A.1 of [Wen2026Refactor]: one natural
        projector is left out by Lemma 1, and the entry is read from the label table
        by Proposition 5 (C.4).
    """
    sector = G_p.sector
    if G_q.sector is not sector:
        raise ValueError("Mappings of different sectors have no Gram entry")

    q_terms = [(j, c) for j, c in enumerate(G_q.coefficients) if c]
    total = Fraction()
    for i, c_i in enumerate(G_p.coefficients):
        if c_i:
            for j, c_j in q_terms:
                total += c_i * c_j * sector.table(i, j)

    return total / (2 * sector.ell + 1)


def get_gram_matrix(mappings: list[Mapping]) -> list[list[Fraction]]:
    """Compute the exact Gram matrix of the mapping tensors.

    Each entry comes from `get_gram_entry`. Only the lower triangle is computed; the
    Gram matrix of real tensors is symmetric, so the rest is mirrored.

    Args:
        mappings: Mappings of one sector.

    Returns:
        Symmetric matrix whose entries are exact Fraction values.

    References:
        Eq. 15 of [Wen2026], computed entry by entry as in A.1 of [Wen2026Refactor].
    """
    num = len(mappings)

    matrix = [[None] * num for _ in range(num)]
    for p in range(num):
        for q in range(p + 1):
            entry = get_gram_entry(mappings[p], mappings[q])
            matrix[p][q] = entry
            matrix[q][p] = entry

    return matrix
