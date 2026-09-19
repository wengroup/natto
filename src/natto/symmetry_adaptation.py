"""Symmetry-adapted mapping tensors, by exact null space.

A Cartesian tensor with an intrinsic symmetry is spanned by fewer mappings than a
generic one of the same rank. Procedure 2 of the paper finds them: each generator of
the symmetry acts on the mapping basis through a mixing matrix, and a combination
survives the symmetry exactly when it lies in the null space of that matrix minus
the generator's sign times the identity, for every generator. The surviving
combinations are the symmetry-adapted mappings.

Both steps are exact. The mixing matrices are built by permuting mappings and
reading their contractions from the label table, and the null space by Gaussian
elimination over the rationals, so a
symmetry-adapted mapping carries no numerical tolerance -- and a multiplicity
that comes out as zero, as weight one does for the third-order elastic tensor,
is a statement rather than a threshold.
"""

from fractions import Fraction

from natto.gram import get_gram_entry
from natto.intrinsic_symmetry import parse_symmetry_generators
from natto.mapping_tensors import Mapping, combine
from natto.rational import matrix_multiply, matrix_null_space


def get_symmetry_adapted_mappings(
    mappings: list[Mapping],
    gram_inverse: list[list[Fraction]],
    symmetry: str,
) -> list[Mapping]:
    """Solve the exact coefficient constraints imposed by internal symmetry.

    Every step is exact: the mixing matrices are built from permuted mappings and
    the label table, and the null space by Gaussian elimination over the rationals,
    so the symmetry-adapted mappings carry no numerical tolerance at all.

    The null space is spanned by its reduced row echelon basis, which is unique for
    the order of `mappings`.

    The symmetry-adapted mappings are ordered by the mappings they contain, compared
    in turn: the first, then the second, and so on. So G1 + G2 + G5 comes before
    G1 + G3 + G4. Only which mappings appear decides the order, not their
    coefficients, and no two contain the same set of mappings, so there are no ties.

    Args:
        mappings: Independent mappings of one sector.
        gram_inverse: Exact inverse of their Gram matrix.
        symmetry: Internal index symmetry of the Cartesian tensor.

    Returns:
        The symmetry-adapted mappings, one per null-space basis vector, in order.

    References:
        Procedure 2 of [Wen2026Reusable], with the symmetry-adapted mappings of Eq. 27.
    """
    if not mappings:
        return []

    generators = parse_symmetry_generators(symmetry, mappings[0].sector.n)
    constraints = []
    for permutation, sign in generators:
        action = get_symmetry_action_matrix(mappings, gram_inverse, permutation)
        for row_index, row in enumerate(action):
            constraints.append(
                [
                    value - sign * Fraction(int(row_index == column_index))
                    for column_index, value in enumerate(row)
                ]
            )

    coefficients = matrix_null_space(constraints, len(mappings))
    coefficients.sort(key=_support)

    return [combine(vector, mappings) for vector in coefficients]


def get_symmetry_action_matrix(
    mappings: list[Mapping],
    gram_inverse: list[list[Fraction]],
    permutation: tuple[int, ...],
) -> list[list[Fraction]]:
    """Evaluate the action of one index permutation on the mapping basis.

    If ``P G[q] = sum_p M[p, q] G[p]``, duality gives

        M[p, q] = (G_dual[p] . P G[q]) / (2*ell + 1)

    That form is not the one evaluated. A dual is a combination of all N mappings,
    so contracting one costs N times a plain contraction, and the matrix costs
    N^3. Expanding the dual moves the inverse Gram matrix outside the contraction,

        M = g_inverse O,   O[p, q] = (G[p] . P G[q]) / (2*ell + 1)

    which is the same matrix from contractions between single mappings. At rank six
    and weight three that is the difference between eighteen minutes and under one.

    Args:
        mappings: Independent mappings of one sector.
        gram_inverse: Exact inverse of their Gram matrix.
        permutation: The generator, as a permutation of the Cartesian indices.

    Returns:
        The exact mixing matrix of this generator.

    Raises:
        ValueError: If the permutation does not permute the Cartesian indices.

    References:
        Eq. 30 of [Wen2026Reusable]. Computed as in Eq. 7 (eq-mixing-matrix) of the
        implementation notes: each permuted mapping is relabelled rather than
        contracted, by Eq. 6 (eq-signed-closure), and its overlaps with the mappings are
        read from the label table.
    """
    # A permuted mapping is another mapping of the same sector, whose candidates are
    # the permuted candidates with their signs.
    permuted = [mapping.permute(permutation) for mapping in mappings]

    overlap = [
        [get_gram_entry(mapping, image) for image in permuted] for mapping in mappings
    ]

    return matrix_multiply(gram_inverse, overlap)


def _support(vector: list[Fraction]) -> tuple[int, ...]:
    """The positions of the nonzero entries of a coefficient vector, increasing.

    It is the sort key of the symmetry-adapted mappings. Each reduced row echelon
    basis vector is 1 at its own free column and 0 at the others', so no two vectors
    share a support and the order has no ties.
    """
    support = tuple(q for q, value in enumerate(vector) if value)

    return support
