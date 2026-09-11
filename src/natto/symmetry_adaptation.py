"""Symmetry-adapted mapping tensors, by exact null space.

A Cartesian tensor with an intrinsic symmetry is spanned by fewer mappings than a
generic one of the same rank. Procedure 2 of the paper finds them: each generator of
the symmetry acts on the mapping basis through a mixing matrix, and a combination
survives the symmetry exactly when it lies in the null space of that matrix minus
the generator's sign times the identity, for every generator. The surviving
combinations are the symmetry-adapted mappings.

Both steps are exact. The mixing matrices are built by symbolic contraction and
the null space by Gaussian elimination over the rationals, so a
symmetry-adapted mapping carries no numerical tolerance -- and a multiplicity
that comes out as zero, as weight one does for the third-order elastic tensor,
is a statement rather than a threshold.

References:
    Eq. 30 of [Wen2026] for the mixing matrix, Eq. 27 for the symmetry-adapted
    mappings, and Procedure 2 for the steps.
"""

from fractions import Fraction

from natto.algebra import simplify_linear_combination
from natto.gram import get_gram_entry
from natto.indices import letter_index, relabel_indices_2
from natto.intrinsic_symmetry import parse_symmetry_generators
from natto.rational import matrix_multiply, matrix_null_space
from natto.symbolic import LinearCombination


def get_symmetry_adapted_mappings(
    ell: int,
    n: int,
    mappings: list[LinearCombination],
    gram_inverse: list[list[Fraction]],
    symmetry: str,
) -> list[LinearCombination]:
    """Solve the exact coefficient constraints imposed by internal symmetry.

    Every step is exact: the mixing matrices are built by symbolic contraction
    and the null space by Gaussian elimination over the rationals, so the
    symmetry-adapted mappings carry no numerical tolerance at all.

    Args:
        ell: Weight of the natural-tensor space.
        n: Rank of the Cartesian tensor space.
        mappings: Independent mappings of that weight.
        gram_inverse: Exact inverse of their Gram matrix.
        symmetry: Internal index symmetry of the Cartesian tensor.

    Returns:
        The symmetry-adapted mappings, one per null-space basis vector.
    """
    generators = parse_symmetry_generators(symmetry, n)
    constraints = []
    for permutation, sign in generators:
        action = get_symmetry_action_matrix(mappings, gram_inverse, ell, n, permutation)
        for row_index, row in enumerate(action):
            constraints.append(
                [
                    value - sign * Fraction(int(row_index == column_index))
                    for column_index, value in enumerate(row)
                ]
            )

    coefficients = matrix_null_space(constraints, len(mappings))
    adapted = []
    for vector in coefficients:
        mapping = sum((c * G for c, G in zip(vector, mappings)), LinearCombination())
        adapted.append(simplify_linear_combination(mapping))

    return adapted


def get_symmetry_action_matrix(
    mappings: list[LinearCombination],
    gram_inverse: list[list[Fraction]],
    ell: int,
    n: int,
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
        mappings: Independent mappings of this weight.
        gram_inverse: Exact inverse of their Gram matrix.
        ell: Weight of the natural-tensor space.
        n: Rank of the Cartesian tensor space.
        permutation: The generator, as a permutation of the Cartesian indices.

    Returns:
        The exact mixing matrix of this generator.

    Raises:
        ValueError: If the permutation does not match the Cartesian n.
    """
    if len(permutation) != n:
        raise ValueError(
            f"Symmetry permutation does not match the Cartesian rank n={n}"
        )

    # Permuting the axes of a tensor renames its indices: the slot that now
    # holds axis `permutation[k]` carries the letter that axis `k` had.
    letters = letter_index(n, upper_case=True)
    relabeling = {letters[permutation[k]]: letters[k] for k in range(n)}
    permuted = [relabel_indices_2(mapping, relabeling) for mapping in mappings]

    overlap = [
        [get_gram_entry(ell, n, mapping, image) for image in permuted]
        for mapping in mappings
    ]

    return matrix_multiply(gram_inverse, overlap)


# TODO, this can be done symbolically. Probably do it.
#  We need:
#  1. symbolic symmetrize() to get T. It is implemented in ops.py, but commented out
#  2. multiply_2() to get X = G \odot^n T
#  3. Simplify_linear_combination() to get the simplified X.
#  4. Compare X to see if they are the same.
