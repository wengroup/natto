"""Orthonormal mapping tensors.

The mappings of one weight are independent but not orthogonal, so extracting an ICT
takes the dual rather than the mapping itself. Rotating them by the inverse square
root of their Gram matrix removes that distinction: an orthonormal mapping is its own
dual, and the same tensor both extracts and embeds.

This is the one place in the package where the arithmetic cannot stay exact. The Gram
matrix is rational, but its inverse square root generally is not, so the
eigendecomposition here is irreducibly numerical; everything upstream of it is done
over the rationals.

The same construction serves the symmetry-adapted mappings, as Section IV of
[Wen2026] notes it must. That is why this is a basis rather than a pair of entry
points: `get_reduction(..., basis="orthonormal")` returns these in place of the
mappings and their duals, whether or not a symmetry was asked for.

References:
    Eq. 21 of [Wen2026] for the orthonormal mappings, Eq. 22 for the self-duality
    that makes them a basis.
"""

import numpy as np

from natto.algebra import simplify_linear_combination
from natto.evaluate import evaluate_tensors
from natto.indices import letter_index
from natto.symbolic import LinearCombination


def orthonormalize_mappings(
    mappings: list[LinearCombination],
    ell: int,
    n: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Orthonormalize mapping tensors with their Cartesian Gram matrix.

    The Gram matrix is the pairwise contraction of the mappings over all their indices,
    divided by 2*ell + 1. Its unique symmetric positive-definite inverse square root
    transforms the input mappings into an orthonormal set.

    Args:
        mappings: Independent symbolic mappings of this weight and n.
        ell: Weight of the ICT space.
        n: Rank of the Cartesian tensor space.

    Returns:
        The numerical input mappings, their Gram matrix, its symmetric inverse square
        root, and the orthonormal numerical mappings.

    Raises:
        ValueError: If `mappings` is empty or its Gram matrix is not symmetric positive
            definite.

    References:
        Eq. 15 of [Wen2026] for the Gram matrix, Eq. 21 for the rotation.
    """
    if not mappings:
        raise ValueError("At least one mapping tensor is required")

    numerical = np.stack(
        [
            evaluate_tensors(simplify_linear_combination(mapping), mode="embedding")
            for mapping in mappings
        ]
    )
    if numerical.ndim != n + ell + 1:
        raise ValueError(
            f"mapping tensor ranks do not match rank n={n} and weight ell={ell}"
        )
    flattened = numerical.reshape(len(mappings), -1)
    gram = flattened @ flattened.T / (2 * ell + 1)
    gram_inverse_sqrt = _symmetric_inverse_square_root(gram)
    orthonormal = np.einsum("pq,q...->p...", gram_inverse_sqrt, numerical)

    return numerical, gram, gram_inverse_sqrt, orthonormal


def get_orthonormal_entries(ell: int, n: int, G: list[LinearCombination]) -> dict:
    """Pack one weight's operators in the self-dual basis of Eq. (26).

    One array both extracts and embeds, so it appears under both keys and only the
    einsum rule tells them apart. There is no symbolic form: the inverse square root
    of a rational Gram matrix is generally irrational.

    Args:
        ell: Weight of the ICT space.
        n: Rank of the Cartesian tensor.
        G: The independent mappings of this weight.

    Returns:
        The operators of this weight, in the form the package publishes.
    """
    _, gram, gram_inverse_sqrt, G_hat = orthonormalize_mappings(G, ell, n)

    lower = letter_index(ell)
    upper = letter_index(n, upper_case=True)
    upper2 = letter_index(n, start=n, upper_case=True)

    rules = {
        "embedding": f"{upper}{lower},...{lower}->...{upper}",
        "extraction": f"{upper}{lower},...{upper}->...{lower}",
        # the decomposition is not a single array here; applying it means
        # extracting and embedding back through the same operator
        "decomposition": f"{upper}{lower},{upper2}{lower},...{upper2}->...{upper}",
    }

    # One array per channel, shared between the keys rather than copied into
    # each: that it is the same operator is the point of this basis.
    operators = list(G_hat)

    entries = {"gram": gram, "gram_inverse_sqrt": gram_inverse_sqrt}
    for key, rule in rules.items():
        entries[key] = [
            {"symbolic": None, "rule": rule, "numerical": operator}
            for operator in operators
        ]

    return entries


def _symmetric_inverse_square_root(
    matrix: np.ndarray, rtol: float = 1e-10, atol: float = 1e-12
) -> np.ndarray:
    """Compute the symmetric inverse square root of a positive-definite matrix."""
    if not np.allclose(matrix, matrix.T, rtol=rtol, atol=atol):
        raise ValueError("Gram matrix must be symmetric")

    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    threshold = atol + rtol * np.max(np.abs(eigenvalues))
    if np.any(eigenvalues <= threshold):
        raise ValueError("Gram matrix must be positive definite")

    inverse_sqrt = eigenvectors @ np.diag(1 / np.sqrt(eigenvalues)) @ eigenvectors.T

    return inverse_sqrt
