r"""Orthonormal mapping tensors.

The mappings of one weight are independent but not orthogonal, so extracting a
natural tensor takes the dual $\widetilde{\mathbf{G}}$ rather than
$\mathbf{G}$ itself. Rotating them by the inverse square root of their Gram
matrix, $\widehat{\mathbf{G}} = \mathbf{g}^{-1/2} \mathbf{G}$ (Eq. 26), removes
that distinction: an orthonormal mapping is its own dual, and the same tensor
both extracts and embeds.

This is the one place in the package where the arithmetic cannot stay exact.
The Gram matrix is rational, but its inverse square root generally is not, so
the eigendecomposition here is irreducibly numerical; everything upstream of it
is done over the rationals.

The same construction serves the symmetry-adapted mappings $\mathbf{Q}$, as
Section IV notes it must -- see docs/notation.md. That is why this is a basis
rather than a pair of entry points: `get_reduction(..., basis="orthonormal")`
returns these in place of the mappings and their duals, whether or not a
symmetry was asked for.
"""

import numpy as np
from numpy.typing import DTypeLike

from natto.algebra import simplify_linear_combination
from natto.evaluate import evaluate_tensors
from natto.symbolic import LinearCombination
from natto.utils import letter_index


def orthonormalize_mappings(
    mappings: list[LinearCombination],
    weight: int,
    rank: int,
    dtype: DTypeLike = np.float64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""Orthonormalize mapping tensors with their Cartesian Gram matrix.

    The Gram matrix is evaluated as
    $$
    g_{pq} = \frac{\mathbf{G}^p \odot^{n+\ell} \mathbf{G}^q}{2\ell + 1}
    $$
    Its unique symmetric positive-definite inverse square root transforms the input
    mappings into an orthonormal set.

    Args:
        mappings: Independent symbolic mappings from weight ``weight`` to Cartesian
            rank ``rank``.
        weight: Weight of the natural-tensor space.
        rank: Rank of the Cartesian tensor space.
        dtype: Floating-point dtype used for evaluation and eigendecomposition.

    Returns:
        Numerical input mappings, their Gram matrix, its symmetric inverse square
        root, and the orthonormal numerical mappings.

    Raises:
        ValueError: If ``mappings`` is empty or its Gram matrix is not symmetric
            positive definite.
    """
    if not mappings:
        raise ValueError("At least one mapping tensor is required")

    numerical = np.stack(
        [
            evaluate_tensors(
                simplify_linear_combination(mapping), mode="embedding", dtype=dtype
            )
            for mapping in mappings
        ]
    )
    if numerical.ndim != rank + weight + 1:
        raise ValueError("Mapping tensor ranks do not match rank and weight")
    flattened = numerical.reshape(len(mappings), -1)
    gram = flattened @ flattened.T / (2 * weight + 1)
    gram_inverse_sqrt = _symmetric_inverse_square_root(gram)
    orthonormal = np.einsum("pq,q...->p...", gram_inverse_sqrt, numerical)

    return numerical, gram, gram_inverse_sqrt, orthonormal


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


def _orthonormal_mapping_rules(weight: int, rank: int) -> tuple[str, str]:
    """Build extraction and embedding rules for a self-dual numerical mapping."""
    lower = letter_index(weight)
    upper = letter_index(rank, upper_case=True)
    extraction_rule = f"{upper}{lower},...{upper}->...{lower}"
    embedding_rule = f"{upper}{lower},...{lower}->...{upper}"

    return extraction_rule, embedding_rule
