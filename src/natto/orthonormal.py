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
Section IV notes it must -- see docs/notation.md.
"""

import torch
from torch import Tensor

from natto.evaluate import evaluate_tensors
from natto.mappings import get_G_H_of_j, get_G_H_S_of_j
from natto.ops import simplify_linear_combination
from natto.symbolic import LinearCombination
from natto.utils import letter_index


def orthonormalize_mappings(
    mappings: list[LinearCombination],
    weight: int,
    rank: int,
    dtype: torch.dtype = torch.float64,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
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

    numerical = torch.stack(
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
    orthonormal = torch.einsum("pq,q...->p...", gram_inverse_sqrt, numerical)

    return numerical, gram, gram_inverse_sqrt, orthonormal


def get_orthonormal_G(n: int, dtype: torch.dtype = torch.float64) -> dict:
    r"""Get orthonormal mapping tensors for an unrestricted Cartesian tensor.

    Args:
        n: Rank of the Cartesian tensor.
        dtype: Floating-point dtype used for evaluation and eigendecomposition.

    Returns:
        A dictionary keyed by weight, with the exact embedding operators, their
        Gram matrix and its symmetric inverse square root, and the orthonormal
        operators built from them.
    """
    out = {}
    for j in range(n + 1):
        G, _, _, _ = get_G_H_of_j(j, n)
        if not G:
            continue

        G_numerical, g, g_inverse_sqrt, G_hat = orthonormalize_mappings(G, j, n, dtype)
        extraction_rule, embedding_rule = _orthonormal_mapping_rules(j, n)
        out[j] = {
            "embedding": [
                {"symbolic": str(G_p), "numerical": G_p_numerical}
                for G_p, G_p_numerical in zip(G, G_numerical)
            ],
            "gram": g,
            "gram_inverse_sqrt": g_inverse_sqrt,
            "orthonormal": [
                {
                    "numerical": G_hat_p,
                    "extraction_rule": extraction_rule,
                    "embedding_rule": embedding_rule,
                }
                for G_hat_p in G_hat
            ],
        }

    return out


def get_orthonormal_Q(
    n: int, symmetry: str = None, dtype: torch.dtype = torch.float64
) -> dict:
    r"""Get symmetry-adapted orthonormal mapping tensors.

    For each weight ``j``, the exact symmetry-adapted embedding tensors ``Q`` are
    obtained from :func:`get_G_H_S_of_j`. Their Gram matrix is evaluated as

    $$
    g_{pq} = \frac{\mathbf{Q}^p \odot^{n+\ell} \mathbf{Q}^q}{2\ell + 1}
    $$

    The returned $\widehat{\mathbf{Q}} = \mathbf{g}^{-1/2}\mathbf{Q}$ are orthonormal and can be used
    for both extraction from and embedding into the target symmetry class.

    Args:
        n: Rank of the Cartesian tensor.
        symmetry: Internal index symmetry of the Cartesian tensor. See
            :func:`get_G_H_S` for examples. If ``None``, the symmetry-free mapping
            tensors are orthonormalized.
        dtype: Floating-point dtype used for the eigendecomposition and returned
            numerical tensors.

    Returns:
        A dictionary keyed by weight, with the exact symmetry-adapted operators,
        their Gram matrix and its symmetric inverse square root, and the
        orthonormal operators built from them. Each orthonormal entry carries an
        einsum rule for extraction and one for embedding, since the same tensor
        performs both.
    """
    out = {}
    for j in range(n + 1):
        Q, _, _, _, _ = get_G_H_S_of_j(j, n, symmetry)
        if len(Q) == 0:
            continue

        Q_numerical, g, g_inverse_sqrt, Q_hat = orthonormalize_mappings(Q, j, n, dtype)

        extraction_rule, embedding_rule = _orthonormal_mapping_rules(j, n)
        out[j] = {
            "embedding": [
                {"symbolic": str(Q_p), "numerical": Q_p_numerical}
                for Q_p, Q_p_numerical in zip(Q, Q_numerical)
            ],
            "gram": g,
            "gram_inverse_sqrt": g_inverse_sqrt,
            "orthonormal": [
                {
                    "numerical": Q_hat_p,
                    "extraction_rule": extraction_rule,
                    "embedding_rule": embedding_rule,
                }
                for Q_hat_p in Q_hat
            ],
        }

    return out


def _symmetric_inverse_square_root(
    matrix: Tensor, rtol: float = 1e-10, atol: float = 1e-12
) -> Tensor:
    """Compute the symmetric inverse square root of a positive-definite matrix."""
    if not torch.allclose(matrix, matrix.T, rtol=rtol, atol=atol):
        raise ValueError("Gram matrix must be symmetric")

    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    threshold = atol + rtol * torch.max(torch.abs(eigenvalues))
    if torch.any(eigenvalues <= threshold):
        raise ValueError("Gram matrix must be positive definite")

    inverse_sqrt = eigenvectors @ torch.diag(eigenvalues.rsqrt()) @ eigenvectors.T

    return inverse_sqrt


def _orthonormal_mapping_rules(j: int, n: int) -> tuple[str, str]:
    """Build extraction and embedding rules for a self-dual numerical mapping."""
    lower = letter_index(j)
    upper = letter_index(n, upper_case=True)
    extraction_rule = f"{upper}{lower},...{upper}->...{lower}"
    embedding_rule = f"{upper}{lower},...{lower}->...{upper}"

    return extraction_rule, embedding_rule
