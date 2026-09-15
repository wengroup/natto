"""Orthonormal mapping tensors.

The mappings of one weight are independent but not orthogonal, so extracting an ICT
takes the dual rather than the mapping itself. Rotating them by the inverse square
root of their Gram matrix removes that distinction: an orthonormal mapping is its own
dual, and the same tensor both extracts and embeds.

This is the one place in the package where the arithmetic cannot stay exact. The Gram
matrix is rational, but its inverse square root generally is not, so an
`OrthonormalOperator` carries float coefficients where an `Operator` carries exact
ones, and does nothing but evaluate.

The same construction serves the symmetry-adapted mappings, as Section IV of
[Wen2026Reusable] notes it must. That is why this is a basis rather than a pair of entry
points: `natto.reduction` returns these in place of the mappings and their duals when
asked for `basis="orthonormal"`, whether or not a symmetry was asked for.
"""

from collections.abc import Sequence
from fractions import Fraction
from types import MappingProxyType

import numpy as np

from natto.mapping_tensors import Mapping
from natto.rational import float_matrix
from natto.symbolic import Signature, Term, evaluate_terms


class OrthonormalOperator:
    """An orthonormal mapping tensor: a sum of terms with float coefficients.

    It holds what an `Operator` holds, a signature and its terms, but its coefficients
    are floats, so it offers evaluation only. Build them with
    `get_orthonormal_operators`.

    Args:
        signature: The index groups of the operator, including which is the input.
        terms: Each term with its float coefficient.

    References:
        Eq. 21 of [Wen2026Reusable].
    """

    __slots__ = ("_signature", "_terms")

    def __init__(self, signature: Signature, terms: dict[Term, float]):
        self._signature = signature
        self._terms = dict(terms)

    @property
    def signature(self) -> Signature:
        """The index groups of the operator."""
        return self._signature

    @property
    def terms(self) -> MappingProxyType:
        """The terms and their float coefficients; read-only."""
        return MappingProxyType(self._terms)

    def evaluate(self, order: Sequence[str] | None = None) -> np.ndarray:
        """Evaluate the operator into an array.

        Args:
            order: Group names in the order their axes should appear. Defaults to the
                signature's own order.

        Returns:
            An array with one axis of length 3 per slot, in the requested group order.
        """
        result = evaluate_terms(self._signature, self._terms, order)

        return result


def get_orthonormal_operators(
    mappings: Sequence[Mapping], gram: list[list[Fraction]], signature: Signature
) -> list[OrthonormalOperator]:
    """The orthonormal mappings of one weight, one per channel.

    Each is a row of the inverse square root of the Gram matrix times the mappings.
    The mappings are vectors over the candidates of one sector, so the rotation mixes
    those vectors, in floats, and each result is expanded once into its terms, as a
    dual is.

    Args:
        mappings: The independent mappings of one weight, of one sector.
        gram: Their exact Gram matrix.
        signature: The signature to give the operators, which marks their input.

    Returns:
        One orthonormal operator per mapping.

    References:
        Eq. 21 of [Wen2026Reusable].
    """
    vectors = np.array([[float(c) for c in G_q.coefficients] for G_q in mappings])
    rotated = get_inverse_square_root(gram) @ vectors
    sector = mappings[0].sector

    operators = [
        OrthonormalOperator(signature, sector.combine_terms(row.tolist()))
        for row in rotated
    ]

    return operators


def get_inverse_square_root(
    gram: list[list[Fraction]], rtol: float = 1e-10, atol: float = 1e-12
) -> np.ndarray:
    """The symmetric inverse square root of an exact Gram matrix.

    Only this step is numerical, since the square root is irrational in general.

    Args:
        gram: The exact Gram matrix of independent mappings.
        rtol: Relative tolerance of the symmetry and positive-definiteness checks.
        atol: Absolute tolerance of the same checks.

    Returns:
        The unique symmetric positive-definite inverse square root. Its rows are the
        weights of the orthonormal mappings.

    Raises:
        ValueError: If the matrix is not symmetric positive definite.

    References:
        Eq. 21 of [Wen2026Reusable].
    """
    matrix = np.array(float_matrix(gram))
    if not np.allclose(matrix, matrix.T, rtol=rtol, atol=atol):
        raise ValueError("Gram matrix must be symmetric")

    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    threshold = atol + rtol * np.max(np.abs(eigenvalues))
    if np.any(eigenvalues <= threshold):
        raise ValueError("Gram matrix must be positive definite")

    inverse_sqrt = eigenvectors @ np.diag(1 / np.sqrt(eigenvalues)) @ eigenvectors.T

    return inverse_sqrt
