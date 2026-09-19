"""Orthonormal mapping tensors.

The mappings of one weight are independent but not orthogonal, so extracting an ICT
takes the dual rather than the mapping itself. Orthonormalizing them removes that
distinction: an orthonormal mapping is its own dual, and the same tensor both extracts
and embeds.

The mappings are orthonormalized in their order, as Gram-Schmidt does: the first is
normalized, the second has its part along the first removed and is normalized, and so
on. Everything but the normalization is rational, and each normalization is one
factor `1 / sqrt(d)` with `d` rational. That factor is a rational times the square
root of a square-free integer, which an `Operator` carries as its radicand, so an
orthonormal mapping is an exact `Operator` like any other.

The same construction serves the symmetry-adapted mappings, as Section IV of
[Wen2026Reusable] notes it must. That is why this is a basis rather than a pair of entry
points: `natto.reduction` returns these in place of the mappings and their duals when
asked for `basis="orthonormal"`, whether or not a symmetry was asked for.
"""

from collections.abc import Sequence
from fractions import Fraction

from natto.mapping_tensors import Mapping, combine
from natto.rational import (
    inverse_square_root,
    ldl_decomposition,
    unit_lower_triangular_inverse,
)
from natto.symbolic import Operator, Signature


def get_orthonormal_operators(
    mappings: Sequence[Mapping], gram: list[list[Fraction]], signature: Signature
) -> list[Operator]:
    """The orthonormal mappings of one weight, one per channel.

    The Gram matrix factors exactly as `g = L D L^T`, with `L` unit lower triangular
    and `D` diagonal. The rows of `L^-1` combine the mappings into orthogonal ones,
    the one of channel p with squared norm `d_p`, and dividing each by `sqrt(d_p)`
    makes it orthonormal. Channel p is thus mapping p with its parts along the
    earlier channels removed, and the orthonormal basis inherits the order of the
    mappings.

    Args:
        mappings: The independent mappings of one weight, of one sector, in order.
        gram: Their exact Gram matrix.
        signature: The signature to give the operators, which marks their input.

    Returns:
        One orthonormal operator per mapping, exact, with the root of its
        normalization as its radicand.

    References:
        Eq. 21 of [Wen2026Reusable]. Computed as in Eq. 13 (eq-ldl-orthonormal) of the
        implementation notes: the Gram-Schmidt basis in place of the symmetric one.
    """
    lower, pivots = ldl_decomposition(gram)
    rows = unit_lower_triangular_inverse(lower)
    sector = mappings[0].sector

    operators = []
    for row, pivot in zip(rows, pivots):
        scale, radicand = inverse_square_root(pivot)
        orthonormal = combine([scale * value for value in row], mappings)
        terms = sector.combine_terms(orthonormal.coefficients)
        operators.append(
            Operator(signature, [(c, term) for term, c in terms.items()], radicand)
        )

    return operators
