"""Choosing which candidate mapping tensors to keep.

Every choice of which indices to contract gives a candidate mapping tensor, and from
rank five on some of them are combinations of the others. Exactly which subset is
kept is not a detail: it fixes which duals are the canonical ones, and so the basis
every ICT downstream is expressed in. Three functions are offered. Each visits the
candidates in order and keeps one when it is not a combination of those kept before it
(the Gram-Schmidt residual test, or its QR form), so they keep the earliest independent
candidates; they differ in the arithmetic and in what they look at:

- `select_independent_mappings_gram_schmidt_exact` (selection `gram_schmidt_exact`):
  the exact one, and what the reduction uses. It borders the rational Gram matrix of
  the kept mappings with each candidate in turn, so no tolerance enters the decision
  and nothing is evaluated numerically.
- `select_independent_mappings_qr_exact` (selection `qr_exact`): factorizes the
  evaluated mappings in integer arithmetic without normalizing the columns. It makes the
  same decisions with no tolerance, but evaluates every candidate in full.
- `select_independent_mappings_gram_schmidt_numerical` and
  `select_independent_mappings_qr_numerical` (selections `gram_schmidt_numerical` and
  `qr_numerical`): evaluate each mapping in full and test those vectors in floating
  point against a tolerance, by in-order Gram-Schmidt or by a pivoted QR, which may keep
  a different subset.

The numerical schemes hand the actual independence test to `qr`, and are kept to
cross-check the exact ones.
"""

import functools
import math
from fractions import Fraction

import numpy as np

from natto.gram import get_gram_entry
from natto.mapping_tensors import Mapping
from natto.qr import DEFAULT_TOLERANCE, Method, find_independent_tensors
from natto.rational import bordered_inverse
from natto.symbolic import evaluate_terms


def select_independent_mappings_gram_schmidt_exact(
    candidates: list[Mapping],
) -> tuple[list[int], list[list[Fraction]]]:
    """Select a maximal independent set of mapping tensors, and their Gram matrix.

    The candidates of a weight are generally not independent: every choice of which
    indices to contract gives one, and at rank five and above some are combinations
    of the others. Which subset is kept has lasting consequences, since it fixes which
    duals are canonical and hence the basis every downstream ICT is expressed in.

    The candidates are visited in order and each is kept when it is independent of those
    already kept. Independence is decided on the Gram matrix: appending a candidate
    borders the kept set's Gram matrix with its contractions against them, and that
    bordered matrix is nonsingular exactly when the candidate lies outside their span.
    The Schur complement of the border is the squared norm of the candidate's residual,
    so this is the residual test of Gram-Schmidt carried out over the rationals -- no
    tolerance. The selection is therefore a pure function of the weight and the n, the
    same on every machine.

    Only the contractions against the kept set are needed, not the full
    candidate-by-candidate matrix, and the kept set's own Gram matrix accumulates as a
    by-product rather than needing a second pass. Its inverse accumulates beside it, so
    each residual is one product with the border rather than a fresh elimination.

    The number of mappings to keep is known in advance, `get_multiplicity`, so the scan
    stops once it is reached: every later candidate would be rejected. Reaching it is
    also a check that the selection found every channel of the weight.

    Args:
        candidates: All candidate mapping tensors of one sector, in the order they are
            preferred.

    Returns:
        The indices of the kept candidates, and their exact Gram matrix.

    Raises:
        RuntimeError: If the candidates span fewer mappings than the weight has.

    References:
        Eq. 15 of [Wen2026Reusable] for the Gram matrix this decides on, and Table II
        for the number of mappings. The scan stops and decides as in
        Eq. 11 (eq-multiplicity-recursion) and Eq. 12 (eq-schur-residual) of the
        implementation notes.
    """
    kept: list[int] = []
    gram: list[list[Fraction]] = []
    if not candidates:
        return kept, gram

    sector = candidates[0].sector
    multiplicity = get_multiplicity(sector.n, sector.ell)

    inverse: list[list[Fraction]] = []
    for i, c in enumerate(candidates):
        if len(kept) == multiplicity:
            break

        border = [get_gram_entry(candidates[k], c) for k in kept]
        diagonal = get_gram_entry(c, c)

        # Singular exactly when the Schur complement of the border, proportional to the
        # squared norm of the candidate's residual, is zero.
        bordered = bordered_inverse(inverse, border, diagonal)
        if bordered is None:
            continue

        kept.append(i)
        gram = [row + [b] for row, b in zip(gram, border)] + [border + [diagonal]]
        inverse = bordered

    if len(kept) != multiplicity:
        raise RuntimeError(
            f"Kept {len(kept)} mappings of weight {sector.ell} and rank {sector.n}, "
            f"expected {multiplicity}"
        )

    return kept, gram


def qr_exact_pivots(candidates: list[Mapping]) -> tuple[list[int], list[Fraction]]:
    """Factorize the evaluated candidates exactly and read the pivots.

    The candidates are the columns of a matrix `A` with rational entries. In rational
    arithmetic `A = Q U` with the columns `q_k` of `Q` mutually orthogonal, not
    normalized, and `U` unit upper triangular, so no square roots arise. A column that
    is a combination of the earlier ones gives `q_k = 0`; the others have
    `q_k . q_k != 0`, and these squared lengths are the pivots. The columns are taken
    in the given order.

    The arithmetic is done on integers. A column is multiplied by the common
    denominator of its coefficients, which changes neither the span nor whether a
    column is dependent, and its pivot is divided by the square of that factor
    afterwards. The orthogonal columns are kept as integer vectors, each known up to a
    factor: `v -> (q.q) v - (q.v) q` removes the part along `q` with no division, and
    the common content of a vector is divided out.

    Args:
        candidates: Candidate mapping tensors of one sector, in the order they are
            preferred.

    Returns:
        The indices of the columns with a nonzero pivot, and those pivots.

    Raises:
        ValueError: If an evaluated candidate is too large for int64, which it is not
            for any weight through rank six.
    """
    kept: list[int] = []
    pivots: list[Fraction] = []
    basis: list[np.ndarray] = []  # integer vectors, orthogonal to each other
    squares: list[int] = []  # their squared lengths

    for i, c in enumerate(candidates):
        operator = c.expand()
        denominator = math.lcm(
            *[Fraction(x).denominator for x in operator.terms.values()]
        )
        cleared = {t: int(Fraction(x) * denominator) for t, x in operator.terms.items()}
        a = evaluate_terms(
            operator.signature, cleared, ("gct", "ict"), dtype=int
        ).ravel()
        if np.abs(a).max() >= 2**40:
            raise ValueError("The evaluated candidate is too large for int64.")
        v = a.astype(object)

        scale = Fraction(1)  # v = scale * (the exact residual so far)
        for q, qq in zip(basis, squares):
            v = qq * v - (q @ v) * q
            scale *= qq
            if any(v):
                content = math.gcd(*[int(x) for x in v])
                v = v // content
                scale /= content

        if not any(v):
            continue

        # The residual is v / scale for the column cleared by `denominator`.
        pivot = Fraction(int(v @ v)) / scale**2 / denominator**2

        basis.append(v)
        squares.append(int(v @ v))
        kept.append(i)
        pivots.append(pivot)

    return kept, pivots


def select_independent_mappings_qr_exact(candidates: list[Mapping]) -> list[int]:
    """Select independent mapping tensors by an exact QR factorization of the arrays.

    This is Algorithm 1 of the paper: the flattened mappings are the columns of a matrix
    that is factorized in rational arithmetic, without normalizing the columns, and a
    column is independent of the earlier ones exactly when its orthogonalized column
    has a nonzero squared length. No tolerance enters. The arrays grow as 3^(n+ell),
    so this evaluates every candidate in full;
    `select_independent_mappings_gram_schmidt_exact` decides the same question from the
    Gram matrix, as the pivots of its LDL^T factorization, without the arrays, and is
    what the reduction uses.

    Args:
        candidates: Candidate mapping tensors of one sector, in the order they are
            preferred.

    Returns:
        The indices of the independent candidates.
    """
    if not candidates:
        return []

    kept, _ = qr_exact_pivots(candidates)

    return kept


def _select_numerical(
    candidates: list[Mapping], tolerance: float, method: Method
) -> list[int]:
    """Evaluate every candidate in full and test the arrays for independence."""
    if not candidates:
        return []

    evaluated = [c.expand().evaluate(("gct", "ict")) for c in candidates]

    _, indices = find_independent_tensors(evaluated, tolerance=tolerance, method=method)

    return indices


def select_independent_mappings_gram_schmidt_numerical(
    candidates: list[Mapping], tolerance: float = DEFAULT_TOLERANCE
) -> list[int]:
    """Select independent mapping tensors by an in-order Gram-Schmidt in floating point.

    Each candidate is evaluated in full -- all 3^(n+ell) components of the n-(n+l)
    array -- and kept when the norm of its residual, after projecting out the kept ones,
    exceeds `tolerance`. It is the numerical counterpart of
    `select_independent_mappings_gram_schmidt_exact` and
    `select_independent_mappings_qr_exact`, which settle the same question exactly, and
    is kept as a cross-check: agreement is evidence that the exact Gram matrix and the
    evaluated arrays describe the same mappings. Prefer the exact schemes for anything
    whose answer is recorded.

    The arrays grow as 3^(n+ell), which at n = l = 6 is 3^12 components per candidate,
    so this costs real memory and time at high n.

    Args:
        candidates: Candidate mapping tensors of one sector, in the order they are
            preferred.
        tolerance: Residual norm below which a candidate is taken as dependent.

    Returns:
        The indices of the independent candidates.
    """
    return _select_numerical(candidates, tolerance, "gram_schmidt")


def select_independent_mappings_qr_numerical(
    candidates: list[Mapping], tolerance: float = DEFAULT_TOLERANCE
) -> list[int]:
    """Select independent mapping tensors by a pivoted QR in floating point.

    The evaluated candidates are the columns of a matrix, and a column is kept when its
    diagonal entry of the pivoted QR factor exceeds `tolerance`. It keeps the same
    number as the exact schemes but may keep a different subset, since pivoting
    reorders the columns. Like `select_independent_mappings_gram_schmidt_numerical` it
    evaluates every candidate in full and is kept as a cross-check.

    Args:
        candidates: Candidate mapping tensors of one sector, in the order they are
            preferred.
        tolerance: Diagonal entry below which a column is taken as dependent.

    Returns:
        The indices of the independent candidates.
    """
    return _select_numerical(candidates, tolerance, "scipy_qr")


@functools.cache
def get_multiplicity(n: int, ell: int) -> int:
    """The number of independent mapping tensors of a weight, in a generic tensor.

    Coupling one more vector index to weight j gives weights j - 1, j and j + 1 for
    j >= 1, and weight 1 alone for j = 0, so the multiplicities of rank n follow from
    those of rank n - 1:

        N(n, ell) = N(n - 1, ell + 1) + [ell >= 1] (N(n - 1, ell - 1) + N(n - 1, ell)),

    with N(0, ell) one for ell = 0 and zero otherwise. The arithmetic is exact.

    Args:
        n: Rank of the Cartesian tensor, at least zero.
        ell: Weight of the ICT.

    Returns:
        The multiplicity, zero when `ell` is negative or above `n`.

    References:
        Table II of [Wen2026Reusable] for the values, computed by
        Eq. 11 (eq-multiplicity-recursion) of the implementation notes.
    """
    if ell < 0 or ell > n:
        return 0
    if n == 0:
        return 1

    multiplicity = get_multiplicity(n - 1, ell + 1)
    if ell >= 1:
        multiplicity += get_multiplicity(n - 1, ell - 1) + get_multiplicity(n - 1, ell)

    return multiplicity
