r"""
Find linearly independent tensors using QR decomposition.

Two implementations are provided, with the same signature and return values:

- `find_independent_tensors_gram_schmidt`: sequential (original-order) Gram-Schmidt
  with reorthogonalization. Rank-revealing, and prefers earlier tensors in the list.
- `find_independent_tensors_scipy_qr`: `scipy.linalg.qr` with column pivoting, which
  is what makes `AP = QR` the rank-revealing factorization. There is no way to turn
  the pivoting off: the unpivoted diagonal is not a rank test, and reading it was the
  original bug here -- it reported rank 2 for the weight-one sector of the
  photoelastic class, which has rank 3, and it never looked past
  `min(matrix.shape)` columns at all. Both mistakes lose an ICT channel.

Note that a rank-deficient set has more than one valid independent subset. The
Gram-Schmidt scheme keeps the earliest tensors, while pivoted QR keeps the ones with
the largest residual norms, so the two can return different (equally valid) subsets.

`find_independent_tensors` dispatches to the Gram-Schmidt scheme by default, i.e. to
the order-preserving one. Householder QR is the more accurate algorithm, but that is
not what decides it here: the coefficient matrices are small and well conditioned
(built from exact fractions upstream), so both are at machine precision on them,
whereas which subset is selected has lasting consequences. The selection fixes
which duals are the canonical ones, and so the coeff = M^-1 N that is stored
alongside them and the basis any downstream natural-tensor components are expressed
in. Keeping the earliest tensors makes that a pure function of the input: the same
answer on every machine, forever. Pivoted QR instead orders the columns by residual
norm, and how LAPACK breaks ties between columns of equal norm can differ between
LAPACK builds and versions, so the canonical dual could silently change under an
unrelated BLAS upgrade.

These decide independence for a plain list of arrays, and know nothing about
mapping tensors. `independence` is what applies them to the candidates of a weight,
and what offers the exact alternative to both.
"""

from typing import Literal

import numpy as np
import scipy.linalg

Method = Literal["gram_schmidt", "scipy_qr"]

#: Residual norm below which a tensor counts as dependent on the ones before it. The
#: coefficient matrices here come from exact fractions upstream and are small and
#: well conditioned, so the gap between a genuine residual and rounding noise is
#: many orders wide and nothing hinges on the exact value.
DEFAULT_TOLERANCE = 1e-4


def find_independent_tensors(
    tensors: list[np.ndarray],
    tolerance: float = DEFAULT_TOLERANCE,
    method: Method = "gram_schmidt",
) -> tuple[list[np.ndarray], list[int]]:
    """Find linearly independent tensors.

    Args:
        tensors: list of tensors
        tolerance: tolerance below which a tensor is considered dependent
        method: which implementation to use, `gram_schmidt` or `scipy_qr`. Both are
            rank-revealing, so they always agree on how many tensors are
            independent. Only `gram_schmidt` preserves the order of the tensors,
            which is why it is the default; see the module docstring. Use `scipy_qr`
            to cross-check the rank, but note that it may return a different
            (equally valid) subset.

    Returns:
        independent_tensors: list of linearly independent tensors
        independent_indices: indices of the independent tensors in the original list

    Raises:
        ValueError: If `method` is not one of the two.
    """
    if method == "gram_schmidt":
        return find_independent_tensors_gram_schmidt(tensors, tolerance)
    elif method == "scipy_qr":
        return find_independent_tensors_scipy_qr(tensors, tolerance)
    else:
        raise ValueError(
            f"Unknown method: {method}. Supported are: gram_schmidt, scipy_qr."
        )


def find_independent_tensors_gram_schmidt(
    tensors: list[np.ndarray], tolerance: float = DEFAULT_TOLERANCE
) -> tuple[list[np.ndarray], list[int]]:
    """Find linearly independent tensors by sequential Gram-Schmidt, rank-revealing.

    A tensor is kept if its residual, after projecting out the span of the
    tensors already kept, has norm larger than `tolerance`; earlier tensors in
    the list are preferred. This is modified Gram-Schmidt with a second
    orthogonalization pass. Unlike reading the diagonal of an unpivoted QR factor,
    it remains correct when a zero or dependent tensor appears before an
    independent one.

    The tensors are visited in the order given, so of several equally valid subsets
    this returns the one made of the earliest tensors. That makes the selection
    deterministic and platform independent, which is why this is the scheme
    `find_independent_tensors` uses by default; see the module docstring.

    Args:
        tensors: list of tensors
        tolerance: residual norm below which a tensor is considered dependent

    Returns:
        independent_tensors: list of linearly independent tensors
        independent_indices: indices of the independent tensors in the original list
    """
    vectors = [np.asarray(t, dtype=np.float64).ravel() for t in tensors]

    independent_indices = []
    basis: list[np.ndarray] = []
    for i, v in enumerate(vectors):
        r = v.copy()
        # Project out the current basis twice for numerical stability
        # (modified Gram-Schmidt with reorthogonalization).
        for _ in range(2):
            for q in basis:
                r = r - (r @ q) * q
        norm = np.linalg.norm(r)
        if norm > tolerance:
            independent_indices.append(i)
            basis.append(r / norm)

    independent_tensors = [tensors[i] for i in independent_indices]

    return independent_tensors, independent_indices


def find_independent_tensors_scipy_qr(
    tensors: list[np.ndarray], tolerance: float = DEFAULT_TOLERANCE
) -> tuple[list[np.ndarray], list[int]]:
    """Find linearly independent tensors using scipy's QR decomposition.

    The tensors are flattened into the columns of a matrix `A`, which is factorized
    as `AP = QR`. Column pivoting orders the diagonal of `R` as
    `|R_11| >= |R_22| >= ...`, so that each diagonal entry is the distance from the
    corresponding column to the span of the already-selected columns, which is what
    makes the diagonal test a valid rank test. The selected columns are the first
    `rank` entries of the permutation `P`, returned in increasing order.

    Pivoting is not optional, and there is no flag to turn it off. Without it
    `|R_ii|` is the distance from column `i` to the span of the columns before it
    only while no degenerate column precedes it, so the diagonal stops being a rank
    test the moment one does -- the Householder step on a zero column applies an
    arbitrary reflector. Unpivoted, the factorization also only reaches
    `min(matrix.shape)` columns, so trailing tensors are never considered at all.
    Either way the rank comes out too small, which in a reduction means an ICT
    channel silently disappears.

    Pivoting does mean the order of the tensors is not preserved: the columns are
    visited largest residual first, so of several equally valid subsets this returns
    the one of largest norms rather than the one of earliest tensors. Which columns
    of equal norm win is up to how LAPACK breaks the tie, so the subset is not
    guaranteed to be stable across LAPACK builds. Use this to cross-check the rank,
    which is unambiguous; prefer `find_independent_tensors_gram_schmidt` when the
    selection itself is recorded.

    Args:
        tensors: list of tensors
        tolerance: `|R_ii|` below which a tensor is considered dependent

    Returns:
        independent_tensors: list of linearly independent tensors
        independent_indices: indices of the independent tensors in the original list
    """
    if not tensors:
        return [], []

    # each column of the matrix is a flattened tensor
    matrix = np.stack(
        [np.asarray(t, dtype=np.float64).ravel() for t in tensors], axis=1
    )

    # `pivoting` must stay True. Without it the diagonal of R is not a rank test at
    # all, and this returns too small a rank -- which in a reduction means a missing
    # ICT channel. See the module docstring for the two ways it goes wrong.
    _, R, P = scipy.linalg.qr(matrix, mode="economic", pivoting=True)
    # the pivoted diagonal is non-increasing, so the rank is the count of entries
    # above the tolerance
    rank = int(np.sum(np.abs(np.diag(R)) > tolerance))
    # `P[:rank]` is in pivot order, largest residual first, so *which* columns these
    # are was decided by norm and not by position. Sorting only makes the return
    # value stable; it does not make the selection order-preserving, and on a
    # rank-deficient input this is generally a different subset from the one
    # `find_independent_tensors_gram_schmidt` returns. Both are valid.
    independent_indices = sorted(int(i) for i in P[:rank])

    independent_tensors = [tensors[i] for i in independent_indices]

    return independent_tensors, independent_indices
