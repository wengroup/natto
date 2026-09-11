r"""
Find linearly independent tensors using QR decomposition.

Three implementations are provided, all with the same signature and return values:

- `find_independent_tensors_qr_unpivoted`: the original scheme, reading the diagonal
  of an *unpivoted* QR factor. It is not rank-revealing and gives wrong answers in
  two situations (see its docstring); kept only for reference and regression tests.
- `find_independent_tensors_gram_schmidt`: sequential (original-order) Gram-Schmidt
  with reorthogonalization. Rank-revealing, and prefers earlier tensors in the list.
- `find_independent_tensors_scipy_qr`: `scipy.linalg.qr`, with column pivoting by
  default. Pivoted QR (`AP = QR`) is the rank-revealing factorization; with
  `pivoting=False` it reproduces the (broken) behavior of the unpivoted scheme.

Note that a rank-deficient set has more than one valid independent subset. The
Gram-Schmidt scheme keeps the earliest tensors, while pivoted QR keeps the ones with
the largest residual norms, so the two can return different (equally valid) subsets.

`find_independent_tensors` dispatches to the Gram-Schmidt scheme by default, i.e. to
the order-preserving one. Householder QR is the more accurate algorithm, but that is
not what decides it here: the coefficient matrices are small and well conditioned
(built from exact fractions upstream), so both schemes are at machine precision on
them, whereas which subset is selected has lasting consequences. The selection fixes
which duals are the canonical ones, and so the coeff = M^-1 N that is stored
alongside them and the basis any downstream natural-tensor components are expressed
in. Keeping the earliest tensors makes that a pure function of the input: the same
answer on every machine, forever. Pivoted QR instead orders the columns by residual
norm, and how LAPACK breaks ties between columns of equal norm can differ between
LAPACK builds and versions, so the canonical dual could silently change under an
unrelated BLAS upgrade.

Those three decide independence for a list of arrays. Above them sit three ways to
decide it for the *mapping tensors* of a weight, which differ in what they look at:

- `select_independent_mappings`: the exact one, and what the reduction uses. It
  works on the rational Gram matrix of the mappings, so there is no tolerance and
  nothing numerical anywhere in the decision.
- `select_independent_mappings_components`: evaluates each mapping in full, all
  $3^{n+\ell}$ components of it, and tests those vectors for independence.
- `select_independent_mappings_probe`: contracts one fixed random ICT through each
  mapping and tests the much smaller rank-$\ell$ results. This is what the
  reduction used before the exact scheme; it is the cheapest and the least direct,
  since a probe can in principle land where two independent mappings agree.

All three agree on every sector through rank six. The numerical two are kept to
cross-check the exact one, which is the only one whose answer does not depend on a
tolerance.
"""

from fractions import Fraction
from typing import Literal

import numpy as np
import scipy.linalg

from natto.algebra import simplify_linear_combination
from natto.evaluate import embed, evaluate_tensors
from natto.gram import get_gram_entry
from natto.rational import is_nonsingular
from natto.symbolic import LinearCombination
from natto.symmetric_traceless import get_random_natural_tensor

Method = Literal["gram_schmidt", "scipy_qr", "qr_unpivoted"]


def find_independent_tensors(
    tensors: list[np.ndarray], tolerance=1e-4, method: Method = "gram_schmidt"
) -> tuple[list[np.ndarray], list[int]]:
    """Find linearly independent tensors.

    Args:
        tensors: list of tensors
        tolerance: tolerance below which a tensor is considered dependent
        method: which implementation to use, one of `gram_schmidt`, `scipy_qr`, and
            `qr_unpivoted`. Only the first two are rank-revealing; `qr_unpivoted` is
            broken and provided for reference only. Of the two, only `gram_schmidt`
            preserves the order of the tensors, which is why it is the default; see
            the module docstring. Use `scipy_qr` to cross-check the rank, but note
            that it may return a different (equally valid) subset.

    Returns:
        independent_tensors: list of linearly independent tensors
        independent_indices: indices of the independent tensors in the original list
    """
    if method == "gram_schmidt":
        return find_independent_tensors_gram_schmidt(tensors, tolerance)
    elif method == "scipy_qr":
        return find_independent_tensors_scipy_qr(tensors, tolerance)
    elif method == "qr_unpivoted":
        return find_independent_tensors_qr_unpivoted(tensors, tolerance)
    else:
        raise ValueError(f"Unknown method: {method}")


def find_independent_tensors_gram_schmidt(
    tensors: list[np.ndarray], tolerance=1e-4
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
    tensors: list[np.ndarray], tolerance=1e-4, pivoting: bool = True
) -> tuple[list[np.ndarray], list[int]]:
    """Find linearly independent tensors using scipy's QR decomposition.

    The tensors are flattened into the columns of a matrix `A`, which is factorized
    as `AP = QR`. Column pivoting orders the diagonal of `R` as
    `|R_11| >= |R_22| >= ...`, so that each diagonal entry is the distance from the
    corresponding column to the span of the already-selected columns, which is what
    makes the diagonal test a valid rank test. The selected columns are the first
    `rank` entries of the permutation `P`, returned in increasing order.

    Pivoting is what makes this correct, but it also means the order of the tensors
    is not preserved: the columns are visited largest residual first, so of several
    equally valid subsets this returns the one of largest norms rather than the one
    of earliest tensors. Which columns of equal norm win is up to how LAPACK breaks
    the tie, so the subset is not guaranteed to be stable across LAPACK builds. Use
    this to cross-check the rank, which is unambiguous; prefer
    `find_independent_tensors_gram_schmidt` when the selection itself is recorded.

    Args:
        tensors: list of tensors
        tolerance: `|R_ii|` below which a tensor is considered dependent
        pivoting: whether to use column pivoting. Keep this True; without pivoting
            the diagonal test is not a rank test and the result is wrong whenever a
            zero or dependent column precedes an independent one, or whenever there
            are more tensors than components (see
            `find_independent_tensors_qr_unpivoted`).

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

    if pivoting:
        _, R, P = scipy.linalg.qr(matrix, mode="economic", pivoting=True)
        # the pivoted diagonal is non-increasing, so the rank is the count of entries
        # above the tolerance
        rank = int(np.sum(np.abs(np.diag(R)) > tolerance))
        independent_indices = sorted(int(i) for i in P[:rank])
    else:
        _, R = scipy.linalg.qr(matrix, mode="economic", pivoting=False)
        independent_indices = [
            i for i in range(min(matrix.shape)) if abs(R[i, i]) > tolerance
        ]

    independent_tensors = [tensors[i] for i in independent_indices]

    return independent_tensors, independent_indices


# TODO, this can give wrong results (see the docstring below); it is kept only so the
#  tests can pin the failure modes. Delete it once we are confident nothing depends on
#  it, and use `find_independent_tensors_scipy_qr` instead.
def find_independent_tensors_qr_unpivoted(
    tensors: list[np.ndarray], tolerance=1e-4
) -> tuple[list[np.ndarray], list[int]]:
    """Find linearly independent tensors from the diagonal of an unpivoted QR factor.

    This is the original implementation. It is NOT rank-revealing and is kept only
    for reference and regression tests -- do not use it. In unpivoted QR, `|R_ii|` is
    the distance from column `i` to the span of columns `0..i-1` only as long as no
    degenerate column precedes it, so the diagonal test fails in two ways:

    1. A zero (or dependent) column early in the list. The Householder step on a zero
       column is degenerate (an arbitrary reflector is applied), after which the later
       diagonal entries are no longer residual distances. For example, for the `j=1`
       sector of the photoelastic class `(ij)kl` the coefficient matrix has rank 3 but
       this function finds only 2 independent tensors.
    2. More tensors than components. The loop only examines `i < min(matrix.shape)`,
       so with 6 tensors of 3 components each, tensors 3-5 are never even considered
       as candidates. This is harmless only when the leading tensors happen to be
       independent.

    Args:
        tensors: list of tensors
        tolerance: tolerance for checking diagonal elements is non-zero

    Returns:
        independent_tensors: list of linearly independent tensors
        independent_indices: indices of the independent tensors in the original list
    """
    vectors = [np.asarray(t, dtype=np.float64).ravel() for t in tensors]
    matrix = np.vstack(vectors)
    _, R = np.linalg.qr(matrix.T, mode="complete")

    # Check all diagonal elements
    independent_indices = []
    for i in range(min(matrix.shape)):
        if abs(R[i, i]) > tolerance:
            independent_indices.append(i)

    independent_tensors = [tensors[i] for i in independent_indices]

    return independent_tensors, independent_indices


def select_independent_mappings(
    weight: int, rank: int, candidates: list[LinearCombination]
) -> tuple[list[int], list[list[Fraction]]]:
    r"""Select a maximal independent set of mapping tensors, exactly.

    The candidates of a weight are generally not independent: every choice
    $\mathcal{D}_p$ of which indices to contract gives one, and at rank five and
    above some are combinations of the others. Which subset is kept has lasting
    consequences, since it fixes which duals are canonical and hence the basis every
    downstream ICT is expressed in.

    The candidates are visited in order and each is kept when it is independent of
    those already kept. Independence is decided on the Gram matrix: appending a
    candidate borders the kept set's Gram matrix with its contractions against them,

    $$
    \begin{pmatrix} \mathbf{g} & \mathbf{b} \\ \mathbf{b}^{\mathsf T} & d
    \end{pmatrix},
    $$

    and that bordered matrix is nonsingular exactly when the candidate lies outside
    their span. Its Schur complement $d - \mathbf{b}^{\mathsf T} \mathbf{g}^{-1}
    \mathbf{b}$ is the squared norm of the candidate's residual, so this is the
    residual test of Gram-Schmidt carried out over the rationals: no tolerance, and
    no random tensor to probe the mappings with. The selection is therefore a pure
    function of `weight` and `rank` -- the same on every machine, in every version.

    Only the contractions against the kept set are needed, not the full
    candidate-by-candidate matrix, and the kept set's own Gram matrix accumulates as
    a by-product rather than needing a second pass.

    Args:
        weight: Weight of the ICT space.
        rank: Rank of the Cartesian tensor space.
        candidates: Candidate mapping tensors, in the order they are preferred.

    Returns:
        The indices of the kept candidates, and their exact Gram matrix.
    """
    kept: list[int] = []
    gram: list[list[Fraction]] = []

    for index, candidate in enumerate(candidates):
        border = [get_gram_entry(weight, rank, candidates[k], candidate) for k in kept]
        diagonal = get_gram_entry(weight, rank, candidate, candidate)
        bordered = [row + [b] for row, b in zip(gram, border)]
        bordered.append(border + [diagonal])

        if is_nonsingular(bordered):
            kept.append(index)
            gram = bordered

    return kept, gram


def select_independent_mappings_components(
    weight: int,
    rank: int,
    candidates: list[LinearCombination],
    tolerance: float = 1e-4,
    method: Method = "gram_schmidt",
) -> list[int]:
    r"""Select independent mapping tensors from their full numerical components.

    Each candidate is evaluated in full -- a rank-$(n + \ell)$ array, so all
    $3^{n+\ell}$ components of it -- and those arrays are tested for linear
    independence directly. Unlike `select_independent_mappings_probe` nothing is
    contracted away first, so the vectors being compared carry the whole mapping and
    no information can hide in the part that was dropped.

    This is the numerical counterpart of `select_independent_mappings`, which settles
    the same question exactly. It is kept as a cross-check: agreement between the two
    is evidence that the exact Gram matrix and the evaluated arrays describe the same
    mappings. Prefer the exact scheme for anything whose answer is recorded.

    The arrays grow as $3^{n+\ell}$, which at rank six and weight six is $3^{12}$
    components per candidate, so this costs real memory and time at high rank.

    Args:
        weight: Weight of the ICT space.
        rank: Rank of the Cartesian tensor space.
        candidates: Candidate mapping tensors, in the order they are preferred.
        tolerance: Residual norm below which a candidate is taken as dependent.
        method: Which array-level scheme decides independence; see
            `find_independent_tensors`.

    Returns:
        The indices of the independent candidates.
    """
    if not candidates:
        return []

    evaluated = [
        evaluate_tensors(simplify_linear_combination(candidate), mode="embedding")
        for candidate in candidates
    ]
    for array in evaluated:
        if array.ndim != rank + weight:
            raise ValueError(
                f"Mapping tensor has rank {array.ndim}, expected {rank + weight}"
            )

    _, independent_indices = find_independent_tensors(
        evaluated, tolerance=tolerance, method=method
    )

    return independent_indices


def select_independent_mappings_probe(
    weight: int,
    rank: int,
    candidates: list[LinearCombination],
    tolerance: float = 1e-4,
    method: Method = "gram_schmidt",
) -> list[int]:
    r"""Select independent mapping tensors by their action on one random ICT.

    One fixed-seed random ICT of the weight is contracted through each candidate,
    leaving a rank-$\ell$ tensor rather than a rank-$(n + \ell)$ one, and those are
    tested for independence. This is what the reduction used before the exact scheme,
    and it is much the cheapest of the three.

    It is also the least direct. Embedding is linear in the mapping, so dependent
    mappings always give dependent results and a dependence is never missed. The
    converse does not hold: the map $\mathbf{G} \mapsto \mathbf{G} \odot^\ell
    \mathbf{X}$ has a kernel in general, so for an unlucky probe two independent
    mappings can give dependent tensors, and a channel would be lost. That cannot
    happen with `select_independent_mappings`, which looks at the mappings
    themselves.

    Args:
        weight: Weight of the ICT space.
        rank: Rank of the Cartesian tensor space.
        candidates: Candidate mapping tensors, in the order they are preferred.
        tolerance: Residual norm below which a candidate is taken as dependent.
        method: Which array-level scheme decides independence; see
            `find_independent_tensors`.

    Returns:
        The indices of the independent candidates.
    """
    if not candidates:
        return []

    X = get_random_natural_tensor(weight)
    embedded = [embed(candidate, X) for candidate in candidates]
    _, independent_indices = find_independent_tensors(
        embedded, tolerance=tolerance, method=method
    )

    return independent_indices
