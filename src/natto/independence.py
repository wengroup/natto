"""Choosing which candidate mapping tensors to keep.

Every choice of which indices to contract gives a candidate mapping tensor, and from
rank five on some of them are combinations of the others. Exactly which subset is
kept is not a detail: it fixes which duals are the canonical ones, and so the basis
every ICT downstream is expressed in. Three schemes are offered, differing in what
they look at to decide:

- `select_independent_mappings_and_gram`: the exact one, and what the reduction
  uses. It borders the rational Gram matrix of the kept mappings with each candidate
  in turn, so no tolerance enters the decision and nothing is evaluated numerically.
- `select_independent_mappings_via_components`: evaluates each mapping in full and
  tests those vectors. It compares what the mappings *are*.
- `select_independent_mappings_via_embeddings`: contracts one fixed ICT through each
  mapping and tests the embeddings that come back, which are smaller. It compares
  what the mappings *do*, which is why it is the cheapest and the least direct. This
  is what the reduction used before the exact scheme.

The two numerical schemes hand the actual independence test to `qr`, and are kept to
cross-check the exact one. All three agree on every sector through rank six.
"""

import functools
from fractions import Fraction

import numpy as np

from natto.gram import get_gram_entry
from natto.mapping_tensors import Mapping
from natto.natural_projector import get_random_natural_tensor
from natto.qr import DEFAULT_TOLERANCE, Method, find_independent_tensors
from natto.rational import bordered_inverse


def select_independent_mappings_and_gram(
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
    tolerance, and no random tensor to probe the mappings with. The selection is
    therefore a pure function of the weight and the n, the same on every machine.

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
        Eq. 15 of [Wen2026] for the Gram matrix this decides on, and Table II for the
        number of mappings. B.1 of [Wen2026Refactor] for stopping the scan.
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


def select_independent_mappings_via_components(
    candidates: list[Mapping],
    tolerance: float = DEFAULT_TOLERANCE,
    method: Method = "gram_schmidt",
) -> list[int]:
    """Select independent mapping tensors via their numerical components.

    Each candidate is evaluated in full -- all 3^(n+ell) components of the n-(n+l)
    array -- and those arrays are tested for linear independence directly. Unlike
    `select_independent_mappings_via_embeddings` nothing is contracted away first, so the
    vectors being compared carry the whole mapping and no information can hide in the
    part that was dropped.

    With `method="scipy_qr"` this is Algorithm 1 of the paper: the flattened mappings
    are the columns of a matrix whose rank-revealing QR with column pivoting picks the
    independent ones. It is also the numerical counterpart of
    `select_independent_mappings_and_gram`, which settles the same question exactly. It is kept as a cross-check: agreement between the
    two is evidence that the exact Gram matrix and the evaluated arrays describe the same
    mappings. Prefer the exact scheme for anything whose answer is recorded.

    The arrays grow as 3^(n+ell), which at n = l = 6 is 3^12 components per candidate,
    so this costs real memory and time at high n.

    Args:
        candidates: Candidate mapping tensors of one sector, in the order they are
            preferred.
        tolerance: Residual norm below which a candidate is taken as dependent.
        method: Which array-level scheme decides independence; see
            `find_independent_tensors`.

    Returns:
        The indices of the independent candidates.

    Raises:
        ValueError: If `method` is not a known scheme, from `find_independent_tensors`.
    """
    if not candidates:
        return []

    evaluated = [c.expand().evaluate(("gct", "ict")) for c in candidates]

    _, indices = find_independent_tensors(evaluated, tolerance=tolerance, method=method)

    return indices


def select_independent_mappings_via_embeddings(
    candidates: list[Mapping],
    tolerance: float = DEFAULT_TOLERANCE,
    method: Method = "gram_schmidt",
) -> list[int]:
    """Select independent mapping tensors via the embeddings they produce.

    One fixed-seed random ICT of the weight is contracted through each candidate. That
    contracts away the mapping's ell weight indices, leaving a rank-n tensor rather
    than the n-(n+l) one the candidate itself is, so there are 3^n components to
    compare instead of 3^(n+ell). This is what the reduction used before the exact scheme, and
    it is much the cheapest of the three.

    It is also the least direct. Embedding is linear in the mapping, so dependent
    mappings always give dependent results and a dependence is never missed. The converse
    does not hold: contracting a mapping with a fixed ICT has a kernel in general, so for
    an unlucky probe two independent mappings can give dependent tensors, and a channel
    would be lost. That cannot happen with `select_independent_mappings_and_gram`, which
    looks at the mappings themselves.

    Args:
        candidates: Candidate mapping tensors of one sector, in the order they are
            preferred.
        tolerance: Residual norm below which a candidate is taken as dependent.
        method: Which array-level scheme decides independence; see
            `find_independent_tensors`.

    Returns:
        The indices of the independent candidates.

    Raises:
        ValueError: If `method` is not a known scheme, from `find_independent_tensors`.
    """
    if not candidates:
        return []

    ell = candidates[0].sector.ell
    X = get_random_natural_tensor(ell)

    # The weight axes come last, and are contracted with the ICT in order.
    embedded = [
        np.tensordot(c.expand().evaluate(("gct", "ict")), X, axes=ell)
        for c in candidates
    ]

    _, indices = find_independent_tensors(embedded, tolerance=tolerance, method=method)

    return indices


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
        Table II of [Wen2026] for the values. B.1 of [Wen2026Refactor] for the
        recursion.
    """
    if ell < 0 or ell > n:
        return 0
    if n == 0:
        return 1

    multiplicity = get_multiplicity(n - 1, ell + 1)
    if ell >= 1:
        multiplicity += get_multiplicity(n - 1, ell - 1) + get_multiplicity(n - 1, ell)

    return multiplicity
