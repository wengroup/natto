r"""Choosing which candidate mapping tensors to keep.

Every choice of which indices to contract gives a candidate mapping tensor, and from
rank five on some of them are combinations of the others. Exactly which subset is
kept is not a detail: it fixes which duals are the canonical ones, and so the basis
every ICT downstream is expressed in. Three schemes are offered, differing in what
they look at to decide:

- `select_independent_mappings`: the exact one, and what the reduction uses. It
  borders the rational Gram matrix of the kept mappings with each candidate in turn,
  so no tolerance enters the decision and nothing is evaluated numerically.
- `select_independent_mappings_components`: evaluates each mapping in full, all
  $3^{n+\ell}$ components of it, and tests those vectors.
- `select_independent_mappings_probe`: contracts one fixed weight-$\ell$ ICT through
  each mapping, which leaves rank-$n$ tensors rather than rank-$(n+\ell)$ ones, and
  tests those. This is what the reduction used before the exact scheme; it is the
  cheapest and the least direct.

The two numerical schemes hand the actual independence test to `qr`, and are kept to
cross-check the exact one. All three agree on every sector through rank six.
"""

from fractions import Fraction

import numpy as np

from natto.algebra import simplify_linear_combination
from natto.evaluate import embed, evaluate_tensors
from natto.gram import get_gram_entry
from natto.qr import DEFAULT_TOLERANCE, Method, find_independent_tensors
from natto.rational import is_nonsingular
from natto.symbolic import LinearCombination
from natto.symmetric_traceless import get_random_natural_tensor


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
    tolerance: float = DEFAULT_TOLERANCE,
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

    Raises:
        ValueError: If `method` is not a known scheme, from
            `find_independent_tensors`.
    """

    def evaluate(candidate: LinearCombination) -> np.ndarray:
        array = evaluate_tensors(
            simplify_linear_combination(candidate), mode="embedding"
        )
        if array.ndim != rank + weight:
            raise ValueError(
                f"Mapping tensor has rank {array.ndim}, expected {rank + weight}"
            )

        return array

    return _select_numerically(candidates, evaluate, tolerance, method)


def select_independent_mappings_probe(
    weight: int,
    rank: int,
    candidates: list[LinearCombination],
    tolerance: float = DEFAULT_TOLERANCE,
    method: Method = "gram_schmidt",
) -> list[int]:
    r"""Select independent mapping tensors by their action on one random ICT.

    One fixed-seed random weight-$\ell$ ICT is contracted through each candidate.
    That contracts away the $\ell$ Greek indices of the mapping and leaves a
    rank-$n$ tensor rather than the rank-$(n + \ell)$ one the candidate itself is,
    so there are $3^n$ components to compare instead of $3^{n+\ell}$. This is what
    the reduction used before the exact scheme, and it is much the cheapest of the
    three.

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

    Raises:
        ValueError: If `method` is not a known scheme, from
            `find_independent_tensors`.
    """
    X = get_random_natural_tensor(weight)

    return _select_numerically(
        candidates, lambda candidate: embed(candidate, X), tolerance, method
    )


def _select_numerically(
    candidates: list[LinearCombination],
    to_array,
    tolerance: float,
    method: Method,
) -> list[int]:
    """Turn each candidate into an array with `to_array`, then test those.

    The two numerical schemes differ only in what they turn a candidate into, so
    everything after that is here.
    """
    if not candidates:
        return []

    _, independent_indices = find_independent_tensors(
        [to_array(candidate) for candidate in candidates],
        tolerance=tolerance,
        method=method,
    )

    return independent_indices
