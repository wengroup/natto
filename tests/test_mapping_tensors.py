"""Candidate mapping tensors.

Each candidate must be the natural projector applied to its rank-lowering tensor. The
reference builds both as arrays straight from the label, so it shares nothing with
how a candidate is expanded symbolically.
"""

import numpy as np
import pytest

from natto.lowering import LoweringLabel
from natto.mapping_tensors import Sector
from natto.natural_projector import get_natural_projector
from natto.utils import dij, eijk


@pytest.mark.parametrize(
    "n, ell", [(2, 0), (2, 1), (2, 2), (3, 0), (3, 1), (3, 2), (4, 1), (4, 2), (4, 3)]
)
def test_candidates(n: int, ell: int):
    """Each candidate is the projector applied to its rank-lowering tensor."""
    sector = Sector(ell, n)
    projector = get_natural_projector(ell).evaluate()

    for candidate, label in zip(sector.candidates(), sector.labels):
        expected = np.tensordot(projector, lowering_array(label, n, ell), axes=ell)

        np.testing.assert_allclose(
            candidate.expand().evaluate(),
            expected,
            atol=1e-12,
            err_msg=str(label),
        )


def test_candidate_order():
    """Candidates are ordered by delta pairs, then epsilon slots, both ascending.

    At weight one and rank four both fields vary, so ordering by the epsilon slots
    first, or keeping the enumeration order, would put delta_23 epsilon_01 first.
    """
    labels = [(label.deltas, label.epsilon) for label in Sector(1, 4).labels]

    assert labels == [
        (((0, 1),), (2, 3)),
        (((0, 2),), (1, 3)),
        (((0, 3),), (1, 2)),
        (((1, 2),), (0, 3)),
        (((1, 3),), (0, 2)),
        (((2, 3),), (0, 1)),
    ]


def lowering_array(label: LoweringLabel, n: int, ell: int) -> np.ndarray:
    """The rank-lowering tensor of a label, with its sigma axes before its rank axes.

    The free rank slots feed the sigma axes through deltas, and a Levi-Civita symbol
    paired off with two rank slots takes the last sigma axis as its tau index.
    """
    rank = [ell + slot for slot in range(n)]
    used = {slot for pair in label.deltas for slot in pair} | set(label.epsilon)
    free = [slot for slot in range(n) if slot not in used]

    operands = []
    for sigma_axis, slot in enumerate(free):
        operands += [dij(), [sigma_axis, rank[slot]]]
    for i, j in label.deltas:
        operands += [dij(), [rank[i], rank[j]]]
    if len(label.epsilon) == 2:
        operands += [eijk(), [ell - 1] + [rank[slot] for slot in label.epsilon]]
    elif label.epsilon:
        operands += [eijk(), [rank[slot] for slot in label.epsilon]]

    return np.einsum(*operands, list(range(ell + n)))
