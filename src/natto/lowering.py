"""Rank-lowering tensors.

To reach an ICT of some weight from a generic tensor of higher rank, the surplus
indices must first be contracted away in pairs against Kronecker deltas and, when
the parity of n - ell requires it, one Levi-Civita symbol. Which indices
are paired is a choice, and each choice leaves different information behind -- which
is exactly why a single tensor can carry several ICTs of the same weight.

A rank-lowering tensor is given by its label: the pairs of rank indices its deltas
join, and the rank indices on its Levi-Civita symbol if it has one. The rank indices
it leaves free, with the tau index of the symbol, are what the natural projector
takes; `mapping_tensors` is where they are put to work.

The labels of a weight are ordered by their delta pairs, then their epsilon slots,
both ascending. The order is part of the result, not of the enumeration: the
independent mappings are the earliest independent candidates in it, so it fixes
which mappings a weight keeps and how its channels are numbered.
"""

import itertools
from collections.abc import Sequence
from dataclasses import dataclass

from natto.indices import get_slot_partitions
from natto.symbolic import sort_with_sign


@dataclass(frozen=True, order=True)
class LoweringLabel:
    """A rank-lowering tensor, as the blocks of rank slots it contracts.

    Labels can be sorted, and their order is the order of the candidates; see
    `get_lowering_labels`. What makes them sortable is `order=True` on the dataclass,
    which generates `<`, `<=`, `>` and `>=` comparing the fields as a tuple in the
    order they are declared: `(deltas, epsilon)`. Tuples compare element by element,
    into nested tuples, so labels are ordered by their first delta pair, then their
    second, and so on, and by `epsilon` only when all the deltas agree. The fields
    are canonical -- each pair increasing, the pairs sorted, the epsilon slots
    increasing -- so equal tensors compare equal. Reordering the two fields would
    change the order of the candidates, and so the mappings a weight keeps.

    Examples:
        The deltas decide first:

        >>> LoweringLabel(((0, 1), (2, 3))) < LoweringLabel(((0, 2), (1, 3)))
        True

        The epsilon slots break a tie in the deltas:

        >>> LoweringLabel(((0, 1),), (2, 3)) < LoweringLabel(((0, 1),), (2, 4))
        True

        So sorting the candidates of weight 3 and rank 4, which have no deltas, orders
        them by their epsilon slots:

        >>> labels = [LoweringLabel((), e) for e in [(1, 2), (0, 3), (0, 1)]]
        >>> [label.epsilon for label in sorted(labels)]
        [(0, 1), (0, 3), (1, 2)]

    Attributes:
        deltas: The pairs of rank slots its Kronecker deltas join, each increasing and
            all sorted.
        epsilon: The rank slots on its Levi-Civita symbol, increasing. Empty when
            n - ell is even; a pair `(i, j)`, standing for the symbol with the tau index
            first, when n - ell is odd and ell >= 1; a triple when n - ell is odd and
            ell = 0.
    """

    deltas: tuple[tuple[int, int], ...]
    epsilon: tuple[int, ...] = ()

    def free_slots(self, n: int) -> tuple[int, ...]:
        """The rank slots the tensor leaves to the natural projector, increasing."""
        used = {slot for pair in self.deltas for slot in pair} | set(self.epsilon)

        return tuple(slot for slot in range(n) if slot not in used)

    def relabel(self, new_slot: Sequence[int]) -> tuple[int, "LoweringLabel"]:
        """Rename every rank slot `s` to `new_slot[s]`.

        Args:
            new_slot: The new slot of each rank slot, a permutation of `range(n)`.

        Returns:
            The sign of the renamed tensor relative to the canonical label, and the
            label.
        """
        deltas = sorted(
            tuple(sorted((new_slot[i], new_slot[j]))) for i, j in self.deltas
        )
        epsilon, sign = sort_with_sign([new_slot[slot] for slot in self.epsilon])

        return sign, LoweringLabel(tuple(deltas), epsilon)


def get_lowering_labels(ell: int, n: int) -> list[LoweringLabel]:
    """The rank-lowering tensors of a weight, whichever parity applies.

    Which form the rank-lowering tensor takes depends on the parity of n - ell:
    deltas alone when it is even, and one Levi-Civita symbol alongside them when it
    is odd. This dispatches on that, so a caller supplies only the weight and rank.

    The labels are sorted by their delta pairs, then their epsilon slots, both
    ascending -- the comparison `order=True` gives `LoweringLabel`, which `sorted`
    uses -- so the first contracts the earliest slots: at ell = 2, n = 4 they are
    delta_01, delta_02, delta_03, delta_12, delta_13, delta_23. The same rule serves
    both parities, the epsilon slots being only the second key. Since the independent
    mappings are the earliest independent candidates, this order is what decides
    which are kept; the order the labels are enumerated in plays no part.

    Args:
        ell: Weight of the ICT.
        n: Rank of the Cartesian tensor.

    Returns:
        The rank-lowering tensors, one per choice of contracted indices, in order.

    References:
        Eq. 2 of [Wen2026Reusable], with Eq. 3 for even n - ell and Eq. 5 for odd;
        Sec. II C for the discussion.
    """
    if (n - ell) % 2 == 0:
        labels = get_lowering_labels_even(ell, n)
    else:
        labels = get_lowering_labels_odd(ell, n)

    return sorted(labels)


def get_lowering_labels_even(ell: int, n: int) -> list[LoweringLabel]:
    """The rank-lowering tensors of even n - ell.

    With the parity even the rank-lowering tensor is built from Kronecker deltas alone,
    (n - ell) / 2 of them, pairing off the surplus indices.

    Args:
        ell: Weight of the ICT.
        n: Rank of the Cartesian tensor.

    Returns:
        The rank-lowering tensors, one per choice of contracted indices.

    References:
        Eq. 3 of [Wen2026Reusable].
    """
    labels = []
    for _, pairs in get_slot_partitions([ell], (n - ell) // 2):
        labels.append(LoweringLabel(pairs))

    return labels


def get_lowering_labels_odd(ell: int, n: int) -> list[LoweringLabel]:
    """The rank-lowering tensors of odd n - ell.

    With the parity odd, one Levi-Civita symbol is needed alongside the Kronecker
    deltas. Its tau index is the one the natural projector takes; its other two
    indices are chosen from those the deltas leave.

    Args:
        ell: Weight of the ICT.
        n: Rank of the Cartesian tensor.

    Returns:
        The rank-lowering tensors, one per choice of contracted indices.

    References:
        Eq. 5 of [Wen2026Reusable].
    """
    if ell == 0:
        return get_lowering_labels_odd_weight_zero(ell, n)

    labels = []
    for (free,), pairs in get_slot_partitions([ell + 1], (n - ell - 1) // 2):
        for epsilon in itertools.combinations(free, 2):
            labels.append(LoweringLabel(pairs, epsilon))

    return labels


def get_lowering_labels_odd_weight_zero(ell: int, n: int) -> list[LoweringLabel]:
    """The rank-lowering tensors of weight zero and odd n.

    Every index the deltas leave goes to the Levi-Civita symbol, which then contracts
    three rank indices and has no tau index.

    Args:
        ell: Weight of the ICT, which must be zero.
        n: Rank of the Cartesian tensor, odd and at least three.

    Returns:
        The rank-lowering tensors, one per choice of contracted indices.

    Raises:
        ValueError: If `ell` is not zero, or `n` is not odd and at least three.
    """
    if ell != 0:
        raise ValueError(f"weight (ell) must be 0, got ell={ell}")
    if n % 2 != 1:
        raise ValueError(f"rank (n) must be odd, got n={n}")
    if n < 3:
        raise ValueError(f"rank (n) must be at least 3, got n={n}")

    labels = []
    for (epsilon,), pairs in get_slot_partitions([3], (n - 3) // 2):
        labels.append(LoweringLabel(pairs, epsilon))

    return labels
