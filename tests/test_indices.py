import itertools

import pytest

from natto.indices import get_slot_partitions


@pytest.mark.parametrize(
    "sizes, num_pairs", [([2], 3), ([2, 3], 2), ([1, 2, 2], 1), ([2, 0, 1], 2)]
)
def test_partitions_order(sizes: list[int], num_pairs: int):
    """The placements come in the order all permutations first meet them."""
    labels = [("group", g) for g, size in enumerate(sizes) for _ in range(size)]
    labels += [p for p in range(num_pairs) for _ in range(2)]
    starts = [sum(sizes[:g]) for g in range(len(sizes))]

    expected, seen = [], set()
    for perm in itertools.permutations(range(len(labels))):
        placed = [labels[i] for i in perm]
        pattern = tuple(x if isinstance(x, tuple) else placed.index(x) for x in placed)
        if pattern in seen:
            continue
        seen.add(pattern)
        slot = [perm.index(i) for i in range(len(labels))]
        groups = tuple(
            tuple(slot[start : start + size]) for start, size in zip(starts, sizes)
        )
        offset = sum(sizes)
        pairs = tuple(
            (slot[offset + 2 * p], slot[offset + 2 * p + 1]) for p in range(num_pairs)
        )
        expected.append((groups, pairs))

    assert get_slot_partitions(sizes, num_pairs) == expected
