import itertools

import pytest

from natto.indices import (
    double_index,
    get_permutations,
    get_permutations_2,
    get_permutations_delta,
    remove_trace_rule,
)


def test_multi_double_index():
    assert double_index(2) == ["ab", "cd"]
    assert double_index(3, start=1) == ["bc", "de", "fg"]


def test_get_permutations():
    assert get_permutations("aaaa") == [[0, 1, 2, 3]]
    assert get_permutations("aaaa", start_dim=2) == [[0, 1, 2, 3, 4, 5]]

    ref = [
        [0, 1, 2, 3, 4],
        [0, 1, 3, 2, 4],
        [0, 1, 3, 4, 2],
        [0, 3, 1, 2, 4],
        [0, 3, 1, 4, 2],
        [0, 3, 4, 1, 2],
        [3, 0, 1, 2, 4],
        [3, 0, 1, 4, 2],
        [3, 0, 4, 1, 2],
        [3, 4, 0, 1, 2],
    ]
    perms = get_permutations("aaabb")
    assert perms == ref

    perms = get_permutations("aaabb", start_dim=2)
    assert perms == [[0, 1] + [2 + i for i in sub] for sub in ref]


def test_get_permutations_2():
    perms = get_permutations_2(m=2, num_delta=1)
    assert perms == [[0, 1]]


@pytest.mark.parametrize(
    "symmetry, delta_indices",
    [("zzaabbcc", "abc"), ("abbccAABB", "AB"), ("aAAbBBc", "AB"), ("aabbc", "")],
)
def test_permutations_order(symmetry: str, delta_indices: str):
    """The patterns come in the order all permutations first meet them.

    The order decides which candidates are selected as independent, so it must not
    move.
    """
    expected, seen = [], set()
    for perm in itertools.permutations(range(len(symmetry))):
        pattern = []
        for letter in (symmetry[i] for i in perm):
            if letter in delta_indices:
                partner = [symmetry[i] for i in perm].index(letter)
                pattern.append(partner)
            else:
                pattern.append(letter)
        if tuple(pattern) not in seen:
            seen.add(tuple(pattern))
            expected.append(list(perm))

    assert get_permutations_delta(symmetry, delta_indices) == expected


def test_remove_trace_rule():
    rule = remove_trace_rule(5, 2)
    assert rule == "...aabbc,de,fg->...cdefg"
