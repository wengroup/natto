from natto.indices import (
    double_index,
    get_permutations,
    get_permutations_2,
    relabel_indices_2,
    remove_trace_rule,
)
from natto.mappings import get_dual_pair_of_weight


def test_multi_double_index():
    assert double_index(2) == ["ab", "cd"]
    assert double_index(3, start=1) == ["bc", "de", "fg"]


def test_relabel_indices_is_simultaneous():
    """A transposition of two letters must not chain into a collapse."""
    mapping, _, _, _ = get_dual_pair_of_weight(2, 4)
    swapped = relabel_indices_2(mapping[0], {"A": "B", "B": "A"})

    terms = {
        term.replace("A", "*").replace("B", "A").replace("*", "B")
        for term in str(swapped).split("  ")
    }

    assert terms == set(str(mapping[0]).split("  "))


def test_relabel_indices_leaves_other_letters_alone():
    """Letters absent from the mapping are untouched, including lower case."""
    mapping, _, _, _ = get_dual_pair_of_weight(1, 3)
    relabeled = relabel_indices_2(mapping[0], {"A": "C", "C": "A"})

    assert "δ_a" in str(relabeled)


def test_relabel_indices_round_trips():
    """Applying a permutation and its inverse returns the original."""
    mapping, _, _, _ = get_dual_pair_of_weight(2, 4)
    forward = {"A": "B", "B": "C", "C": "A"}
    backward = {new: old for old, new in forward.items()}

    there_and_back = relabel_indices_2(relabel_indices_2(mapping[0], forward), backward)

    assert str(there_and_back) == str(mapping[0])


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


def test_remove_trace_rule():
    rule = remove_trace_rule(5, 2)
    assert rule == "...aabbc,de,fg->...cdefg"
