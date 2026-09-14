"""Index letters, and the ways of placing indices into slots.

Two kinds of bookkeeping recur often enough to live together here: naming a run of
letters for the indices of an `einsum` rule, lower-case for one group and upper-case
for another, and enumerating the distinct ways of placing groups of interchangeable
indices and pairs of indices into slots, which builds the terms of the operators.

Nothing here knows what the operators mean; it is the plumbing they share.
"""

import string
from collections.abc import Hashable, Sequence


def letter_index(n: int, start: int = 0, upper_case: bool = False) -> str:
    """
    Get a list of letters 'abc...' of length n.

    Args:
        n: the length of the letters
        start: the starting index
        upper_case: whether to use upper case letters
    """
    if upper_case:
        return string.ascii_uppercase[start : start + n]
    else:
        return string.ascii_lowercase[start : start + n]


def get_slot_partitions(
    sizes: Sequence[int], num_pairs: int
) -> list[tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, int], ...]]]:
    """The distinct ways of placing groups of interchangeable indices and pairs in slots.

    There are `sum(sizes) + 2 * num_pairs` slots. The indices of a group are
    interchangeable, and so are the two of a pair and the pairs among themselves, so a
    placement is fixed by the slots each group takes and the slots that are paired.
    These are the index patterns of products of Kronecker deltas: the pairs are the
    deltas, and each group the slots one symmetric tensor takes.

    Args:
        sizes: Number of indices in each group.
        num_pairs: Number of pairs.

    Returns:
        One `(groups, pairs)` per placement: the slots of each group, increasing, and
        the pairs of slots, each increasing and ordered by their first slot. The
        placements come in the order `itertools.permutations` first meets them, which
        decides the candidates selected as independent, so it must not change.

    Examples:
        >>> get_slot_partitions([1], 1)
        [(((0,),), ((1, 2),)), (((1,),), ((0, 2),)), (((2,),), ((0, 1),))]

    References:
        C.8 of [Wen2026Refactor].
    """
    literals = {}
    start = 0
    for group, size in enumerate(sizes):
        literals[group] = list(range(start, start + size))
        start += size
    pairs = [[start + 2 * p, start + 2 * p + 1] for p in range(num_pairs)]
    size = start + 2 * num_pairs

    patterns = []
    _extend_patterns(
        size, literals, pairs, dict.fromkeys(literals, 0), [], 0, [], patterns
    )

    partitions = []
    for perm in patterns:
        slot = [0] * size
        for position, index in enumerate(perm):
            slot[index] = position
        groups = tuple(tuple(slot[i] for i in indices) for indices in literals.values())
        paired = tuple((slot[a], slot[b]) for a, b in pairs)
        partitions.append((groups, paired))

    return partitions


def _extend_patterns(
    size: int,
    literals: dict[Hashable, list[int]],
    pairs: list[list[int]],
    used: dict[Hashable, int],
    open_pairs: list[int],
    opened: int,
    perm: list[int],
    patterns: list[list[int]],
):
    """Extend a partial pattern by every choice for its next position, recursively.

    A pattern is a permutation: position `p` takes index `perm[p]`. Each position takes
    the smallest index a choice allows: the next index of a group, the second index of
    a pair already opened, or the first index of the next pair. Trying the choices in
    increasing index order yields the patterns in the order `itertools.permutations`
    first meets them.

    Args:
        size: Number of positions.
        literals: Indices of each group, increasing.
        pairs: The two indices of each pair, ordered by the first.
        used: How many indices of each group the partial pattern has taken.
        open_pairs: Pairs whose first index is taken and second is not, in order.
        opened: How many pairs have been opened.
        perm: The partial pattern, extended and restored in place.
        patterns: Where each complete pattern is appended.
    """
    if len(perm) == size:
        patterns.append(perm[:])
        return

    choices = [
        (indices[used[group]], "group", group)
        for group, indices in literals.items()
        if used[group] < len(indices)
    ]
    choices += [(pairs[p][1], "close", p) for p in open_pairs]
    if opened < len(pairs):
        choices.append((pairs[opened][0], "open", opened))

    for index, kind, key in sorted(choices):
        perm.append(index)
        if kind == "group":
            used[key] += 1
            _extend_patterns(
                size, literals, pairs, used, open_pairs, opened, perm, patterns
            )
            used[key] -= 1
        elif kind == "close":
            position = open_pairs.index(key)
            open_pairs.pop(position)
            _extend_patterns(
                size, literals, pairs, used, open_pairs, opened, perm, patterns
            )
            open_pairs.insert(position, key)
        else:
            open_pairs.append(key)
            _extend_patterns(
                size, literals, pairs, used, open_pairs, opened + 1, perm, patterns
            )
            open_pairs.pop()
        perm.pop()
