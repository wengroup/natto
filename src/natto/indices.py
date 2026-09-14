"""Index letters, and the permutations built from them.

Two kinds of bookkeeping recur often enough to live together here: naming a fresh run
of letters for the indices of a tensor, lower-case for one group and upper-case for
another, and enumerating the index permutations that symmetrize a tensor or pair its
indices off against deltas, written out as `einsum` rules.

Nothing here knows what the operators mean; it is the plumbing they share.
"""

import string


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


def double_index(n: int, start: int = 0, upper_case: bool = False) -> list[str]:
    """
    Get multiple double indices, like ['ab', 'cd', 'ef'].

    Args:
        n: the number of double indices
        start: the starting index
        upper_case: whether to use upper case letters

    Examples:
        >>> double_index(2)
        ['ab', 'cd']
        >>> double_index(3, start=1)
        ['bc', 'cd', 'de']
    """
    indices = letter_index(2 * n, start, upper_case)
    return [indices[i : i + 2] for i in range(0, 2 * n, 2)]


def repeat_double_index(n: int, start: int = 0, upper_case: bool = False) -> list[str]:
    """
    Get multiple repeated double indices, like ['aa', 'bb', 'cc'].

    Args:
        n: the number of double indices
        start: the starting index
        upper_case: whether to use upper case letters

    Examples:
        >>> repeat_double_index(2)
        ['aa', 'bb']
        >>> repeat_double_index(3, start=1)
        ['bb', 'cc', 'dd']
    """
    indices = letter_index(n, start, upper_case)

    return [s + s for s in indices]


def get_permutations(symmetry: str, start_dim: int = 0) -> list[list[int]]:
    """
    Get the unique permutations of the indices to fully symmetrize a tensor.

    This works for the case where part or all of the indices are symmetric.

    Args:
        symmetry: A string that describes the symmetry already in the tensor. For
            example, `abba` means the first and the fourth indices are symmetric, and
            the second and the third indices are symmetric. The symmetry only applies to
            the indices after the `start_dim`.
        start_dim: the starting dimension to perform the operation. Dimensions before
            `start_dim` will not be used in the operation.

    Example:
        >>> get_permutations('abba')
        [[0, 1, 2, 3],  # abba
         [0, 1, 3, 2],  # abab
         [0, 3, 1, 2],  # aabb
         [1, 0, 2, 3],  # baba
         [1, 0, 3, 2],  # baab
         [1, 2, 0, 3]]  # bbaa
        >>> get_permutations('abba', 2)
        [[0, 1, 2, 3, 4, 5],  # abba
         [0, 1, 2, 3, 5, 4],  # abab
         [0, 1, 2, 5, 3, 4],  # aabb
         [0, 1, 3, 2, 4, 5],  # baba
         [0, 1, 3, 2, 5, 4],  # baab
         [0, 1, 3, 4, 2, 5]]  # bbaa

    Returns:
        Each tuple contains the permutation indices for symmetrization.
    """
    return get_permutations_delta(symmetry, "", start_dim)


def get_permutations_2(m: int, num_delta: int, start_dim: int = 0) -> list[list[int]]:
    """

    Get the unique permutations of the tensor product of a symmetric tensor and deltas.

    For example, we know
    {U_rrss delta_ij delta_kl}
    = U_rrss delta_ij delta_kl
    + U_rsrs delta_ij delta_kl
    + U_rssr delta_ij delta_kl

    This is equivalent to
    1. First get V_ijkl = U_rrss delta_ij delta_kl
    2. Then permute V_ijkl to get V_ikjl and V_iklj
    3. Sum them up to get the result, i.e.
        {U_rrss delta_ij delta_kl} = V_ijkl + V_ikjl + V_iklj


    This function find the permutations of the indices in V.
    There are two types of symmetry to consider in the permutations:
    a. Minor symmetry: the symmetry of the two indices in each delta tensor.
       For example, V_ijkl = V_ijlk = V_jikl = V_jilk
    b. Major symmetry: the symmetry of indices between the deltas. For example,
       V_ijkl = V_klij

    In addition, we consider another symmetry:
    c. The symmetry of the remaining indices of the tensor, e.g. in U_rrstdelta_ij,
        the indices r and s are symmetric.

    Args:
        m: the rank of the symmetric tensor
        num_delta: the number of delta tensors to be contracted
        start_dim: the starting dimension to perform the operation. Dimensions before
            `start_dim` will not be used in the operation.

    Returns:
        Each inner list contains the permutation indices for symmetrization.
    """

    num_remain = m - 2 * num_delta
    assert num_remain >= 0, "The number of remaining indices must be non-negative."

    # Construct the symmetry pattern, e.g., zzaabb
    u_remain = "z" * num_remain
    delta = "".join(repeat_double_index(num_delta))
    symmetry = f"{u_remain}{delta}"

    delta_indices = letter_index(num_delta)

    perms = get_permutations_delta(symmetry, delta_indices, start_dim)

    return perms


# TODO, we can rename this to make it general.
#  We can call this minor and major symmetries. just like elastic tensor.
#  T delta_ij delta_kl = T delta_kl delta_ij we have minor symmetry between i and j,
#  and between k and l. We have major symmetry between (ij) and (kl).
#  This is the same as the elastic tensor ((ij)(kl)).
#  So, we can rename this function to make it more general.


def get_permutations_delta(
    symmetry: str, delta_indices: str, start_dim: int = 0
) -> list[list[int]]:
    """
    Get the permutations that symmetrize a tensor built with delta tensors.

    For example, `symmetry = xxyyaabb`, and `delta_indices = ab` means:
        1. indices 1 and 2 are symmetric (both associated with `x`), and indices 3 and
           4 are symmetric (both associated with `y`)
        2. indices 5 and 6 are symmetric (both associated with `a`), and indices 7 and
           8 are symmetric (both associated with `b`)
        3. indices pairs (5, 6) and (7, 8) are symmetric because both `a` and `b` are in
           `delta_indices`, meaning they are associated with delta tensors.

    So, we consider three types of symmetries:
        1. symmetries in other indices of the tensor
        2. minor symmetry in delta tensors (i.e. symmetry in the indices of each delta)
        3. major symmetry in delta tensors (i.e. symmetry between different deltas)

    For 2 and 3, consider, for example:
        delta_ij delta_kl = delta_ji delta_lk = delta_kl delta_ij,
    So, we have both minor and major symmetries in delta tensors.

    The above example can be though as symmetrizing a tensor Z obtained as:
        Z = X (x) Y (x) delta (x) delta
    where `X` and `Y` are rank-2 symmetric tensors, and `delta` are delta tensors.

    As another example, consider `symmetry = xxxxaabb` and `delta_indices = ab`, then
    it can be thought as symmetrizing a tensor Z obtained as:
        Z = X (x) delta (x) delta (x) delta
    where `X` is a rank-4 symmetric tensor, and `delta` are delta tensors.

    As can be seen, `delta_indices` does not necessarily need to be associated with
    deltas. It can be indices of the same symmetric tensor. For example, the permutation
    of
        Z = X (x) X (x) Y (x) Y
    where `X` and `Y` are rank-2 symmetric tensors,
    can be obtained using `symmetry = xxaabb` and `delta_indices = ab`.


    Args:
        symmetry: A string that describes the symmetry already in the tensor. This
            should be used together with `delta_indices`. See below.
        delta_indices: The indices that are associated with the delta tensors, for which
            need to consider both minor and major symmetries.
        start_dim: the starting dimension to perform the operation. Dimensions before
            `start_dim` will not be used in the operation.

    Returns:
        Each inner tuple contains the permutation indices for symmetrization, each the
        first of its pattern among `itertools.permutations`, and in that order.

    Raises:
        ValueError: If a letter of `delta_indices` does not occur exactly twice.

    References:
        C.8 (Sec. 6.8) of [Wen2026Refactor].
    """
    literals: dict[str, list[int]] = {}
    pair_slots: dict[str, list[int]] = {}
    for index, letter in enumerate(symmetry):
        group = pair_slots if letter in delta_indices else literals
        group.setdefault(letter, []).append(index)
    if any(len(slots) != 2 for slots in pair_slots.values()):
        raise ValueError(f"Each of {delta_indices} must occur twice in {symmetry}")
    pairs = sorted(pair_slots.values())

    patterns = []
    _extend_patterns(
        len(symmetry), literals, pairs, dict.fromkeys(literals, 0), [], 0, [], patterns
    )
    prefix = list(range(start_dim))
    perms = [prefix + [start_dim + index for index in pattern] for pattern in patterns]

    return perms


def remove_trace_rule(m: int, d: int) -> str:
    """
    Get the contraction rule to remove the trace of a symmetric tensor.

    Note, d <= m/2.

    Args:
        m: rank of the symmetric tensor
        d: the number of delta

    Returns:
        rule: the contraction rule
        symmetry: the symmetry of the indices
    """
    u_contracted = "".join(repeat_double_index(d))
    u_remain = letter_index(m - 2 * d, start=d)
    delta = double_index(d, start=m - d)

    return (
        f"...{u_contracted}{u_remain},{','.join(delta)}->...{u_remain}{''.join(delta)}"
    )


def _extend_patterns(
    size: int,
    literals: dict[str, list[int]],
    pairs: list[list[int]],
    used: dict[str, int],
    open_pairs: list[int],
    opened: int,
    perm: list[int],
    patterns: list[list[int]],
):
    """Extend a partial pattern by every choice for its next position, recursively.

    Each position takes the smallest index a choice allows: the next index of a plain
    letter, the second index of a pair already opened, or the first index of the next
    pair. Trying the choices in increasing index order yields the patterns in the order
    `itertools.permutations` first meets them.

    Args:
        size: Length of the symmetry string.
        literals: Indices of each plain letter, increasing.
        pairs: The two indices of each delta letter, ordered by the first.
        used: How many indices of each plain letter the partial pattern has taken.
        open_pairs: Pairs whose first index is taken and second is not, in order.
        opened: How many pairs have been opened.
        perm: The partial pattern, extended and restored in place.
        patterns: Where each complete pattern is appended.
    """
    if len(perm) == size:
        patterns.append(perm[:])
        return

    choices = [
        (indices[used[letter]], "letter", letter)
        for letter, indices in literals.items()
        if used[letter] < len(indices)
    ]
    choices += [(pairs[p][1], "close", p) for p in open_pairs]
    if opened < len(pairs):
        choices.append((pairs[opened][0], "open", opened))

    for index, kind, key in sorted(choices):
        perm.append(index)
        if kind == "letter":
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
