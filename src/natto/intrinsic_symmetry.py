"""Intrinsic permutation symmetry of a tensor.

Some tensors are constrained before any reduction happens: an elasticity tensor
obeys ``C[ijkl] = C[jikl] = C[klij]``, a piezoelectric one ``d[ijk] = d[ikj]``.
Such a constraint is a group of signed index permutations, and this module is
where one is parsed from its index equalities, closed into a group, imposed on a
tensor, or checked.

This is *not* full symmetrization: `impose_symmetry` projects onto whichever
permutation group was asked for, while the natural projector averages over all
permutations and removes the traces. The two are easy to confuse, which is why
they are named apart.
"""

import numpy as np


def impose_symmetry(t: np.ndarray, symmetry: str) -> np.ndarray:
    """
    Symmetrize a generic tensor to obtain a tensor with the specified symmetry.

    This is achieved by averaging the permutations of the tensor indices according to
    the given symmetry. The average, not the sum, is what makes this a projector: it
    leaves a tensor that already has the symmetry unchanged.

    Args:
        t: The input tensor to be symmetrized.
        symmetry: The target symmetry of the output tensor. e.g. 'ijk=ikj=jik'.

    Returns:
        The symmetrized tensor with the specified symmetry.
    """
    n = len(t.shape)
    permutations = generate_permutations(symmetry)
    if len(permutations[0][0]) != n:
        raise ValueError(f"symmetry {symmetry} does not match tensor rank n={n}.")

    transformed = [
        sign * np.transpose(t, permutation) for permutation, sign in permutations
    ]
    return np.mean(np.stack(transformed), axis=0)


def check_symmetry(
    t: np.ndarray,
    symmetry: str,
    rtol: float = 1e-5,
    atol: float = 1e-7,
) -> bool:
    """
    Check if a tensor has the specified symmetry.

    Args:
        t: The input tensor to be checked.
        symmetry: The target symmetry of the output tensor. e.g. 'ijk=ikj=jik'.
        rtol: Relative tolerance used for numerical comparison.
        atol: Absolute tolerance used for numerical comparison. The default is
            deliberately loose, since the tensor being checked is the caller's own
            and may carry error from wherever it came from; tighten it for a
            tensor built exactly.

    Returns:
        True if the tensor has the specified symmetry, False otherwise.
    """
    for permutation, sign in parse_symmetry_generators(symmetry, t.ndim):
        if not np.allclose(
            np.transpose(t, permutation), sign * t, rtol=rtol, atol=atol
        ):
            return False
    return True


def generate_permutations(symmetry: str) -> list[tuple[tuple[int, ...], int]]:
    """
    Get index permutations for a specified tensor symmetry.

    This is the group closure of the generators returned by
    :func:`parse_symmetry_generators`, so it covers both symmetric and antisymmetric
    relations.

    Args:
        symmetry: A string representing the symmetry of the target tensor. For example,
            - "ij=ji" means that the target is a fully symmetric rank-2 tensor (e.g.
                stress tensor);
            - "ij=-ji" means that the target is an antisymmetric rank-2 tensor;
            - "ijk=ikj" means that the target is a rank-3 tensor with the last two
                indices symmetric (e.g. piezoelectric tensor);
            - "ijk=ikj=jik" means that the target is a fully symmetric rank-3 tensor;
            - "ijkl=jikl=klij" means that the target is a rank-4 tensor with both minor
                symmetry (between i and j, and between k and l) and major symmetry (
                between ij and kl). For example, the elastic tensor has this symmetry;
            The number of unique letters gives the rank of the tensor (what letters to
            use does not matter).
    Returns:
        ``(permutation, sign)`` pairs representing
        ``permute(tensor, permutation) = sign * tensor``.

    Raises:
        ValueError: If the relations assign conflicting signs to one permutation.
    """
    generators = parse_symmetry_generators(symmetry)
    n = len(symmetry.split("=")[0].strip().replace(" ", ""))
    identity = tuple(range(n))
    signs = {identity: 1}
    queue = [(identity, 1)]

    while queue:
        permutation, sign = queue.pop(0)
        for generator, generator_sign in generators:
            composed = tuple(generator[axis] for axis in permutation)
            composed_sign = sign * generator_sign
            if composed in signs:
                if signs[composed] != composed_sign:
                    raise ValueError("Inconsistent signed symmetry relations")
                continue
            signs[composed] = composed_sign
            queue.append((composed, composed_sign))

    return list(signs.items())


def parse_symmetry_generators(
    symmetry: str, n: int | None = None
) -> list[tuple[tuple[int, ...], int]]:
    """Parse index equalities into signed permutation generators.

    Args:
        symmetry: Relations such as ``"ijk=ikj"`` or ``"ij=-ji"``. Every term is
            interpreted relative to the unsigned first term.
        n: Expected tensor rank. If provided, it must match the number of indices
            in the reference term.

    Returns:
        ``(permutation, sign)`` pairs representing
        ``permute(tensor, permutation) = sign * tensor``.
    """
    parts = [part.strip().replace(" ", "") for part in symmetry.split("=")]
    if not parts or not parts[0] or parts[0].startswith(("+", "-")):
        raise ValueError("The first symmetry term must be an unsigned index string")

    original = parts[0]
    if len(set(original)) != len(original):
        raise ValueError("Each index must occur once in a symmetry term")
    if n is not None and len(original) != n:
        raise ValueError(f"symmetry {symmetry} does not match tensor rank n={n}")
    index_to_axis = {char: axis for axis, char in enumerate(original)}

    generators = []
    for part in parts[1:]:
        sign = -1 if part.startswith("-") else 1
        term = part[1:] if part.startswith(("+", "-")) else part
        if len(term) != len(original):
            raise ValueError(f"Permutation {part} has wrong length")
        if set(term) != set(original):
            raise ValueError(f"Permutation {part} must contain the indices {original}")
        permutation = tuple(index_to_axis[char] for char in term)
        generators.append((permutation, sign))

    return generators
