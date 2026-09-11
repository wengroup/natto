import itertools

import numpy as np

from natto.intrinsic_symmetry import (
    check_symmetry,
    generate_permutations,
    impose_symmetry,
    parse_symmetry_generators,
)


def unsigned_permutations(symmetry: str) -> set[tuple[int, ...]]:
    """Return the permutations of an unsigned symmetry as a set."""
    permutations = generate_permutations(symmetry)
    assert all(sign == 1 for _, sign in permutations)

    return {permutation for permutation, _ in permutations}


def test_generate_permutations():
    # rank-2 symmetric
    symmetry = "ij=ji"
    perms = unsigned_permutations(symmetry)
    assert perms == set(itertools.permutations([0, 1]))

    # rank-3 partially symmetric
    symmetry = "ijk=ikj"
    perms = unsigned_permutations(symmetry)
    assert perms == {(0, 1, 2), (0, 2, 1)}

    # rank-3 fully symmetric
    symmetry = "ijk=jik=ikj"
    perms = unsigned_permutations(symmetry)
    assert perms == set(itertools.permutations([0, 1, 2]))

    # rank-4 partially symmetric (elastic tensor)
    symmetry = "ijkl=jikl=klij"
    perms = unsigned_permutations(symmetry)
    assert perms == {
        (0, 1, 2, 3),
        (0, 1, 3, 2),
        (1, 0, 2, 3),
        (1, 0, 3, 2),
        (2, 3, 0, 1),
        (2, 3, 1, 0),
        (3, 2, 0, 1),
        (3, 2, 1, 0),
    }

    # rank-4 fully symmetric
    symmetry = "ijkl=jikl=kjil=ljki"
    perms = unsigned_permutations(symmetry)
    assert perms == set(itertools.permutations([0, 1, 2, 3]))


def test_impose_symmetry():
    rng = np.random.default_rng(35)

    t = rng.standard_normal((3, 3))
    symmetry = "ij=ji"
    out = impose_symmetry(t, symmetry)
    assert check_symmetry(out, symmetry)

    t = rng.standard_normal((3, 3, 3))
    symmetry = "ijk=ikj"
    out = impose_symmetry(t, symmetry)
    assert check_symmetry(out, symmetry)

    symmetry = "ijk=jik=ikj"
    out = impose_symmetry(t, symmetry)
    assert check_symmetry(out, symmetry)

    t = rng.standard_normal((3, 3, 3, 3))
    symmetry = "ijkl=jikl=klij"
    out = impose_symmetry(t, symmetry)
    assert check_symmetry(out, symmetry)

    symmetry = "ijkl=jikl=kjil=ljki"
    out = impose_symmetry(t, symmetry)
    assert check_symmetry(out, symmetry)


def test_antisymmetric_rank_two():
    """Check signed parsing and projection onto antisymmetric rank-two tensors."""
    symmetry = "ij=-ji"
    assert parse_symmetry_generators(symmetry) == [((1, 0), -1)]
    assert set(generate_permutations(symmetry)) == {
        ((0, 1), 1),
        ((1, 0), -1),
    }

    tensor = np.random.default_rng(35).standard_normal((3, 3))
    output = impose_symmetry(tensor, symmetry)
    assert check_symmetry(output, symmetry)
    assert np.allclose(output, (tensor - tensor.T) / 2)


def test_fully_antisymmetric_rank_three():
    """Check float32 projection onto the fully antisymmetric rank-three space."""
    symmetry = "ijk=-jik=-ikj"
    output = impose_symmetry(
        np.random.default_rng(35).standard_normal((3, 3, 3)), symmetry
    )

    assert check_symmetry(output, symmetry)
