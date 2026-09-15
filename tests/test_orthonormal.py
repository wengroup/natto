"""The self-dual basis of Eq. 21.

`get_reduction(..., basis="orthonormal")` returns, per channel, an embedding and an
extraction operator with the same terms, differing only in their input. These tests
assert the property that makes that possible -- the mappings of a weight are
orthonormal under the Cartesian inner product -- and that extraction followed by
embedding still recovers the tensor.
"""

import numpy as np
import pytest

from natto.intrinsic_symmetry import check_symmetry, impose_symmetry
from natto.reduction import get_gram_matrices, get_reduction
from natto.symbolic import act, evaluate

#: The operators are built in double precision and the identity they satisfy is
#: exact, so what is left is rounding in the eigendecomposition.
ORTHONORMAL_ATOL = 1e-12

#: Reconstruction accumulates over every weight and channel, so it is looser.
RECONSTRUCTION_TOL = 1e-10


def assert_orthonormal(output: dict, weight: int):
    """Assert the mappings of one weight are orthonormal, as Eq. 22 states."""
    stacked = np.stack(
        [
            operators["embedding"].evaluate()
            for (ell, _), operators in output.items()
            if ell == weight
        ]
    )
    flattened = stacked.reshape(len(stacked), -1)
    gram = flattened @ flattened.T / (2 * weight + 1)

    np.testing.assert_allclose(
        gram, np.eye(len(stacked), dtype=gram.dtype), rtol=0, atol=ORTHONORMAL_ATOL
    )


def reconstruct(output: dict, tensor: np.ndarray) -> list[np.ndarray]:
    """Extract and embed back through each channel, returning the parts."""
    parts = [
        act(operators["embedding"], act(operators["extraction"], tensor))
        for operators in output.values()
    ]

    return parts


def test_the_same_array_extracts_and_embeds():
    """The self-duality that makes `basis="orthonormal"` one operator, not two."""
    output = get_reduction(3, symmetry="ijk=ikj", basis="orthonormal")

    for operators in output.values():
        embedding, embedding_rule = evaluate(operators["embedding"])
        extraction, extraction_rule = evaluate(operators["extraction"])

        np.testing.assert_array_equal(embedding, extraction)
        assert embedding_rule != extraction_rule


def test_piezoelectric_gram_matrix():
    """Pin the piezoelectric weight-1 Gram matrix, and its orthonormal mappings."""
    gram = get_gram_matrices(3, ell=1, symmetry="ijk=ikj")[1]

    assert gram == [[3, 2], [2, 8]]
    output = get_reduction(3, ell=1, symmetry="ijk=ikj", basis="orthonormal")
    assert_orthonormal(output, weight=1)


@pytest.mark.parametrize("rank", [1, 2])
def test_reconstructs_a_general_tensor(rank: int):
    """Unrestricted orthonormal mappings, and their self-dual reconstruction."""
    tensor = np.random.default_rng(35).standard_normal((3,) * rank)
    output = get_reduction(rank, basis="orthonormal")

    for weight in {ell for ell, _ in output}:
        assert_orthonormal(output, weight)

    reconstructed = np.stack(reconstruct(output, tensor)).sum(axis=0)
    np.testing.assert_allclose(
        reconstructed, tensor, rtol=RECONSTRUCTION_TOL, atol=RECONSTRUCTION_TOL
    )


@pytest.mark.parametrize("rank,symmetry", [(3, "ijk=ikj"), (4, "ijkl=jikl=klij")])
def test_reconstructs_a_symmetric_tensor(rank: int, symmetry: str):
    """Self-dual extraction and embedding for two physical symmetry classes.

    Each embedded part must carry the symmetry of the class on its own, not
    only in the sum.
    """
    tensor = impose_symmetry(
        np.random.default_rng(35).standard_normal((3,) * rank), symmetry
    )
    output = get_reduction(rank, symmetry=symmetry, basis="orthonormal")

    parts = reconstruct(output, tensor)
    for part in parts:
        assert check_symmetry(part, symmetry, atol=RECONSTRUCTION_TOL)

    reconstructed = np.stack(parts).sum(axis=0)
    np.testing.assert_allclose(
        reconstructed, tensor, rtol=RECONSTRUCTION_TOL, atol=RECONSTRUCTION_TOL
    )
