"""The self-dual basis of Eq. 21.

`get_reduction(..., basis="orthonormal")` returns, per channel, an embedding and an
extraction operator with the same terms, differing only in their input. These tests
assert the property that makes that possible -- the mappings of a weight are
orthonormal under the Cartesian inner product, exactly, since each is an exact operator
with one root of an integer -- and that extraction followed by embedding still
recovers the tensor.
"""

import numpy as np
import pytest

from natto.algebra import contract
from natto.intrinsic_symmetry import check_symmetry, impose_symmetry
from natto.reduction import get_gram_matrices, get_reduction
from natto.symbolic import Signature, act, evaluate

#: Reconstruction evaluates the operators and accumulates over every weight and
#: channel, so it is compared to a tolerance.
RECONSTRUCTION_TOL = 1e-10


def assert_orthonormal(output: dict, weight: int):
    """Assert the mappings of one weight are orthonormal, exactly, as Eq. 22 states.

    Two mappings contracted over every index give 2 * weight + 1 times their Gram
    entry, and the roots of the two normalizations multiply back to a rational.
    """
    mappings = [
        operators["embedding"]
        for (ell, _), operators in output.items()
        if ell == weight
    ]
    labels = list(range(mappings[0].signature.size))

    for p, first in enumerate(mappings):
        for q, second in enumerate(mappings):
            product = contract([(first, labels), (second, labels)], Signature(()))
            expected = [2 * weight + 1] if p == q else []

            assert product.radicand == 1
            assert list(product.terms.values()) == expected


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

    assert gram == [[8, 2], [2, 3]]
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
