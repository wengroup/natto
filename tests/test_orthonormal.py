"""The self-dual basis of Eq. (26).

`get_reduction(..., basis="orthonormal")` returns one array per channel that
both extracts and embeds. These tests assert the property that makes that
possible -- the mappings of a weight are orthonormal under the Cartesian inner
product -- and that extraction followed by embedding still recovers the tensor.
"""

import pytest
import torch

from natto.mappings import get_reduction
from natto.sym import check_symmetry, symmetrize

#: The operators are built in double precision and the identity they satisfy is
#: exact, so what is left is rounding in the eigendecomposition.
ORTHONORMAL_ATOL = 1e-12

#: Reconstruction accumulates over every weight and channel, so it is looser.
RECONSTRUCTION_TOL = 1e-10


def orthonormal(rank: int, symmetry: str = None, weight: int = None) -> dict:
    """The reduction in the self-dual basis, in double precision."""
    output = get_reduction(rank, symmetry, basis="orthonormal", dtype=torch.float64)

    return output if weight is None else output[weight]


def assert_orthonormal(per_weight: dict, weight: int):
    """Assert the mappings of one weight are orthonormal under Eq. (21)."""
    stacked = torch.stack([entry["numerical"] for entry in per_weight["embedding"]])
    flattened = stacked.reshape(len(stacked), -1)
    gram = flattened @ flattened.T / (2 * weight + 1)

    torch.testing.assert_close(
        gram, torch.eye(len(stacked), dtype=gram.dtype), rtol=0, atol=ORTHONORMAL_ATOL
    )


def reconstruct(output: dict, tensor: torch.Tensor) -> list[torch.Tensor]:
    """Extract and embed back through each channel, returning the parts."""
    parts = []
    for per_weight in output.values():
        for embedding, extraction in zip(
            per_weight["embedding"], per_weight["extraction"]
        ):
            natural = torch.einsum(extraction["rule"], extraction["numerical"], tensor)
            parts.append(
                torch.einsum(embedding["rule"], embedding["numerical"], natural)
            )

    return parts


def test_the_same_array_extracts_and_embeds():
    """The self-duality that makes `basis="orthonormal"` one operator, not two."""
    per_weight = orthonormal(3, "ijk=ikj", weight=1)

    for embedding, extraction in zip(per_weight["embedding"], per_weight["extraction"]):
        assert embedding["numerical"] is extraction["numerical"]
        assert embedding["rule"] != extraction["rule"]


def test_piezoelectric_gram_matrix():
    """Pin the piezoelectric weight-1 Gram matrix, and its orthonormal mappings."""
    per_weight = orthonormal(3, "ijk=ikj", weight=1)

    expected_gram = torch.tensor([[3.0, 2.0], [2.0, 8.0]], dtype=torch.float64)
    torch.testing.assert_close(per_weight["gram"], expected_gram)
    assert_orthonormal(per_weight, weight=1)


@pytest.mark.parametrize("rank", [1, 2])
def test_reconstructs_a_general_tensor(rank: int):
    """Unrestricted orthonormal mappings, and their self-dual reconstruction."""
    torch.manual_seed(35)
    tensor = torch.randn((3,) * rank, dtype=torch.float64)
    output = orthonormal(rank)

    for weight, per_weight in output.items():
        assert_orthonormal(per_weight, weight)

    reconstructed = torch.stack(reconstruct(output, tensor)).sum(dim=0)
    torch.testing.assert_close(
        reconstructed, tensor, rtol=RECONSTRUCTION_TOL, atol=RECONSTRUCTION_TOL
    )


@pytest.mark.parametrize("rank,symmetry", [(3, "ijk=ikj"), (4, "ijkl=jikl=klij")])
def test_reconstructs_a_symmetric_tensor(rank: int, symmetry: str):
    """Self-dual extraction and embedding for two physical symmetry classes.

    Each embedded part must carry the symmetry of the class on its own, not
    only in the sum.
    """
    torch.manual_seed(35)
    tensor = symmetrize(torch.randn((3,) * rank, dtype=torch.float64), symmetry)
    output = orthonormal(rank, symmetry)

    parts = reconstruct(output, tensor)
    for part in parts:
        assert check_symmetry(part, symmetry, atol=RECONSTRUCTION_TOL)

    reconstructed = torch.stack(parts).sum(dim=0)
    torch.testing.assert_close(
        reconstructed, tensor, rtol=RECONSTRUCTION_TOL, atol=RECONSTRUCTION_TOL
    )
