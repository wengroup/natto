import pytest
import torch

from natto.orthonormal import get_orthonormal_G, get_orthonormal_Q
from natto.sym import check_symmetry, symmetrize


def test_piezoelectric_orthonormal_Q():
    """Check the piezoelectric Gram matrix and orthonormal weight-1 mappings."""
    out = get_orthonormal_Q(3, "ijk=ikj")
    out_weight_1 = out[1]

    expected_gram = torch.tensor([[3.0, 2.0], [2.0, 8.0]], dtype=torch.float64)
    assert torch.allclose(out_weight_1["gram"], expected_gram)

    orthonormal = torch.stack(
        [entry["numerical"] for entry in out_weight_1["orthonormal"]]
    )
    orthonormal_flat = orthonormal.reshape(len(orthonormal), -1)
    gram = orthonormal_flat @ orthonormal_flat.T / 3
    assert torch.allclose(gram, torch.eye(2, dtype=orthonormal.dtype), atol=1e-12)


@pytest.mark.parametrize("rank", [1, 2])
def test_orthonormal_G_reconstructs_general_tensor(rank: int):
    """Check unrestricted orthonormal mappings and their self-dual reconstruction."""
    torch.manual_seed(35)
    tensor = torch.randn((3,) * rank, dtype=torch.float64)
    output = get_orthonormal_G(rank)

    embedded_parts = []
    for out_weight in output.values():
        orthonormal = torch.stack(
            [entry["numerical"] for entry in out_weight["orthonormal"]]
        )
        flattened = orthonormal.reshape(len(orthonormal), -1)
        weight = orthonormal.ndim - tensor.ndim - 1
        gram = flattened @ flattened.T / (2 * weight + 1)
        assert torch.allclose(
            gram, torch.eye(len(orthonormal), dtype=orthonormal.dtype), atol=1e-12
        )

        for mapping in out_weight["orthonormal"]:
            natural = torch.einsum(
                mapping["extraction_rule"], mapping["numerical"], tensor
            )
            embedded_parts.append(
                torch.einsum(mapping["embedding_rule"], mapping["numerical"], natural)
            )

    reconstructed = torch.stack(embedded_parts).sum(dim=0)
    assert torch.allclose(reconstructed, tensor, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize(
    "rank,symmetry",
    [(3, "ijk=ikj"), (4, "ijkl=jikl=klij")],
)
def test_orthonormal_Q_reconstructs_symmetric_tensor(rank: int, symmetry: str):
    """Check self-dual extraction and embedding for key physical symmetry classes."""
    torch.manual_seed(35)
    tensor = symmetrize(torch.randn((3,) * rank, dtype=torch.float64), symmetry)
    output = get_orthonormal_Q(rank, symmetry)

    embedded_parts = []
    for out_weight in output.values():
        for orthonormal in out_weight["orthonormal"]:
            natural = torch.einsum(
                orthonormal["extraction_rule"], orthonormal["numerical"], tensor
            )
            embedded = torch.einsum(
                orthonormal["embedding_rule"], orthonormal["numerical"], natural
            )
            assert check_symmetry(embedded, symmetry, atol=1e-10)
            embedded_parts.append(embedded)

    reconstructed = torch.stack(embedded_parts).sum(dim=0)
    assert torch.allclose(reconstructed, tensor, rtol=1e-10, atol=1e-10)
