"""
Example to convert ordinary tensor and natural tensor back and forth.
"""

import torch

from natto.mappings import get_reduction
from natto.sym import symmetrize
from natto.utils import is_symmetric, is_symmetric_traceless, is_traceless


def convert(rank=3, symmetry: str = None):
    """Converting between ordinary tensor and natural tensor.

    For an ordinary tensor T, convert it to natural tensors X (which will be
    symmetric traceless), embed each natural tensor X to be in the original space T' (
    which is not symmetric traceless).
    Or, T' can be directly obtained by applying S to T.

    Args:
        rank: rank of the tensor T
        symmetry: symmetry of the tensor T, e.g. "ij=ji" for a rank-2 tensor.
    """

    torch.manual_seed(35)

    # Create a random tensor T of the given rank and symmetry
    T = torch.randn(*[3] * rank)
    if symmetry is not None:
        T = symmetrize(T, symmetry)

    output = get_reduction(rank, symmetry)

    all_T_prime = []
    for j, out_j in output.items():
        for p, (extraction, embedding, decomposition) in enumerate(
            zip(out_j["extraction"], out_j["embedding"], out_j["decomposition"])
        ):
            # Extract the natural tensor of this weight and channel
            X = torch.einsum(extraction["rule"], extraction["numerical"], T)
            assert is_symmetric_traceless(X), (
                "X is not symmetric traceless for j={j}, p={p}"
            )

            # Embed it back into the Cartesian space
            T_p_1 = torch.einsum(embedding["rule"], embedding["numerical"], X)
            print(
                f"T' (j={j}, p={p}), symmetric:",
                is_symmetric(T_p_1),
                "traceless:",
                is_traceless(T_p_1),
            )

            # The same thing in one step
            T_p_2 = torch.einsum(decomposition["rule"], decomposition["numerical"], T)

            # T_p_1 and T_p_2 should be equal
            assert torch.allclose(T_p_1, T_p_2, rtol=1e-5, atol=1e-6), (
                f"T_p_1 and T_p_2 are not equal for j={j}, p={p}"
            )

            all_T_prime.append(T_p_1)

    sum_T_prime = torch.sum(torch.stack(all_T_prime), dim=0)

    assert torch.allclose(sum_T_prime, T)


if __name__ == "__main__":
    convert(rank=2)
    print("=" * 40)
    convert(rank=3)
