"""Reduce a Cartesian tensor into natural tensors, then rebuild it.

This is the round trip the package exists for. For a Cartesian tensor `T` of
some rank and intrinsic symmetry, the extraction operator of each weight and
channel gives a natural tensor `X` -- symmetric and traceless -- and the
embedding operator puts it back into the Cartesian space as `T'`, the part of
`T` that this weight and channel accounts for. Summing those parts over every
weight and channel returns `T`.

The example also shows the third operator at work: `decomposition` is the
composition of the other two, taking `T` straight to `T'` without forming `X`,
which is what you want when the natural tensor itself is not of interest.

Change the `rank` and `symmetry` arguments at the bottom to try other classes;
`symmetry=None` means a tensor with no assumed symmetry. The elastic tensor,
`ijkl=jikl=klij`, is included as the richest of them.
"""

import numpy as np

from natto.intrinsic_symmetry import impose_symmetry
from natto.reduction import get_reduction
from natto.utils import is_symmetric, is_symmetric_traceless, is_traceless


def reduce_and_reconstruct(rank: int = 3, symmetry: str = None):
    """Reduce a random tensor of this class, then rebuild it from the parts.

    Args:
        rank: rank of the tensor T.
        symmetry: intrinsic symmetry of T, e.g. "ij=ji" for a symmetric rank-2
            tensor. None for a tensor with no assumed symmetry.
    """

    # Create a random tensor T of the given rank and symmetry
    T = np.random.default_rng(35).standard_normal((3,) * rank)
    if symmetry is not None:
        T = impose_symmetry(T, symmetry)

    output = get_reduction(rank, symmetry)

    all_T_prime = []
    for j, out_j in output.items():
        for p, (extraction, embedding, decomposition) in enumerate(
            zip(out_j["extraction"], out_j["embedding"], out_j["decomposition"])
        ):
            # Extract the natural tensor of this weight and channel
            X = np.einsum(extraction["rule"], extraction["numerical"], T)
            assert is_symmetric_traceless(X), (
                "X is not symmetric traceless for j={j}, p={p}"
            )

            # Embed it back into the Cartesian space
            T_p_1 = np.einsum(embedding["rule"], embedding["numerical"], X)
            print(
                f"T' (j={j}, p={p}), symmetric:",
                is_symmetric(T_p_1),
                "traceless:",
                is_traceless(T_p_1),
            )

            # The same thing in one step
            T_p_2 = np.einsum(decomposition["rule"], decomposition["numerical"], T)

            # T_p_1 and T_p_2 should be equal
            assert np.allclose(T_p_1, T_p_2, rtol=1e-5, atol=1e-6), (
                f"T_p_1 and T_p_2 are not equal for j={j}, p={p}"
            )

            all_T_prime.append(T_p_1)

    sum_T_prime = np.sum(np.stack(all_T_prime), axis=0)

    assert np.allclose(sum_T_prime, T)


if __name__ == "__main__":
    # a rank-2 tensor with no assumed symmetry: weights 0, 1 and 2, one channel each
    reduce_and_reconstruct(rank=2)

    print("=" * 40)
    # rank 3, where a weight first appears more than once
    reduce_and_reconstruct(rank=3)

    print("=" * 40)
    # the elastic tensor: minor symmetry within each pair of indices, and major
    # symmetry between the pairs, leaving weights 0, 2 and 4
    reduce_and_reconstruct(rank=4, symmetry="ijkl=jikl=klij")
