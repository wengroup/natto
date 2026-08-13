import contextlib
import functools
from typing import NamedTuple, Optional

import pytest
import torch

import natto.GHS
from natto.EGH import get_g_matrix
from natto.evaluate import evaluate_tensors
from natto.GHS import get_G_H_S, get_G_H_S_natural, get_G_H_S_of_j
from natto.qr import find_independent_tensors
from natto.sym import symmetrize
from natto.symmetrize import get_random_natural_tensor
from natto.utils import letter_index


class TensorClass(NamedTuple):
    """One physical tensor class, i.e. one row of Table 1."""

    # `Example` column, e.g. `photoelastic effect`
    example: str
    # `Rank n` column
    rank: int
    # `Symmetry` column, spelled the way `get_G_H_S` wants it; the table writes it with
    # parentheses around the symmetric index groups, so its `(ij)kl` is `ijkl=jikl`
    # here. None when the indices are unconstrained.
    symmetry: Optional[str]
    # `N_ind` column, the number of independent components
    n_ind: int
    # `Weight m` columns, as {weight m: multiplicity N_m}; weights with N_m = 0 omitted
    multiplicity: dict[int, int]

    @property
    def test_id(self) -> str:
        """Identifier of this class, unique because two rows share an example."""
        return f"rank{self.rank}_{self.example.replace(' ', '_')}"


PHYSICAL_TENSOR_CLASSES = [
    TensorClass("pressure", 0, None, 1, {0: 1}),
    TensorClass("spontaneous polarization", 1, None, 3, {1: 1}),
    TensorClass("optical activity", 2, None, 9, {0: 1, 1: 1, 2: 1}),
    TensorClass("stress and strain", 2, "ij=ji", 6, {0: 1, 2: 1}),
    TensorClass("rate-of-rotation tensor", 2, "ij=-ji", 3, {1: 1}),
    TensorClass("optical mixing", 3, None, 27, {0: 1, 1: 3, 2: 2, 3: 1}),
    TensorClass("piezoelectric effect", 3, "ijk=ikj", 18, {1: 2, 2: 1, 3: 1}),
    TensorClass("Kleinman symmetry in SHG", 3, "ijk=ikj=jik", 10, {1: 1, 3: 1}),
    TensorClass("optical mixing", 4, None, 81, {0: 3, 1: 6, 2: 6, 3: 3, 4: 1}),
    # the j=1 sector of the photoelastic class is what the unpivoted-QR selection in
    # `natto.qr` used to get wrong, see tests/test_qr.py
    TensorClass(
        "photoelastic effect", 4, "ijkl=jikl", 54, {0: 2, 1: 3, 2: 4, 3: 2, 4: 1}
    ),
    TensorClass("Kerr effect", 4, "ijkl=jikl=ijlk", 36, {0: 2, 1: 1, 2: 3, 3: 1, 4: 1}),
    TensorClass(
        "third harmonic generation",
        4,
        "ijkl=ikjl=jikl",
        30,
        {0: 1, 1: 1, 2: 2, 3: 1, 4: 1},
    ),
    TensorClass("elasticity", 4, "ijkl=jikl=klij", 21, {0: 2, 2: 2, 4: 1}),
    TensorClass("Cauchy relations", 4, "ijkl=jikl=kjil=ljki", 15, {0: 1, 2: 1, 4: 1}),
]

# Classes `get_G_H_S` cannot handle yet, as {test_id: reason}. Kept out of the table
# above so that it stays a statement about the physics, not about the code.
NOT_SUPPORTED = {}


def get_tensor_class_params(
    tensor_classes: Optional[list[TensorClass]] = None,
) -> list:
    """Parametrize over tensor classes, xfailing the ones that are not supported yet.

    Args:
        tensor_classes: classes to parametrize over, all of Table 1 by default.
    """
    if tensor_classes is None:
        tensor_classes = PHYSICAL_TENSOR_CLASSES

    params = []
    for tc in tensor_classes:
        reason = NOT_SUPPORTED.get(tc.test_id)
        marks = (
            pytest.mark.xfail(raises=AssertionError, strict=True, reason=reason)
            if reason is not None
            else ()
        )
        params.append(pytest.param(tc, id=tc.test_id, marks=marks))

    return params


@contextlib.contextmanager
def selection_scheme(method: str):
    """Make `GHS` select independent tensors with the given `natto.qr` scheme."""
    original = natto.GHS.find_independent_tensors
    natto.GHS.find_independent_tensors = functools.partial(
        find_independent_tensors, method=method
    )
    try:
        yield
    finally:
        natto.GHS.find_independent_tensors = original


@functools.lru_cache(maxsize=None)
def get_G_H_S_cached(rank: int, symmetry: str, method: str = "gram_schmidt") -> dict:
    """`get_G_H_S`, computed once per (class, scheme); rank 4 takes a couple seconds."""
    with selection_scheme(method):
        return get_G_H_S(rank, symmetry)


@pytest.mark.parametrize("method", ["gram_schmidt", "scipy_qr"])
@pytest.mark.parametrize("tensor_class", get_tensor_class_params())
def test_weight_multiplicity(tensor_class: TensorClass, method: str):
    """Check the weight decomposition of each physical tensor class in Table 1.

    Each weight-m sector appears N_m times, and the multiplicities account for all
    N_ind independent components: N_ind = sum_m N_m (2m + 1).

    The multiplicities are the ranks found when selecting independent H tensors, so
    this is also where a non-rank-revealing selection scheme shows up; it is checked
    for both schemes of `natto.qr` since either may be used.

    Args:
        tensor_class: physical tensor class to check
        method: `natto.qr` scheme used to select the independent H tensors
    """
    output = get_G_H_S_cached(tensor_class.rank, tensor_class.symmetry, method)

    found = {m: len(out_m["H"]) for m, out_m in output.items() if len(out_m["H"]) > 0}

    assert found == tensor_class.multiplicity
    assert sum(N_m * (2 * m + 1) for m, N_m in found.items()) == tensor_class.n_ind


@pytest.mark.parametrize(
    "tensor_class",
    get_tensor_class_params(
        [tc for tc in PHYSICAL_TENSOR_CLASSES if tc.symmetry is not None]
    ),
)
def test_symbolic_symmetry_adapted_gram_matrix(tensor_class: TensorClass):
    """Check exact Gram matrices for every internally symmetric weight sector."""
    for weight in tensor_class.multiplicity:
        Q, _, _, _, _ = get_G_H_S_of_j(weight, tensor_class.rank, tensor_class.symmetry)
        Q_with_zeros = [Q_p + 0 * Q_p for Q_p in Q]
        numerical_Q = torch.stack([evaluate_tensors(Q_p, mode="G") for Q_p in Q])
        flattened_Q = numerical_Q.reshape(len(Q), -1)
        numerical = flattened_Q @ flattened_Q.T / (2 * weight + 1)
        symbolic = torch.tensor(
            [
                [float(value) for value in row]
                for row in get_g_matrix(weight, tensor_class.rank, Q_with_zeros)
            ],
            dtype=numerical.dtype,
        )

        torch.testing.assert_close(symbolic, numerical)


def test_symmetry_rank_must_match_tensor_rank():
    """Reject a symmetry whose reference term has the wrong tensor rank."""
    with pytest.raises(ValueError):
        get_G_H_S(3, "ij", numerical=False)


@pytest.mark.parametrize("tensor_class", get_tensor_class_params())
def test_get_G_H_S(tensor_class: TensorClass):
    """Test the get_G_H_S function.

    For a given tensor T, obtain the natural tensors X (using H), and then obtain the
    embedding T' in the original tensor space (using G). We check that we can recover T.

    Args:
        tensor_class: physical tensor class to check
    """
    rank = tensor_class.rank
    symmetry = tensor_class.symmetry

    torch.manual_seed(35)
    T = torch.randn((3,) * rank)

    # symmetrize the tensor if `symmetry` is not None
    if symmetry is not None:
        T = symmetrize(T, symmetry)

    output = get_G_H_S_cached(rank, symmetry)

    all_T_prime = []
    for j, out_j in output.items():
        for p, (H, G, S) in enumerate(zip(out_j["H"], out_j["G"], out_j["S"])):
            # X = H T
            X = torch.einsum(H["rule"], H["numerical"], T)

            # T' = G X
            T_p_1 = torch.einsum(G["rule"], G["numerical"], X)

            # T' = S T
            T_p_2 = torch.einsum(S["rule"], S["numerical"], T)

            # T_p_1 and T_p_2 should be equal
            assert torch.allclose(T_p_1, T_p_2, rtol=1e-5, atol=1e-6), (
                f"T_p_1 and T_p_2 are not equal for j={j}, p={p}"
            )

            all_T_prime.append(T_p_1)

    sum_T_prime = torch.sum(torch.stack(all_T_prime), dim=0)

    assert torch.allclose(sum_T_prime, T, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    "j1,j2",
    [
        (0, 1),
        (0, 2),
        (1, 1),
        (1, 2),
        (2, 2),
    ],
)
def test_get_G_H_S_natural(j1: int, j2: int):

    # Create T = T1 \otimes T2
    T1 = get_random_natural_tensor(j1, 35)
    T2 = get_random_natural_tensor(j2, 36)
    idx1 = letter_index(j1)
    idx2 = letter_index(j2, start=j1)
    T = torch.einsum(f"{idx1},{idx2}->{idx1}{idx2}", T1, T2)

    output = get_G_H_S_natural(j1, j2)

    all_T_prime = []
    for j, out_j in output.items():
        for p, (H, G, S) in enumerate(zip(out_j["H"], out_j["G"], out_j["S"])):
            # X = H T
            X = torch.einsum(H["rule"], H["numerical"], T)

            # T' = G X
            T_p_1 = torch.einsum(G["rule"], G["numerical"], X)

            # T' = S T
            T_p_2 = torch.einsum(S["rule"], S["numerical"], T)

            # T_p_1 and T_p_2 should be equal
            assert torch.allclose(T_p_1, T_p_2, rtol=1e-5, atol=1e-6), (
                f"T_p_1 and T_p_2 are not equal for j={j}, p={p}"
            )

            all_T_prime.append(T_p_1)

    sum_T_prime = torch.sum(torch.stack(all_T_prime), dim=0)

    assert torch.allclose(sum_T_prime, T, rtol=1e-5, atol=1e-6)
