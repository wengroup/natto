import pytest
import torch

from natto.convert import Converter
from natto.sym import symmetrize

#: Rank four is where the construction costs seconds rather than a fraction of
#: one, so all but the elastic class are marked `slow`; see
#: `test_mappings.DEFAULT_RANK_FOUR_CLASS`.
slow = pytest.mark.slow


@pytest.mark.parametrize(
    "rank,symmetry",
    [
        (2, None),
        (2, "ij=ji"),
        (3, None),
        (3, "ijk=ikj"),
        (3, "ijk=ikj=jik"),
        pytest.param(4, None, marks=slow),
        pytest.param(4, "ijkl=jikl", marks=slow),
        pytest.param(4, "ijkl=jilk", marks=slow),
        (4, "ijkl=jikl=klij"),
        pytest.param(4, "ijkl=jikl=kjil=ljki", marks=slow),
    ],
)
def test_Converter(rank, symmetry):

    torch.manual_seed(35)

    converter = Converter(rank, symmetry)

    T = torch.randn(*[3] * rank)

    # symmetrize the tensor if `symmetry` is not None
    if symmetry is not None:
        T = symmetrize(T, symmetry)

    X = converter.to_natural_tensor(T)
    T_2 = converter.to_ordinary_tensor(X)

    assert torch.allclose(T, T_2, atol=1e-6), f"Failed for rank {rank}."
