"""Conversion between an ordinary Cartesian tensor T and its natural tensors X."""

import torch
import torch.nn as nn
from torch import Tensor

from natto.mappings import get_G_H_S


# TODO, this function should be reimplemented for simplicity, not consider batching
#   and such. In `carten` we have a batched version
class Converter(nn.Module):
    r"""
    Convertor to map between ordinary cartesian tensor T and natural tensors X.

    Writing $\widetilde{\mathbf{G}}$ for the extraction operator and
    $\mathbf{G}$ for the embedding operator, the conversion is

    $$
    \mathbf{X}_\ell^p = \widetilde{\mathbf{G}}^p_{(\ell|n)} \odot^n \mathbf{T}_n
    \qquad
    \mathbf{S}_n^{\ell,p} = \mathbf{G}^p_{(\ell|n)} \odot^\ell \mathbf{X}_\ell^p
    $$

    Args:
        rank: The rank of the tensor T.
        symmetry: symmetry of the Cartesian ordinary tensor, if any. For example,
            - "ij=ji" means that the tensor is a fully symmetric rank-2 tensor (e.g.
                stress tensor);
            - "ijk=ikj" means that the tensor is a rank-3 tensor with the last two
                indices symmetric (e.g. piezoelectric tensor);
            - "ijk=ikj=jik" means that the tensor is a fully symmetric rank-3 tensor;
            - "ijkl=jikl=klij" means that the tensor is a rank-4 tensor with both minor
                symmetry (between i and j, and between k and l) and major symmetry (
                between ij and kl). For example, the elastic tensor has this symmetry.
            The number of unique letters gives the rank of the tensor (what letters to
            use does not matter).
    """

    def __init__(self, rank: int, symmetry: str = None):
        super().__init__()
        self.rank = rank
        self.symmetry = symmetry

        out = get_G_H_S(rank, symmetry)

        self.l_num_p = {}
        for weight, out_weight in out.items():
            self.l_num_p[weight] = len(out_weight["embedding"])
            # TODO, the tensors of different G_j_p might be batched, which can
            #  accelerate the computation
            for p, (embedding, extraction) in enumerate(
                zip(out_weight["embedding"], out_weight["extraction"])
            ):
                self.register_buffer(
                    f"embedding_{weight}_{p}", torch.tensor(embedding["numerical"])
                )
                self.register_buffer(
                    f"extraction_{weight}_{p}", torch.tensor(extraction["numerical"])
                )
                setattr(self, f"embedding_{weight}_{p}_rule", embedding["rule"])
                setattr(self, f"extraction_{weight}_{p}_rule", extraction["rule"])

    def to_natural_tensor(self, T: Tensor) -> dict[int, Tensor]:
        """
        Convert an ordinary cartesian tensor T to natural tensors X.

        Args:
            T: The tensor to convert. Shape (B, 3, ..., 3), where B represents arbitrary
                batch dimensions, and the number of 3s is, of course, equal to the rank
                of the tensor.

        Returns:
            Natural tensors {l: X}, where l is the rank (l=0, 1, ...n) and X is the
            corresponding tensor value. The shape of X is (B, F, 3^l) where B represents
            arbitrary batch dimensions, F is the number of natural tensors of rank l,
            and 3^l is the flattened dimension of the tensor. The second dimension F
            batches natural tensors of the same weight l, one per channel p.
        """
        B = T.shape[: -self.rank]

        # TODO, the looping is very inefficient, need to think about better ways
        out = {}
        for weight, num_p in self.l_num_p.items():
            out[weight] = torch.stack(
                [
                    torch.einsum(
                        getattr(self, f"extraction_{weight}_{p}_rule"),
                        getattr(self, f"extraction_{weight}_{p}"),
                        T,
                    ).reshape(*B, 3**weight)
                    for p in range(num_p)
                ],
                dim=-2,  # stack to create the new F dimension
            )

        return out

    def to_ordinary_tensor(self, X: dict[int, Tensor]) -> Tensor:
        """
        Convert natural tensors X to ordinary cartesian tensor T.

        This is the inverse of `to_natural_tensor`.

        Args:
            X: Natural tensors {l: X}, where l is the rank (l=0, 1, ...n) and X is the
            corresponding tensor value. The shape of X is (B, F, 3^l) where B represents
            arbitrary batch dimensions, F is the number of natural tensors of rank l,
            and 3^l is the flattened dimension of the tensor. The second dimension F
            batches natural tensors of the same weight l, one per channel p.

        Returns:
            An ordinary cartesian tensor T corresponding to the natural tensors X.
            The shape is (B, 3, ..., 3), where B represents arbitrary batch dimensions,
            and the number of 3s is, of course, equal to the rank of the tensor.
        """
        B = X[list(X.keys())[0]].shape[:-2]

        out = []
        # TODO, the looping is very inefficient, need to think about better ways
        for weight, num_p in self.l_num_p.items():
            for p in range(num_p):
                rule = getattr(self, f"embedding_{weight}_{p}_rule")
                G = getattr(self, f"embedding_{weight}_{p}")
                X_ = X[weight][..., p, :]

                # special rule for scalars
                if weight == 0:
                    X_ = X_[..., 0]
                else:
                    X_ = X_.reshape(*B, *(3,) * weight)
                out.append(torch.einsum(rule, G, X_))

        out = torch.sum(torch.stack(out), dim=0)

        return out
