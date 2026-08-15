"""
Evaluating numerical values of G, H, and S tensors and their tensor products.
"""

from functools import lru_cache

import torch
from torch import Tensor

from natto.ops import simplify_linear_combination
from natto.symbolic import Delta, Epsilon, LinearCombination, TensorProduct
from natto.utils import dij, eijk, letter_index

_MAX_CACHED_CONTRACTION_ELEMENTS = 3**8


@lru_cache(maxsize=512)
def _cached_delta_epsilon_contraction(
    rule: str, num_delta: int, num_epsilon: int, dtype: torch.dtype = None
) -> Tensor:
    """Contract and cache a small product of delta and Levi-Civita tensors."""
    data = [dij(dtype=dtype)] * num_delta + [eijk(dtype=dtype)] * num_epsilon

    return torch.einsum(rule, *data)


def _contract_delta_epsilon(
    rule: str, num_delta: int, num_epsilon: int, dtype: torch.dtype = None
) -> Tensor:
    """Contract a product of Kronecker deltas and Levi-Civita symbols.

    The operands are fixed constants whose multiplicities are given by ``num_delta``
    and ``num_epsilon``, so small results depend only on the arguments and are cached.
    A tensor expansion contains many terms sharing the same index pattern -- for the
    rank-six weight-two mappings, 11358 contractions use only 190 distinct rules --
    and ``torch.einsum`` spends most of its time searching for a contraction path
    rather than contracting, so caching removes the bulk of that cost. Results larger
    than ``3**8`` elements bypass the cache to keep its memory use bounded.

    A cached result is shared; callers must not modify it in place.
    """
    if dtype is None:
        dtype = torch.get_default_dtype()

    output_indices = rule.rsplit("->", maxsplit=1)[1]
    output_elements = 3 ** len(output_indices)
    if output_elements <= _MAX_CACHED_CONTRACTION_ELEMENTS:
        return _cached_delta_epsilon_contraction(rule, num_delta, num_epsilon, dtype)

    data = [dij(dtype=dtype)] * num_delta + [eijk(dtype=dtype)] * num_epsilon

    return torch.einsum(rule, *data)


def tp_delta_epsilon(tp: TensorProduct, mode: str, dtype: torch.dtype = None) -> Tensor:
    """Get the tensor product of Kronecker delta and Levi-Civita tensors.

    Note, the order of the indices need to be taken care of.
    Upper-case letters are used to represent tensors in the n space (namely for
    tensors T and such), while lower-case letters are used to represent tensors in the
    j space (namely for tensors X). So:
    1. X = H T: H would consist of both lower case and upper-case letters, and its
       upper-case letters are to be contracted with T. We assume the contracting rule
       is something like X_ab = H_abABC T_ABC.
    2. T' = G X: G would consist of both lower case and upper-case letters, and its
       lower-case letters are to be contracted with X. We assume the contracting rule is
       something like T'_ABC = G_ABCab X_ab.
    3. T' = G H T = S T: S would consist of only upper-case letters. We assume the
       contracting rule is something like T'_ABC = S_ABCDEF T_DEF, where the first
       half of the indices are associated with the embedded tensor T', while the
       latter half of the indices are associated with the original tensor T.

    Args:
        tp: Tensor product of Kronecker delta and Levi-Civita tensors.
        mode: which mode to use, either `G`, `H`, or `S`. This determines how the
            output indices are ordered.
        dtype: Floating-point dtype of the evaluated tensor.

    Returns:
        Tensor product of Kronecker delta and Levi-Civita tensors.
    """
    delta_rules = []
    epsilon_rules = []
    for t in tp.components:
        if isinstance(t, Delta):
            delta_rules.append(t.indices)
        elif isinstance(t, Epsilon):
            epsilon_rules.append(t.indices)
        else:
            raise ValueError(f"Unknown tensor type: {type(t)}")

    # The tensor product actually has no delta or epsilon tensors
    if not delta_rules and not epsilon_rules:
        return torch.tensor(float(tp.factor), dtype=dtype)

    left = ",".join(delta_rules + epsilon_rules)

    # Since the tensors only consists of delta and epsilon, the left rule should be OK,
    # because of the `major` symmetric in them, e.g. d_ij d_kl = d_kl d_ij.
    # But, the right rule should be ordered according to the mode.
    # The order of individual index in lower indices (or upper indices) in the right
    # does not matter when consider Z = HT and such cases, since this function is only
    # since H is symmetric.
    #
    # TODO
    # The order of the indices in the right rule can matter when we do
    # Z = H:XY (see H_tp.py), because H carries three set of indices then. That's why
    # we sort the indices. This is definitely abuse of this function. It is designed
    # only for cases like Z = HT, so we need to create a new function just like this
    # for Z = H:XY.
    right = "".join(delta_rules + epsilon_rules)
    lower = sorted([c for c in right if c.islower()])
    upper = sorted([c for c in right if c.isupper()])
    if mode == "G" or mode == "S":
        right = "".join(upper + lower)
    elif mode == "H":
        right = "".join(lower + upper)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    rule = left + "->" + right

    contracted = _contract_delta_epsilon(
        rule, len(delta_rules), len(epsilon_rules), dtype
    )

    # multiply factor; this is out of place, so the cached tensor is left untouched
    product = contracted * float(tp.factor)

    return product


def evaluate_tensors(
    tensors: LinearCombination, mode: str, dtype: torch.dtype = None
) -> Tensor:
    """
    Evaluate the tensor product of Kronecker delta and Levi-Civita tensors to get
    numerical values.
    """

    # Evaluate each tensor product
    output = 0
    for tp in tensors.components:
        if isinstance(tp, TensorProduct):
            output += tp_delta_epsilon(tp, mode, dtype=dtype)
        else:
            raise ValueError(f"Unknown tensor type: {type(tp)}")

    return output


def extract(H: LinearCombination, T: Tensor) -> Tensor:
    r"""
    Evaluate X(j) = H(j|n) \odot^n T(n).

    In G, lower case indices are for r1, r2, ..., rj, and upper case indices are for
    s1, s2, ..., sn. Here, the upper indices are to be contracted away.

    Args:
        H: the contraction rule.
        T: the ordinary tensor T(n) to contract with H.

    Returns:
        X(j) in the space j.
    """
    # Get numerical values of H

    H = simplify_linear_combination(H)
    H_num = evaluate_tensors(H, mode="H")

    n = T.dim()
    j = H_num.dim() - n

    lower = letter_index(j)
    upper = letter_index(n, upper_case=True)
    H_indices = lower + upper
    T_indices = upper
    X_indices = lower
    rule = f"{H_indices},{T_indices}->{X_indices}"

    out = torch.einsum(rule, H_num, T)

    return out


def embed(G: LinearCombination, X: Tensor) -> Tensor:
    r"""
    Evaluate S(n) = G(n|j) \odot^j X(j).

    In G, lower case indices are for r1, r2, ..., rj, and upper case indices are for
    s1, s2, ..., sn. Here, the lower indices are to be contracted away.

    Args:
        G: the contraction rule.
        X: the natural tensor X(j) to contract with G.

    Return:
        S(n) in the space n.
    """
    # Get numerical values of G
    G = simplify_linear_combination(G)
    G_num = evaluate_tensors(G, mode="G")

    j = X.dim()
    n = G_num.dim() - j

    lower = letter_index(j)
    upper = letter_index(n, upper_case=True)
    G_indices = upper + lower
    X_indices = lower
    S_indices = upper
    rule = f"{G_indices},{X_indices}->{S_indices}"

    out = torch.einsum(rule, G_num, X)

    return out
