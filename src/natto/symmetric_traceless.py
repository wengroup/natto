"""Making a numerical tensor symmetric and traceless.

An ICT is a tensor that is symmetric in all its indices and traceless on every pair
of them. This module is the numerical route to one: average a numpy array over the
permutations of its indices, then subtract the delta terms that carry its traces. It
is the array-level counterpart of `natural_projector`, which does the same thing
symbolically and exactly, and is much the cheaper of the two at high rank because it
works on the array rather than on the projector.

The index permutations and contraction rules the routines here work through are built
in `indices`; this module only applies them to arrays.

References:
    [JCB78] J. Jerphagnon, D. Chemla, and R. Bonneville, The description of the
    physical properties of condensed matter using irreducible tensors, Advances in
    Physics 27, 609 (1978).
"""

import itertools

import numpy as np

from natto.indices import get_permutations, get_permutations_2, remove_trace_rule
from natto.utils import dij


def symmetrize_via_permutation(
    t: np.ndarray, perms: list[list[int]], mode: str = "sum"
) -> np.ndarray:
    """
    Symmetrize a tensor by summing/averaging over all permutations.

    Args:
        t: The tensor to symmetrize.
        perms: Permutations of the indices for symmetrization.
        mode: The mode of symmetrization. For `sum`, summation is performed over all
            permutations. For `mean`, the average is taken.

    Returns:
        The symmetrized tensor.
    """
    if mode == "sum":
        return np.stack([np.transpose(t, p) for p in perms]).sum(axis=0)
    elif mode == "mean":
        return np.stack([np.transpose(t, p) for p in perms]).mean(axis=0)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def symmetrize_and_remove_trace(
    t: np.ndarray, start_dim: int = 0, symmetry: str = None
) -> np.ndarray:
    """
    Symmetrize and remove the trace of a (generic) tensor.

    This only extracts the symmetric traceless part of the tensor at the same rank.
    The antisymmetric part is totally ignored, which can further be decomposed into
    symmetric and traceless tensors of lower ranks.

    Args:
        t: input tensor
        start_dim: the starting dimension to perform the operation. Dimensions before
            `start_dim` will not be used in the operation.
        symmetry: a string that describes the symmetry of the indices. For example,
            `abba` means the first and the fourth indices are symmetric, and the
            second and the third indices are symmetric. Default is None, which means
            there is no symmetry between

    Returns:
        A symmetric traceless tensor of the same rank as the input tensor.
    """
    return remove_trace(symmetrize(t, start_dim, symmetry), start_dim)


def symmetrize(
    t: np.ndarray, start_dim: int = 0, symmetry: str = None, mode: str = "mean"
) -> np.ndarray:
    """
    Symmetrize a tensor.

    The symmetrization is done by averaging/sum over unique permutations of the indices,
    considering the symmetry of the indices.

    Args:
        t: The tensor to symmetrize
        start_dim: the starting dimension to perform the operation. Dimensions before
            `start_dim` will not be used in the operation.
        symmetry: A string that describes the symmetry of the indices. For example,
            `abba` means the first and the fourth indices are symmetric, and the
            second and the third indices are symmetric. Default is None, which means
            there is no symmetry between the indices.
        mode: `mean` or `sum`. If `mean`, the tensor is averaged over the permutations.
            If `sum`, the tensor is summed over the permutations.

    Returns:
        The symmetrized tensor.
    """

    # fully symmetrize the tensor
    if symmetry is None:
        permutations = itertools.permutations(range(start_dim, t.ndim))
        if start_dim > 0:
            prefix = list(range(start_dim))
            permutations = [prefix + list(p) for p in permutations]

    # symmetrize with the given symmetry
    else:
        assert start_dim + len(symmetry) == t.ndim, (
            "The length of the symmetry string must match the tensor shape."
        )
        permutations = get_permutations(symmetry, start_dim)

    if mode == "mean":
        u = np.mean(np.stack([np.transpose(t, p) for p in permutations]), axis=0)
    elif mode == "sum":
        u = np.sum(np.stack([np.transpose(t, p) for p in permutations]), axis=0)
    else:
        raise ValueError("The mode must be either 'mean' or 'sum'.")

    return u


def symmetrize_2(t: np.ndarray, num_delta: int, start_dim: int = 0) -> np.ndarray:
    """
    Symmetrize a tensor that is obtained by contracting a symmetric tensor with deltas.

    Symmetrization is done by summation over unique permutations of the indices,
    considering the three set of symmetries. See `get_permutations_2` for more details.

    Args:
        t: the tensor
        num_delta: number of deltas used to obtain the tensor
        start_dim: the starting dimension to perform the operation. Dimensions before
            `start_dim` will not be used in the operation.

    Returns:
        A symmetrized tensor.
    """
    # rank of the symmetric tensor, after considering the start_dim
    m = t.ndim - start_dim

    # Get unique permutations
    permutations = get_permutations_2(m, num_delta, start_dim)

    # Sum over the permutations
    u = np.sum(np.stack([np.transpose(t, p) for p in permutations]), axis=0)

    return u


# TODO, this can be refactored to be similar as unit_vector.py
def remove_trace(u: np.ndarray, start_dim: int = 0) -> np.ndarray:
    """Remove the trace of a symmetric tensor, leaving an ICT of the same rank.

    The traces are subtracted off as a signed sum over the number of delta pairs taken
    out, each term symmetrized and weighted by a ratio of double factorials. The sum runs
    to half the rank, rounded down.

    Args:
        u: A fully symmetric tensor.
        start_dim: The starting dimension to perform the operation. Dimensions before
            `start_dim` will not be used in the operation.

    Returns:
        An ICT of the same shape as the input tensor.

    References:
        Eq. 10 of [JCB78]. This is one of the few places the implementation does not
        follow [Wen2026]: it is the explicit numerical formula, and it works on the
        rank-ell array rather than on the rank-2*ell natural projector that does
        the same thing exactly, so it stays much the cheaper route at high weight.
    """

    m = u.ndim - start_dim
    D = m // 2

    delta = dij()
    coeff = 1
    out = u
    for d in range(1, D + 1):
        rule = remove_trace_rule(m, d)

        # Contract with multiple deltas to get a tensor of the same rank as u
        prod = np.einsum(rule, u, *([delta] * d))

        prod = symmetrize_2(prod, num_delta=d, start_dim=start_dim)

        # coeff = (-1) ** d / double_factorial(2 * m - 1, 2 * m - 2 * d - 1 + 2)
        coeff = -coeff / (2 * m - 2 * d + 1)

        out = out + coeff * prod

    return out


# TODO, this is a generalization of get_sym_rule_2 and get_sym_rule_3 in tensor_product1.py
#  Can we merge them?


def get_random_natural_tensor(n: int, seed: int = 35) -> np.ndarray:
    """
    Create a random symmetric traceless tensor of the given rank.
    """
    X = np.random.default_rng(seed).standard_normal((3,) * n)
    X = symmetrize_and_remove_trace(X)

    return X
