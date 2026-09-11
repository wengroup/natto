import gzip
import itertools
import math
from pathlib import Path
from typing import Optional

import numpy as np
import yaml
from numpy.typing import DTypeLike

from natto.indices import letter_index


def dij(dtype: DTypeLike = None) -> np.ndarray:
    """Kronecker delta tensor."""
    return np.eye(3, dtype=dtype)


def eijk(dtype: DTypeLike = None) -> np.ndarray:
    """Levi-Civita tensor."""
    e = np.zeros((3, 3, 3), dtype=dtype)
    e[0, 1, 2] = 1.0
    e[1, 2, 0] = 1.0
    e[2, 0, 1] = 1.0
    e[0, 2, 1] = -1.0
    e[1, 0, 2] = -1.0
    e[2, 1, 0] = -1.0

    return e


def factorial(n: int) -> int:
    """Get the factorial of a number.

    Exact, as a Python integer. These counts appear in the denominators of the
    exact coefficients, where a float would silently stop being exact well
    before the weights this package reaches.

    Args:
        n: The number to take the factorial of.

    Returns:
        `n!`.
    """
    return math.factorial(n)


def double_factorial(n: int, lower_bound: Optional[int] = None) -> int:
    """Get the double factorial of a number, exactly.

    Args:
        n: The number to calculate the double factorial.
        lower_bound: The lower bound of the double factorial. If given, this is
            calculated as n * (n-2) * ... * lower_bound. Default is None,
            meaning 1 if n is odd and 2 if n is even.

    Returns:
        `n!!`, or the bounded product; 1 when the product is empty, which is the
        convention that makes `(-1)!! = 1`.
    """
    if n == 0 or n == 1:
        return 1

    if n % 2 == 0:
        if lower_bound is None:
            lower_bound = 2
        else:
            assert lower_bound % 2 == 0, "lower_bound must be even"
    else:
        if lower_bound is None:
            lower_bound = 1
        else:
            assert lower_bound % 2 == 1, "lower_bound must be odd"

    return math.prod(range(lower_bound, n + 2, 2))


def get_trace(T: np.ndarray, i: int, j: int) -> np.ndarray:
    """
    Trace of a tensor between two indices.

    Args:
        T: input tensor
        i: first index
        j: second index

    Example:
        T_ijkl -> T_ijil
    """

    assert i < T.ndim and j < T.ndim, "Index out of range"

    indices = letter_index(T.ndim)
    rule = indices.replace(indices[j], indices[i])
    trace = np.einsum(rule, T)

    return trace


def is_symmetric(
    T: np.ndarray, start_dim: int = 0, atol: float = 1e-6, rtol: float = 1e-5
) -> bool:
    """
    Check if a tensor is fully symmetric.

    Args:
        T: input tensor
        start_dim: the starting dimension to check symmetry
    """

    if T.ndim - start_dim <= 1:
        return True

    for p in itertools.permutations(range(start_dim, T.ndim)):
        p = list(range(start_dim)) + list(p)
        permuted = np.transpose(T, p)
        if not np.allclose(T, permuted, atol=atol, rtol=rtol):
            return False

    return True


def is_traceless(T, start_dim: int = 0, atol: float = 1e-6, rtol: float = 1e-5) -> bool:
    """Check if a tensor is traceless.

    Args:
        T: input tensor
        start_dim: the starting dimension to check tracelessness
    """

    rank = T.ndim - start_dim

    if rank <= 1:
        return True

    for i, j in itertools.combinations(range(start_dim, T.ndim), 2):
        trace = get_trace(T, i, j)
        if not np.allclose(trace, np.zeros_like(trace), atol=atol, rtol=rtol):
            return False

    return True


def is_symmetric_traceless(
    T: np.ndarray, atol: float = 1e-6, rtol: float = 1e-5
) -> bool:
    """Check if a tensor is symmetric and traceless."""
    return is_symmetric(T, atol=atol, rtol=rtol) and is_traceless(
        T, atol=atol, rtol=rtol
    )


def yaml_dump(obj: dict, filename: Path, compress: bool = True) -> None:
    """Dump a dictionary to a yaml file.

    Args:
        obj: The dictionary to dump.
        filename: The path to the yaml file.
        compress: Whether to compress the file using gzip. Default is True.
    """
    if compress:
        filename = filename.with_suffix(filename.suffix + ".gz")
        with gzip.open(filename, "wt") as f:
            yaml.dump(obj, f)
    else:
        with open(filename, "w") as f:
            yaml.dump(obj, f)
