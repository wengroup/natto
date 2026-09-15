"""Natural tensors: the irreducible parts of Cartesian tensors.

Every builder returns an exact operator, and `evaluate` turns one into an array
together with the einsum rule that applies it, while `act` applies it directly.
"""

from ._version import __version__
from .coupling import get_coupling_operator
from .harmonics import get_harmonic_operator
from .natural_projector import get_natural_projector
from .reduction import (
    get_composed_operators,
    get_embedding_operators,
    get_extraction_operators,
    get_gram_matrices,
    get_reduction,
)
from .symbolic import act, evaluate

__all__ = [
    "__version__",
    "act",
    "evaluate",
    "get_composed_operators",
    "get_coupling_operator",
    "get_embedding_operators",
    "get_extraction_operators",
    "get_gram_matrices",
    "get_harmonic_operator",
    "get_natural_projector",
    "get_reduction",
]
