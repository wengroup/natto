"""Snapshots of the operators natto publishes.

One snapshot per physical tensor class of Table III of [Wen2026Reusable]: for each
weight, the exact Gram matrix of its mappings and the symbolic terms of every embedding
operator, which together record the basis natto publishes. The terms are stored one per
line, so a diff shows which ones changed. Regenerate with `NATTO_REGOLD=1 pytest`, and
read the diff before committing.

The extraction operators are fixed by the mappings and their Gram matrix, and
`test_reduction` checks them by exact duality and the round trip. Their terms are still
stored up to rank three, where several weights have more than one channel and the
inverse Gram matrix mixes them, as a guard on how the duals are formed. Above rank four
only the Gram matrices are stored: the operators' terms run to megabytes there, and a
change of basis changes the Gram matrices too.
"""

import pytest

from natto.rational import fraction_matrix
from natto.reduction import get_gram_matrices
from tests.golden import assert_snapshot
from tests.test_reduction import get_reduction_cached, get_tensor_class_params

#: `Operator.__str__` joins the terms of an operator with two spaces.
TERM_SEPARATOR = "  "

#: The highest rank whose extraction operator terms are stored.
MAX_EXTRACTION_RANK = 3

#: The highest rank whose embedding operator terms are stored; above it, only Gram
#: matrices.
MAX_TERMS_RANK = 4


def operator_content(rank: int, symmetry: str | None) -> dict:
    """The Gram matrices and operator terms of one class, keyed by weight.

    Args:
        rank: Rank of the Cartesian tensor.
        symmetry: Intrinsic symmetry of the class, or None for a generic tensor.

    Returns:
        For each weight, its Gram matrix, up to `MAX_TERMS_RANK` the terms of each
        embedding operator, and up to `MAX_EXTRACTION_RANK` those of each extraction
        operator, in channel order.
    """
    grams = get_gram_matrices(rank, symmetry=symmetry)
    if rank > MAX_TERMS_RANK:
        content = {
            weight: {"gram": fraction_matrix(gram)} for weight, gram in grams.items()
        }

        return content

    keys = ["embedding"]
    if rank <= MAX_EXTRACTION_RANK:
        keys.append("extraction")

    content = {}
    for (weight, _), operators in get_reduction_cached(rank, symmetry).items():
        if weight not in content:
            content[weight] = {"gram": fraction_matrix(grams[weight])}
            content[weight].update({key: [] for key in keys})
        for key in keys:
            terms = str(operators[key]).split(TERM_SEPARATOR)
            content[weight][key].append(terms)

    return content


@pytest.mark.parametrize("tensor_class", get_tensor_class_params())
def test_operator_content(tensor_class):
    """Pin the operators and Gram matrices of one Table III class."""
    assert_snapshot(
        tensor_class.test_id,
        operator_content(tensor_class.rank, tensor_class.symmetry),
    )
