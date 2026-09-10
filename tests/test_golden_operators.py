"""Snapshots of the operator content natto publishes.

One snapshot per physical tensor class of the paper's Table III, at rank four
and below. Each records, per weight, the exact symbolic form of every embedding
and extraction operator, the exact rational Gram matrix and its inverse, the
einsum rules, and a fingerprint of the evaluated arrays.

The symbolic form is the operator's content, and it is what makes these
snapshots worth reading: a change of basis rewrites those strings, so a diff
here says what moved rather than merely that something did. The evaluated
arrays are not stored in full because they are a deterministic function of the
symbolic form and run to hundreds of kilobytes per class; `golden.fingerprint`
guards the evaluation instead.

`S` is not snapshotted. It is the composition `G` and the extraction operator,
carrying nothing they do not, and `test_GHS.test_get_G_H_S` already asserts
`G (H T) == S T` for every class, so pinning the two pins `S` with them.

The classes come from `test_GHS.PHYSICAL_TENSOR_CLASSES` rather than being
restated here, so the table stays one statement about the physics.
"""

import functools

import pytest

from natto.GHS import get_orthonormal_Q
from tests.golden import assert_snapshot, fingerprint
from tests.test_GHS import get_G_H_S_cached, get_tensor_class_params

#: `LinearCombination.__str__` joins the terms of an operator with two spaces.
TERM_SEPARATOR = "  "

#: The package evaluates operators in float32 by default, so the fingerprint of a
#: rank-8 operator sums a few thousand such entries and re-associating that sum
#: moves the total by around 1e-5 relative. The symbolic form is what pins the
#: content exactly; this tolerance only has to be tight enough to notice a
#: different basis, which is order one.
FINGERPRINT_RTOL = 1e-5


@functools.lru_cache(maxsize=None)
def get_orthonormal_Q_cached(rank: int, symmetry: str | None) -> dict:
    """`get_orthonormal_Q`, computed once per class; rank 4 takes a few seconds."""
    return get_orthonormal_Q(rank, symmetry)


def dual_pair_content(rank: int, symmetry: str | None) -> dict:
    """Collect the dual-pair operators of one class into snapshot form.

    Args:
        rank: Rank of the Cartesian tensor.
        symmetry: Intrinsic symmetry of the class, or None for a generic tensor.

    Returns:
        The operators keyed by weight.
    """
    output = get_G_H_S_cached(rank, symmetry)

    content = {}
    for weight, per_weight in output.items():
        content[weight] = {
            "gram": per_weight["g_pq"]["symbolic"],
            "gram_inverse": per_weight["h_pq"]["symbolic"],
            "embedding": [_operator(entry) for entry in per_weight["G"]],
            "extraction": [_operator(entry) for entry in per_weight["H"]],
        }

    return content


def orthonormal_content(rank: int, symmetry: str | None) -> dict:
    """Collect the orthonormal operators of one class into snapshot form.

    These have no symbolic form -- the inverse square root of the Gram matrix is
    irrational -- so the Gram matrix and its inverse square root are stored in
    full, being small, and the operators themselves by fingerprint.

    Args:
        rank: Rank of the Cartesian tensor.
        symmetry: Intrinsic symmetry of the class, or None for a generic tensor.

    Returns:
        The operators keyed by weight.
    """
    output = get_orthonormal_Q_cached(rank, symmetry)

    content = {}
    for weight, per_weight in output.items():
        content[weight] = {
            "gram": per_weight["g_Q"],
            "gram_inverse_sqrt": per_weight["g_Q_inverse_sqrt"],
            "orthonormal": [
                {
                    "extraction_rule": entry["extraction_rule"],
                    "embedding_rule": entry["embedding_rule"],
                    "numerical": fingerprint(entry["numerical"]),
                }
                for entry in per_weight["Q_tilde"]
            ],
        }

    return content


def _operator(entry: dict) -> dict:
    """Snapshot form of one operator: symbolic exactly, evaluated by fingerprint.

    The symbolic form is stored as one string per term rather than as the single
    joined string, which is what makes a diff worth reading: a changed
    coefficient rewrites one short line instead of one line of several thousand
    characters. `LinearCombination.__str__` joins `to_str_list()` with two
    spaces, so splitting on two spaces recovers exactly those terms.
    """
    return {
        "symbolic": entry["symbolic"].split(TERM_SEPARATOR),
        "rule": entry["rule"],
        "numerical": fingerprint(entry["numerical"]),
    }


@pytest.mark.parametrize("tensor_class", get_tensor_class_params())
def test_dual_pair_operator_content(tensor_class):
    """Pin the embedding and extraction operators of one Table III class.

    Args:
        tensor_class: physical tensor class to snapshot
    """
    assert_snapshot(
        f"dual_pair_{tensor_class.test_id}",
        dual_pair_content(tensor_class.rank, tensor_class.symmetry),
        rtol=FINGERPRINT_RTOL,
    )


@pytest.mark.parametrize("tensor_class", get_tensor_class_params())
def test_orthonormal_operator_content(tensor_class):
    """Pin the orthonormal operators of one Table III class.

    Args:
        tensor_class: physical tensor class to snapshot
    """
    assert_snapshot(
        f"orthonormal_{tensor_class.test_id}",
        orthonormal_content(tensor_class.rank, tensor_class.symmetry),
        rtol=FINGERPRINT_RTOL,
    )
