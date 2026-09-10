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
carrying nothing they do not, and `test_mappings.test_reduction_round_trip` already asserts
`G (H T) == S T` for every class, so pinning the two pins `S` with them.

The classes come from `test_mappings.PHYSICAL_TENSOR_CLASSES` rather than being
restated here, so the table stays one statement about the physics.

Rank six is snapshotted too, since it carries the paper's new result, but only
at weights zero to three; the rest cost minutes each.

Most of these snapshots are marked `slow` and left out of the default run: the
rank-six pair, and every rank-four class but the elastic one. Regolding is
therefore something to do with `pytest -m full`, since a plain `NATTO_REGOLD=1
pytest` rewrites only the snapshots the default run reaches and leaves the rest
holding their old values.

Nothing above rank four is committed. Those snapshots are large -- a megabyte
for the generic rank-six class -- and slow to produce, so they live in the
gitignored `local` directory and their test skips when they are absent.
Generate them once at the start of a porting session and they guard the rest of
it; a fresh clone gets the rank-four coverage and a skip.
"""

import functools
from pathlib import Path

import pytest
import torch

from natto.algebra import simplify_linear_combination
from natto.evaluate import evaluate_tensors
from natto.mappings import (
    get_dual_pair_of_weight,
    get_reduction,
    get_reduction_of_weight,
)
from natto.matrix import fraction_matrix
from natto.utils import letter_index
from tests.golden import (
    LOCAL_SNAPSHOT_DIR,
    assert_snapshot,
    fingerprint,
    regolding,
    snapshot_path,
)
from tests.test_mappings import get_reduction_cached, get_tensor_class_params

#: `LinearCombination.__str__` joins the terms of an operator with two spaces.
TERM_SEPARATOR = "  "

#: The package evaluates operators in float32 by default, so the fingerprint of a
#: rank-8 operator sums a few thousand such entries and re-associating that sum
#: moves the total by around 1e-5 relative. The symbolic form is what pins the
#: content exactly; this tolerance only has to be tight enough to notice a
#: different basis, which is order one.
FINGERPRINT_RTOL = 1e-5

#: Rank six is where the paper's new result lives, so it is snapshotted even
#: though no rank-six row is in `PHYSICAL_TENSOR_CLASSES`.
RANK_SIX = 6

#: Only the affordable sectors. Rank-six weight 4 takes about three minutes and
#: weight 5 about thirty-four, at nearly four gigabytes, so both are left out;
#: the multiplicities there are covered by the audit rather than by a test.
RANK_SIX_WEIGHTS = (0, 1, 2, 3)

#: Path named in the skip message, so it can be pasted into a shell.
TEST_FILE = (
    Path(__file__).relative_to(Path.cwd())
    if Path(__file__).is_relative_to(Path.cwd())
    else Path(__file__)
)

#: The third-order elastic tensor, class ((ij)(kl)(mn)), written with the
#: paper's minimal generating set: exchange within the first pair, exchange of
#: the first pair with the second, and of the second with the third.
THIRD_ORDER_ELASTIC = "ijklmn=jiklmn=klijmn=ijmnkl"


@functools.lru_cache(maxsize=None)
def get_orthonormal_cached(rank: int, symmetry: str | None) -> dict:
    """The reduction in the self-dual basis, once per class; rank 4 costs seconds."""
    return get_reduction(rank, symmetry, basis="orthonormal", dtype=torch.float64)


def dual_pair_content(rank: int, symmetry: str | None) -> dict:
    """Collect the dual-pair operators of one class into snapshot form.

    Args:
        rank: Rank of the Cartesian tensor.
        symmetry: Intrinsic symmetry of the class, or None for a generic tensor.

    Returns:
        The operators keyed by weight.
    """
    output = get_reduction_cached(rank, symmetry)

    content = {}
    for weight, per_weight in output.items():
        content[weight] = {
            "gram": per_weight["gram"]["symbolic"],
            "gram_inverse": per_weight["gram_inverse"]["symbolic"],
            "embedding": [_operator(entry) for entry in per_weight["embedding"]],
            "extraction": [_operator(entry) for entry in per_weight["extraction"]],
        }

    return content


def orthonormal_content(rank: int, symmetry: str | None) -> dict:
    """Collect the orthonormal operators of one class into snapshot form.

    These have no symbolic form -- the inverse square root of the Gram matrix is
    irrational -- so the Gram matrix and its inverse square root are stored in
    full, being small, and the operators themselves by fingerprint.

    One array per channel now serves as both embedding and extraction, so the
    rules could be recorded once for the weight instead of once per operator.
    They are not: keeping the shape the stored snapshots already have is what
    lets these files stay byte-identical across the move to a `basis` argument,
    and unchanged files are the evidence that the move touched no numbers.

    Args:
        rank: Rank of the Cartesian tensor.
        symmetry: Intrinsic symmetry of the class, or None for a generic tensor.

    Returns:
        The operators keyed by weight.
    """
    output = get_orthonormal_cached(rank, symmetry)

    content = {}
    for weight, per_weight in output.items():
        content[weight] = {
            "gram": per_weight["gram"],
            "gram_inverse_sqrt": per_weight["gram_inverse_sqrt"],
            "orthonormal": [
                {
                    "extraction_rule": extraction["rule"],
                    "embedding_rule": embedding["rule"],
                    "numerical": fingerprint(embedding["numerical"]),
                }
                for embedding, extraction in zip(
                    per_weight["embedding"], per_weight["extraction"]
                )
            ],
        }

    return content


def rank_six_content(symmetry: str | None) -> dict:
    """Collect the affordable rank-six sectors of one class into snapshot form.

    `get_reduction` cannot be used here. It walks every weight, which would pull in
    the two that cost minutes, and it evaluates `S`, which at rank six is a
    rank-12 array of half a million entries per channel. This goes weight by
    weight instead and builds only what is snapshotted.

    Args:
        symmetry: Intrinsic symmetry of the class, or None for a generic tensor.

    Returns:
        The operators keyed by weight. A weight the symmetry extinguishes is
        recorded as a multiplicity of zero rather than omitted, since its
        absence is a result in its own right.
    """
    lower_by_weight = {weight: letter_index(weight) for weight in RANK_SIX_WEIGHTS}
    upper = letter_index(RANK_SIX, upper_case=True)

    content = {}
    for weight in RANK_SIX_WEIGHTS:
        if symmetry is None:
            embedding, extraction, gram, gram_inverse = get_dual_pair_of_weight(
                weight, RANK_SIX
            )
        else:
            embedding, extraction, _, gram, gram_inverse = get_reduction_of_weight(
                weight, RANK_SIX, symmetry
            )

        if not embedding:
            content[weight] = {"multiplicity": 0}
            continue

        lower = lower_by_weight[weight]
        content[weight] = {
            "multiplicity": len(embedding),
            "gram": fraction_matrix(gram),
            "gram_inverse": fraction_matrix(gram_inverse),
            "embedding": [
                _rank_six_operator(
                    operator, "embedding", f"{upper}{lower},...{lower}->...{upper}"
                )
                for operator in embedding
            ],
            "extraction": [
                _rank_six_operator(
                    operator, "extraction", f"{lower}{upper},...{upper}->...{lower}"
                )
                for operator in extraction
            ],
        }

    return content


def _rank_six_operator(operator, mode: str, rule: str) -> dict:
    """Snapshot form of one rank-six operator, evaluated on the spot.

    The rank-four path takes its operators from `get_reduction`, which has already
    simplified and evaluated them. Here they arrive raw, so both steps happen
    here.
    """
    simplified = simplify_linear_combination(operator)

    return {
        "symbolic": str(simplified).split(TERM_SEPARATOR),
        "rule": rule,
        "numerical": fingerprint(evaluate_tensors(simplified, mode=mode)),
    }


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


@pytest.mark.slow
@pytest.mark.parametrize(
    "name,symmetry", [("generic", None), ("third_order_elastic", THIRD_ORDER_ELASTIC)]
)
def test_rank_six_operator_content(name: str, symmetry: str | None):
    """Pin the affordable rank-six sectors of the paper's two rank-six rows.

    Weights 0 to 3 only; see `RANK_SIX_WEIGHTS` for why the rest are left out.

    Args:
        name: Short label used as the snapshot name.
        symmetry: Intrinsic symmetry of the class, or None for a generic tensor.
    """
    snapshot = f"rank6_{name}"
    if not regolding() and not snapshot_path(snapshot, LOCAL_SNAPSHOT_DIR).exists():
        pytest.skip(
            f"No local snapshot for {snapshot}. Rank-six snapshots are not "
            f"committed; create them with "
            f"NATTO_REGOLD=1 pytest -m full {TEST_FILE}::test_rank_six_operator_content"
        )

    assert_snapshot(
        snapshot,
        rank_six_content(symmetry),
        rtol=FINGERPRINT_RTOL,
        directory=LOCAL_SNAPSHOT_DIR,
    )
