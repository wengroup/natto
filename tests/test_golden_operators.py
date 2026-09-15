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
carrying nothing they do not, and `test_reduction.test_reduction_round_trip` already asserts
`G (H T) == S T` for every class, so pinning the two pins `S` with them.

The classes come from `test_reduction.PHYSICAL_TENSOR_CLASSES` rather than being
restated here, so the table stays one statement about the physics.

Rank six is snapshotted too, since it carries the paper's new result, but only
at weights zero to three; the rest cost minutes each.

Regolding is `NATTO_REGOLD=1 pytest`, which rewrites every snapshot the run
reaches.

Nothing above rank four is committed. Those snapshots are large -- a megabyte
for the generic rank-six class -- and slow to produce, so they live in the
gitignored `local` directory and their test skips when they are absent.
Generate them once at the start of a porting session and they guard the rest of
it; a fresh clone gets the rank-four coverage and a skip.
"""

from pathlib import Path

import pytest

from natto.orthonormal import get_inverse_square_root
from natto.rational import float_matrix, fraction_matrix, matrix_inverse
from natto.reduction import (
    get_embedding_operators,
    get_extraction_operators,
    get_gram_matrices,
)
from natto.symbolic import evaluate
from tests.golden import (
    LOCAL_SNAPSHOT_DIR,
    assert_snapshot,
    fingerprint,
    regolding,
    snapshot_path,
)
from tests.test_reduction import (
    get_orthonormal_cached,
    get_reduction_cached,
    get_tensor_class_params,
)

#: `Operator.__str__` joins the terms of an operator with two spaces.
TERM_SEPARATOR = "  "

#: The fingerprint of a rank-8 operator sums a few thousand float64 entries, and
#: re-associating that sum -- between BLAS builds, say -- moves the total in its
#: last bits. The suite passes at 1e-12; this leaves two orders on that. The
#: symbolic form is what pins the content exactly, so this only has to be tight
#: enough to notice a different basis, which is order one.
FINGERPRINT_RTOL = 1e-10

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


def dual_pair_content(rank: int, symmetry: str | None) -> dict:
    """Collect the dual-pair operators of one class into snapshot form.

    Args:
        rank: Rank of the Cartesian tensor.
        symmetry: Intrinsic symmetry of the class, or None for a generic tensor.

    Returns:
        The operators keyed by weight.
    """
    output = get_reduction_cached(rank, symmetry)
    grams = get_gram_matrices(rank, symmetry=symmetry)

    content = {}
    for (weight, _), operators in output.items():
        if weight not in content:
            content[weight] = {
                "gram": fraction_matrix(grams[weight]),
                "gram_inverse": fraction_matrix(matrix_inverse(grams[weight])),
                "embedding": [],
                "extraction": [],
            }
        content[weight]["embedding"].append(_operator(operators["embedding"]))
        content[weight]["extraction"].append(_operator(operators["extraction"]))

    return content


def orthonormal_content(rank: int, symmetry: str | None) -> dict:
    """Collect the orthonormal operators of one class into snapshot form.

    These have no symbolic form -- the inverse square root of the Gram matrix is
    irrational -- so the Gram matrix and its inverse square root, whose rows are the
    weights of the operators, are stored in full, being small, and the operators
    themselves by fingerprint.

    Args:
        rank: Rank of the Cartesian tensor.
        symmetry: Intrinsic symmetry of the class, or None for a generic tensor.

    Returns:
        The operators keyed by weight.
    """
    output = get_orthonormal_cached(rank, symmetry)
    grams = get_gram_matrices(rank, symmetry=symmetry)

    content = {}
    for (weight, _), operators in output.items():
        if weight not in content:
            content[weight] = {
                "gram": float_matrix(grams[weight]),
                "gram_inverse_sqrt": get_inverse_square_root(grams[weight]),
                "orthonormal": [],
            }
        embedding, embedding_rule = evaluate(operators["embedding"])
        _, extraction_rule = evaluate(operators["extraction"])
        content[weight]["orthonormal"].append(
            {
                "extraction_rule": extraction_rule,
                "embedding_rule": embedding_rule,
                "numerical": fingerprint(embedding),
            }
        )

    return content


def rank_six_content(symmetry: str | None) -> dict:
    """Collect the affordable rank-six sectors of one class into snapshot form.

    Only the weights snapshotted are built, since the others cost the most.

    Args:
        symmetry: Intrinsic symmetry of the class, or None for a generic tensor.

    Returns:
        The operators keyed by weight. A weight the symmetry extinguishes is
        recorded as a multiplicity of zero rather than omitted, since its
        absence is a result in its own right.
    """
    content = {}
    for weight in RANK_SIX_WEIGHTS:
        embeddings = get_embedding_operators(RANK_SIX, ell=weight, symmetry=symmetry)
        if not embeddings:
            content[weight] = {"multiplicity": 0}
            continue

        extractions = get_extraction_operators(RANK_SIX, ell=weight, symmetry=symmetry)
        gram = get_gram_matrices(RANK_SIX, ell=weight, symmetry=symmetry)[weight]
        content[weight] = {
            "multiplicity": len(embeddings),
            "gram": fraction_matrix(gram),
            "gram_inverse": fraction_matrix(matrix_inverse(gram)),
            "embedding": [_operator(operator) for operator in embeddings.values()],
            "extraction": [_operator(operator) for operator in extractions.values()],
        }

    return content


def _operator(operator) -> dict:
    """Snapshot form of one operator: symbolic exactly, evaluated by fingerprint.

    The symbolic form is stored as one string per term rather than as the single
    joined string, which is what makes a diff worth reading: a changed
    coefficient rewrites one short line instead of one line of several thousand
    characters. `Operator.__str__` joins the terms with two spaces, so splitting
    on two spaces recovers exactly those terms.
    """
    array, rule = evaluate(operator)

    return {
        "symbolic": str(operator).split(TERM_SEPARATOR),
        "rule": rule,
        "numerical": fingerprint(array),
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
            f"NATTO_REGOLD=1 pytest {TEST_FILE}::test_rank_six_operator_content"
        )

    assert_snapshot(
        snapshot,
        rank_six_content(symmetry),
        rtol=FINGERPRINT_RTOL,
        directory=LOCAL_SNAPSHOT_DIR,
    )
