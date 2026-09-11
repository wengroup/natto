"""Choosing which candidate mapping tensors to keep.

The exact scheme is what the reduction uses, so what matters here is that it keeps
the earliest independent candidates, that the Gram matrix it returns is the one for
what it kept, and that the two numerical schemes reach the same answer by looking at
something else entirely.
"""

import pytest

from natto.gram import get_gram_matrix
from natto.independence import (
    select_independent_mappings_and_gram,
    select_independent_mappings_via_components,
    select_independent_mappings_via_embeddings,
)
from natto.mapping_tensors import get_mappings
from natto.rational import is_nonsingular


def get_candidates(weight: int, rank: int):
    return get_mappings(weight, rank)


#: (rank, weight, candidates, kept) for sectors that exercise both the full-rank
#: case and the rank-deficient one. Rank three weight two is the smallest sector
#: with a dependent candidate: three choices of which index pair to contract, but
#: only two independent mappings.
SECTORS = [
    (2, 0, 1, [0]),
    (2, 2, 1, [0]),
    (3, 1, 3, [0, 1, 2]),
    (3, 2, 3, [0, 1]),
    (4, 0, 3, [0, 1, 2]),
    (4, 3, 6, [0, 1, 2]),
    (5, 0, 10, [0, 1, 2, 3, 4, 5]),
]


@pytest.mark.parametrize("rank, weight, n_candidates, expected", SECTORS)
def test_selection_keeps_the_earliest_independent_candidates(
    rank, weight, n_candidates, expected
):
    """The selection is order preserving, so it is reproducible by construction."""
    candidates = get_candidates(weight, rank)
    assert len(candidates) == n_candidates

    kept, _ = select_independent_mappings_and_gram(weight, rank, candidates)

    assert kept == expected


@pytest.mark.parametrize("rank, weight, n_candidates, expected", SECTORS)
def test_selection_returns_the_gram_matrix_of_what_it_kept(
    rank, weight, n_candidates, expected
):
    """The Gram matrix accumulates during selection rather than in a second pass."""
    candidates = get_candidates(weight, rank)

    kept, gram = select_independent_mappings_and_gram(weight, rank, candidates)

    assert gram == get_gram_matrix(weight, rank, [candidates[i] for i in kept])


@pytest.mark.parametrize("rank, weight", [(3, 2), (4, 3), (5, 0)])
def test_every_discarded_candidate_is_genuinely_dependent(rank, weight):
    """Adding back any rejected candidate must make the Gram matrix singular.

    The rank of a Gram matrix is the rank of the tensors it is built from, so a
    rejected candidate that still raised the rank would mean a mapping was lost.
    """
    candidates = get_candidates(weight, rank)
    kept, gram = select_independent_mappings_and_gram(weight, rank, candidates)
    rejected = [i for i in range(len(candidates)) if i not in kept]
    assert rejected, "this sector is meant to be rank deficient"

    for index in rejected:
        subset = [candidates[i] for i in kept] + [candidates[index]]
        enlarged = get_gram_matrix(weight, rank, subset)

        assert len(matrix_rank_rows(enlarged)) == len(gram)


def matrix_rank_rows(matrix):
    """Pivot rows of an exact rational matrix, by Gaussian elimination."""
    rows = [row.copy() for row in matrix]
    pivots, pivot_row = [], 0
    for column in range(len(rows[0])):
        row = next(
            (r for r in range(pivot_row, len(rows)) if rows[r][column] != 0), None
        )
        if row is None:
            continue
        rows[pivot_row], rows[row] = rows[row], rows[pivot_row]
        pivot = rows[pivot_row][column]
        for other in range(len(rows)):
            if other != pivot_row and rows[other][column] != 0:
                factor = rows[other][column] / pivot
                rows[other] = [
                    v - factor * pv for v, pv in zip(rows[other], rows[pivot_row])
                ]
        pivots.append(column)
        pivot_row += 1
        if pivot_row == len(rows):
            break
    return pivots


def test_no_candidates_selects_nothing():
    assert select_independent_mappings_and_gram(2, 2, []) == ([], [])


@pytest.mark.parametrize("rank, weight, n_candidates, expected", SECTORS)
def test_numerical_schemes_agree_with_the_exact_one(
    rank, weight, n_candidates, expected
):
    """The two numerical schemes must reach the exact scheme's answer.

    They look at different things -- the full components of each mapping, and the
    mappings' action on one random ICT -- so agreement is evidence that the rational
    Gram matrix and the evaluated arrays describe the same mappings.
    """
    candidates = get_candidates(weight, rank)

    kept, _ = select_independent_mappings_and_gram(weight, rank, candidates)

    assert select_independent_mappings_via_components(weight, rank, candidates) == kept
    assert select_independent_mappings_via_embeddings(weight, rank, candidates) == kept


def test_components_scheme_rejects_a_mapping_of_the_wrong_rank():
    """The evaluated array must have rank n + l; a mismatch is a caller error."""
    candidates = get_candidates(2, 4)

    with pytest.raises(ValueError, match="expected"):
        select_independent_mappings_via_components(2, 5, candidates)


@pytest.mark.parametrize(
    "select",
    [
        select_independent_mappings_via_components,
        select_independent_mappings_via_embeddings,
    ],
)
def test_numerical_schemes_select_nothing_from_nothing(select):
    assert select(2, 2, []) == []


#: Sectors where the candidates really are rank deficient, so there is more than
#: one valid subset and the choice between them is visible. A full-rank sector
#: cannot tell the schemes apart.
DEFICIENT_SECTORS = [
    pytest.param(3, 2, id="rank3_weight2"),
    pytest.param(4, 3, id="rank4_weight3"),
    pytest.param(5, 0, id="rank5_weight0"),
]


@pytest.mark.parametrize(
    "select",
    [
        select_independent_mappings_via_components,
        select_independent_mappings_via_embeddings,
    ],
)
@pytest.mark.parametrize("rank, weight", DEFICIENT_SECTORS)
def test_gram_schmidt_reaches_the_exact_selection(select, rank, weight):
    """The float Gram-Schmidt route must agree with the exact one, index for index.

    Both prefer the earliest candidates, so they are the same algorithm over
    different arithmetic and there is no licence for them to differ.
    """
    candidates = get_candidates(weight, rank)
    kept, _ = select_independent_mappings_and_gram(weight, rank, candidates)

    assert select(weight, rank, candidates, method="gram_schmidt") == kept


@pytest.mark.parametrize(
    "select",
    [
        select_independent_mappings_via_components,
        select_independent_mappings_via_embeddings,
    ],
)
@pytest.mark.parametrize("rank, weight", DEFICIENT_SECTORS)
def test_pivoted_qr_finds_a_valid_subset_of_the_same_size(select, rank, weight):
    """Pivoted QR may keep a different subset, and that is allowed.

    It orders the candidates by residual norm rather than by position, so of
    several equally valid subsets it need not pick the earliest. What it may not do
    is find the wrong number of them, or keep a dependent one -- so the count must
    match the exact selection and the Gram matrix of what it kept must be
    nonsingular, both checked over the rationals.
    """
    candidates = get_candidates(weight, rank)
    kept, _ = select_independent_mappings_and_gram(weight, rank, candidates)

    chosen = select(weight, rank, candidates, method="scipy_qr")

    assert len(chosen) == len(kept)
    assert is_nonsingular(
        get_gram_matrix(weight, rank, [candidates[i] for i in chosen])
    )


@pytest.mark.parametrize(
    "select",
    [
        select_independent_mappings_via_components,
        select_independent_mappings_via_embeddings,
    ],
)
def test_mapping_schemes_reject_an_unknown_method(select):
    with pytest.raises(ValueError, match="Unknown"):
        select(2, 4, get_candidates(2, 4), method="not_a_method")
