"""Tests for the linear-independence selection schemes in `natto.qr`.

Two situations broke the original unpivoted-QR diagonal test, and both are covered
here for every implementation:

1. A zero (or dependent) tensor early in the list, which makes the later diagonal
   entries of an unpivoted QR factor meaningless.
2. More tensors than components, where the unpivoted loop never even considers the
   trailing tensors as candidates.
"""

import functools
from typing import Callable, NamedTuple, Optional

import numpy as np
import pytest

from natto.gram import get_gram_matrix
from natto.mapping_tensors import get_mappings_even, get_mappings_odd
from natto.qr import (
    find_independent_tensors,
    find_independent_tensors_gram_schmidt,
    find_independent_tensors_qr_unpivoted,
    find_independent_tensors_scipy_qr,
    select_independent_mappings,
    select_independent_mappings_components,
    select_independent_mappings_probe,
)


class Scheme(NamedTuple):
    """One selection scheme of `natto.qr`, and the properties it is expected to have."""

    # identifies the scheme, and used as the test id
    name: str
    # the scheme itself, called as `select(tensors)`
    select: Callable
    # whether it finds the rank even when a degenerate tensor precedes an independent
    # one, or when there are more tensors than components
    rank_revealing: bool
    # whether it walks the tensors in the original order, and so selects the earliest
    # tensors of an equally valid subset. Pivoted QR does not: it reorders the columns
    # by residual norm.
    order_preferring: bool


SCHEMES = [
    Scheme("gram_schmidt", find_independent_tensors_gram_schmidt, True, True),
    # the scipy scheme is exercised both at its default and with pivoting spelled out,
    # to pin the default down
    Scheme("scipy_qr", find_independent_tensors_scipy_qr, True, False),
    Scheme(
        "scipy_qr_pivoting_true",
        functools.partial(find_independent_tensors_scipy_qr, pivoting=True),
        True,
        False,
    ),
    Scheme(
        "scipy_qr_pivoting_false",
        functools.partial(find_independent_tensors_scipy_qr, pivoting=False),
        False,
        True,
    ),
    Scheme("qr_unpivoted", find_independent_tensors_qr_unpivoted, False, True),
]


def get_scheme_params(
    rank_revealing: Optional[bool] = None, order_preferring: Optional[bool] = None
) -> list:
    """Parametrize over the selection schemes with the given properties.

    Args:
        rank_revealing: keep only schemes that are (or are not) rank-revealing;
            None keeps both.
        order_preferring: keep only schemes that are (or are not) order-preferring;
            None keeps both.
    """
    params = []
    for scheme in SCHEMES:
        if rank_revealing is not None and scheme.rank_revealing != rank_revealing:
            continue
        if order_preferring is not None and scheme.order_preferring != order_preferring:
            continue
        params.append(pytest.param(scheme.select, id=scheme.name))

    return params


def get_e(dim: int) -> list[np.ndarray]:
    """Unit basis vectors of dimension `dim`."""
    return list(np.eye(dim))


def assert_valid_selection(tensors, selected, indices, expected_rank: int):
    """Check that `indices` selects a maximal independent subset of `tensors`.

    A rank-deficient set has more than one valid independent subset (the schemes
    differ in which one they pick), so we check the defining properties instead of
    a specific set of indices: the right number of tensors is selected, they are
    independent, and they span the same space as the full set.
    """
    assert indices == sorted(set(indices))
    assert len(indices) == expected_rank
    assert all(t is tensors[i] for t, i in zip(selected, indices))

    if expected_rank == 0:
        return

    matrix_all = np.vstack([t.flatten() for t in tensors]).astype(np.float64)
    matrix_selected = np.vstack([t.flatten() for t in selected]).astype(np.float64)

    assert np.linalg.matrix_rank(matrix_selected) == expected_rank
    assert np.linalg.matrix_rank(matrix_all) == expected_rank


@pytest.mark.parametrize("scheme", get_scheme_params())
def test_valid_regime(scheme):
    """Every scheme finds the rank when the leading tensors are independent.

    This is the regime where the unpivoted diagonal test is valid: no degenerate
    column precedes an independent one, and there are no more tensors than
    components. It is the regime all the existing natto results were computed in,
    so no scheme may disagree about the rank here.
    """
    e = get_e(4)
    tensors = [e[0], e[1], e[2], e[0] + e[1], 2 * e[2]]

    selected, indices = scheme(tensors)

    assert_valid_selection(tensors, selected, indices, expected_rank=3)


@pytest.mark.parametrize("scheme", get_scheme_params(order_preferring=True))
def test_valid_regime_selects_the_leading_tensors(scheme):
    """In that same regime, the order-preferring schemes select the same tensors.

    This pins the claim that replacing the unpivoted scheme changes nothing where
    it used to work.
    """
    e = get_e(4)
    tensors = [e[0], e[1], e[2], e[0] + e[1], 2 * e[2]]

    _, indices = scheme(tensors)

    assert indices == [0, 1, 2]


@pytest.mark.parametrize("scheme", get_scheme_params())
def test_shape_is_irrelevant(scheme):
    """Tensors are flattened, so only the components matter, not the shape."""
    rng = np.random.default_rng(35)
    a = rng.standard_normal((3, 3))
    b = rng.standard_normal((3, 3))
    tensors = [a, b, 2 * a - b]

    selected, indices = scheme(tensors)

    assert_valid_selection(tensors, selected, indices, expected_rank=2)


@pytest.mark.parametrize("scheme", get_scheme_params())
def test_all_independent(scheme):
    """A full-rank set is returned unchanged."""
    e = get_e(3)

    selected, indices = scheme(e)

    assert indices == [0, 1, 2]
    assert_valid_selection(e, selected, indices, expected_rank=3)


#
# Case 1: a zero or dependent tensor early in the list.
#


@pytest.mark.parametrize("scheme", get_scheme_params(rank_revealing=True))
def test_zero_tensor_first(scheme):
    """A leading zero tensor must not hide the independent tensors after it."""
    e = get_e(4)
    tensors = [np.zeros(4), e[0], e[1], e[2]]

    selected, indices = scheme(tensors)

    assert_valid_selection(tensors, selected, indices, expected_rank=3)


@pytest.mark.parametrize("scheme", get_scheme_params(rank_revealing=True))
def test_dependent_tensor_early(scheme):
    """A leading (nonzero) dependent tensor must not hide the ones after it."""
    e = get_e(4)
    tensors = [e[0], 2 * e[0], e[1], e[2]]

    selected, indices = scheme(tensors)

    assert_valid_selection(tensors, selected, indices, expected_rank=3)


def test_zero_tensor_first_gram_schmidt_prefers_earlier():
    """Gram-Schmidt keeps the earliest tensors of an equally valid subset."""
    e = get_e(4)

    _, indices = find_independent_tensors_gram_schmidt([np.zeros(4), e[0], e[1], e[2]])
    assert indices == [1, 2, 3]

    _, indices = find_independent_tensors_gram_schmidt([e[0], 2 * e[0], e[1], e[2]])
    assert indices == [0, 2, 3]


def test_zero_tensor_first_unpivoted_is_wrong():
    """The unpivoted diagonal test breaks on a degenerate leading column.

    Pinned here so it is clear why this scheme must not be used: the Householder
    step on the zero column is degenerate, and every diagonal entry after it stops
    being a residual distance -- all four are reported as zero, giving rank 0.
    """
    e = get_e(4)
    tensors = [np.zeros(4), e[0], e[1], e[2]]

    _, indices = find_independent_tensors_qr_unpivoted(tensors)
    assert indices == []

    _, indices = find_independent_tensors_scipy_qr(tensors, pivoting=False)
    assert indices == []


#
# Case 2: more tensors than components.
#


@pytest.mark.parametrize("scheme", get_scheme_params(rank_revealing=True))
def test_more_tensors_than_components(scheme):
    """Tensors beyond the `min(matrix.shape)` cutoff must still be considered.

    Here the leading tensors are *not* independent, so a rank-3 subset can only be
    formed by reaching past index 2 -- into the range the unpivoted loop never
    examines.
    """
    e = get_e(3)
    tensors = [e[0], 2 * e[0], 3 * e[0], e[1], e[2], e[0] + e[1]]

    selected, indices = scheme(tensors)

    assert_valid_selection(tensors, selected, indices, expected_rank=3)


def test_more_tensors_than_components_gram_schmidt_prefers_earlier():
    """Gram-Schmidt keeps the earliest tensors of an equally valid subset."""
    e = get_e(3)
    tensors = [e[0], 2 * e[0], 3 * e[0], e[1], e[2], e[0] + e[1]]

    _, indices = find_independent_tensors_gram_schmidt(tensors)

    assert indices == [0, 3, 4]


def test_more_tensors_than_components_unpivoted_is_wrong():
    """The unpivoted loop stops at `min(matrix.shape)`, so it finds only rank 1."""
    e = get_e(3)
    tensors = [e[0], 2 * e[0], 3 * e[0], e[1], e[2], e[0] + e[1]]

    _, indices = find_independent_tensors_qr_unpivoted(tensors)
    assert indices == [0]

    _, indices = find_independent_tensors_scipy_qr(tensors, pivoting=False)
    assert indices == [0]


#
# Both cases at once, on the data that originally exposed the bug.
#

# Coefficient matrix of the j=1 sector of the photoelastic class `(ij)kl`, recorded
# from the numerical grouping route that the reduction used to carry. It hits both failure
# modes at once: the first row is identically zero, and there are 6 rows of only 3
# components each, so the rank-3 subset needs the last row.
PHOTOELASTIC_J1_U = [
    [0.0, 0.0, 0.0],
    [0.2, -0.1, 0.1],
    [-0.1, 0.2, -0.1],
    [0.2, -0.1, 0.1],
    [-0.1, 0.2, -0.1],
    [0.2, -0.2, 0.3],
]


@pytest.mark.parametrize("scheme", get_scheme_params(rank_revealing=True))
def test_photoelastic_j1_coefficients(scheme):
    tensors = [np.array(row) for row in PHOTOELASTIC_J1_U]

    selected, indices = scheme(tensors)

    assert_valid_selection(tensors, selected, indices, expected_rank=3)


def test_photoelastic_j1_coefficients_unpivoted_is_wrong():
    """The original failure: rank 3, but the unpivoted scheme reports 2."""
    tensors = [np.array(row) for row in PHOTOELASTIC_J1_U]

    _, indices = find_independent_tensors_qr_unpivoted(tensors)
    assert indices == [1, 2]

    _, indices = find_independent_tensors_scipy_qr(tensors, pivoting=False)
    assert indices == [1, 2]


#
# Dispatcher.
#


@pytest.mark.parametrize(
    "method,expected",
    [
        ("gram_schmidt", [0, 3, 4]),
        ("qr_unpivoted", [0]),
    ],
)
def test_find_independent_tensors_method(method, expected):
    e = get_e(3)
    tensors = [e[0], 2 * e[0], 3 * e[0], e[1], e[2], e[0] + e[1]]

    _, indices = find_independent_tensors(tensors, method=method)

    assert indices == expected


def test_find_independent_tensors_method_scipy_qr():
    e = get_e(3)
    tensors = [e[0], 2 * e[0], 3 * e[0], e[1], e[2], e[0] + e[1]]

    selected, indices = find_independent_tensors(tensors, method="scipy_qr")

    assert_valid_selection(tensors, selected, indices, expected_rank=3)


def test_find_independent_tensors_default_is_gram_schmidt():
    e = get_e(3)
    tensors = [e[0], 2 * e[0], 3 * e[0], e[1], e[2], e[0] + e[1]]

    _, indices = find_independent_tensors(tensors)

    assert indices == find_independent_tensors_gram_schmidt(tensors)[1]


def test_find_independent_tensors_unknown_method():
    with pytest.raises(ValueError, match="Unknown method"):
        find_independent_tensors(get_e(3), method="not_a_method")


# ---------------------------------------------------------------------------
# Selecting independent mapping tensors, the level above the array schemes.
# ---------------------------------------------------------------------------


def get_candidates(weight: int, rank: int):
    if (rank - weight) % 2 == 0:
        return get_mappings_even(weight, rank)
    return get_mappings_odd(weight, rank)


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

    kept, _ = select_independent_mappings(weight, rank, candidates)

    assert kept == expected


@pytest.mark.parametrize("rank, weight, n_candidates, expected", SECTORS)
def test_selection_returns_the_gram_matrix_of_what_it_kept(
    rank, weight, n_candidates, expected
):
    """The Gram matrix accumulates during selection rather than in a second pass."""
    candidates = get_candidates(weight, rank)

    kept, gram = select_independent_mappings(weight, rank, candidates)

    assert gram == get_gram_matrix(weight, rank, [candidates[i] for i in kept])


@pytest.mark.parametrize("rank, weight", [(3, 2), (4, 3), (5, 0)])
def test_every_discarded_candidate_is_genuinely_dependent(rank, weight):
    """Adding back any rejected candidate must make the Gram matrix singular.

    The rank of a Gram matrix is the rank of the tensors it is built from, so a
    rejected candidate that still raised the rank would mean a mapping was lost.
    """
    candidates = get_candidates(weight, rank)
    kept, gram = select_independent_mappings(weight, rank, candidates)
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
    assert select_independent_mappings(2, 2, []) == ([], [])


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

    kept, _ = select_independent_mappings(weight, rank, candidates)

    assert select_independent_mappings_components(weight, rank, candidates) == kept
    assert select_independent_mappings_probe(weight, rank, candidates) == kept


def test_components_scheme_rejects_a_mapping_of_the_wrong_rank():
    """The evaluated array must have rank n + l; a mismatch is a caller error."""
    candidates = get_candidates(2, 4)

    with pytest.raises(ValueError, match="expected"):
        select_independent_mappings_components(2, 5, candidates)


@pytest.mark.parametrize(
    "select",
    [
        select_independent_mappings_components,
        select_independent_mappings_probe,
    ],
)
def test_numerical_schemes_select_nothing_from_nothing(select):
    assert select(2, 2, []) == []
