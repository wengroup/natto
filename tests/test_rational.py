from fractions import Fraction

import pytest

from natto.rational import is_nonsingular, matrix_null_space


def test_matrix_null_space_exact_relations():
    """Check exact rational solutions of simultaneous homogeneous constraints."""
    constraints = [
        [Fraction(1), Fraction(-1), Fraction(0)],
        [Fraction(0), Fraction(1), Fraction(-2)],
    ]

    assert matrix_null_space(constraints, 3) == [
        [Fraction(2), Fraction(2), Fraction(1)]
    ]


def test_is_nonsingular_full_rank():
    assert is_nonsingular([[Fraction(2), Fraction(1)], [Fraction(1), Fraction(1)]])


def test_is_nonsingular_detects_a_dependent_row():
    """A row that is a rational multiple of another leaves no pivot."""
    assert not is_nonsingular(
        [[Fraction(1), Fraction(2)], [Fraction(1, 2), Fraction(1)]]
    )


def test_is_nonsingular_is_exact_where_floats_would_not_be():
    """1/3 and 1/7 have no exact binary form, so a float test would see noise.

    The second row is exactly three times the first, which makes the matrix
    singular as a fact rather than to within a tolerance.
    """
    matrix = [
        [Fraction(1, 3), Fraction(1, 7)],
        [Fraction(1), Fraction(3, 7)],
    ]

    assert not is_nonsingular(matrix)


def test_is_nonsingular_empty_matrix():
    """A zero-by-zero matrix is trivially full rank."""
    assert is_nonsingular([])


def test_is_nonsingular_rejects_a_non_square_matrix():
    with pytest.raises(ValueError, match="not square"):
        is_nonsingular([[Fraction(1), Fraction(2)]])
