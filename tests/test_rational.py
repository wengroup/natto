from fractions import Fraction

from natto.rational import matrix_null_space


def test_matrix_null_space_exact_relations():
    """Check exact rational solutions of simultaneous homogeneous constraints."""
    constraints = [
        [Fraction(1), Fraction(-1), Fraction(0)],
        [Fraction(0), Fraction(1), Fraction(-2)],
    ]

    assert matrix_null_space(constraints, 3) == [
        [Fraction(2), Fraction(2), Fraction(1)]
    ]
