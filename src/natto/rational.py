"""Exact linear algebra over the rationals.

The mapping tensors, and the mixing matrices that adapt them to a symmetry,
carry coefficients that are exact rationals. Inverting a Gram matrix or taking
a null space in floating point would turn those into approximations, and the
null-space dimension -- which counts how many ICTs of a weight survive the
symmetry -- would then depend on a tolerance. So the matrices here are plain
nested lists of :class:`fractions.Fraction`, and every operation on them is
exact. An inverse square root, which normalizing needs, is kept exact too, as a
rational times the square root of a square-free integer. ``float_matrix`` and
``fraction_matrix`` are the deliberate exits, used only once a result is being
handed out for numerical work or printing.
"""

import math
from fractions import Fraction


def matrix_inverse(matrix: list[list[Fraction]]) -> list[list[Fraction]]:
    """
    Calculate the inverse of a matrix containing Fraction objects.

    Returns the inverse matrix with Fraction elements.

    Args:
        matrix: List of lists containing Fraction objects

    Returns:
        Inverse matrix as list of lists with Fraction objects
    """
    n = len(matrix)

    # Create augmented matrix [A|I]
    augmented = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(matrix[i][j])
        for j in range(n):
            row.append(Fraction(1) if i == j else Fraction(0))
        augmented.append(row)

    # Gaussian elimination
    for i in range(n):
        # Find a nonzero pivot, exchanging rows when the diagonal entry is zero
        pivot_row = next((k for k in range(i, n) if augmented[k][i] != 0), None)
        if pivot_row is None:
            raise ValueError("Matrix is not invertible")
        augmented[i], augmented[pivot_row] = augmented[pivot_row], augmented[i]
        pivot = augmented[i][i]

        # Scale row to make pivot 1
        for j in range(2 * n):
            augmented[i][j] = augmented[i][j] / pivot

        # Eliminate column
        for k in range(n):
            if k != i:
                factor = augmented[k][i]
                for j in range(2 * n):
                    augmented[k][j] -= factor * augmented[i][j]

    # Extract inverse matrix
    inverse = []
    for i in range(n):
        inverse.append([])
        for j in range(n):
            inverse[i].append(augmented[i][j + n])

    return inverse


def matrix_multiply(
    m1: list[list[Fraction]], m2: list[list[Fraction]]
) -> list[list[Fraction]]:
    """Perform matrix multiplication on two matrices containing Fraction objects.

    Args:
        m1: First matrix
        m2: Second matrix

    Returns:
        Resulting matrix as list of lists with Fraction objects
    """
    n = len(m1)
    m = len(m2[0])
    p = len(m2)

    if len(m1[0]) != p:
        raise ValueError("Incompatible matrix dimensions for multiplication")

    result = []
    for i in range(n):
        row = []
        for j in range(m):
            value = Fraction(0)
            for k in range(p):
                value += m1[i][k] * m2[k][j]
            row.append(value)
        result.append(row)

    return result


def is_nonsingular(matrix: list[list[Fraction]]) -> bool:
    """Whether a square rational matrix has full rank, decided exactly.

    Gaussian elimination over :class:`fractions.Fraction`, so the answer is a fact
    about the matrix rather than a statement about a tolerance. A singular matrix is
    found the moment a column has no nonzero entry left to pivot on.

    Args:
        matrix: Square matrix with exact rational entries.

    Returns:
        True when the matrix has full rank.
    """
    rows = [row.copy() for row in matrix]
    size = len(rows)
    if any(len(row) != size for row in rows):
        raise ValueError("Matrix is not square")

    for column in range(size):
        pivot_row = next(
            (row for row in range(column, size) if rows[row][column] != 0), None
        )
        if pivot_row is None:
            return False

        rows[column], rows[pivot_row] = rows[pivot_row], rows[column]
        pivot = rows[column][column]
        for row in range(column + 1, size):
            if rows[row][column] != 0:
                factor = rows[row][column] / pivot
                rows[row] = [
                    value - factor * pivot_value
                    for value, pivot_value in zip(rows[row], rows[column])
                ]

    return True


def bordered_inverse(
    inverse: list[list[Fraction]], border: list[Fraction], diagonal: Fraction
) -> list[list[Fraction]] | None:
    """The inverse of a symmetric matrix bordered by one more row and column.

    For the symmetric matrix `[[A, b], [b^T, d]]` with `A` invertible, the Schur
    complement `s = d - b^T A^-1 b` decides the answer, since the determinant is
    `det(A) s`. When `s` is not zero the inverse is
    `[[A^-1 + p p^T / s, -p / s], [-p^T / s, 1 / s]]` with `p = A^-1 b`, which costs a
    product with the border rather than a fresh elimination.

    Args:
        inverse: The exact inverse of `A`, empty when `A` has no rows.
        border: The new column `b`, one entry per row of `A`.
        diagonal: The new diagonal entry `d`.

    Returns:
        The exact inverse of the bordered matrix, or None when it is singular.
    """
    projection = [
        sum((value * b for value, b in zip(row, border)), Fraction(0))
        for row in inverse
    ]
    schur = diagonal - sum((b * p for b, p in zip(border, projection)), Fraction(0))
    if schur == 0:
        return None

    bordered = [
        [value + p * q / schur for value, q in zip(row, projection)] + [-p / schur]
        for row, p in zip(inverse, projection)
    ]
    bordered.append([-p / schur for p in projection] + [1 / schur])

    return bordered


def matrix_null_space(
    matrix: list[list[Fraction]], n_columns: int
) -> list[list[Fraction]]:
    """Return a basis for the null space of an exact rational matrix.

    Each inner list is one basis vector satisfying ``matrix @ vector = 0``. Gaussian
    elimination is performed entirely with
    :class:`fractions.Fraction` objects.

    Args:
        matrix: Constraint matrix with exact rational entries. It may have no rows.
        n_columns: Number of columns, needed when ``matrix`` has no rows.

    Returns:
        Independent null-space vectors of length ``n_columns``.
    """
    if not matrix:
        return [
            [Fraction(int(i == j)) for i in range(n_columns)] for j in range(n_columns)
        ]
    if any(len(row) != n_columns for row in matrix):
        raise ValueError("Constraint matrix has inconsistent dimensions")

    reduced = [row.copy() for row in matrix]
    pivot_columns = []
    pivot_row = 0
    for column in range(n_columns):
        row = next(
            (
                candidate
                for candidate in range(pivot_row, len(reduced))
                if reduced[candidate][column] != 0
            ),
            None,
        )
        if row is None:
            continue

        reduced[pivot_row], reduced[row] = reduced[row], reduced[pivot_row]
        pivot = reduced[pivot_row][column]
        reduced[pivot_row] = [value / pivot for value in reduced[pivot_row]]
        for other_row in range(len(reduced)):
            if other_row == pivot_row:
                continue
            factor = reduced[other_row][column]
            if factor != 0:
                reduced[other_row] = [
                    value - factor * pivot_value
                    for value, pivot_value in zip(
                        reduced[other_row], reduced[pivot_row]
                    )
                ]

        pivot_columns.append(column)
        pivot_row += 1
        if pivot_row == len(reduced):
            break

    free_columns = [
        column for column in range(n_columns) if column not in pivot_columns
    ]
    basis = []
    for free_column in free_columns:
        vector = [Fraction(0) for _ in range(n_columns)]
        vector[free_column] = Fraction(1)
        for row, pivot_column in enumerate(pivot_columns):
            vector[pivot_column] = -reduced[row][free_column]
        basis.append(vector)

    return basis


def ldl_decomposition(
    matrix: list[list[Fraction]],
) -> tuple[list[list[Fraction]], list[Fraction]]:
    """Factor a symmetric positive-definite matrix as `L diag(d) L^T`, exactly.

    `L` is unit lower triangular and every pivot in `d` is positive. Unlike a
    Cholesky factor, which needs the square roots of the pivots, both stay rational.

    Args:
        matrix: A symmetric positive-definite matrix with exact rational entries.

    Returns:
        `L`, and the pivots `d`.

    Raises:
        ValueError: If a pivot is not positive, so the matrix is not positive
            definite.
    """
    size = len(matrix)
    reduced = [[Fraction(value) for value in row] for row in matrix]
    lower = [[Fraction(int(i == j)) for j in range(size)] for i in range(size)]
    pivots = []
    for k in range(size):
        pivot = reduced[k][k]
        if pivot <= 0:
            raise ValueError("Matrix must be positive definite")
        pivots.append(pivot)
        for i in range(k + 1, size):
            factor = reduced[i][k] / pivot
            lower[i][k] = factor
            for j in range(k, size):
                reduced[i][j] -= factor * reduced[k][j]

    return lower, pivots


def unit_lower_triangular_inverse(
    lower: list[list[Fraction]],
) -> list[list[Fraction]]:
    """The inverse of a unit lower triangular matrix, by forward substitution.

    Row p of the inverse is the unit vector p minus `lower[p][q]` times row q, summed
    over the earlier rows q, so no pivoting or division is needed.

    Args:
        lower: A unit lower triangular matrix with exact rational entries.

    Returns:
        Its exact inverse, unit lower triangular as well.
    """
    size = len(lower)
    inverse = []
    for p in range(size):
        row = [Fraction(int(p == j)) for j in range(size)]
        for q in range(p):
            factor = lower[p][q]
            if factor:
                for j in range(q + 1):
                    row[j] -= factor * inverse[q][j]
        inverse.append(row)

    return inverse


def square_free_split(n: int) -> tuple[int, int]:
    """Write a positive integer as `k**2 * s` with `s` square-free.

    An integer is square-free when it cannot be written as `j**2 * m` with a whole
    number `j > 1`, that is, when no prime divides it twice. So 1, 2, 5, 6 = 2 * 3
    and 30 = 2 * 3 * 5 are square-free, while 8 = 2**2 * 2, 12 = 2**2 * 3 and
    36 = 6**2 are not. Moving the largest square out of `n` leaves a square-free
    part, so `sqrt(n) = k * sqrt(s)` with the smallest possible root.

    This is what gives a square root one written form. An orthonormal mapping carries
    `sqrt(s)` as the radicand of its `Operator`, and with `s` square-free, equal
    operators have equal radicands, and a product of roots that is rational, such as
    `sqrt(6) * sqrt(6) = 6`, comes out with no root left.

    Trial division runs only up to the cube root of what is left: past that, the
    remainder has at most two prime factors, so it is 1, a prime, a product of two
    distinct primes, or the square of one, and an integer square root tells which.

    Args:
        n: A positive integer.

    Returns:
        `k` and the square-free part `s`.

    Raises:
        ValueError: If `n` is not positive.

    Examples:
        24 = 2**2 * 6, so sqrt(24) = 2 sqrt(6):

        >>> square_free_split(24)
        (2, 6)

        A perfect square leaves no root:

        >>> square_free_split(36)
        (6, 1)

        A square-free number comes back unchanged:

        >>> square_free_split(7)
        (1, 7)
    """
    if n < 1:
        raise ValueError(f"n must be positive, got {n}")

    k, s, rest, p = 1, 1, n, 2
    while p**3 <= rest:
        exponent = 0
        while rest % p == 0:
            rest //= p
            exponent += 1
        k *= p ** (exponent // 2)
        s *= p ** (exponent % 2)
        p += 1

    root = math.isqrt(rest)
    if root * root == rest:
        k *= root
    else:
        s *= rest

    return k, s


def inverse_square_root(value: Fraction) -> tuple[Fraction, int]:
    """Write `1 / sqrt(value)` exactly, as a rational times the root of an integer.

    With `value = a / b` in lowest terms, `1 / sqrt(value) = sqrt(a b) / a`, and the
    square factor of `a b` moves out of the root.

    Args:
        value: A positive rational.

    Returns:
        `r` and the square-free `s` with `1 / sqrt(value) = r * sqrt(s)`.

    Raises:
        ValueError: If `value` is not positive.

    Examples:
        >>> inverse_square_root(Fraction(8, 3))
        (Fraction(1, 4), 6)
    """
    value = Fraction(value)
    if value <= 0:
        raise ValueError(f"value must be positive, got {value}")

    k, s = square_free_split(value.numerator * value.denominator)
    scale = Fraction(k, value.numerator)

    return scale, s


def float_matrix(m: list[list[Fraction]]) -> list[list[float]]:
    """
    Convert a matrix of Fraction objects to a matrix of floats.

    Args:
        m: List of lists containing Fraction objects

    Returns:
        List of lists containing floats
    """
    return [[float(fraction) for fraction in row] for row in m]


def fraction_matrix(m: list[list[Fraction]]) -> list[list[str]]:
    """
    Convert a matrix of Fraction objects to a matrix of strings.

    Each Fraction is represented as a string in the form "numerator/denominator".

    Args:
        m: List of lists containing Fraction objects

    Returns:
        List of lists containing strings representing the fractions
    """
    return [[str(fraction) for fraction in row] for row in m]
