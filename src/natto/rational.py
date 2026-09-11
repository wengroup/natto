"""Exact linear algebra over the rationals.

The mapping tensors, and the mixing matrices that adapt them to a symmetry,
carry coefficients that are exact rationals. Inverting a Gram matrix or taking
a null space in floating point would turn those into approximations, and the
null-space dimension -- which counts how many ICTs of a weight survive the
symmetry -- would then depend on a tolerance. So the matrices here are plain
nested lists of :class:`fractions.Fraction`, and every operation on them is
exact. ``float_matrix`` and ``fraction_matrix`` are the deliberate exits, used
only once a result is being handed out for numerical work or printing.
"""

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
        # Find pivot
        pivot = augmented[i][i]
        if pivot == 0:
            raise ValueError("Matrix is not invertible")

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


def matrix_transpose(m: list[list[Fraction]]) -> list[list[Fraction]]:
    """Transpose a matrix containing Fraction objects.

    Args:
        m: Matrix to be transposed

    Returns:
        Transposed matrix as list of lists with Fraction objects
    """
    return [[m[j][i] for j in range(len(m))] for i in range(len(m[0]))]


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
