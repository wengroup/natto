"""Everything that can be exact is.

The reduction carries no numerical tolerance: the mapping tensors, their Gram
matrix and its inverse, the duals, the mixing matrices of an intrinsic symmetry and
the null space those define are all built with `fractions.Fraction`. That is a
property of the whole package rather than of any one function, and it regresses the
first moment someone writes `/` where they meant `Fraction` -- Python hands back a
float and nothing else complains. These tests are what notices.

Exactness is structural, not a matter of size: a rank-three reduction runs the same
functions a rank-six one does, so nothing here goes above rank three. Rank two and
three together already reach both parities of `rank - weight`, a weight carrying
several channels, and a symmetry whose null space extinguishes one.

Two places are deliberately numerical, and are not checked here: evaluating an
operator into an array, and the orthonormal mappings, whose inverse square root is
irrational in general.
"""

import functools
from fractions import Fraction

import pytest

from natto.coupling import coeff_C_even, coeff_C_odd, get_coupling_symbolic
from natto.harmonics import coeff_harmonic, get_harmonic_symbolic
from natto.natural_projector import get_natural_projector
from natto.rational import matrix_null_space
from natto.reduction import get_dual_pair, get_independent_mappings
from natto.symmetry_adaptation import get_symmetry_action_matrix

#: `Fraction` is the exact type. A plain `int` is exact too; `bool` is an `int` by
#: inheritance but is never a coefficient, so it is rejected deliberately.
EXACT = (Fraction, int)

#: Rank and symmetry of the classes checked; see the module docstring.
CASES = [
    pytest.param(2, None, id="rank2"),
    pytest.param(3, None, id="rank3"),
    pytest.param(3, "ijk=ikj", id="rank3_piezoelectric"),
]


def assert_exact(label: str, values):
    """Every value must be an exact rational, not a float that happens to look it."""
    for value in values:
        assert isinstance(value, EXACT) and not isinstance(value, bool), (
            f"{label}: {type(value).__name__} {value!r} is not exact"
        )


def coefficients(combination):
    """Every scalar in a linear combination.

    A float can hide in a term's own factor or in any of its tensors', so both are
    collected.
    """
    values = []
    for term in combination:
        values.append(term.factor)
        values.extend(tensor.factor for tensor in term)

    return values


def entries(matrix):
    return [value for row in matrix for value in row]


@functools.lru_cache(maxsize=None)
def reduce_weight(rank: int, symmetry: str, weight: int):
    """The whole exact pipeline for one sector, built once and shared.

    Each test below would otherwise rebuild it, and at rank four that is seconds.
    """
    G, gram = get_independent_mappings(weight, rank, symmetry)
    if not G:
        return None

    _, G_tilde, S, gram_inverse = get_dual_pair(G, gram, rank)

    return G, gram, G_tilde, S, gram_inverse


def sectors(rank: int, symmetry: str):
    """Every weight of a class that has any mapping at all."""
    for weight in range(rank + 1):
        built = reduce_weight(rank, symmetry, weight)
        if built is not None:
            yield weight, built


@pytest.mark.parametrize("rank, symmetry", CASES)
def test_reduction_is_exact(rank: int, symmetry: str):
    """The mappings, the Gram matrix, its inverse, the duals and S."""
    checked = 0
    for weight, (G, gram, G_tilde, S, gram_inverse) in sectors(rank, symmetry):
        tag = f"rank {rank} weight {weight} symmetry {symmetry}"

        for p, G_p in enumerate(G):
            assert_exact(f"{tag} G[{p}]", coefficients(G_p))
        for p, G_tilde_p in enumerate(G_tilde):
            assert_exact(f"{tag} G_tilde[{p}]", coefficients(G_tilde_p))
        for p, S_p in enumerate(S):
            assert_exact(f"{tag} S[{p}]", coefficients(S_p))
        assert_exact(f"{tag} gram", entries(gram))
        assert_exact(f"{tag} gram inverse", entries(gram_inverse))
        checked += 1

    assert checked, f"rank {rank} symmetry {symmetry} produced no sector to check"


@pytest.mark.parametrize("rank, symmetry", CASES)
def test_symmetry_mixing_matrix_and_null_space_are_exact(rank: int, symmetry: str):
    """The multiplicity of a weight comes from this null space.

    Were it computed in floating point, a weight that is extinguished -- as weight
    one is for the third-order elastic tensor -- would become a threshold rather
    than a statement.
    """
    permutation = tuple([1, 0] + list(range(2, rank)))

    for weight, (G, gram, _, _, gram_inverse) in sectors(rank, symmetry):
        tag = f"rank {rank} weight {weight}"
        mixing = get_symmetry_action_matrix(G, gram_inverse, weight, rank, permutation)
        assert_exact(f"{tag} mixing matrix", entries(mixing))
        assert_exact(
            f"{tag} null space",
            [v for row in matrix_null_space(mixing, len(G)) for v in row],
        )


@pytest.mark.parametrize("weight", range(4))
def test_natural_projector_is_exact(weight: int):
    assert_exact(f"E({weight})", coefficients(get_natural_projector(weight)))


@pytest.mark.parametrize("weight", range(4))
def test_harmonic_is_exact(weight: int):
    """The harmonic operator, and the constant that normalizes it."""
    H, _, _ = get_harmonic_symbolic(weight)

    assert_exact(f"V({weight})", coefficients(H))
    assert isinstance(coeff_harmonic(weight), Fraction)


@pytest.mark.parametrize("l1, l2, l3", [(1, 1, 1), (1, 1, 2), (1, 2, 3)])
def test_coupling_operator_is_exact(l1: int, l2: int, l3: int):
    """The coupling operator, and both normalization constants.

    The constants are ratios of factorials, so there is no reason for either to be
    a float, and `int / int` in Python would quietly make them one. The two cases
    are reached by the parity of `l1 + l2 + l3`.
    """
    K, _, _, _ = get_coupling_symbolic(l1, l2, l3)
    assert_exact(f"K({l1},{l2},{l3})", coefficients(K))

    even = (l1 + l2 + l3) % 2 == 0
    coefficient = coeff_C_even(l1, l2, l3) if even else coeff_C_odd(l1, l2, l3)

    assert isinstance(coefficient, Fraction)
