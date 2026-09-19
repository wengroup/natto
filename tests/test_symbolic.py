"""Printing, parsing and evaluating operators.

Every operator prints in the delta and epsilon notation and parses back from it, in
Unicode or in plain ASCII, to exactly the same operator, and evaluates to the sum of
its terms contracted as dense arrays.
"""

import numpy as np
import pytest

from natto.coupling import get_coupling_operator
from natto.natural_projector import get_natural_projector
from natto.reduction import (
    get_composed_operators,
    get_embedding_operators,
    get_extraction_operators,
)
from natto.symbolic import Operator
from natto.utils import dij, eijk


def package_operators() -> list[Operator]:
    """Projectors, coupling operators, and the operators of a symmetric reduction."""
    operators = [get_natural_projector(ell) for ell in range(5)]
    for triple in [(1, 1, 1), (1, 1, 2), (2, 2, 3), (2, 3, 3)]:
        operators.append(get_coupling_operator(*triple))
    for build in (get_embedding_operators, get_extraction_operators):
        operators += list(build(3, symmetry="ijk=ikj").values())
    operators += list(get_composed_operators(3, symmetry="ijk=ikj").values())
    operators += list(
        get_embedding_operators(3, symmetry="ijk=ikj", basis="orthonormal").values()
    )

    return operators


def test_round_trip():
    """Unicode and ASCII both parse back to the same operator."""
    for operator in package_operators():
        signature = operator.signature

        assert Operator.parse(str(operator), signature) == operator
        assert Operator.parse(operator.to_string(ascii=True), signature) == operator


def test_evaluate():
    """Evaluation equals the dense terms contracted by einsum, in a reversed order."""
    delta, epsilon = dij(), eijk()
    for operator in package_operators():
        order = [group.name for group in operator.signature.groups][::-1]
        axes = [slot for name in order for slot in operator.signature.slots(name)]

        expected = np.zeros((3,) * operator.signature.size)
        for term, coefficient in operator.terms.items():
            operands = []
            for pair in term.deltas:
                operands += [delta, list(pair)]
            for triple in term.epsilons:
                operands += [epsilon, list(triple)]
            dense = np.einsum(*operands, axes) if operands else 1.0
            expected = expected + float(coefficient) * dense

        expected = expected * np.sqrt(operator.radicand)

        np.testing.assert_allclose(operator.evaluate(order), expected, atol=1e-12)


def test_different_radicands_do_not_add():
    """A sum of different roots is not an operator's form, so it is refused."""
    signature = get_natural_projector(1).signature
    root_two = Operator.parse("sqrt(2) * (d_Aa)", signature)
    root_three = Operator.parse("sqrt(3) * (d_Aa)", signature)

    assert root_two + root_two == Operator.parse("sqrt(2) * (2 d_Aa)", signature)
    with pytest.raises(ValueError):
        root_two + root_three
