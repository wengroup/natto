"""Printing, parsing and evaluating operators.

Every operator prints in the delta and epsilon notation and parses back from it, in
Unicode or in plain ASCII, to exactly the same operator, and evaluates to the sum of
its terms contracted as dense arrays.
"""

import numpy as np

from natto.coupling import get_coupling_symbolic
from natto.mapping_tensors import get_decomposition_operators, get_extraction_operators
from natto.natural_projector import get_natural_projector
from natto.reduction import get_dual_pair, get_independent_mappings
from natto.symbolic import Operator
from natto.utils import dij, eijk


def package_operators() -> list[Operator]:
    """Projectors, coupling operators, and the operators of a symmetric reduction."""
    operators = [get_natural_projector(ell) for ell in range(5)]
    for triple in [(1, 1, 1), (1, 1, 2), (2, 2, 3), (2, 3, 3)]:
        operators.append(get_coupling_symbolic(*triple))
    for ell in range(4):
        G, gram = get_independent_mappings(ell, 3, "ijk=ikj")
        if G:
            embedding, extraction, gram_inverse = get_dual_pair(G, gram)
            duals = get_extraction_operators(gram_inverse, G)
            decomposition = get_decomposition_operators(G, duals)
            operators += embedding + extraction + decomposition

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

        np.testing.assert_allclose(operator.evaluate(order), expected, atol=1e-12)
