"""Printing and parsing operators.

Every operator prints in the delta and epsilon notation and parses back from it, in
Unicode or in plain ASCII, to exactly the same operator.
"""

from natto.coupling import get_coupling_symbolic
from natto.natural_projector import get_natural_projector
from natto.reduction import get_dual_pair, get_independent_mappings
from natto.symbolic import Operator


def package_operators() -> list[Operator]:
    """Projectors, coupling operators, and the operators of a symmetric reduction."""
    operators = [get_natural_projector(ell) for ell in range(5)]
    for triple in [(1, 1, 1), (1, 1, 2), (2, 2, 3), (2, 3, 3)]:
        operators.append(get_coupling_symbolic(*triple))
    for ell in range(4):
        G, gram = get_independent_mappings(ell, 3, "ijk=ikj")
        if G:
            embedding, extraction, decomposition, _ = get_dual_pair(G, gram)
            operators += embedding + extraction + decomposition

    return operators


def test_round_trip():
    """Unicode and ASCII both parse back to the same operator."""
    for operator in package_operators():
        signature = operator.signature

        assert Operator.parse(str(operator), signature) == operator
        assert Operator.parse(operator.to_string(ascii=True), signature) == operator
