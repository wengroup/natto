"""Contraction of operators.

The reference for every case is `numpy.einsum` on the evaluated operators, which
shares nothing with the walk that reduces the terms.
"""

import numpy as np
import pytest

from natto.algebra import contract, contract_fully
from natto.symbolic import IndexGroup, Operator, Signature


def operator(text: str, size: int) -> Operator:
    """An operator over one group of lower-case indices."""
    return Operator.parse(text, Signature((IndexGroup("x", size, upper=False),)))


#: (factors as (printed operator, slot count, labels), free slots of the result)
CASES = {
    "delta chain": (
        [("d_ab", 2, [0, "i"]), ("d_ab", 2, ["i", "j"]), ("d_ab", 2, ["j", 1])],
        2,
    ),
    "closed loop": ([("d_ab", 2, ["i", "j"]), ("d_ab", 2, ["j", "i"])], 0),
    "trace": ([("d_ab", 2, ["i", "i"])], 0),
    "symbol renamed": ([("e_abc", 3, [0, 1, "i"]), ("d_ab", 2, ["i", 2])], 3),
    "symbol vanishes": ([("e_abc", 3, [0, "i", "j"]), ("d_ab", 2, ["i", "j"])], 1),
    "symbols share one": ([("e_abc", 3, [0, 1, "i"]), ("e_abc", 3, ["i", 2, 3])], 4),
    "symbols share two": (
        [("e_abc", 3, [0, "i", "j"]), ("e_abc", 3, ["j", "i", 1])],
        2,
    ),
    "symbols share three": (
        [("e_abc", 3, ["i", "j", "k"]), ("e_abc", 3, ["j", "i", "k"])],
        0,
    ),
    "symbols through a delta": (
        [("e_abc", 3, [0, 1, "i"]), ("d_ab", 2, ["i", "j"]), ("e_abc", 3, ["j", 2, 3])],
        4,
    ),
    "symbols apart": ([("e_abc", 3, [0, 1, 2]), ("e_abc", 3, [3, 4, 5])], 6),
    "radicands multiply": (
        [("sqrt(2) * (d_ab)", 2, [0, "i"]), ("sqrt(6) * (d_ab)", 2, ["i", 1])],
        2,
    ),
    "radicands cancel": (
        [("sqrt(2) * (d_ab)", 2, [0, "i"]), ("sqrt(2) * (d_ab)", 2, ["i", 1])],
        2,
    ),
    "sums of terms": (
        [
            ("+1/2 d_ab d_cd  +1/2 d_ac d_bd  -1/3 d_ad d_bc", 4, [0, 1, "i", "j"]),
            ("+2 d_ab d_cd  -1 d_ac d_bd", 4, ["i", "j", 2, 3]),
        ],
        4,
    ),
}


@pytest.mark.parametrize("factors, size", CASES.values(), ids=CASES.keys())
def test_contract(factors, size):
    """The reduced product evaluates to the product summed over its shared labels."""
    built = [(operator(text, slots), labels) for text, slots, labels in factors]
    result = contract(built, Signature((IndexGroup("x", size, upper=False),)))

    ids = {}
    operands = []
    for op, labels in built:
        operands += [
            op.evaluate(),
            [ids.setdefault(label, len(ids)) for label in labels],
        ]
    expected = np.einsum(*operands, [ids[slot] for slot in range(size)])

    np.testing.assert_allclose(result.evaluate(), expected, atol=1e-12)


#: (deltas, Levi-Civita symbols, value) of products summed over every label
FULL_CASES = {
    "trace": ([("i", "i")], [], 3),
    "two loops": ([("i", "j"), ("j", "i"), ("k", "k")], [], 9),
    "symbols aligned": ([], [("i", "j", "k"), ("i", "j", "k")], 6),
    "symbols swapped": ([], [("i", "j", "k"), ("j", "i", "k")], -6),
    "symbols through deltas": (
        [("i", "m"), ("j", "l"), ("k", "p"), ("q", "q")],
        [("i", "j", "k"), ("l", "m", "p")],
        -18,
    ),
    "path on one symbol": (
        [("i", "j"), ("l", "m"), ("k", "p")],
        [("i", "j", "k"), ("l", "m", "p")],
        0,
    ),
    "triangle": ([("i", "j"), ("j", "k"), ("k", "i")], [], 3),
}


@pytest.mark.parametrize(
    "deltas, epsilons, value", FULL_CASES.values(), ids=FULL_CASES.keys()
)
def test_contract_fully(deltas, epsilons, value):
    """Counting cycles gives the value of the fully contracted product."""
    assert contract_fully(deltas, epsilons) == value
