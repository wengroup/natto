"""Index slots, terms and operators.

The slot-based `Operator` must describe exactly the operators the letter-based classes
build today: the same tensor, printed in canonical form. Evaluation is the reference
for "the same tensor", since it shares none of the canonicalisation.
"""

import functools

import numpy as np

from natto.algebra import simplify_linear_combination
from natto.evaluate import evaluate_tensors
from natto.natural_projector import get_natural_projector
from natto.reduction import get_dual_pair, get_independent_mappings
from natto.symbolic import IndexGroup, Operator, Signature


@functools.cache
def package_operators() -> list:
    """Operators the package builds today, with their signature and evaluation order.

    Each entry is `(label, combination, signature, mode, order)`: `mode` is how
    `evaluate_tensors` orders the axes, and `order` names the same groups for
    `Operator.evaluate`.
    """
    cases = []

    for n, symmetry in [(2, None), (3, None), (3, "ijk=ikj"), (4, None)]:
        for ell in range(n + 1):
            G, gram = get_independent_mappings(ell, n, symmetry)
            if not G:
                continue
            G_simplified, G_tilde, S, _ = get_dual_pair(G, gram, n)
            mapping = Signature(
                (
                    IndexGroup("rank", n, upper=True),
                    IndexGroup("weight", ell, upper=False),
                )
            )
            decomposition = Signature(
                (
                    IndexGroup("rank", n, upper=True),
                    IndexGroup("rank_in", n, upper=True),
                )
            )
            for p, (embedding, extraction, composed) in enumerate(
                zip(G_simplified, G_tilde, S)
            ):
                label = f"n={n} {symmetry} ell={ell} channel {p}"
                cases += [
                    (f"{label} G", embedding, mapping, "embedding", ("rank", "weight")),
                    (
                        f"{label} G~",
                        extraction,
                        mapping,
                        "extraction",
                        ("weight", "rank"),
                    ),
                    (
                        f"{label} S",
                        composed,
                        decomposition,
                        "decomposition",
                        ("rank", "rank_in"),
                    ),
                ]

    for ell in range(5):
        signature = Signature(
            (
                IndexGroup("weight", ell, upper=False),
                IndexGroup("sigma", ell, upper=True),
            )
        )
        E = simplify_linear_combination(get_natural_projector(ell))
        cases.append((f"E({ell})", E, signature, "extraction", ("weight", "sigma")))

    return cases


def test_evaluation():
    """The same tensor whatever the printed form, including reordered symbols' signs."""
    for label, combination, signature, mode, order in package_operators():
        operator = Operator.from_linear_combination(combination, signature)

        np.testing.assert_allclose(
            operator.evaluate(order),
            evaluate_tensors(combination, mode=mode),
            atol=1e-12,
            err_msg=label,
        )


def test_delta_printing():
    """Without a Levi-Civita symbol the published strings must not move."""
    checked = 0
    for label, combination, signature, *_ in package_operators():
        if "ε" in str(combination):
            continue
        operator = Operator.from_linear_combination(combination, signature)

        assert str(operator) == str(combination), label
        checked += 1

    assert checked > 0


def test_round_trip():
    """Unicode, ASCII and today's printed string all parse back to the same operator.

    Parsing today's string is the check of Sec. 6.10 of [Wen2026Refactor]:
    canonicalising it must give the converted operator exactly.
    """
    for label, combination, signature, *_ in package_operators():
        operator = Operator.from_linear_combination(combination, signature)

        assert Operator.parse(str(operator), signature) == operator, label
        assert Operator.parse(operator.to_string(ascii=True), signature) == operator, (
            label
        )
        assert Operator.parse(str(combination), signature) == operator, label
