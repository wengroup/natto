"""Contraction of operators.

Every operator of the package is a sum of products of Kronecker deltas and Levi-Civita
symbols, so multiplying operators and summing over the indices they share is the one
algebraic operation the package needs. `contract` does it: it multiplies terms, lets
each shared index close a chain of deltas or join two symbols, and returns the result
as an operator whose terms are canonical and collected.
"""

import itertools
import math
from collections import Counter
from collections.abc import Hashable, Sequence
from fractions import Fraction

from natto.rational import square_free_split
from natto.symbolic import Operator, Signature, Term, sort_with_sign


def contract(
    factors: Sequence[tuple[Operator, Sequence[Hashable]]], signature: Signature
) -> Operator:
    """Multiply operators and sum over the indices they share.

    Each operator comes with one label per slot. A label on two slots is summed over;
    a label on one slot is a free index of the result and must be one of its slots,
    given by number. Every term of the product is reduced by the rules of the walk:
    a chain of deltas collapses to one delta, a closed loop of them is a factor of 3,
    a Levi-Civita symbol with a repeated index vanishes, and two symbols sharing an
    index expand into deltas by their determinant identity.

    The radicands of the operators multiply: the product is written as `k**2 * s`
    with `s` square-free, `k` joins the coefficients, and `s` is the radicand of the
    result. An orthonormal mapping contracted with itself is therefore rational again.

    Args:
        factors: The operators, each with the labels of its slots in order.
        signature: The signature of the result.

    Returns:
        The product, its terms canonical and collected.

    Raises:
        ValueError: If a label occurs more than twice, if the free labels are not
            exactly the slots of `signature`, or if an operator gets the wrong
            number of labels.

    References:
        Eq. 8 (eq-delta-chain) and Eq. 9 (eq-epsilon-determinant) of the implementation
        notes for the reduction of each term, and Eq. 10 (eq-cycle-count) for its full
        contractions.
    """
    counts = Counter()
    for operator, labels in factors:
        if len(labels) != operator.signature.size:
            raise ValueError(
                f"An operator with {operator.signature.size} slots got {len(labels)} "
                "labels"
            )
        counts.update(labels)
    if any(count > 2 for count in counts.values()):
        raise ValueError("A label occurs more than twice")
    free = {label for label, count in counts.items() if count == 1}
    if free != set(range(signature.size)):
        raise ValueError(
            f"The free labels {sorted(free, key=str)} are not the slots of the result"
        )

    radicand = math.prod(operator.radicand for operator, _ in factors)
    root, radicand = square_free_split(radicand)

    terms = []
    choices = [list(operator.terms.items()) for operator, _ in factors]
    for choice in itertools.product(*choices):
        coefficient = Fraction(root)
        deltas, epsilons = [], []
        for (term, value), (_, labels) in zip(choice, factors):
            coefficient *= value
            deltas += [[labels[i], labels[j]] for i, j in term.deltas]
            epsilons += [[labels[slot] for slot in triple] for triple in term.epsilons]

        for factor, kept_deltas, kept_epsilons in _reduce(deltas, epsilons):
            sign, reduced = Term.from_blocks(kept_deltas, kept_epsilons)
            terms.append((sign * factor * coefficient, reduced))

    return Operator(signature, terms, radicand)


def contract_fully(
    deltas: Sequence[Sequence[Hashable]], epsilons: Sequence[Sequence[Hashable]] = ()
) -> int:
    """Sum a product of deltas and Levi-Civita symbols over every index.

    Every label occurs exactly twice, so the deltas form closed cycles, each a factor
    of 3, and paths whose ends are slots of the symbols. A path joining two slots of
    one symbol makes the product vanish; otherwise the paths join each slot of the
    first symbol to one of the second, and the two symbols give 6 times the sign of
    that bijection.

    Args:
        deltas: Pairs of labels, one per Kronecker delta.
        epsilons: Triples of labels, one per Levi-Civita symbol, at most two.

    Returns:
        The value of the product.

    Raises:
        ValueError: If a label does not occur exactly twice, or there are more than two
            Levi-Civita symbols.

    References:
        Eq. 10 (eq-cycle-count) of the implementation notes.
    """
    if len(epsilons) > 2:
        raise ValueError("At most two Levi-Civita symbols can be contracted fully")

    # The two ends at each label: a delta by its number, or a symbol's slot as a pair
    ends = {}
    for k, (u, v) in enumerate(deltas):
        ends.setdefault(u, []).append(k)
        ends.setdefault(v, []).append(k)
    for s, triple in enumerate(epsilons):
        for position, label in enumerate(triple):
            ends.setdefault(label, []).append((s, position))
    if any(len(pair) != 2 for pair in ends.values()):
        raise ValueError("Every label must occur exactly twice")

    visited = [False] * len(deltas)
    factor = 1
    if epsilons:
        targets = []
        for position, label in enumerate(epsilons[0]):
            end, here = (0, position), label
            while True:
                first, second = ends[here]
                end = second if first == end else first
                if isinstance(end, tuple):
                    break
                visited[end] = True
                u, v = deltas[end]
                here = v if u == here else u
            if end[0] == 0:
                return 0
            targets.append(end[1])
        factor = 6 * sort_with_sign(targets)[1]

    cycles = 0
    for k in range(len(deltas)):
        if visited[k]:
            continue
        cycles += 1
        edge, here = k, deltas[k][0]
        while not visited[edge]:
            visited[edge] = True
            u, v = deltas[edge]
            here = v if u == here else u
            first, second = ends[here]
            edge = second if first == edge else first

    return factor * 3**cycles


def _reduce(deltas: list, epsilons: list) -> list[tuple[int, list, list]]:
    """Sum over the repeated labels of one product of deltas and Levi-Civita symbols.

    Args:
        deltas: Pairs of labels, as mutable lists.
        epsilons: Triples of labels, as mutable lists.

    Returns:
        `(factor, deltas, epsilons)` for each product the sum leaves, every label in it
        occurring once; empty when the product vanishes.
    """
    factor, deltas, epsilons = _contract_deltas(deltas, epsilons)

    if any(len(set(triple)) < 3 for triple in epsilons):
        return []

    for first, second in itertools.combinations(range(len(epsilons)), 2):
        if set(epsilons[first]) & set(epsilons[second]):
            a, b = epsilons[first], epsilons[second]
            others = [e for k, e in enumerate(epsilons) if k not in (first, second)]
            reduced = []
            for permutation in itertools.permutations(range(3)):
                sign = sort_with_sign(permutation)[1]
                expanded = [pair[:] for pair in deltas]
                expanded += [[a[i], b[permutation[i]]] for i in range(3)]
                remaining = [triple[:] for triple in others]
                for inner, kept, symbols in _reduce(expanded, remaining):
                    reduced.append((sign * factor * inner, kept, symbols))

            return reduced

    return [(factor, deltas, epsilons)]


def _contract_deltas(deltas: list, epsilons: list) -> tuple[int, list, list]:
    """Collapse every delta that carries a repeated label.

    A delta with a label that occurs elsewhere moves its other label there and
    disappears; a delta whose two labels coincide is a closed loop, a factor of 3.

    Returns:
        The integer factor, the deltas left (each label in them free), and the
        Levi-Civita symbols with their labels renamed.
    """
    pending = [pair[:] for pair in deltas]
    symbols = [triple[:] for triple in epsilons]
    factor = 1
    kept = []
    while pending:
        a, b = pending.pop()
        if a == b:
            factor *= 3
        elif not (_replace(a, b, pending, symbols) or _replace(b, a, pending, symbols)):
            kept.append([a, b])

    return factor, kept, symbols


def _replace(old: Hashable, new: Hashable, deltas: list, epsilons: list) -> bool:
    """Replace one occurrence of a label among the blocks; whether one was found."""
    for block in (*deltas, *epsilons):
        for position, label in enumerate(block):
            if label == old:
                block[position] = new
                return True

    return False
