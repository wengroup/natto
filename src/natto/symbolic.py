"""Symbolic representation of the isotropic operators the package builds.

Every operator is a sum of products of Kronecker deltas and Levi-Civita symbols with
exact rational coefficients, times at most the square root of one square-free integer.
`Signature` names its groups of indices, `Term` is one canonical product over integer
index slots, and `Operator` is a sum of terms collected by construction. Each prints
and parses in the familiar delta and epsilon notation, and evaluates to an array in any
order of its groups. `evaluate` gives that array with the einsum rule that applies the
operator, and `act` applies it.
"""

import math
import re
import string
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from fractions import Fraction
from functools import cached_property
from types import MappingProxyType

import numpy as np

from natto.rational import square_free_split


@dataclass(frozen=True)
class IndexGroup:
    """One named group of index slots in a signature.

    Attributes:
        name: What the indices are, e.g. `ict` or `gct`.
        size: How many indices the group has.
        upper: Whether the group prints with upper-case letters. Groups of the same
            case take consecutive letters in signature order.
        input: Whether an array is contracted into this group when the operator is
            applied; see `evaluate`.
    """

    name: str
    size: int
    upper: bool
    input: bool = False

    def __post_init__(self):
        if self.size < 0:
            raise ValueError(f"Group {self.name!r} has negative size {self.size}")


@dataclass(frozen=True)
class Signature:
    """The index groups of an operator, in order.

    An operator's slots are the integers 0, ..., size - 1, numbered group by group.
    The signature decides the letters an operator prints with and the axis order of
    any array made from it. Lower-case groups take the letters a, b, c, ... in
    order, and upper-case groups A, B, C, ..., which reproduces the letters the
    package prints: for a mapping tensor, ICT indices `a...` then Cartesian tensor
    indices `A...`.

    Args:
        groups: The index groups.

    Raises:
        ValueError: If two groups share a name, or one case needs more than 26 letters.
    """

    groups: tuple[IndexGroup, ...]

    def __post_init__(self):
        names = [group.name for group in self.groups]
        if len(set(names)) != len(names):
            raise ValueError(f"Group names must be distinct, got {names}")
        for upper in (True, False):
            letters = sum(group.size for group in self.groups if group.upper == upper)
            if letters > 26:
                raise ValueError(f"A signature needs {letters} letters of one case")

    @property
    def size(self) -> int:
        """The number of index slots."""
        return sum(group.size for group in self.groups)

    def slots(self, name: str) -> range:
        """The slots of the group called `name`.

        Raises:
            ValueError: If no group has that name.
        """
        start = 0
        for group in self.groups:
            if group.name == name:
                return range(start, start + group.size)
            start += group.size

        raise ValueError(f"No index group named {name!r}")

    @cached_property
    def letters(self) -> str:
        """The letter each slot prints with, in slot order."""
        letters = []
        counts = {True: 0, False: 0}
        for group in self.groups:
            alphabet = string.ascii_uppercase if group.upper else string.ascii_lowercase
            start = counts[group.upper]
            letters.append(alphabet[start : start + group.size])
            counts[group.upper] += group.size

        return "".join(letters)

    def letters_of(self, name: str) -> str:
        """The letters of the group called `name`, in slot order."""
        return "".join(self.letters[slot] for slot in self.slots(name))

    def slot_of(self, letter: str) -> int:
        """The slot a printed letter stands for.

        Raises:
            ValueError: If no slot prints with that letter.
        """
        slot = self.letters.find(letter)
        if slot < 0:
            raise ValueError(f"Letter {letter!r} is not an index of this signature")

        return slot


@dataclass(frozen=True)
class Term:
    """A canonical product of Kronecker deltas and Levi-Civita symbols over slots.

    Each delta is a pair of slots and each Levi-Civita symbol a triple, and no slot
    occurs twice. Canonical means every pair and every triple is in increasing order
    and both collections are sorted, so equal products are equal terms. Build one from
    arbitrary blocks with `Term.from_blocks`, which returns the sign that putting the
    triples in order introduces, or from its printed factors with `Term.parse`.

    Attributes:
        deltas: The delta pairs.
        epsilons: The Levi-Civita triples, at most two.

    Raises:
        ValueError: If the blocks are not canonical, a slot occurs twice, or there are
            more than two Levi-Civita symbols.
    """

    deltas: tuple[tuple[int, int], ...] = ()
    epsilons: tuple[tuple[int, int, int], ...] = ()

    def __post_init__(self):
        for pair in self.deltas:
            if len(pair) != 2 or not pair[0] < pair[1]:
                raise ValueError(f"A delta must be an increasing pair, got {pair}")
        for triple in self.epsilons:
            if len(triple) != 3 or not triple[0] < triple[1] < triple[2]:
                raise ValueError(
                    f"A Levi-Civita symbol must be increasing, got {triple}"
                )
        if list(self.deltas) != sorted(self.deltas):
            raise ValueError("The deltas of a term must be sorted")
        if list(self.epsilons) != sorted(self.epsilons):
            raise ValueError("The Levi-Civita symbols of a term must be sorted")
        if len(self.epsilons) > 2:
            raise ValueError("A term carries at most two Levi-Civita symbols")
        slots = [slot for block in (*self.deltas, *self.epsilons) for slot in block]
        if len(set(slots)) != len(slots):
            raise ValueError(f"A slot occurs twice in {slots}; contract it first")

    @classmethod
    def from_blocks(
        cls,
        deltas: Iterable[Sequence[int]] = (),
        epsilons: Iterable[Sequence[int]] = (),
    ) -> tuple[int, "Term"]:
        """Build the canonical term of a product given by its blocks, in any order.

        Args:
            deltas: Pairs of slots, one per Kronecker delta.
            epsilons: Triples of slots, one per Levi-Civita symbol, each in the order
                of that symbol's indices.

        Returns:
            The sign of the product relative to the canonical term, and the term.
        """
        sign = 1
        triples = []
        for triple in epsilons:
            ordered, permutation_sign = sort_with_sign(triple)
            sign *= permutation_sign
            triples.append(ordered)
        pairs = sorted(tuple(sorted(pair)) for pair in deltas)
        term = cls(tuple(pairs), tuple(sorted(triples)))

        return sign, term

    @classmethod
    def parse(cls, text: str, signature: Signature) -> tuple[int, "Term"]:
        """Build a term from its printed factors, such as `d_Aa d_BC` or `δ_Aa δ_BC`.

        Args:
            text: Factors separated by spaces; see `Operator.parse` for their names.
            signature: The signature whose letters the factors use.

        Returns:
            The sign of the product relative to the canonical term, and the term.

        Raises:
            ValueError: If a factor is malformed or uses a letter outside the signature.
        """
        return _parse_factors(text.split(), signature)

    @cached_property
    def slots(self) -> tuple[int, ...]:
        """The slots the term uses, sorted.

        Cached, since every operator built from the term checks it, and terms are shared
        between operators.
        """
        return tuple(
            sorted(slot for block in (*self.deltas, *self.epsilons) for slot in block)
        )


class Operator:
    """A sum of canonical terms over the slots of a signature, with exact coefficients.

    Terms are collected by construction: equal terms merge into the position of the
    first, and a coefficient that reaches zero removes its term. Every term uses every
    slot of the signature exactly once, since all of an operator's indices are free.
    Operators are immutable; arithmetic returns new ones.

    The whole sum may carry one irrational factor, the square root of the square-free
    integer `radicand`: the operator is `sqrt(radicand)` times its terms. Square-free
    means no prime divides it twice -- 2, 5 and 6 are, 8 = 2**2 * 2 is not -- which
    gives each root one written form; see `natto.rational.square_free_split`. It is
    what an orthonormal mapping needs, `1 / sqrt(d)` for a rational `d`, written as a
    rational, which goes into the coefficients, times such a root. The default, 1, is
    an ordinary rational operator. Operators with different radicands cannot be added,
    since the sum is no longer of this form; contracting them multiplies the radicands,
    and the square part of the product moves into the coefficients.

    Build one from `(coefficient, term)` pairs or from its printed form with
    `Operator.parse`; `natto.algebra.contract` builds one out of others.

    Args:
        signature: The index groups of the operator.
        terms: `(coefficient, term)` pairs in the order they should print, each
            coefficient an `int` or a `Fraction`.
        radicand: The square-free positive integer whose square root multiplies the
            terms. A zero operator always has radicand 1.

    Raises:
        TypeError: If a coefficient is not exact, or `radicand` is not an `int`.
        ValueError: If a term does not use every slot of the signature exactly once,
            or `radicand` is not a square-free positive integer.
    """

    __slots__ = ("_signature", "_terms", "_radicand")

    def __init__(
        self,
        signature: Signature,
        terms: Iterable[tuple[int | Fraction, Term]] = (),
        radicand: int = 1,
    ):
        if isinstance(radicand, bool) or not isinstance(radicand, int):
            raise TypeError(f"The radicand must be an int, got {radicand!r}")
        if radicand < 1 or square_free_split(radicand)[0] != 1:
            raise ValueError(
                f"The radicand must be a square-free positive integer, got {radicand}"
            )

        all_slots = tuple(range(signature.size))
        collected: dict[Term, Fraction] = {}
        for coefficient, term in terms:
            if isinstance(coefficient, bool) or not isinstance(
                coefficient, (int, Fraction)
            ):
                raise TypeError(f"Coefficients must be exact, got {coefficient!r}")
            if term.slots != all_slots:
                raise ValueError(
                    f"A term must use every slot once, got the slots {term.slots}"
                )
            collected[term] = collected.get(term, Fraction(0)) + coefficient

        self._signature = signature
        self._terms = {term: c for term, c in collected.items() if c != 0}
        self._radicand = radicand if self._terms else 1

    @classmethod
    def parse(cls, text: str, signature: Signature) -> "Operator":
        """Build an operator from its printed form.

        A term is a coefficient followed by its factors, as the package prints them:
        `+1/2 δ_Aa δ_Bb  -1/3 δ_AB δ_ab`. A factor is a tensor name, an underscore
        and its index letters, and the name may be typed in plain ASCII: `d` or `delta`
        for the Kronecker delta, `e`, `eps` or `epsilon` for the Levi-Civita symbol,
        as well as the printed `δ` and `ε`. Factors before any coefficient form a
        term with coefficient 1, so `d_AB d_CD` is a single term. An operator with a
        radicand prints as `√6 · (...)`, or `sqrt(6) * (...)` in ASCII, and parses
        back from either.

        Args:
            text: The printed operator; an empty string is the zero operator.
            signature: The signature whose letters the factors use.

        Returns:
            The operator, with its terms canonical and collected.

        Raises:
            ValueError: If a factor is malformed or uses a letter outside the signature.
        """
        radicand = 1
        radical = _RADICAL.match(text)
        if radical:
            radicand = int(radical.group(1) or radical.group(2))
            text = radical.group(3)

        pieces: list[tuple[Fraction, list[str]]] = []
        for token in text.split():
            if _COEFFICIENT.match(token):
                pieces.append((Fraction(token), []))
            else:
                if not pieces:
                    pieces.append((Fraction(1), []))
                pieces[-1][1].append(token)

        terms = []
        for coefficient, factors in pieces:
            sign, term = _parse_factors(factors, signature)
            terms.append((sign * coefficient, term))

        return cls(signature, terms, radicand)

    @property
    def signature(self) -> Signature:
        """The index groups of the operator."""
        return self._signature

    @property
    def terms(self) -> MappingProxyType:
        """The terms and their coefficients, in print order; read-only."""
        return MappingProxyType(self._terms)

    @property
    def radicand(self) -> int:
        """The square-free integer whose square root multiplies the terms."""
        return self._radicand

    def to_string(self, ascii: bool = False) -> str:
        """The printed form, as `str` gives it or in plain ASCII.

        Args:
            ascii: Name the tensors `d` and `e` rather than `δ` and `ε`.

        Returns:
            The terms joined by two spaces, inside `√s · (...)` when the radicand `s`
            is not 1; the zero operator prints as an empty string.
        """
        letters = self._signature.letters
        delta, epsilon = ("d", "e") if ascii else ("δ", "ε")

        printed = []
        for term, coefficient in self._terms.items():
            # A delta is symmetric, so its letters can simply be sorted; a Levi-Civita
            # symbol is sorted by its printed letters, with the sign that takes.
            sign = 1
            factors = sorted(
                f"{delta}_" + "".join(sorted(letters[i] + letters[j]))
                for i, j in term.deltas
            )
            symbols = []
            for triple in term.epsilons:
                ordered, permutation_sign = sort_with_sign([letters[s] for s in triple])
                sign *= permutation_sign
                symbols.append(f"{epsilon}_" + "".join(ordered))
            factors += sorted(symbols)

            value = sign * coefficient
            head = f"+{value}" if value >= 0 else f"{value}"
            printed.append(" ".join([head, *factors]))
        body = "  ".join(printed)

        if self._radicand != 1:
            root = f"sqrt({self._radicand}) *" if ascii else f"√{self._radicand} ·"
            body = f"{root} ({body})"

        return body

    def evaluate(self, order: Sequence[str] | None = None) -> np.ndarray:
        """Evaluate the operator into an array.

        Args:
            order: Group names in the order their axes should appear. Defaults to the
                signature's own order.

        Returns:
            An array with one axis of length 3 per slot, in the requested group order.

        Raises:
            ValueError: If `order` does not name every group exactly once.
        """
        result = evaluate_terms(self._signature, self._terms, order)
        if self._radicand != 1:
            result = result * math.sqrt(self._radicand)

        return result

    def __eq__(self, other) -> bool:
        if not isinstance(other, Operator):
            return NotImplemented

        return (
            self._signature == other._signature
            and self._terms == other._terms
            and self._radicand == other._radicand
        )

    __hash__ = None

    def __len__(self) -> int:
        return len(self._terms)

    def __add__(self, other: "Operator") -> "Operator":
        if not isinstance(other, Operator):
            return NotImplemented
        if other._signature != self._signature:
            raise ValueError("Operators with different signatures cannot be added")
        if self._terms and other._terms and self._radicand != other._radicand:
            raise ValueError("Operators with different radicands cannot be added")

        radicand = self._radicand if self._terms else other._radicand

        return Operator(
            self._signature,
            [(c, t) for t, c in (*self._terms.items(), *other._terms.items())],
            radicand,
        )

    def __neg__(self) -> "Operator":
        return self * -1

    def __sub__(self, other: "Operator") -> "Operator":
        return self + (-other)

    def __mul__(self, scalar: int | Fraction) -> "Operator":
        if isinstance(scalar, bool) or not isinstance(scalar, (int, Fraction)):
            return NotImplemented

        return Operator(
            self._signature,
            [(c * scalar, t) for t, c in self._terms.items()],
            self._radicand,
        )

    def __rmul__(self, scalar: int | Fraction) -> "Operator":
        return self.__mul__(scalar)

    def __str__(self) -> str:
        return self.to_string()

    def __repr__(self) -> str:
        return f"Operator({self._signature!r}, {self.to_string()!r})"


def evaluate(operator: Operator) -> tuple[np.ndarray, str]:
    """Evaluate an operator into an array, with the einsum rule that applies it.

    The array has its axes in the order of the operator's index groups, which for
    every operator of the package puts the output groups first, so ICT indices come
    before Cartesian tensor indices. The rule contracts one array into each input
    group and leaves the other groups free, so that `numpy.einsum(rule, array,
    *inputs)` applies the operator. Every operand and the output carry a leading
    ellipsis, so a batch of inputs applies as readily as one. The two are returned
    together because the rule is only valid for this axis order.

    Args:
        operator: The operator.

    Returns:
        The evaluated array and its einsum rule.
    """
    signature = operator.signature
    inputs = [group for group in signature.groups if group.input]
    outputs = [group for group in signature.groups if not group.input]

    operands = "".join(f",...{signature.letters_of(group.name)}" for group in inputs)
    output = "".join(signature.letters_of(group.name) for group in outputs)
    rule = f"{signature.letters}{operands}->...{output}"
    array = operator.evaluate()

    return array, rule


def act(operator: Operator, *inputs: np.ndarray) -> np.ndarray:
    """Apply an operator to arrays, one per input group.

    This is `evaluate` followed by `numpy.einsum`, for when only the result is needed.

    Args:
        operator: The operator.
        *inputs: One array per input group, in the order of the groups, each with the
            group's indices last and any batch dimensions before them.

    Returns:
        The operator applied to the inputs.

    Raises:
        ValueError: If there is not one array per input group.
    """
    expected = sum(group.input for group in operator.signature.groups)
    if len(inputs) != expected:
        raise ValueError(f"The operator takes {expected} inputs, got {len(inputs)}")

    array, rule = evaluate(operator)
    result = np.einsum(rule, array, *inputs)

    return result


def evaluate_terms(
    signature: Signature,
    terms: dict[Term, int | Fraction | float],
    order: Sequence[str] | None = None,
) -> np.ndarray:
    """Evaluate a sum of terms over the slots of a signature into an array.

    `Operator.evaluate` is this, times the square root of the operator's radicand.

    Args:
        signature: The index groups the terms' slots belong to.
        terms: Each term with its coefficient.
        order: Group names in the order their axes should appear. Defaults to the
            signature's own order.

    Returns:
        An array with one axis of length 3 per slot, in the requested group order.

    Raises:
        ValueError: If `order` does not name every group exactly once.
    """
    names = [group.name for group in signature.groups]
    order = names if order is None else list(order)
    if sorted(order) != sorted(names):
        raise ValueError(f"The order must name the groups {names}, got {order}")
    axes = [slot for name in order for slot in signature.slots(name)]

    # The stride of each slot in the flattened array, with the axes in `order`
    size = signature.size
    stride = [0] * size
    for position, slot in enumerate(axes):
        stride[slot] = 3 ** (size - 1 - position)

    # A product of deltas and Levi-Civita symbols is zero almost everywhere, so each
    # term adds its coefficient only where it is nonzero. Every entry receives the
    # same additions in the same order as summing the dense terms would give.
    flat = np.zeros(3**size)
    for term, coefficient in terms.items():
        indices, signs = _nonzero_entries(term, stride)
        flat[indices] += float(coefficient) * signs
    result = flat.reshape((3,) * size)

    return result


def sort_with_sign(items: Sequence) -> tuple[tuple, int]:
    """Sort a short sequence of distinct items, and return the sign of the permutation.

    Args:
        items: Distinct, comparable items, such as the slots of a Levi-Civita symbol.

    Returns:
        The items in increasing order, and the sign of the permutation that sorts them.
    """
    ordered = list(items)
    sign = 1
    for i in range(1, len(ordered)):
        j = i
        while j > 0 and ordered[j - 1] > ordered[j]:
            ordered[j - 1], ordered[j] = ordered[j], ordered[j - 1]
            sign = -sign
            j -= 1

    return tuple(ordered), sign


#: A coefficient token of a printed operator, such as `+1`, `-1/3` or `2`.
_COEFFICIENT = re.compile(r"^[+-]?\d+(/\d+)?$")

#: The radical around a printed operator: `√6 · (...)` or `sqrt(6) * (...)`.
_RADICAL = re.compile(r"^\s*(?:√(\d+)|sqrt\((\d+)\))\s*[·*]\s*\((.*)\)\s*$", re.DOTALL)

#: A factor token: a tensor name, an underscore and the index letters.
_FACTOR = re.compile(r"^(δ|ε|delta|epsilon|eps|d|e)_([A-Za-z]+)$")

#: The index values at which a Levi-Civita symbol is nonzero, and its value there.
_PERMUTATIONS = np.array(
    [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]
)
_PERMUTATION_SIGNS = np.array([1.0, -1.0, -1.0, 1.0, 1.0, -1.0])

_DELTA_NAMES = {"δ", "d", "delta"}


def _nonzero_entries(
    term: Term, stride: Sequence[int]
) -> tuple[np.ndarray, np.ndarray]:
    """The flat positions and values of the nonzero entries of one term.

    A delta is nonzero, and one, where its two indices agree; a Levi-Civita symbol where
    its three indices are a permutation of 0, 1 and 2, with that permutation's sign.
    The nonzero entries of the product are every combination of those.

    Args:
        term: A term using every slot of its signature once.
        stride: The stride of each slot in the flattened array.

    Returns:
        The flat positions, distinct, and the value at each, 1 or -1.
    """
    indices = np.zeros(1, dtype=np.int64)
    signs = np.ones(1)
    for a, b in term.deltas:
        offsets = np.arange(3) * (stride[a] + stride[b])
        indices = (indices[:, None] + offsets[None, :]).ravel()
        signs = np.repeat(signs, 3)
    for triple in term.epsilons:
        offsets = _PERMUTATIONS @ np.array([stride[slot] for slot in triple])
        indices = (indices[:, None] + offsets[None, :]).ravel()
        signs = (signs[:, None] * _PERMUTATION_SIGNS[None, :]).ravel()

    return indices, signs


def _parse_factors(factors: Sequence[str], signature: Signature) -> tuple[int, Term]:
    """The sign and canonical term of a product given by its printed factors."""
    deltas, epsilons = [], []
    for factor in factors:
        match = _FACTOR.match(factor)
        if match is None:
            raise ValueError(f"Cannot read the factor {factor!r}")
        name, letters = match.groups()
        slots = [signature.slot_of(letter) for letter in letters]
        if name in _DELTA_NAMES:
            if len(slots) != 2:
                raise ValueError(f"A delta takes two indices, got {factor!r}")
            deltas.append(slots)
        else:
            if len(slots) != 3:
                raise ValueError(
                    f"A Levi-Civita symbol takes three indices, got {factor!r}"
                )
            epsilons.append(slots)

    return Term.from_blocks(deltas, epsilons)
