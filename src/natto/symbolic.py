"""Symbolic representation of the isotropic operators the package builds.

Every operator is a sum of products of Kronecker deltas and Levi-Civita symbols with
exact rational coefficients. `Signature` names its groups of indices, `Term` is one
canonical product over integer index slots, and `Operator` is a sum of terms collected
by construction. Each prints and parses in the familiar delta and epsilon notation, and
evaluates to an array in any order of its groups.
"""

import re
import string
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from fractions import Fraction
from functools import cached_property
from types import MappingProxyType

import numpy as np

from natto.utils import dij, eijk


@dataclass(frozen=True)
class IndexGroup:
    """One named group of index slots in a signature.

    Attributes:
        name: What the indices are, e.g. `rank` or `weight`.
        size: How many indices the group has.
        upper: Whether the group prints with upper-case letters. Groups of the same
            case take consecutive letters in signature order.
    """

    name: str
    size: int
    upper: bool

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
    package prints: for a mapping tensor, rank indices `A...` then weight indices
    `a...`.

    Args:
        groups: The index groups.

    Raises:
        ValueError: If two groups share a name, or one case needs more than 26 letters.

    References:
        Definition 1 (Sec. 6.1) of [Wen2026Refactor].
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

    References:
        Definition 2 and Proposition 3 (Sec. 6.2) of [Wen2026Refactor].
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
    def from_letters(
        cls,
        signature: Signature,
        deltas: Iterable[str] = (),
        epsilons: Iterable[str] = (),
    ) -> tuple[int, "Term"]:
        """Build the canonical term of a product written in a signature's letters.

        Args:
            signature: The signature the letters belong to.
            deltas: Two letters per Kronecker delta, e.g. `["Aa", "BC"]`.
            epsilons: Three letters per Levi-Civita symbol, in the symbol's order.

        Returns:
            The sign of the product relative to the canonical term, and the term.

        Raises:
            ValueError: If a letter is not in the signature.
        """
        return cls.from_blocks(
            [[signature.slot_of(letter) for letter in pair] for pair in deltas],
            [[signature.slot_of(letter) for letter in triple] for triple in epsilons],
        )

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

    @property
    def slots(self) -> tuple[int, ...]:
        """The slots the term uses, sorted."""
        return tuple(
            sorted(slot for block in (*self.deltas, *self.epsilons) for slot in block)
        )


class Operator:
    """A sum of canonical terms over the slots of a signature, with exact coefficients.

    Terms are collected by construction: equal terms merge into the position of the
    first, and a coefficient that reaches zero removes its term. Every term uses every
    slot of the signature exactly once, since all of an operator's indices are free.
    Operators are immutable; arithmetic returns new ones.

    Build one from `(coefficient, term)` pairs or from its printed form with
    `Operator.parse`; `natto.algebra.contract` builds one out of others.

    Args:
        signature: The index groups of the operator.
        terms: `(coefficient, term)` pairs in the order they should print, each
            coefficient an `int` or a `Fraction`.

    Raises:
        TypeError: If a coefficient is not exact.
        ValueError: If a term does not use every slot of the signature exactly once.

    References:
        Definition 3 (Sec. 6.3) of [Wen2026Refactor].
    """

    __slots__ = ("_signature", "_terms")

    def __init__(
        self, signature: Signature, terms: Iterable[tuple[int | Fraction, Term]] = ()
    ):
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

    @classmethod
    def parse(cls, text: str, signature: Signature) -> "Operator":
        """Build an operator from its printed form.

        A term is a coefficient followed by its factors, as the package prints them:
        `+1/2 δ_Aa δ_Bb  -1/3 δ_AB δ_ab`. A factor is a tensor name, an underscore
        and its index letters, and the name may be typed in plain ASCII: `d` or `delta`
        for the Kronecker delta, `e`, `eps` or `epsilon` for the Levi-Civita symbol,
        as well as the printed `δ` and `ε`. Factors before any coefficient form a
        term with coefficient 1, so `d_AB d_CD` is a single term.

        Args:
            text: The printed operator; an empty string is the zero operator.
            signature: The signature whose letters the factors use.

        Returns:
            The operator, with its terms canonical and collected.

        Raises:
            ValueError: If a factor is malformed or uses a letter outside the signature.
        """
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

        return cls(signature, terms)

    @property
    def signature(self) -> Signature:
        """The index groups of the operator."""
        return self._signature

    @property
    def terms(self) -> MappingProxyType:
        """The terms and their coefficients, in print order; read-only."""
        return MappingProxyType(self._terms)

    def to_string(self, ascii: bool = False) -> str:
        """The printed form, as `str` gives it or in plain ASCII.

        Args:
            ascii: Name the tensors `d` and `e` rather than `δ` and `ε`.

        Returns:
            The terms joined by two spaces; the zero operator prints as an empty string.
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

        return "  ".join(printed)

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
        names = [group.name for group in self._signature.groups]
        order = names if order is None else list(order)
        if sorted(order) != sorted(names):
            raise ValueError(f"The order must name the groups {names}, got {order}")
        axes = [slot for name in order for slot in self._signature.slots(name)]

        delta, epsilon = dij(), eijk()
        result = np.zeros((3,) * self._signature.size)
        for term, coefficient in self._terms.items():
            operands = []
            for pair in term.deltas:
                operands += [delta, list(pair)]
            for triple in term.epsilons:
                operands += [epsilon, list(triple)]
            if operands:
                result = result + float(coefficient) * np.einsum(*operands, axes)
            else:
                result = result + float(coefficient)

        return result

    def __eq__(self, other) -> bool:
        if not isinstance(other, Operator):
            return NotImplemented

        return self._signature == other._signature and self._terms == other._terms

    __hash__ = None

    def __len__(self) -> int:
        return len(self._terms)

    def __add__(self, other: "Operator") -> "Operator":
        if not isinstance(other, Operator):
            return NotImplemented
        if other._signature != self._signature:
            raise ValueError("Operators with different signatures cannot be added")

        return Operator(
            self._signature,
            [(c, t) for t, c in (*self._terms.items(), *other._terms.items())],
        )

    def __neg__(self) -> "Operator":
        return self * -1

    def __sub__(self, other: "Operator") -> "Operator":
        return self + (-other)

    def __mul__(self, scalar: int | Fraction) -> "Operator":
        if isinstance(scalar, bool) or not isinstance(scalar, (int, Fraction)):
            return NotImplemented

        return Operator(
            self._signature, [(c * scalar, t) for t, c in self._terms.items()]
        )

    def __rmul__(self, scalar: int | Fraction) -> "Operator":
        return self.__mul__(scalar)

    def __str__(self) -> str:
        return self.to_string()

    def __repr__(self) -> str:
        return f"Operator({self._signature!r}, {self.to_string()!r})"


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

#: A factor token: a tensor name, an underscore and the index letters.
_FACTOR = re.compile(r"^(δ|ε|delta|epsilon|eps|d|e)_([A-Za-z]+)$")

_DELTA_NAMES = {"δ", "d", "delta"}


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
