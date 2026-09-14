"""
Symbolic representation of the isotropic tensors, their products, and sums of those.

Two representations live here while the package moves from the first to the second.
The classes built on letters -- `IsotropicTensor`, `Delta`, `Epsilon`,
`IsotropicProduct` and `LinearCombination` -- are what the pipeline uses today.
`Signature`, `Term` and `Operator` describe the same objects over integer index slots
in named groups: every term is canonical and every sum collected by construction, and
each prints and parses in the familiar delta and epsilon notation.
"""

import re
import string
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from fractions import Fraction
from functools import cached_property
from types import MappingProxyType
from typing import Union

import numpy as np


class IsotropicTensor:
    """One isotropic tensor, carrying named indices.

    Isotropic means the components are the same in every rotated frame, and in three
    dimensions only two such tensors exist: the Kronecker delta and the Levi-Civita
    symbol. Every operator this package builds is made of those two alone, which is
    why they are the only atoms here. `Scalar` is the rank-zero case.

    Note this is not the Cartesian tensor being reduced, which is the caller's own
    array and never has a symbolic form. Repeated index pairs are allowed, as in ii
    or jj, since a contraction can leave them.

    Args:
        indices: The indices of the tensor.
        factor: The scalar factor multiplied to the tensor, default is 1.
        symbol: The symbol of the tensor, default is "T".
    """

    def __init__(self, indices: str, factor: int | Fraction = 1, symbol: str = "T"):
        self._check_indices(indices)
        self._indices = indices

        if isinstance(factor, int):
            self._factor = Fraction(factor)
        elif isinstance(factor, Fraction):
            self._factor = factor
        else:
            raise ValueError("The `factor` must be a Fraction object")

        self._symbol = symbol

    @property
    def indices(self):
        return self._indices

    @property
    def symbol(self):
        return self._symbol

    @property
    def factor(self):
        return self._factor

    @property
    def rank(self):
        return len(self.indices)

    def permute_indices(
        self, permute: list[int], factor: int | Fraction = 1
    ) -> "IsotropicTensor":
        """
        Permute the indices of the tensor.

        For example, if the tensor is T_ijk and permute is [1, 2, 0], the new tensor
        is T_jki.

        Args:
            permute: The new order of the indices.
            factor: The factor to be multiplied to the tensor, default is 1.

        Returns:
            The tensor with permuted indices.
        """
        indices = "".join([self.indices[i] for i in permute])
        if Counter(self.indices) != Counter(indices):
            raise ValueError("The new indices must contain the same indices")

        return self.__class__(indices, factor * self.factor, self.symbol)

    @staticmethod
    def _check_indices(indices: str):
        """Check indices are repeated at most twice.

        Only check for alphabetic indices, not integer indices. When the indices are
        evaluated, there can be many repeated values of the same integer.
        """
        for i in indices:
            if i.isalpha() and indices.count(i) > 2:
                raise ValueError("Indices can be repeated at most twice")

    def __contains__(self, item):
        return item in self.indices

    def __iter__(self):
        return iter(self.indices)

    def __getitem__(self, index):
        return self.indices[index]

    def __mul__(self, other: int | Fraction):
        """Multiply the tensor by a scalar."""
        return self.__class__(self.indices, self.factor * other, self.symbol)

    def __rmul__(self, other: int | Fraction):
        return self.__mul__(other)

    def __eq__(self, other):
        idx2pos_self = defaultdict(list)
        idx2pos_other = defaultdict(list)
        for p, i in enumerate(self.indices):
            idx2pos_self[i].append(p)
        for p, i in enumerate(other.indices):
            idx2pos_other[i].append(p)

        # check if the indices are the same, ignoring repeated indices
        idx2pos_self = {k: v for k, v in idx2pos_self.items() if len(v) == 1}
        idx2pos_other = {k: v for k, v in idx2pos_other.items() if len(v) == 1}
        if idx2pos_self != idx2pos_other:
            return False

        return self.symbol == other.symbol and self.factor == other.factor

    def __str__(self):
        if self.factor == Fraction(1):
            return f"{self.symbol}_{self.indices}"
        else:
            return f"({self.factor}) {self.symbol}_{self.indices}"


class Scalar(IsotropicTensor):
    """
    Zero rank tensor, a scalar.
    """

    def __init__(self, factor: int | Fraction = 1):
        super().__init__("", factor, "Const")

    def __str__(self):
        return f"{self.symbol}({self.factor})"


class Zero(Scalar):
    """The scalar zero."""

    def __init__(self):
        super().__init__(0)


class Delta(IsotropicTensor):
    """
    The Kronecker delta tensor.

    Args:
        indices: The indices of the delta tensor.
        factor: The scalar factor multiplied to the delta tensor, default is 1.
        symbol: The symbol of the delta tensor, default is "δ".
    """

    def __init__(self, indices: str, factor: int | Fraction = 1, symbol: str = "δ"):
        assert len(indices) == 2, "The delta tensor must have two indices"
        super().__init__(indices, factor, symbol)

    def __eq__(self, other):
        # should not compare symbol, but just make sure it is a Delta tensor
        if not isinstance(other, Delta):
            return False

        if not self.factor == other.factor:
            return False

        if self.indices != other.indices and self.indices != other.indices[::-1]:
            return False

        return True


class Epsilon(IsotropicTensor):
    """
    The Levi-Civita tensor.

    Args:
        indices: The indices of the epsilon tensor.
        factor: The scalar factor multiplied to the epsilon tensor, default is 1.
        symbol: The symbol of the epsilon tensor, default is "ε".
    """

    def __init__(self, indices: str, factor: int | Fraction = 1, symbol: str = "ε"):
        assert len(indices) == 3, "The epsilon tensor must have three indices"
        super().__init__(indices, factor, symbol)

    def __eq__(self, other):
        # should not compare symbol, but just make sure it is an Epsilon tensor
        if not isinstance(other, Epsilon):
            return False

        if not self.factor == other.factor:
            return False

        # even permutations of indices are equal
        indices = other.indices
        if (
            self.indices != indices
            and self.indices != indices[1] + indices[2] + indices[0]
            and self.indices != indices[2] + indices[0] + indices[1]
        ):
            return False

        return True


class IsotropicProduct:
    """A product of isotropic tensors.

    Deltas and Levi-Civita symbols multiplied together, which is the form every term
    of every operator here takes. It is not a general tensor product: nothing but
    those two, and scalars, can appear in it.

    Args:
        tensors: The constituting tensors.
        factor: Additional factor multiplied to the product. Each tensor in the
            product can have its own factor, so the overall factor is the product of
            theirs and this one.
        combine_scalars: If True, scalars are folded into the product's factor and
            removed from the product itself. Default is True.
    """

    def __init__(
        self,
        *tensors: IsotropicTensor | Epsilon | Delta | Scalar,
        factor: int | Fraction = 1,
        combine_scalars: bool = True,
    ):
        self.combine_scalars = combine_scalars

        if not combine_scalars:
            self._factor = factor
            self._tensors = list(tensors)
        else:
            # get overall factor
            for t in tensors:
                factor *= t.factor
            self._factor = factor

            if self._factor == 0:
                self._tensors = [Zero()]

            else:
                # set the factor of the constituting tensors to 1
                self._tensors = []
                for t in tensors:
                    if isinstance(t, Scalar):
                        pass  # scalars already been included in the factor
                    else:
                        self._tensors.append(t.__class__(t.indices, 1, t.symbol))

    @property
    def factor(self):
        """The overall factor of the isotropic product."""
        return self._factor

    @property
    def components(self):
        """Constituting tensors of the product, without considering the factor."""
        return self._tensors

    @property
    def indices(self):
        """The indices of the isotropic product."""
        return "".join([t.indices for t in self._tensors])

    def permute_indices(
        self, permute: list[int], factor: int | Fraction = 1
    ) -> "IsotropicProduct":
        """
        Permute the indices of the isotropic product.

        For example,
        if the tensor is D_ab T_ijk and permute is [2,4,0,1,3], the new tensor is
        D_ik T_abj.

        Args:
            permute: The new order of the indices.
            factor: Additional factor to be multiplied to the tensor, default is 1.

        Returns:
            The isotropic product with permuted indices.
        """

        indices = self.indices

        i = 0
        tensors = []
        for t in self._tensors:
            perm = permute[i : i + len(t.indices)]
            permuted_indices = "".join([indices[p] for p in perm])
            nt = t.__class__(permuted_indices, t.factor, t.symbol)
            tensors.append(nt)
            i += len(t.indices)

        return IsotropicProduct(*tensors, factor=factor * self.factor)

    def canonize(self):
        """
        Canonicalize the isotropic product.

        1. The canonized form will be like: delta_... epsilon_... T_...
        2. For a delta, the indices will be ordered, e.g. delta_ji -> delta_ij
        3. For an epsilon, the indices will be shifted such that the first index is the
           smallest one. e.g. epsilon_jik -> epsilon_ikj, epsilon_jki -> epsilon_ijk,
           while keeping the relative order of the indices.
        4. For a general tensor, no ordering of the indices are performed.
        5. For tensors of the same type (e.g. two delta tensors), they will be ordered
           by their first indices. For example, delta_ij delta_ab -> delta_ab delta_ij,
           since a < i. This is similarly for the epsilon tensors and general tensors.

        Returns:
            A canonized isotropic product.
        """
        # TODO, this assumes the factor of each component is 1, which may not be true
        #  in general.

        deltas = []
        epsilons = []
        general = []
        for t in self._tensors:
            if isinstance(t, Delta):
                deltas.append(t)
            elif isinstance(t, Epsilon):
                epsilons.append(t)
            else:
                general.append(t)

        # canonize deltas
        all_indices = ["".join(sorted(t.indices)) for t in deltas]
        all_indices = sorted(all_indices)
        deltas = [Delta(indices) for indices in all_indices]

        # canonize epsilons
        def sort_circular(s: str):
            """Shift the indices so the smallest comes first.

            For example, kij -> ijk.
            """
            # Find the index of the smallest character
            min_index = s.index(min(s))
            # Rotate the string to bring the smallest character to the front
            return s[min_index:] + s[:min_index]

        all_indices = [sort_circular(t.indices) for t in epsilons]
        all_indices = sorted(all_indices)
        epsilons = [Epsilon(indices) for indices in all_indices]

        # canonize general tensors
        all_indices = sorted([t.indices for t in general])
        general = [IsotropicTensor(indices) for indices in all_indices]

        # Create the isotropic product
        tensors = deltas + epsilons + general
        return IsotropicProduct(*tensors, factor=self.factor)

    def __eq__(self, other: Union[IsotropicTensor, "IsotropicProduct"]):
        if len(self) != len(other):
            return False

        if self.factor != other.factor:
            return False

        # compare symbol and indices of the constituting tensors
        if not str(self.canonize()) == str(other.canonize()):
            return False

        return True

    def __mul__(self, other: int | Fraction):
        """Multiply the tensor by a scalar."""
        return self.__class__(*self._tensors, factor=self.factor * other)

    def __rmul__(self, other: int | Fraction):
        return self.__mul__(other)

    def __iter__(self):
        return iter(self._tensors)

    def __getitem__(self, item):
        return self._tensors[item]

    def __len__(self):
        return len(self._tensors)

    def __str__(self):
        rep = self.str_rep_without_factor()
        if self.factor >= 0:
            factor = f"+{self.factor}"
        else:
            factor = self.factor
        return f"{factor}{rep}"

    def str_rep_without_factor(self):
        """Get the string representation of the isotropic product without the factor."""
        rep = ""
        for t in self._tensors:
            # scalars will be included in the factor, so we skip them here
            if not isinstance(t, Scalar):
                rep += f" {t.symbol}_{t.indices}"

        return rep


class LinearCombination:
    """A sum of isotropic tensors and products of them, with rational coefficients.

    This is what an operator is: `simplify_linear_combination` reduces one to its
    canonical form, and `evaluate_tensors` turns it into an array.
    """

    def __init__(self, *tensors: IsotropicTensor | Delta | Epsilon | IsotropicProduct):
        self._tensors = tensors

    @property
    def components(self):
        """The constituting isotropic products."""
        return self._tensors

    def to_str_list(self, including_zero: bool = False) -> list[str]:
        """
        Convert the tensors to string representation.

        Args:
            including_zero: If True, include zero tensors in the output.
        """
        return [str(t) for t in self._tensors if including_zero or t.factor != 0]

    def __eq__(self, other: "LinearCombination"):
        # TODO, we just implement the case that the constituting tensors are the same
        #  and in the same order. Of course, this is not general.

        if len(self) != len(other):
            return False

        for x, y in zip(self._tensors, other._tensors):
            if x != y:
                return False

        return True

    def __len__(self):
        return len(self._tensors)

    def __iter__(self):
        return iter(self._tensors)

    def __getitem__(self, item):
        return self._tensors[item]

    def __add__(self, other: "LinearCombination"):
        return LinearCombination(*self._tensors, *other._tensors)

    def __radd__(self, other: "LinearCombination"):
        # Handle the sum() case. Note, sum([X]) is expanded as 0 + X
        if other == 0:
            return self
        return self.__add__(other)

    def __mul__(self, other: int | Fraction):
        """Multiply the tensor by a scalar."""
        return LinearCombination(*[t * other for t in self._tensors])

    def __rmul__(self, other: int | Fraction):
        return self.__mul__(other)

    def __str__(self):
        str_rep = self.to_str_list(including_zero=False)

        return "  ".join(str_rep)


def create_delta_epsilon_tensors(
    rule: list[str], epsilon: str = None, factor: int | Fraction = 1
) -> IsotropicProduct:
    """Create a IsotropicProduct of deltas and epsilons.

    Currently, we only support a single epsilon tensor in the product, because it is
    all needed to create the E, G, H tensors.

    Args:
        rule: Each string contains a pair of indices for a delta tensor.
        epsilon: A three letter string for the epsilon tensor.
        factor: additional factor to multiply with the isotropic product

    Returns:
        List of IsotropicProduct objects.
    """

    tensors = [Delta(pair) for pair in rule]

    if epsilon is not None:
        e = Epsilon(epsilon)
        tensors.append(e)

    tp = IsotropicProduct(*tensors, factor=factor)

    return tp


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
            ordered, permutation_sign = _sort_with_sign(triple)
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

    Build one from `(coefficient, term)` pairs, from its printed form with
    `Operator.parse`, or from a `LinearCombination` with `from_linear_combination`.

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

    @classmethod
    def from_linear_combination(
        cls, combination: "LinearCombination", signature: Signature
    ) -> "Operator":
        """Convert a collected `LinearCombination` into an operator.

        Args:
            combination: Products of deltas and Levi-Civita symbols whose letters are
                those of `signature`, each occurring once in a product.
            signature: The signature the letters belong to.

        Returns:
            The same operator, with its terms canonical and collected.

        Raises:
            ValueError: If a letter is not in the signature or repeats within a product.
        """
        terms = []
        for product in combination:
            if product.factor == 0:
                continue
            deltas, epsilons = [], []
            for tensor in product:
                slots = [signature.slot_of(letter) for letter in tensor.indices]
                if isinstance(tensor, Delta):
                    deltas.append(slots)
                elif isinstance(tensor, Epsilon):
                    epsilons.append(slots)
                else:
                    raise ValueError(
                        f"Only deltas and Levi-Civita symbols, got {tensor}"
                    )
            sign, term = Term.from_blocks(deltas, epsilons)
            terms.append((sign * product.factor, term))

        return cls(signature, terms)

    @property
    def signature(self) -> Signature:
        """The index groups of the operator."""
        return self._signature

    @property
    def terms(self) -> MappingProxyType:
        """The terms and their coefficients, in print order; read-only."""
        return MappingProxyType(self._terms)

    def to_linear_combination(self) -> "LinearCombination":
        """Convert into a `LinearCombination` over the signature's letters."""
        letters = self._signature.letters
        products = []
        for term, coefficient in self._terms.items():
            tensors = [Delta(letters[i] + letters[j]) for i, j in term.deltas]
            tensors += [
                Epsilon("".join(letters[slot] for slot in triple))
                for triple in term.epsilons
            ]
            products.append(IsotropicProduct(*tensors, factor=coefficient))

        return LinearCombination(*products)

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
                ordered, permutation_sign = _sort_with_sign(
                    [letters[s] for s in triple]
                )
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
        # `natto.utils` imports this module through `natto.indices`, so the two
        # isotropic tensors are imported here rather than at the top.
        from natto.utils import dij, eijk

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


def _sort_with_sign(items: Sequence) -> tuple[tuple, int]:
    """Sort a short sequence of distinct items, and return the permutation's sign."""
    ordered = list(items)
    sign = 1
    for i in range(1, len(ordered)):
        j = i
        while j > 0 and ordered[j - 1] > ordered[j]:
            ordered[j - 1], ordered[j] = ordered[j], ordered[j - 1]
            sign = -sign
            j -= 1

    return tuple(ordered), sign
