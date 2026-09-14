"""Mapping tensors between a Cartesian tensor and its irreducible parts.

The candidate mapping tensors, the extraction duals, and the decomposition operators
that compose the two. Everything here is symbolic and exact.

A mapping tensor is the natural projector applied to a combination of rank-lowering
tensors. All mappings of one weight and rank combine the same candidates, so a
`Sector` numbers those candidates once and caches every contraction between two of
them, and a `Mapping` is a vector of rational coefficients over them. Which candidates
are independent is settled in `independence`, against the Gram matrix of `gram`;
`reduction` runs the whole pipeline and publishes the result.

References:
    Eq. 13 of [Wen2026] for the mapping tensors, Eq. 16 for the duals, and Eq. 19
    for the decomposition operators.

    [Wen2026] M. Wen, Reusable Operators for Irreducible Cartesian Tensor
    Decomposition and Coupling, arXiv:2609.05971 (2026).
"""

import string
from collections.abc import Sequence
from fractions import Fraction

from natto.algebra import contract
from natto.lowering import LoweringLabel, get_lowering_labels
from natto.natural_projector import get_natural_projector
from natto.symbolic import IndexGroup, Operator, Signature, Term


class Sector:
    """The candidate mapping tensors of one weight and rank, numbered once.

    Every mapping of a weight and rank is a combination of the same candidates, one per
    rank-lowering tensor, so the sector holds them and caches the contraction of any
    two, the label table. A `Mapping` of the sector is a vector over these candidates.

    Args:
        ell: Weight of the ICT.
        n: Rank of the Cartesian tensor.

    References:
        Definition 5 (Sec. 6.4) of [Wen2026Refactor].
    """

    def __init__(self, ell: int, n: int):
        self.ell = ell
        self.n = n
        self.labels = tuple(get_lowering_labels(ell, n))
        self.signature = Signature(
            (IndexGroup("rank", n, upper=True), IndexGroup("weight", ell, upper=False))
        )

        self._index = {label: i for i, label in enumerate(self.labels)}
        self._projector = None
        self._candidates = {}
        self._table = {}

    def __len__(self) -> int:
        return len(self.labels)

    def index(self, label: LoweringLabel) -> int:
        """The number of a candidate's label.

        Raises:
            ValueError: If the label is not a candidate of this sector.
        """
        if label not in self._index:
            raise ValueError(f"{label} is not a candidate of this sector")

        return self._index[label]

    def candidates(self) -> list["Mapping"]:
        """The candidate mapping tensors, one unit vector per label."""
        size = len(self)

        return [Mapping(self, [int(i == p) for i in range(size)]) for p in range(size)]

    def table(self, i: int, j: int) -> Fraction:
        """The contraction of candidates `i` and `j` over all their indices.

        Both carry the natural projector, which is symmetric and idempotent, so one of
        the two is left out: the entry is the contraction of the rank-lowering tensor
        of one candidate with the other candidate. The table is symmetric, so each pair
        is contracted once and cached.

        Returns:
            The exact entry L_ij, which is 2 * ell + 1 times the Gram entry.

        References:
            Lemma 1 (Eq. 15, Sec. 4.1) and Proposition 5 (Sec. 6.4) of
            [Wen2026Refactor].
        """
        key = (i, j) if i <= j else (j, i)
        if key not in self._table:
            sigma = [("sigma", k) for k in range(self.ell)]
            rank = [("rank", a) for a in range(self.n)]
            factors = [
                (self._lowering(key[0]), sigma + rank),
                (self._candidate(key[1]), rank + sigma),
            ]
            contracted = contract(factors, Signature(()))
            self._table[key] = contracted.terms.get(Term(), Fraction(0))

        return self._table[key]

    def _candidate(self, i: int) -> Operator:
        """Candidate `i`: the natural projector applied to rank-lowering tensor `i`."""
        if i not in self._candidates:
            if self._projector is None:
                self._projector = get_natural_projector(self.ell)
            sigma = [("sigma", k) for k in range(self.ell)]
            weight = [self.n + k for k in range(self.ell)]
            factors = [
                (self._projector, weight + sigma),
                (self._lowering(i), sigma + list(range(self.n))),
            ]
            self._candidates[i] = contract(factors, self.signature)

        return self._candidates[i]

    def _lowering(self, i: int) -> Operator:
        """Rank-lowering tensor `i`, its `sigma` indices first and its `rank` after.

        The free rank slots feed the sigma indices through deltas; a Levi-Civita symbol
        with a pair of rank slots takes the last sigma index as its tau index.
        """
        ell, n = self.ell, self.n
        label = self.labels[i]
        deltas = [(k, ell + slot) for k, slot in enumerate(label.free_slots(n))]
        deltas += [(ell + a, ell + b) for a, b in label.deltas]
        if len(label.epsilon) == 2:
            epsilons = [(ell - 1, ell + label.epsilon[0], ell + label.epsilon[1])]
        elif label.epsilon:
            epsilons = [tuple(ell + slot for slot in label.epsilon)]
        else:
            epsilons = []

        signature = Signature(
            (IndexGroup("sigma", ell, upper=False), IndexGroup("rank", n, upper=True))
        )
        sign, term = Term.from_blocks(deltas, epsilons)

        return Operator(signature, [(sign, term)])


class Mapping:
    """A mapping tensor: the natural projector applied to a combination of candidates.

    The coefficients are one exact rational per candidate of the sector. A candidate is
    a unit vector, and its dual, a symmetry-adapted mapping or a permuted mapping is
    another vector of the same sector, so mappings add, scale and permute as vectors
    and their contractions come from the sector's table. `expand` gives the delta and
    Levi-Civita terms, for printing and evaluation.

    Args:
        sector: The sector whose candidates the coefficients refer to.
        coefficients: One `int` or `Fraction` per candidate.

    Raises:
        ValueError: If there is not one coefficient per candidate.
        TypeError: If a coefficient is not exact.

    References:
        Definition 6 and Proposition 4 (Sec. 6.4) of [Wen2026Refactor].
    """

    __slots__ = ("sector", "coefficients", "_expansion")

    def __init__(self, sector: Sector, coefficients: Sequence[int | Fraction]):
        coefficients = tuple(coefficients)
        if len(coefficients) != len(sector):
            raise ValueError(
                f"This sector takes {len(sector)} coefficients, got {len(coefficients)}"
            )
        for c in coefficients:
            if isinstance(c, bool) or not isinstance(c, (int, Fraction)):
                raise TypeError(f"Coefficients must be exact, got {c!r}")

        self.sector = sector
        self.coefficients = tuple(Fraction(c) for c in coefficients)
        self._expansion = None

    def permute(self, permutation: Sequence[int]) -> "Mapping":
        """The mapping with its rank indices permuted, as `numpy.transpose` would.

        Args:
            permutation: Axis `k` of the permuted mapping is axis `permutation[k]` of
                this one.

        Returns:
            The permuted mapping, in the same sector.

        Raises:
            ValueError: If `permutation` does not permute the rank indices.

        References:
            Proposition 2 (Sec. 4.4) and Proposition 4 (Sec. 6.4) of
            [Wen2026Refactor].
        """
        n = self.sector.n
        if sorted(permutation) != list(range(n)):
            raise ValueError(
                f"Expected a permutation of {n} indices, got {permutation}"
            )

        new_slot = [0] * n
        for k, axis in enumerate(permutation):
            new_slot[axis] = k

        coefficients = [Fraction(0)] * len(self.sector)
        for i, c in enumerate(self.coefficients):
            if c:
                sign, label = self.sector.labels[i].relabel(new_slot)
                coefficients[self.sector.index(label)] += sign * c

        return Mapping(self.sector, coefficients)

    def expand(self) -> Operator:
        """The mapping as an operator, rank indices first and weight indices after."""
        if self._expansion is None:
            terms = []
            for i, c in enumerate(self.coefficients):
                if c:
                    candidate = self.sector._candidate(i)
                    terms += [
                        (c * value, term) for term, value in candidate.terms.items()
                    ]
            self._expansion = Operator(self.sector.signature, terms)

        return self._expansion

    def __add__(self, other: "Mapping") -> "Mapping":
        if not isinstance(other, Mapping):
            return NotImplemented

        return combine([1, 1], [self, other])

    def __sub__(self, other: "Mapping") -> "Mapping":
        if not isinstance(other, Mapping):
            return NotImplemented

        return combine([1, -1], [self, other])

    def __neg__(self) -> "Mapping":
        return self * -1

    def __mul__(self, scalar: int | Fraction) -> "Mapping":
        if isinstance(scalar, bool) or not isinstance(scalar, (int, Fraction)):
            return NotImplemented

        return Mapping(self.sector, [scalar * c for c in self.coefficients])

    def __rmul__(self, scalar: int | Fraction) -> "Mapping":
        return self.__mul__(scalar)

    def __eq__(self, other) -> bool:
        if not isinstance(other, Mapping):
            return NotImplemented

        return self.sector is other.sector and self.coefficients == other.coefficients

    __hash__ = None

    def __str__(self) -> str:
        letters = string.ascii_uppercase
        tau = letters[self.sector.n]

        parts = []
        for label, c in zip(self.sector.labels, self.coefficients):
            if not c:
                continue
            factors = [f"δ_{letters[i]}{letters[j]}" for i, j in label.deltas]
            if len(label.epsilon) == 2:
                factors.append("ε_" + tau + "".join(letters[s] for s in label.epsilon))
            elif label.epsilon:
                factors.append("ε_" + "".join(letters[s] for s in label.epsilon))
            head = f"+{c}" if c >= 0 else f"{c}"
            parts.append(f"{head} [{' '.join(factors)}]")

        return f"E({self.sector.ell}) · (" + "  ".join(parts) + ")"


def get_mappings(ell: int, n: int) -> list[Mapping]:
    """The candidate mapping tensors of a weight.

    A mapping tensor is a rank-lowering tensor followed by the natural projector, so
    there is one candidate per choice of which indices the rank lowering contracts
    away. The projector takes the indices that choice leaves unused.

    The parity of n - ell decides what the rank lowering looks like -- deltas alone,
    or deltas with one Levi-Civita symbol -- but that is `lowering`'s concern, and a
    caller supplies only the weight and rank.

    Args:
        ell: Weight of the ICT.
        n: Rank of the Cartesian tensor to map onto.

    Returns:
        One mapping per choice of contracted indices: the candidates of a new sector.

    References:
        Eq. 13 of [Wen2026].
    """
    return Sector(ell, n).candidates()


def combine(
    coefficients: Sequence[int | Fraction], mappings: Sequence[Mapping]
) -> Mapping:
    """The linear combination of mappings of one sector.

    Args:
        coefficients: One exact coefficient per mapping.
        mappings: Mappings of one sector.

    Returns:
        The combined mapping.

    Raises:
        ValueError: If there are no mappings, the counts differ, or the mappings belong
            to different sectors.
    """
    if not mappings:
        raise ValueError("At least one mapping is needed")
    if len(coefficients) != len(mappings):
        raise ValueError("One coefficient per mapping is needed")

    sector = mappings[0].sector
    total = [Fraction(0)] * len(sector)
    for c, mapping in zip(coefficients, mappings):
        if mapping.sector is not sector:
            raise ValueError("Mappings of different sectors cannot be combined")
        if c:
            for i, value in enumerate(mapping.coefficients):
                total[i] += c * value

    return Mapping(sector, total)


def get_extraction_operators(
    gram_inverse: list[list[Fraction]], embedding: list[Mapping]
) -> list[Mapping]:
    """Build the extraction operators dual to a set of embedding operators.

    Each dual is the combination of the embedding operators whose coefficients are the
    corresponding row of the inverse Gram matrix. Contracted with a Cartesian tensor,
    each returns the ICT of its weight and channel.

    Args:
        gram_inverse: Exact inverse of the embedding operators' Gram matrix.
        embedding: The embedding operators, in the order the matrix indexes.

    Returns:
        One extraction operator per row of `gram_inverse`, a mapping of the same sector.

    References:
        Eq. 16 of [Wen2026].
    """
    return [combine(row, embedding) for row in gram_inverse]


def get_decomposition_operators(
    G: list[Mapping], G_tilde: list[Mapping]
) -> list[Operator]:
    """Get the decomposition operators of a mapping and its dual.

    Each is a mapping composed with its own dual, so contracting one with a Cartesian
    tensor gives that tensor's part of this weight and channel directly, without
    forming the ICT on the way.

    Careful with the paper's S: there it is a rank-n *tensor*, one weight's part of a
    particular T, while here it is the rank-2n *operator* that produces it. Same
    letter, one the map and one its output.

    Args:
        G: Mapping tensors.
        G_tilde: The duals, in the order of the mappings they correspond to.

    Returns:
        One decomposition operator per channel.

    References:
        Eq. 19 of [Wen2026].
    """
    return [compose(G_i, dual_i) for G_i, dual_i in zip(G, G_tilde)]


def compose(mapping: Mapping, other: Mapping) -> Operator:
    """Compose two mappings of one sector over their weight indices.

    The result carries the rank indices of `mapping` and then those of `other`.
    `other` already carries the natural projector, so `mapping` contributes only its
    rank-lowering tensors.

    Args:
        mapping: The mapping whose rank indices come first.
        other: The mapping whose rank indices come second.

    Returns:
        The composition, with signature `rank` then `rank_in`.

    Raises:
        ValueError: If the mappings belong to different sectors.

    References:
        Lemma 2 (Eq. 16, Sec. 4.2) of [Wen2026Refactor].
    """
    sector = mapping.sector
    if other.sector is not sector:
        raise ValueError("Mappings of different sectors cannot be composed")
    n, ell = sector.n, sector.ell

    signature = Signature(
        (IndexGroup("rank", n, upper=True), IndexGroup("rank_in", n, upper=True))
    )
    sigma = [("sigma", k) for k in range(ell)]
    dual = other.expand()

    terms = []
    for i, c in enumerate(mapping.coefficients):
        if c:
            factors = [
                (sector._lowering(i), sigma + list(range(n))),
                (dual, [n + a for a in range(n)] + sigma),
            ]
            part = contract(factors, signature)
            terms += [(c * value, term) for term, value in part.terms.items()]

    return Operator(signature, terms)
