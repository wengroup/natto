"""The reduction of a Cartesian tensor into irreducible Cartesian tensors.

This is the public entry point for the operators that move between a Cartesian tensor
and its ICTs. Every operator is keyed by `(ell, p)`, the weight of its ICT and the
multiplicity index p = 1, ..., N_ell of its channel, as the paper labels them.

- `get_extraction_operators` takes a Cartesian tensor to the ICT of each weight and
  channel.
- `get_embedding_operators` takes that ICT back to the tensor's part of the weight and
  channel. Summing those parts over every weight and channel returns the tensor.
- `get_reduction` gives the two together.
- `get_composed_operators` gives the two composed into one rank-2n operator that takes
  the tensor straight to its part of a weight and channel. The paper applies the two
  operators in turn instead, and at rank n the composed operator has 3^(2n) entries
  per channel, so it is built only on request.
- `get_gram_matrices` gives the exact Gram matrix of each weight.

Each operator is exact; `natto.evaluate` turns one into an array together with the
einsum rule that applies it, and `natto.act` applies it directly.

The dual basis is exact throughout: the mappings, and the duals built from them with
the inverse of their exact Gram matrix. The orthonormal basis rotates the mappings by
the inverse square root of that matrix instead, which is irrational in general, so
its operators carry float coefficients; see `natto.orthonormal`.
"""

import functools
from dataclasses import replace
from fractions import Fraction
from typing import Literal

from natto.gram import get_gram_matrix
from natto.independence import (
    select_independent_mappings_and_gram,
    select_independent_mappings_via_components,
    select_independent_mappings_via_embeddings,
)
from natto.mapping_tensors import Mapping, compose, get_dual_mappings, get_mappings
from natto.orthonormal import OrthonormalOperator, get_orthonormal_operators
from natto.rational import matrix_inverse
from natto.symbolic import Operator, Signature
from natto.symmetry_adaptation import get_symmetry_adapted_mappings

#: The basis of each weight's operators: `dual`, exact, or `orthonormal`, self-dual.
Basis = Literal["dual", "orthonormal"]

#: What to judge the independence of the candidate mappings on; see
#: `natto.independence`.
Selection = Literal["symbolic", "qr", "components", "embeddings"]


def get_reduction(
    n: int, ell: int | None = None, symmetry: str | None = None, basis: Basis = "dual"
) -> dict[tuple[int, int], dict[str, Operator | OrthonormalOperator]]:
    """The extraction and embedding operators of a Cartesian tensor space.

    The operators depend only on the rank and the symmetry, not on any particular
    tensor, so one call serves every tensor of that class.

    Args:
        n: Rank of the Cartesian tensor.
        ell: The one weight to build, or None for every weight.
        symmetry: Intrinsic symmetry of the Cartesian tensor, if any. For example,
            - "ij=ji" is a fully symmetric rank-2 tensor, e.g. the stress tensor;
            - "ij=-ji" is an antisymmetric rank-2 tensor;
            - "ijk=ikj" is a rank-3 tensor with the last two indices symmetric,
              e.g. the piezoelectric tensor;
            - "ijk=ikj=jik" is a fully symmetric rank-3 tensor;
            - "ijkl=jikl=klij" is a rank-4 tensor with both minor symmetry
              (between i and j, and between k and l) and major symmetry (between
              ij and kl), e.g. the elastic tensor.
            The number of distinct letters gives the rank; which letters are used
            does not matter.
        basis: `dual` gives the mappings as embedding operators and their duals as
            extraction operators, exactly. `orthonormal` gives the mappings of Eq. 21
            instead, which are self-dual, so the embedding and extraction operator of
            a channel have the same float terms and differ only in their input.

    Returns:
        A dict mapping `(ell, p)` to the operators of that weight and channel, under
        the keys `embedding` and `extraction`. A weight the symmetry extinguishes has
        no channels.

    Raises:
        ValueError: If `ell` is not between 0 and `n`, `basis` is not recognized, or
            `symmetry` does not have rank `n`.

    References:
        Eq. 18 of [Wen2026] for the extraction, Eq. 19 for the embedding, and
        Eq. 20 for the sum that returns the tensor.
    """
    embeddings = get_embedding_operators(n, ell, symmetry, basis)
    extractions = get_extraction_operators(n, ell, symmetry, basis)
    reduction = {
        key: {"embedding": embeddings[key], "extraction": extractions[key]}
        for key in embeddings
    }

    return reduction


def get_embedding_operators(
    n: int, ell: int | None = None, symmetry: str | None = None, basis: Basis = "dual"
) -> dict[tuple[int, int], Operator | OrthonormalOperator]:
    """The embedding operators, which take an ICT to its part of a Cartesian tensor.

    Each has the ICT indices as its `ict` group, the input, and the tensor indices as
    its `gct` group.

    Args:
        n: Rank of the Cartesian tensor.
        ell: The one weight to build, or None for every weight.
        symmetry: Intrinsic symmetry of the Cartesian tensor, if any, as for
            `get_reduction`.
        basis: `dual` or `orthonormal`, as for `get_reduction`.

    Returns:
        A dict mapping `(ell, p)` to the embedding operator of that weight and channel.

    Raises:
        ValueError: If `ell` is not between 0 and `n`, `basis` is not recognized, or
            `symmetry` does not have rank `n`.

    References:
        Eq. 13 of [Wen2026] for the mappings, used as in Eq. 19, and Eq. 21 for the
        orthonormal basis.
    """
    operators = _get_operators(n, ell, symmetry, basis, extraction=False)

    return operators


def get_extraction_operators(
    n: int, ell: int | None = None, symmetry: str | None = None, basis: Basis = "dual"
) -> dict[tuple[int, int], Operator | OrthonormalOperator]:
    """The extraction operators, which take a Cartesian tensor to its ICTs.

    Each has the ICT indices as its `ict` group and the tensor indices as its `gct`
    group, the input.

    Args:
        n: Rank of the Cartesian tensor.
        ell: The one weight to build, or None for every weight.
        symmetry: Intrinsic symmetry of the Cartesian tensor, if any, as for
            `get_reduction`.
        basis: `dual` or `orthonormal`, as for `get_reduction`.

    Returns:
        A dict mapping `(ell, p)` to the extraction operator of that weight and
        channel.

    Raises:
        ValueError: If `ell` is not between 0 and `n`, `basis` is not recognized, or
            `symmetry` does not have rank `n`.

    References:
        Eq. 16 of [Wen2026] for the duals, used as in Eq. 18, and Eq. 21 for the
        orthonormal basis.
    """
    operators = _get_operators(n, ell, symmetry, basis, extraction=True)

    return operators


def get_composed_operators(
    n: int, ell: int | None = None, symmetry: str | None = None
) -> dict[tuple[int, int], Operator]:
    """The composed operators of a Cartesian tensor space, in the dual basis.

    Each is the embedding operator of one weight and channel composed with its
    extraction operator over the ICT indices: a rank-2n operator that takes a tensor
    straight to its part of that weight and channel. Its `gct` group carries the
    output tensor indices and its `gct_in` group the input ones. The paper applies the
    two operators in turn, Eq. 18 and then Eq. 19, and never forms this one; at rank n
    it has 3^(2n) entries per channel.

    Args:
        n: Rank of the Cartesian tensor.
        ell: The one weight to build, or None for every weight.
        symmetry: Intrinsic symmetry of the Cartesian tensor, if any, as for
            `get_reduction`.

    Returns:
        A dict mapping `(ell, p)` to the composed operator of that weight and channel.

    Raises:
        ValueError: If `ell` is not between 0 and `n`, or `symmetry` does not have
            rank `n`.

    References:
        Eq. 18 and Eq. 19 of [Wen2026], composed. Computed as in A.2 of
        [Wen2026Refactor].
    """
    operators = {}
    for weight in _get_weights(n, ell):
        G, G_tilde, _ = _get_channels(weight, n, symmetry)
        for p, (G_p, G_tilde_p) in enumerate(zip(G, G_tilde), start=1):
            operators[weight, p] = compose(G_p, G_tilde_p)

    return operators


def get_gram_matrices(
    n: int, ell: int | None = None, symmetry: str | None = None
) -> dict[int, list[list[Fraction]]]:
    """The exact Gram matrix of the mappings of each weight.

    Entry (p, q), counted from zero, pairs channel p + 1 with channel q + 1, so a
    matrix belongs to a weight rather than to a channel.

    Args:
        n: Rank of the Cartesian tensor.
        ell: The one weight to build, or None for every weight.
        symmetry: Intrinsic symmetry of the Cartesian tensor, if any, as for
            `get_reduction`.

    Returns:
        A dict mapping each weight that has channels to its Gram matrix.

    Raises:
        ValueError: If `ell` is not between 0 and `n`, or `symmetry` does not have
            rank `n`.

    References:
        Eq. 15 of [Wen2026].
    """
    grams = {}
    for weight in _get_weights(n, ell):
        _, _, gram = _get_channels(weight, n, symmetry)
        if gram:
            grams[weight] = [list(row) for row in gram]

    return grams


def get_independent_mappings(
    ell: int,
    n: int,
    symmetry: str | None = None,
    selection: Selection = "symbolic",
) -> tuple[list[Mapping], list[list[Fraction]]]:
    """The independent mapping tensors of one weight, and their exact Gram matrix.

    This is the first stage of the reduction, and the only one both bases share: it
    enumerates the candidate mappings, keeps an independent subset of them, and -- if the
    tensor has an intrinsic symmetry -- mixes those into the symmetry-adapted mappings.

    Each mapping is a vector over the candidates of its weight and rank; `expand`
    gives its terms.

    The symmetry adaptation is exact whatever `selection` is. Its null space is what
    yields the multiplicity of the weight, and a weight that comes out empty -- as weight
    one does for the third-order elastic tensor -- is a statement rather than a threshold.

    Args:
        ell: Weight of the ICT space.
        n: Rank of the Cartesian tensor.
        symmetry: Intrinsic index symmetry, as index equalities, or None.
        selection: What to judge the independence of the candidate mappings on.
            `symbolic` is the default: the mappings' Gram matrix is contracted
            symbolically and the decision made over the rationals, so it is the
            same on every machine, keeping the earliest independent candidates.
            `qr` is Algorithm 1 of the paper: a rank-revealing QR with column
            pivoting on the evaluated mappings, which keeps the same number but
            may keep a different subset. `components` uses the mappings evaluated
            in full, `embeddings` their action on one probe tensor. The last three
            are numerical, and so decided against a tolerance; see
            `natto.independence`.

    Returns:
        The independent mappings and their exact Gram matrix. Both are empty when there
        is no mapping of this weight, or none the symmetry admits.

    Raises:
        ValueError: If `selection` is not recognized.

    References:
        Eq. 13 of [Wen2026] for the mappings, Algorithm 1 for the `qr` selection, and
        Eq. 27 for the symmetry-adapted mappings.
    """
    # No isotropic n-one mapping exists from a scalar ICT.
    if n == 1 and ell == 0:
        return [], []

    candidates = get_mappings(ell, n)

    if selection == "symbolic":
        independent_indices, gram = select_independent_mappings_and_gram(candidates)
    elif selection == "qr":
        independent_indices = select_independent_mappings_via_components(
            candidates, method="scipy_qr"
        )
        gram = get_gram_matrix([candidates[i] for i in independent_indices])
    elif selection == "components":
        independent_indices = select_independent_mappings_via_components(candidates)
        gram = get_gram_matrix([candidates[i] for i in independent_indices])
    elif selection == "embeddings":
        independent_indices = select_independent_mappings_via_embeddings(candidates)
        gram = get_gram_matrix([candidates[i] for i in independent_indices])
    else:
        raise ValueError(
            f"Unknown selection: {selection}. "
            "Supported are: symbolic, qr, components, embeddings."
        )

    G = [candidates[i] for i in independent_indices]

    # An intrinsic symmetry admits fewer mappings, and mixes them into new ones.
    if symmetry is not None:
        G = get_symmetry_adapted_mappings(G, matrix_inverse(gram), symmetry)
        if not G:
            return [], []
        gram = get_gram_matrix(G)

    return G, gram


@functools.cache
def _get_channels(
    ell: int, n: int, symmetry: str | None
) -> tuple[tuple[Mapping, ...], tuple[Mapping, ...], tuple[tuple[Fraction, ...], ...]]:
    """The mappings of one weight, their duals and their Gram matrix, built once.

    The result is cached for the life of the process, so asking for the embedding and
    then the extraction operators of a class selects its mappings once. It is shared
    between callers, so it is returned as tuples, and the public functions build fresh
    operators and matrices from it.

    Returns:
        The mappings, their duals and their exact Gram matrix; all empty when the
        weight has no channel.
    """
    G, gram = get_independent_mappings(ell, n, symmetry)
    if not G:
        return (), (), ()

    G_tilde = get_dual_mappings(matrix_inverse(gram), G)
    channels = (tuple(G), tuple(G_tilde), tuple(tuple(row) for row in gram))

    return channels


def _get_operators(
    n: int, ell: int | None, symmetry: str | None, basis: Basis, extraction: bool
) -> dict[tuple[int, int], Operator | OrthonormalOperator]:
    """The embedding or extraction operators; see `get_embedding_operators`."""
    if basis not in ("dual", "orthonormal"):
        raise ValueError(f"Unknown basis: {basis}. Supported are: dual, orthonormal.")

    group = "gct" if extraction else "ict"
    operators = {}
    for weight in _get_weights(n, ell):
        G, G_tilde, gram = _get_channels(weight, n, symmetry)
        if not G:
            continue

        if basis == "orthonormal":
            # The self-dual basis rotates the mappings themselves, for both uses.
            signature = _with_input_group(G[0].sector.signature, group)
            gram_matrix = [list(row) for row in gram]
            orthonormal = get_orthonormal_operators(G, gram_matrix, signature)
            for p, operator in enumerate(orthonormal, start=1):
                operators[weight, p] = operator
        else:
            chosen = G_tilde if extraction else G
            for p, mapping in enumerate(chosen, start=1):
                operators[weight, p] = _with_input(mapping.expand(), group)

    return operators


def _get_weights(n: int, ell: int | None) -> range | list[int]:
    """Every weight of rank `n`, or only `ell`.

    Raises:
        ValueError: If `ell` is not between 0 and `n`.
    """
    if ell is None:
        return range(n + 1)
    if not 0 <= ell <= n:
        raise ValueError(f"The weight must be between 0 and {n}, got {ell}")

    return [ell]


def _with_input(operator: Operator, name: str) -> Operator:
    """The same operator, with the group called `name` as its only input."""
    signature = _with_input_group(operator.signature, name)
    terms = [(coefficient, term) for term, coefficient in operator.terms.items()]

    return Operator(signature, terms)


def _with_input_group(signature: Signature, name: str) -> Signature:
    """The same signature, with the group called `name` as its only input."""
    groups = tuple(
        replace(group, input=group.name == name) for group in signature.groups
    )

    return Signature(groups)
