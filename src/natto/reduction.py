r"""The reduction of a Cartesian tensor into irreducible Cartesian tensors.

This is the public entry point. `get_reduction` returns, for each weight $\ell$ and
channel $p$ of a rank-$n$ tensor, the operators that move between the two spaces.
Writing $\mathbf{G}$ for the embedding operator, $\widetilde{\mathbf{G}}$ for the
extraction operator dual to it, and $\mathbf{S}$ for their composition:

$$
\mathbf{X}_\ell^p = \widetilde{\mathbf{G}}^p_{(\ell|n)} \odot^n \mathbf{T}_n
\qquad\text{(Eq. 24)}
$$

$$
\mathbf{S}_n^{\ell,p} = \mathbf{G}^p_{(\ell|n)} \odot^\ell \mathbf{X}_\ell^p
\qquad\text{(Eq. 25)}
$$

so that $\mathbf{S} = \mathbf{G} \odot^\ell \widetilde{\mathbf{G}}$ takes
$\mathbf{T}$ straight to its weight-$\ell$, channel-$p$ part without forming
$\mathbf{X}$ on the way. Summing that part over every weight and channel returns
$\mathbf{T}$ (Eq. 30). They are published under the keys `embedding`, `extraction`
and `decomposition`; see docs/notation.md for the correspondence with the paper.

The work is staged, because the two bases need different amounts of it.
`get_independent_mappings` is common to both: it keeps an independent subset of the
candidates built in `mapping_tensors` and adapts them to any intrinsic symmetry.
From there the dual basis calls `get_dual_pair`, which inverts the exact Gram
matrix, while the orthonormal basis goes to `orthonormal` instead and builds no
duals at all -- its operator is its own dual, and is irrational, so the exact work
would only be discarded.
"""

from fractions import Fraction
from typing import Literal

from natto.algebra import simplify_linear_combination
from natto.evaluate import evaluate_tensors
from natto.gram import get_gram_matrix
from natto.independence import (
    select_independent_mappings_and_gram,
    select_independent_mappings_via_components,
    select_independent_mappings_via_embeddings,
)
from natto.indices import letter_index
from natto.mapping_tensors import (
    get_decomposition_operators,
    get_extraction_operators,
    get_mappings_even,
    get_mappings_odd,
)
from natto.orthonormal import get_orthonormal_entries
from natto.rational import float_matrix, fraction_matrix, matrix_inverse
from natto.symbolic import LinearCombination
from natto.symmetry_adaptation import get_symmetry_adapted_mappings

#: How to decide which candidate mappings are independent; see `natto.qr`.
Selection = Literal["symbolic", "components", "embeddings"]


def get_reduction(
    rank: int,
    symmetry: str = None,
    basis: str = "dual",
    numerical: bool = True,
    selection: Selection = "symbolic",
) -> dict:
    """Reduce a Cartesian tensor space into its irreducible parts.

    The operators depend only on the rank and the symmetry, not on any
    particular tensor, so one call serves every tensor of that class.

    Args:
        rank: Rank of the Cartesian tensor.
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
        basis: `dual` returns the mappings and the duals that extract through
            them, exactly, with rational coefficients. `orthonormal` returns
            the mappings of Eq. (26) instead, rotated by the inverse square root
            of their Gram matrix. Those are self-dual, so `embedding` and
            `extraction` carry the same array and differ only in the einsum rule
            applying it; the rotation is irrational, so they have no symbolic
            form.
        numerical: Whether to evaluate the operators as well as building them
            symbolically. Ignored for `basis="orthonormal"`, which is numerical
            by construction.
        selection: What to judge the independence of the candidate mappings on.
            `symbolic` is the default: the mappings' Gram matrix is contracted
            symbolically and the decision made over the rationals, so it is the
            same on every machine. `components` uses the mappings evaluated in
            full, `embeddings` their action on one probe tensor -- both
            numerically, and so both against a tolerance; see
            `natto.independence`. All three agree on every sector tested, but
            only `symbolic` is free of a tolerance, which matters because the
            choice fixes which duals are canonical.

    Returns:
        The embedding, extraction and decomposition operators keyed by weight,
        each with the einsum rule that applies it. A weight the symmetry
        extinguishes is absent rather than empty. The Gram matrix comes with its
        inverse in the dual basis and with its inverse square root in the
        orthonormal one.

    Raises:
        ValueError: If `basis` is neither `dual` nor `orthonormal`, or if
            `selection` is not one of `symbolic`, `components` and `embeddings`.
    """
    if basis not in ("dual", "orthonormal"):
        raise ValueError(f"Unknown basis: {basis}. Supported are: dual, orthonormal.")

    out = {}
    for weight in range(rank + 1):
        G, gram = get_independent_mappings(weight, rank, symmetry, selection)

        # No ICT of this weight, or none that the symmetry admits
        if not G:
            continue

        if basis == "orthonormal":
            # The self-dual basis needs no duals, so none are built.
            out[weight] = get_orthonormal_entries(weight, rank, G)
        else:
            G_simplified, G_tilde, S, gram_inverse = get_dual_pair(G, gram, rank)
            out[weight] = assemble_operator_entries(
                weight,
                rank,
                G_simplified,
                G_tilde,
                S,
                gram,
                gram_inverse,
                numerical,
                include_gram=True,
                include_gram_inverse=True,
            )

    return out


def get_independent_mappings(
    weight: int,
    rank: int,
    symmetry: str = None,
    selection: Selection = "symbolic",
) -> tuple[list[LinearCombination], list[list[Fraction]]]:
    r"""The independent mapping tensors of one weight, and their exact Gram matrix.

    This is the first stage of the reduction, and the only one both bases share: it
    enumerates the candidate mappings, keeps an independent subset of them, and -- if
    the tensor has an intrinsic symmetry -- mixes those into the symmetry-adapted
    $\mathbf{Q}^p$ of Eq. (36).

    The mappings are returned unsimplified. Combining like terms would reorder the
    index patterns that `gram` contracts, so the exact Gram entries must be taken
    from this form; `get_dual_pair` simplifies once it is done with them.

    The symmetry adaptation is exact whatever `selection` is. Its null space is what
    yields the multiplicity of the weight, and a weight that comes out empty -- as
    weight one does for the third-order elastic tensor -- is a statement rather than
    a threshold.

    Args:
        weight: Weight of the ICT space.
        rank: Rank of the Cartesian tensor.
        symmetry: Intrinsic index symmetry, as index equalities, or None.
        selection: How to decide which candidates are independent; see
            `natto.qr`. The default is exact.

    Returns:
        The independent mappings and their exact Gram matrix. Both are empty when
        there is no mapping of this weight, or none the symmetry admits.
    """
    # No isotropic rank-one mapping exists from a scalar ICT.
    if rank == 1 and weight == 0:
        return [], []

    if (rank - weight) % 2 == 0:
        candidates = get_mappings_even(weight, rank)
    else:
        candidates = get_mappings_odd(weight, rank)

    if selection == "symbolic":
        independent_indices, gram = select_independent_mappings_and_gram(
            weight, rank, candidates
        )
    elif selection == "components":
        independent_indices = select_independent_mappings_via_components(
            weight, rank, candidates
        )
        gram = get_gram_matrix(
            weight, rank, [candidates[i] for i in independent_indices]
        )
    elif selection == "embeddings":
        independent_indices = select_independent_mappings_via_embeddings(
            weight, rank, candidates
        )
        gram = get_gram_matrix(
            weight, rank, [candidates[i] for i in independent_indices]
        )
    else:
        raise ValueError(
            f"Unknown selection: {selection}. Supported are: symbolic, components, embeddings."
        )

    G = [candidates[i] for i in independent_indices]

    # An intrinsic symmetry admits fewer mappings, and mixes them into new ones.
    if symmetry is not None:
        G = get_symmetry_adapted_mappings(
            weight, rank, G, matrix_inverse(gram), symmetry
        )
        if not G:
            return [], []
        gram = get_gram_matrix(weight, rank, G)

    return G, gram


def get_dual_pair(
    G: list[LinearCombination], gram: list[list[Fraction]], rank: int
) -> tuple[
    list[LinearCombination],
    list[LinearCombination],
    list[LinearCombination],
    list[list[Fraction]],
]:
    r"""Complete the mappings into an extraction-and-embedding pair, exactly.

    The duals of Eq. (22) are $\widetilde{\mathbf{G}}^p = \sum_q (g^{-1})_{pq}
    \mathbf{G}^q$, so inverting the Gram matrix over the rationals is all that is
    needed; their composition gives the decomposition operators of Eq. (25).

    Args:
        G: The independent mappings, unsimplified, from `get_independent_mappings`.
        gram: Their exact Gram matrix.
        rank: Rank of the Cartesian tensor.

    Returns:
        The mappings, their duals and the decomposition operators, all simplified,
        and the exact inverse of the Gram matrix.
    """
    gram_inverse = matrix_inverse(gram)
    G_tilde_raw = get_extraction_operators(gram_inverse, G)

    G_simplified = [simplify_linear_combination(G_p) for G_p in G]
    G_tilde = [simplify_linear_combination(G_p) for G_p in G_tilde_raw]
    S = get_decomposition_operators(G_simplified, G_tilde, rank)

    return G_simplified, G_tilde, S, gram_inverse


def assemble_operator_entries(
    weight: int,
    rank: int,
    G: list[LinearCombination],
    G_tilde: list[LinearCombination],
    S: list[LinearCombination],
    gram: list[list[Fraction]],
    gram_inverse: list[list[Fraction]],
    numerical: bool = True,
    include_gram: bool = True,
    include_gram_inverse: bool = True,
) -> dict:
    """Pack the operators of one weight into the form the package publishes.

    Each operator is paired with the einsum rule that applies it, and optionally
    with its evaluated array. The rules carry a leading ellipsis, so an operator
    applies to a batch of tensors as readily as to one.

    Args:
        weight: Weight of the natural-tensor space.
        rank: Rank of the Cartesian tensor.
        G: Embedding operators, one per channel.
        G_tilde: Extraction operators dual to them.
        S: Their composition, one per channel.
        gram: Gram matrix of the embedding operators.
        gram_inverse: Its exact inverse.
        numerical: Whether to evaluate each operator as well as recording it
            symbolically.
        include_gram: Whether to report the Gram matrix.
        include_gram_inverse: Whether to report its inverse.

    Returns:
        The operators under the keys `embedding`, `extraction` and
        `decomposition`, plus `gram` and `gram_inverse` when asked for.
    """
    out_weight = {"embedding": [], "extraction": [], "decomposition": []}

    if include_gram:
        out_weight["gram"] = {
            "symbolic": fraction_matrix(gram),
            "numerical": float_matrix(gram),
        }

    if include_gram_inverse:
        out_weight["gram_inverse"] = {
            "symbolic": fraction_matrix(gram_inverse),
            "numerical": float_matrix(gram_inverse),
        }

    lower = letter_index(weight)
    upper = letter_index(rank, upper_case=True)
    upper2 = letter_index(rank, start=rank, upper_case=True)

    for G_p, G_tilde_p, S_p in zip(G, G_tilde, S):
        out_weight["embedding"].append(
            {"symbolic": str(G_p), "rule": f"{upper}{lower},...{lower}->...{upper}"}
        )
        if numerical:
            out_weight["embedding"][-1]["numerical"] = evaluate_tensors(
                G_p, mode="embedding"
            )

        out_weight["extraction"].append(
            {
                "symbolic": str(G_tilde_p),
                "rule": f"{lower}{upper},...{upper}->...{lower}",
            }
        )
        if numerical:
            out_weight["extraction"][-1]["numerical"] = evaluate_tensors(
                G_tilde_p, mode="extraction"
            )

        out_weight["decomposition"].append(
            {"symbolic": str(S_p), "rule": f"{upper}{upper2},...{upper2}->...{upper}"}
        )
        if numerical:
            out_weight["decomposition"][-1]["numerical"] = evaluate_tensors(
                S_p, mode="decomposition"
            )

    return out_weight
