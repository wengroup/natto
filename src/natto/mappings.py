r"""
Symbolic and numerical mapping operators between a Cartesian tensor and its
irreducible parts.

Three operators per weight $\ell$ and channel $p$, built from the Kronecker
delta and the Levi-Civita symbol alone. Writing $\mathbf{G}$ for the embedding
operator, $\widetilde{\mathbf{G}}$ for the extraction operator dual to it, and
$\mathbf{S}$ for their composition:

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
$\mathbf{X}$ on the way. Summing that part over every weight and channel
returns $\mathbf{T}$ (Eq. 30).

They are returned under the keys `embedding`, `extraction` and `decomposition`;
see docs/notation.md for the correspondence with the paper throughout.

Two steps of the construction live in their own modules: the exact null space
that adapts the mappings to an intrinsic symmetry, in `symmetry_adapted`, and
the numerical rotation to a self-dual basis, in `orthonormal`.
"""

from fractions import Fraction

from natto.algebra import simplify_linear_combination
from natto.evaluate import embed, evaluate_tensors
from natto.matrix import float_matrix, fraction_matrix, matrix_inverse
from natto.operators import (
    get_extraction_operators,
    get_G_even,
    get_G_odd,
    get_gram_matrix,
    get_S,
)
from natto.qr import find_independent_tensors
from natto.symbolic import LinearCombination
from natto.symmetrize import get_random_natural_tensor
from natto.symmetry_adapted import get_symmetry_adapted_mappings
from natto.utils import letter_index


def get_reduction(rank: int, symmetry: str = None, numerical: bool = True) -> dict:
    """Reduce a Cartesian tensor space into its irreducible parts.

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
        numerical: Whether to evaluate the operators as well as building them
            symbolically.

    Returns:
        The embedding, extraction and decomposition operators keyed by weight,
        with the Gram matrix and its inverse. A weight the symmetry extinguishes
        is absent rather than empty.
    """
    out = {}
    for weight in range(rank + 1):
        G, G_tilde, S, gram, gram_inverse = get_reduction_of_weight(
            weight, rank, symmetry
        )

        # No natural tensor of this weight
        if len(G) == 0:
            continue

        out[weight] = assemble_operator_entries(
            weight,
            rank,
            G,
            G_tilde,
            S,
            gram,
            gram_inverse,
            numerical,
            include_gram=True,
            include_gram_inverse=True,
        )

    return out


def get_reduction_of_weight(
    weight: int, rank: int, symmetry: str = None
) -> tuple[
    list[LinearCombination],
    list[LinearCombination],
    list[LinearCombination],
    list[list[Fraction]],
    list[list[Fraction]],
]:
    """Build the operators of one weight, with or without an intrinsic symmetry.

    Args:
        weight: Weight of the natural-tensor space.
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

    Returns:
        The embedding operators, one per channel; the extraction operators dual
        to them; their composition; the Gram matrix; and its inverse. All five
        are empty when the symmetry leaves no mapping of this weight.
    """
    # The mappings independent for a Cartesian tensor with no assumed symmetry
    ind_G, ind_G_tilde, gram, gram_inverse = get_dual_pair_of_weight(weight, rank)

    # An intrinsic symmetry admits fewer of them, and mixes them into new ones.
    if symmetry is not None:
        ind_G = get_symmetry_adapted_mappings(
            weight, rank, ind_G, gram_inverse, symmetry
        )
        if not ind_G:
            return [], [], [], [], []

        gram = get_gram_matrix(weight, rank, ind_G)
        gram_inverse = matrix_inverse(gram)
        ind_G_tilde = get_extraction_operators(gram_inverse, ind_G)

    G = [simplify_linear_combination(G_p) for G_p in ind_G]
    G_tilde = [simplify_linear_combination(G_tilde_p) for G_tilde_p in ind_G_tilde]
    S = get_S(G, G_tilde, rank)

    return G, G_tilde, S, gram, gram_inverse


def get_dual_pair_of_weight(
    weight: int, rank: int
) -> tuple[
    list[LinearCombination],
    list[LinearCombination],
    list[list[Fraction]],
    list[list[Fraction]],
]:
    """Build the mappings of one weight and the duals extracting through them.

    The independence here is that of a Cartesian tensor with no assumed
    symmetry. A tensor with an intrinsic symmetry admits fewer, which
    `get_symmetry_adapted_mappings` selects from these.

    Args:
        weight: Weight of the natural-tensor space.
        rank: Rank of the Cartesian tensor.

    Returns:
        The independent mappings, one per channel; the duals corresponding to
        them; their Gram matrix; and its exact inverse.
    """
    # No isotropic rank-one mapping exists from a scalar natural tensor.
    if rank == 1 and weight == 0:
        return [], [], [], []

    if (rank - weight) % 2 == 0:
        candidates = get_G_even(weight, rank)
    else:
        candidates = get_G_odd(weight, rank)

    # WARNING, the candidates must not be simplified here: `get_gram_matrix` below
    # is set up to work with the mappings in their original form.

    # Embed one random natural tensor through each candidate; the candidates are
    # independent exactly when the tensors they produce are.
    X = get_random_natural_tensor(weight)
    embedded = [embed(candidate, X) for candidate in candidates]
    _, independent_indices = find_independent_tensors(embedded)
    ind_G = [candidates[i] for i in independent_indices]

    gram = get_gram_matrix(weight, rank, ind_G)
    gram_inverse = matrix_inverse(gram)
    ind_G_tilde = get_extraction_operators(gram_inverse, ind_G)

    return ind_G, ind_G_tilde, gram, gram_inverse


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
