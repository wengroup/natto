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

A mapping tensor is a rank-lowering tensor followed by the natural projector
(Eq. 19), so the candidates of a weight are assembled here out of `lowering` and
`natural_projector`. Which of them are independent, and how the embedding
operators turn into their extraction duals, is settled by the exact Gram matrix
in `gram`. Two further steps have their own modules: the exact null space that
adapts the mappings to an intrinsic symmetry, in `symmetry_adaptation`, and the
numerical rotation to a self-dual basis, in `orthonormal`.
"""

from fractions import Fraction

from natto.algebra import multiply_2, simplify_linear_combination
from natto.evaluate import embed, evaluate_tensors
from natto.gram import get_gram_matrix
from natto.indices import letter_index, shift_index_2
from natto.lowering import get_lowering_rules_even, get_lowering_rules_odd
from natto.natural_projector import get_natural_projector
from natto.orthonormal import orthonormalize_mappings
from natto.qr import find_independent_tensors
from natto.rational import float_matrix, fraction_matrix, matrix_inverse
from natto.symbolic import (
    Epsilon,
    LinearCombination,
    Scalar,
    create_delta_epsilon_tensors,
)
from natto.symmetric_traceless import get_random_natural_tensor
from natto.symmetry_adaptation import get_symmetry_adapted_mappings


def get_reduction(
    rank: int,
    symmetry: str = None,
    basis: str = "dual",
    numerical: bool = True,
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

    Returns:
        The embedding, extraction and decomposition operators keyed by weight,
        each with the einsum rule that applies it. A weight the symmetry
        extinguishes is absent rather than empty. The Gram matrix comes with its
        inverse in the dual basis and with its inverse square root in the
        orthonormal one.

    Raises:
        ValueError: If `basis` is neither `dual` nor `orthonormal`.
    """
    if basis not in ("dual", "orthonormal"):
        raise ValueError(f"Unknown basis: {basis}. Supported are: dual, orthonormal.")

    out = {}
    for weight in range(rank + 1):
        G, G_tilde, S, gram, gram_inverse = get_reduction_of_weight(
            weight, rank, symmetry
        )

        # No natural tensor of this weight
        if len(G) == 0:
            continue

        if basis == "orthonormal":
            out[weight] = _orthonormal_entries(weight, rank, G)
        else:
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


def _orthonormal_entries(weight: int, rank: int, G: list[LinearCombination]) -> dict:
    """Pack one weight's operators in the self-dual basis of Eq. (26).

    One array both extracts and embeds, so it appears under both keys and only
    the einsum rule tells them apart. There is no symbolic form: the inverse
    square root of a rational Gram matrix is generally irrational.
    """
    _, gram, gram_inverse_sqrt, G_hat = orthonormalize_mappings(G, weight, rank)

    lower = letter_index(weight)
    upper = letter_index(rank, upper_case=True)
    upper2 = letter_index(rank, start=rank, upper_case=True)

    rules = {
        "embedding": f"{upper}{lower},...{lower}->...{upper}",
        "extraction": f"{upper}{lower},...{upper}->...{lower}",
        # the decomposition is not a single array here; applying it means
        # extracting and embedding back through the same operator
        "decomposition": f"{upper}{lower},{upper2}{lower},...{upper2}->...{upper}",
    }

    # One array per channel, shared between the keys rather than copied into
    # each: that it is the same operator is the point of this basis.
    operators = list(G_hat)

    entries = {"gram": gram, "gram_inverse_sqrt": gram_inverse_sqrt}
    for key, rule in rules.items():
        entries[key] = [
            {"symbolic": None, "rule": rule, "numerical": operator}
            for operator in operators
        ]

    return entries


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
    S = get_decomposition_operators(G, G_tilde, rank)

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
        candidates = get_mappings_even(weight, rank)
    else:
        candidates = get_mappings_odd(weight, rank)

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


def get_mappings_even(j: int, n: int) -> list[LinearCombination]:
    r"""
    Mapping operator G to map minimal rank tensor subspaces j onto the space n.

    G(n|j)^q = E_j \otimes^{n-j} f_{n-j}^q.

    This is for even n-j.

    Reference: Eq. 2.4 of [AG82].

    Args:
        j: the minimal tensor subspace
        n: the space to map to

    Returns:
        A list of Tensors objects, each corresponding to a q in f_{n-j}^q.
    """

    assert (n - j) % 2 == 0, f"n-j must be even, got n={n}, j={j}"

    E_s_letters, delta_rules = get_lowering_rules_even(j, n)

    all_G = []
    for si, rule in zip(E_s_letters, delta_rules):
        E_j = get_natural_projector(j, s_letters=si)
        f_q = create_delta_epsilon_tensors(rule)
        G = multiply_2(E_j, f_q)
        all_G.append(G)

    return all_G


def get_mappings_odd(j: int, n: int) -> list[LinearCombination]:
    r"""
    Mapping operator G to map minimal rank tensor subspaces j onto the space n.


    G(n|j)^q = E_j \otimes^{n-j} f_{n-j}^q.

    This is for odd n-j.

    Reference: Eq. 2.5 of [AG82].

    Args:
        j: the minimal tensor subspace
        n: the space to map to

    Returns:
        A list of Tensors objects, each corresponding to a q in f_{n-j}^q.
    """
    assert (n - j) % 2 == 1, f"n-j must be odd, got n={n}, j={j}"

    E_s_letters, f_epsilon_rules, f_delta_rules = get_lowering_rules_odd(j, n)

    all_G = []
    for si, e_rule, d_rule in zip(E_s_letters, f_epsilon_rules, f_delta_rules):
        E_j = get_natural_projector(j, s_letters=si)
        f_q_epsilon = Epsilon(e_rule)
        f_q_delta = create_delta_epsilon_tensors(d_rule)
        G = multiply_2(E_j, f_q_epsilon, f_q_delta)
        all_G.append(G)

    return all_G


def get_extraction_operators(
    gram_inverse: list[list[Fraction]], embedding: list[LinearCombination]
) -> list[LinearCombination]:
    r"""Build the extraction operators dual to a set of embedding operators.

    The paper's Eq. (22),

    $$
    \widetilde{\mathbf{G}}^p_{(\ell|n)}
        = \sum_q (\mathbf{g}^{-1})_{pq} \mathbf{G}^q_{(\ell|n)}
    $$

    Contracted with a Cartesian tensor, each returns the natural tensor of its
    weight and channel.

    Args:
        gram_inverse: Exact inverse of the embedding operators' Gram matrix.
        embedding: The embedding operators, in the order the matrix indexes.

    Returns:
        One extraction operator per row of `gram_inverse`.
    """
    extraction = []
    for row in gram_inverse:
        terms = []
        for c, G in zip(row, embedding):
            if c:
                terms.extend(multiply_2(Scalar(c), G))
        extraction.append(LinearCombination(*terms))

    return extraction


def get_decomposition_operators(
    G: list[LinearCombination], G_tilde: list[LinearCombination], n: int
) -> list[LinearCombination]:
    r"""
    Get the decomposition operators of a mapping and its dual.

    S = G \odot^j G~

    Args:
        G: mapping tensors
        G_tilde: the duals, in the order of the mappings they correspond to.
        n: rank of the Cartesian tensor.

    Returns:
        S: one decomposition operator per channel
    """
    S = []
    for G_i, dual_i in zip(G, G_tilde):
        # Shift upper letters of the dual to distinguish them from those of G
        dual_i = shift_index_2(dual_i, n, letter_index(24, upper_case=True))

        S_i = multiply_2(G_i, dual_i)
        S_i = simplify_linear_combination(S_i)

        S.append(S_i)

    return S
