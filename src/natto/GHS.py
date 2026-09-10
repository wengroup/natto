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
"""

from fractions import Fraction
from pprint import pprint

import torch
from torch import Tensor

from natto.EGH import (
    get_extraction_operators,
    get_G_even,
    get_G_odd,
    get_gram_entry,
    get_gram_matrix,
    get_S,
    relabel_indices_2,
)
from natto.evaluate import embed, evaluate_tensors
from natto.matrix import (
    float_matrix,
    fraction_matrix,
    matrix_inverse,
    matrix_multiply,
    matrix_null_space,
)
from natto.ops import simplify_linear_combination
from natto.qr import find_independent_tensors
from natto.sym import parse_symmetry_generators
from natto.symbolic import LinearCombination
from natto.symmetrize import get_random_natural_tensor
from natto.utils import letter_index


def get_G_H_S(n: int, symmetry: str = None, numerical: bool = True) -> dict:
    """
    Get all the G, H, S tensors of dimension n.

    Args:
        n: dim of the space T is in
        symmetry: symmetry of the tensor in space n, if any. For example,
            - "ij=ji" means that the target is a fully symmetric rank-2 tensor (e.g.
                stress tensor);
            - "ij=-ji" means that the target is an antisymmetric rank-2 tensor;
            - "ijk=ikj" means that the target is a rank-3 tensor with the last two
                indices symmetric (e.g. piezoelectric tensor);
            - "ijk=ikj=jik" means that the target is a fully symmetric rank-3 tensor;
            - "ijkl=jikl=klij" means that the target is a rank-4 tensor with both minor
                symmetry (between i and j, and between k and l) and major symmetry (
                between ij and kl). For example, the elastic tensor has this symmetry;
            The number of unique letters gives the rank of the tensor (what letters to
            use does not matter).
        numerical: whether to return numerical values of G, H, S.

    Returns:
        The embedding, extraction and decomposition operators, with the Gram matrix
        and its inverse.
    """
    out = {}
    for j in range(n + 1):
        G, H, S, g, h = get_G_H_S_of_j(j, n, symmetry)

        # No natural tensor of this rank
        if len(G) == 0:
            continue

        # Get rules and numerical values
        out_j = get_G_H_S_rules_and_values(
            j, n, G, H, S, g, h, numerical, include_g=True, include_h=True
        )

        out[j] = out_j

    return out


def orthonormalize_mappings(
    mappings: list[LinearCombination],
    weight: int,
    rank: int,
    dtype: torch.dtype = torch.float64,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    r"""Orthonormalize mapping tensors with their Cartesian Gram matrix.

    The Gram matrix is evaluated as
    $$
    g_{pq} = \frac{\mathbf{G}^p \odot^{n+\ell} \mathbf{G}^q}{2\ell + 1}
    $$
    Its unique symmetric positive-definite inverse square root transforms the input
    mappings into an orthonormal set.

    Args:
        mappings: Independent symbolic mappings from weight ``weight`` to Cartesian
            rank ``rank``.
        weight: Weight of the natural-tensor space.
        rank: Rank of the Cartesian tensor space.
        dtype: Floating-point dtype used for evaluation and eigendecomposition.

    Returns:
        Numerical input mappings, their Gram matrix, its symmetric inverse square
        root, and the orthonormal numerical mappings.

    Raises:
        ValueError: If ``mappings`` is empty or its Gram matrix is not symmetric
            positive definite.
    """
    if not mappings:
        raise ValueError("At least one mapping tensor is required")

    numerical = torch.stack(
        [
            evaluate_tensors(
                simplify_linear_combination(mapping), mode="embedding", dtype=dtype
            )
            for mapping in mappings
        ]
    )
    if numerical.ndim != rank + weight + 1:
        raise ValueError("Mapping tensor ranks do not match rank and weight")
    flattened = numerical.reshape(len(mappings), -1)
    gram = flattened @ flattened.T / (2 * weight + 1)
    gram_inverse_sqrt = _symmetric_inverse_square_root(gram)
    orthonormal = torch.einsum("pq,q...->p...", gram_inverse_sqrt, numerical)

    return numerical, gram, gram_inverse_sqrt, orthonormal


def get_orthonormal_G(n: int, dtype: torch.dtype = torch.float64) -> dict:
    r"""Get orthonormal mapping tensors for an unrestricted Cartesian tensor.

    Args:
        n: Rank of the Cartesian tensor.
        dtype: Floating-point dtype used for evaluation and eigendecomposition.

    Returns:
        A dictionary keyed by weight, with the exact embedding operators, their
        Gram matrix and its symmetric inverse square root, and the orthonormal
        operators built from them.
    """
    out = {}
    for j in range(n + 1):
        G, _, _, _ = get_G_H_of_j(j, n)
        if not G:
            continue

        G_numerical, g, g_inverse_sqrt, G_hat = orthonormalize_mappings(G, j, n, dtype)
        extraction_rule, embedding_rule = _orthonormal_mapping_rules(j, n)
        out[j] = {
            "embedding": [
                {"symbolic": str(G_p), "numerical": G_p_numerical}
                for G_p, G_p_numerical in zip(G, G_numerical)
            ],
            "gram": g,
            "gram_inverse_sqrt": g_inverse_sqrt,
            "orthonormal": [
                {
                    "numerical": G_hat_p,
                    "extraction_rule": extraction_rule,
                    "embedding_rule": embedding_rule,
                }
                for G_hat_p in G_hat
            ],
        }

    return out


def get_orthonormal_Q(
    n: int, symmetry: str = None, dtype: torch.dtype = torch.float64
) -> dict:
    r"""Get symmetry-adapted orthonormal mapping tensors.

    For each weight ``j``, the exact symmetry-adapted embedding tensors ``Q`` are
    obtained from :func:`get_G_H_S_of_j`. Their Gram matrix is evaluated as

    $$
    g_{pq} = \frac{\mathbf{Q}^p \odot^{n+\ell} \mathbf{Q}^q}{2\ell + 1}
    $$

    The returned $\widehat{\mathbf{Q}} = \mathbf{g}^{-1/2}\mathbf{Q}$ are orthonormal and can be used
    for both extraction from and embedding into the target symmetry class.

    Args:
        n: Rank of the Cartesian tensor.
        symmetry: Internal index symmetry of the Cartesian tensor. See
            :func:`get_G_H_S` for examples. If ``None``, the symmetry-free mapping
            tensors are orthonormalized.
        dtype: Floating-point dtype used for the eigendecomposition and returned
            numerical tensors.

    Returns:
        A dictionary keyed by weight, with the exact symmetry-adapted operators,
        their Gram matrix and its symmetric inverse square root, and the
        orthonormal operators built from them. Each orthonormal entry carries an
        einsum rule for extraction and one for embedding, since the same tensor
        performs both.
    """
    out = {}
    for j in range(n + 1):
        Q, _, _, _, _ = get_G_H_S_of_j(j, n, symmetry)
        if len(Q) == 0:
            continue

        Q_numerical, g, g_inverse_sqrt, Q_hat = orthonormalize_mappings(Q, j, n, dtype)

        extraction_rule, embedding_rule = _orthonormal_mapping_rules(j, n)
        out[j] = {
            "embedding": [
                {"symbolic": str(Q_p), "numerical": Q_p_numerical}
                for Q_p, Q_p_numerical in zip(Q, Q_numerical)
            ],
            "gram": g,
            "gram_inverse_sqrt": g_inverse_sqrt,
            "orthonormal": [
                {
                    "numerical": Q_hat_p,
                    "extraction_rule": extraction_rule,
                    "embedding_rule": embedding_rule,
                }
                for Q_hat_p in Q_hat
            ],
        }

    return out


def get_G_H_S_of_j(
    j: int, n: int, symmetry: str = None
) -> tuple[
    list[LinearCombination],
    list[LinearCombination],
    list[LinearCombination],
    list[list[Fraction]],
    list[list[Fraction]],
]:
    """
    Get the G, H, S tensors for a given weight j and rank n.

    This can deal with / without symmetry.

    Args:
        j: weight
        n: dim of the space T is in
        symmetry: symmetry of the tensor in space n, if any. For example,
            - "ij=ji" means that the target is a fully symmetric rank-2 tensor (e.g.
                stress tensor);
            - "ij=-ji" means that the target is an antisymmetric rank-2 tensor;
            - "ijk=ikj" means that the target is a rank-3 tensor with the last two
                indices symmetric (e.g. piezoelectric tensor);
            - "ijk=ikj=jik" means that the target is a fully symmetric rank-3 tensor;
            - "ijkl=jikl=klij" means that the target is a rank-4 tensor with both minor
                symmetry (between i and j, and between k and l) and major symmetry (
                between ij and kl). For example, the elastic tensor has this symmetry;
            The number of unique letters gives the rank of the tensor (what letters to
            use does not matter).

    Returns:
        G: independent embedding operators, one per channel p
        H: H corresponding to G
        S: S corresponding to G and H
        g: Gram matrix
        h: Inverse Gram matrix
    """
    # Get independent G and H tensors for a general tensor
    ind_G, ind_H, g, h = get_G_H_of_j(j, n)

    # Further reduce the mappings for tensors with internal symmetry.
    if symmetry is not None:
        ind_G = _get_symmetry_adapted_mappings(j, n, ind_G, h, symmetry)
        if not ind_G:
            return [], [], [], [], []

        g = get_gram_matrix(j, n, ind_G)
        h = matrix_inverse(g)
        ind_H = get_extraction_operators(h, ind_G)

    # Get S tensors
    G = [simplify_linear_combination(G) for G in ind_G]
    H = [simplify_linear_combination(H) for H in ind_H]
    S = get_S(G, H, n)

    return G, H, S, g, h


def get_G_H_of_j(
    j: int, n: int
) -> tuple[
    list[LinearCombination],
    list[LinearCombination],
    list[list[Fraction]],
    list[list[Fraction]],
]:
    """
    Get the independent G and H tensors for a given weight j and rank n.

    Note, here, the independence of G and H are for a general tensor T. For tensors with
    certain symmetry (e.g. tensor that is the product of two natural tensors),
    further processing is needed to get the independent G and H tensors.

    Args:
        j: weight of the natural tensor X
        n: rank of the ordinary tensor T

    Returns:
        G: independent G tensors for ordinary tensor
        H: independent H tensors for ordinary tensor, corresponding to G
        g: Gram matrix
        h: Inverse Gram matrix
    """
    # No isotropic rank-one mapping exists from a scalar natural tensor.
    if n == 1 and j == 0:
        return [], [], [], []

    # create G mapping operator
    if (n - j) % 2 == 0:
        all_G = get_G_even(j, n)
    else:
        all_G = get_G_odd(j, n)

    # WARNING, should not simplify G using the below function, as get_gram_matrix() below
    # is set up to work with the original G tensors.
    # all_G = [simplify_linear_combination(g) for g in all_G]

    # Get numerical S tensors, embedding a random natura tensor X in space j to space n
    X = get_random_natural_tensor(j)
    all_num_S = [embed(G, X) for G in all_G]

    # Get linearly independent S tensors
    _, independent_indices = find_independent_tensors(all_num_S)

    # Get linearly independent G tensors
    ind_G = [all_G[i] for i in independent_indices]

    # Gram matrix of the independent embedding operators
    g = get_gram_matrix(j, n, ind_G)

    # and its exact inverse
    h = matrix_inverse(g)

    # Get H tensors, corresponding to independent G
    ind_H = get_extraction_operators(h, ind_G)

    return ind_G, ind_H, g, h


def get_G_H_S_rules_and_values(
    j: int,
    n: int,
    G: list[LinearCombination],
    H: list[LinearCombination],
    S: list[LinearCombination],
    g: list[list[Fraction]],
    h: list[list[Fraction]],
    numerical: bool = True,
    include_g: bool = True,
    include_h: bool = True,
):
    """
    Get the numerical values of G, H, S tensors and the rules for performing tensor
    products.

    Args:
        j:
        n:
        G:
        H:
        S:
        g:
        h:
        numerical:
        include_g:
        include_h:

    Returns:

    """

    out_j = {"embedding": [], "extraction": [], "decomposition": []}

    if include_g:
        out_j["gram"] = {"symbolic": fraction_matrix(g), "numerical": float_matrix(g)}

    if include_h:
        out_j["gram_inverse"] = {
            "symbolic": fraction_matrix(h),
            "numerical": float_matrix(h),
        }

    for G_p, G_tilde_p, S_p in zip(G, H, S):
        lower = letter_index(j)
        upper = letter_index(n, upper_case=True)
        upper2 = letter_index(n, start=n, upper_case=True)

        out_j["embedding"].append(
            {
                "symbolic": str(G_p),
                "rule": (f"{upper}{lower},...{lower}->...{upper}"),
            },
        )
        if numerical:
            out_j["embedding"][-1]["numerical"] = evaluate_tensors(
                G_p, mode="embedding"
            )

        out_j["extraction"].append(
            {
                "symbolic": str(G_tilde_p),
                "rule": f"{lower}{upper},...{upper}->...{lower}",
            }
        )
        if numerical:
            out_j["extraction"][-1]["numerical"] = evaluate_tensors(
                G_tilde_p, mode="extraction"
            )

        out_j["decomposition"].append(
            {
                "symbolic": str(S_p),
                "rule": f"{upper}{upper2},...{upper2}->...{upper}",
            }
        )
        if numerical:
            out_j["decomposition"][-1]["numerical"] = evaluate_tensors(
                S_p, mode="decomposition"
            )

    return out_j


def _symmetric_inverse_square_root(
    matrix: Tensor, rtol: float = 1e-10, atol: float = 1e-12
) -> Tensor:
    """Compute the symmetric inverse square root of a positive-definite matrix."""
    if not torch.allclose(matrix, matrix.T, rtol=rtol, atol=atol):
        raise ValueError("Gram matrix must be symmetric")

    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    threshold = atol + rtol * torch.max(torch.abs(eigenvalues))
    if torch.any(eigenvalues <= threshold):
        raise ValueError("Gram matrix must be positive definite")

    inverse_sqrt = eigenvectors @ torch.diag(eigenvalues.rsqrt()) @ eigenvectors.T

    return inverse_sqrt


def _orthonormal_mapping_rules(j: int, n: int) -> tuple[str, str]:
    """Build extraction and embedding rules for a self-dual numerical mapping."""
    lower = letter_index(j)
    upper = letter_index(n, upper_case=True)
    extraction_rule = f"{upper}{lower},...{upper}->...{lower}"
    embedding_rule = f"{upper}{lower},...{lower}->...{upper}"

    return extraction_rule, embedding_rule


def _get_symmetry_adapted_mappings(
    weight: int,
    rank: int,
    mappings: list[LinearCombination],
    gram_inverse: list[list[Fraction]],
    symmetry: str,
) -> list[LinearCombination]:
    """Solve the exact coefficient constraints imposed by internal symmetry.

    Every step is exact: the mixing matrices are built by symbolic contraction
    and the null space by Gaussian elimination over the rationals, so the
    symmetry-adapted mappings carry no numerical tolerance at all.

    Args:
        weight: Weight of the natural-tensor space.
        rank: Rank of the Cartesian tensor space.
        mappings: Independent mappings of that weight.
        gram_inverse: Exact inverse of their Gram matrix.
        symmetry: Internal index symmetry of the Cartesian tensor.

    Returns:
        The symmetry-adapted mappings, one per null-space basis vector.
    """
    generators = parse_symmetry_generators(symmetry, rank=rank)
    constraints = []
    for permutation, sign in generators:
        action = _get_symmetry_action_matrix(
            mappings, gram_inverse, weight, rank, permutation
        )
        for row_index, row in enumerate(action):
            constraints.append(
                [
                    value - sign * Fraction(int(row_index == column_index))
                    for column_index, value in enumerate(row)
                ]
            )

    coefficients = matrix_null_space(constraints, len(mappings))
    adapted = []
    for vector in coefficients:
        mapping = sum((c * G for c, G in zip(vector, mappings)), LinearCombination())
        adapted.append(simplify_linear_combination(mapping))

    return adapted


def _get_symmetry_action_matrix(
    mappings: list[LinearCombination],
    gram_inverse: list[list[Fraction]],
    weight: int,
    rank: int,
    permutation: tuple[int, ...],
) -> list[list[Fraction]]:
    r"""Evaluate the action of one index permutation on the mapping basis.

    If ``P G[q] = sum_p M[p, q] G[p]``, duality gives
    $$
    M^a_{pq} = \frac{\widetilde{\mathbf{G}}^p \odot^{n+\ell}
        (\Pi_a \mathbf{G}^q)}{2\ell + 1}
    $$

    That form is not the one evaluated. A dual is a combination of all ``N``
    mappings, so contracting one costs ``N`` times a plain contraction, and the
    matrix costs ``N^3``. Expanding the dual moves the inverse Gram matrix
    outside the contraction,

    $$
    \mathbf{M}^a = \mathbf{g}^{-1} \mathbf{O},
    \qquad
    O_{pq} = \frac{\mathbf{G}^p \odot^{n+\ell} (\Pi_a \mathbf{G}^q)}{2\ell + 1}
    $$

    which is the same matrix from contractions between single mappings. At rank
    six and weight three that is the difference between eighteen minutes and
    under one.

    Args:
        mappings: Independent mappings of this weight.
        gram_inverse: Exact inverse of their Gram matrix.
        weight: Weight of the natural-tensor space.
        rank: Rank of the Cartesian tensor space.
        permutation: The generator, as a permutation of the Cartesian indices.

    Returns:
        The exact mixing matrix of this generator.

    Raises:
        ValueError: If the permutation does not match the Cartesian rank.
    """
    if len(permutation) != rank:
        raise ValueError("Symmetry permutation does not match the Cartesian rank")

    # Permuting the axes of a tensor renames its indices: the slot that now
    # holds axis `permutation[k]` carries the letter that axis `k` had.
    letters = letter_index(rank, upper_case=True)
    relabeling = {letters[permutation[k]]: letters[k] for k in range(rank)}
    permuted = [relabel_indices_2(mapping, relabeling) for mapping in mappings]

    overlap = [
        [get_gram_entry(weight, rank, mapping, image) for image in permuted]
        for mapping in mappings
    ]

    return matrix_multiply(gram_inverse, overlap)


# TODO, this can be done symbolically. Probably do it.
#  We need:
#  1. symbolic symmetrize() to get T. It is implemented in ops.py, but commented out
#  2. multiply_2() to get X = G \odot^n T
#  3. Simplify_linear_combination() to get the simplified X.
#  4. Compare X to see if they are the same.


if __name__ == "__main__":
    # # elastic tensor
    # j = 4
    # rank = 4
    # symmetry = "ijkl=jikl=klij"
    # get_G_H_S_of_j(j, rank, symmetry)

    ######
    rank = 2
    symmetry = "ij=ji"
    # rank = 3
    # symmetry = "ijk=ijk"
    # rank = 4
    # rank = 4
    # symmetry = "ijkl=jikl=klij"
    out = get_G_H_S(rank, symmetry, numerical=False)
    pprint(out)

    # import yaml
    # with open("out_new.yaml", "w") as f:
    #     yaml.dump(out, f)
