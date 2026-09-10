"""
Symbolic and numerical G, H, S tensors.

G, H, and S are made of only the Kronecker delta and Levi-Civita symbols.

G, H, and S can be used to map a general tensor T and a natural tensor X.
S = G H
X = H T
T' = G X = (G H) T = S T
where T' is the embedding of X in the T space.
"""

from fractions import Fraction
from pprint import pprint

import torch
from torch import Tensor

from natto.EGH import (
    get_G_even,
    get_g_matrix,
    get_G_odd,
    get_g_pq,
    get_H,
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
        G, H, S, and g_pq, h_pq information.
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
    ``g[p, q] = (mapping[p] \odot^(rank+weight) mapping[q]) / (2*weight + 1)``.
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
                simplify_linear_combination(mapping), mode="G", dtype=dtype
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
        A dictionary keyed by weight. Each value contains the exact ``G`` tensors,
        their numerical Gram matrix and its inverse square root, and the numerical
        orthonormal tensors ``G_tilde``.
    """
    out = {}
    for j in range(n + 1):
        G, _, _, _ = get_G_H_of_j(j, n)
        if not G:
            continue

        G_numerical, g_G, g_G_inverse_sqrt, G_tilde = orthonormalize_mappings(
            G, j, n, dtype
        )
        extraction_rule, embedding_rule = _orthonormal_mapping_rules(j, n)
        out[j] = {
            "G": [
                {"symbolic": str(G_p), "numerical": G_p_numerical}
                for G_p, G_p_numerical in zip(G, G_numerical)
            ],
            "g_G": g_G,
            "g_G_inverse_sqrt": g_G_inverse_sqrt,
            "G_tilde": [
                {
                    "numerical": G_tilde_p,
                    "extraction_rule": extraction_rule,
                    "embedding_rule": embedding_rule,
                }
                for G_tilde_p in G_tilde
            ],
        }

    return out


def get_orthonormal_Q(
    n: int, symmetry: str = None, dtype: torch.dtype = torch.float64
) -> dict:
    r"""Get symmetry-adapted orthonormal mapping tensors.

    For each weight ``j``, the exact symmetry-adapted embedding tensors ``Q`` are
    obtained from :func:`get_G_H_S_of_j`. Their Gram matrix is evaluated as

    ``g_Q[p, q] = (Q[p] \odot^(n+j) Q[q]) / (2*j + 1)``.

    The returned tensors ``Q_tilde = g_Q^(-1/2) Q`` are orthonormal and can be used
    for both extraction from and embedding into the target symmetry class.

    Args:
        n: Rank of the Cartesian tensor.
        symmetry: Internal index symmetry of the Cartesian tensor. See
            :func:`get_G_H_S` for examples. If ``None``, the symmetry-free mapping
            tensors are orthonormalized.
        dtype: Floating-point dtype used for the eigendecomposition and returned
            numerical tensors.

    Returns:
        A dictionary keyed by weight. Each value contains the exact ``Q`` tensors,
        their numerical Gram matrix ``g_Q``, its symmetric inverse square root
        ``g_Q_inverse_sqrt``, and the numerical orthonormal tensors ``Q_tilde``.
        Every ``Q_tilde`` entry provides separate einsum rules for extraction and
        embedding because the same numerical tensor performs both operations.
    """
    out = {}
    for j in range(n + 1):
        Q, _, _, _, _ = get_G_H_S_of_j(j, n, symmetry)
        if len(Q) == 0:
            continue

        Q_numerical, g_Q, g_Q_inverse_sqrt, Q_tilde = orthonormalize_mappings(
            Q, j, n, dtype
        )

        extraction_rule, embedding_rule = _orthonormal_mapping_rules(j, n)
        out[j] = {
            "Q": [
                {"symbolic": str(Q_p), "numerical": Q_p_numerical}
                for Q_p, Q_p_numerical in zip(Q, Q_numerical)
            ],
            "g_Q": g_Q,
            "g_Q_inverse_sqrt": g_Q_inverse_sqrt,
            "Q_tilde": [
                {
                    "numerical": Q_tilde_p,
                    "extraction_rule": extraction_rule,
                    "embedding_rule": embedding_rule,
                }
                for Q_tilde_p in Q_tilde
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
        G: independent G tensors of different seniority p
        H: H corresponding to G
        S: S corresponding to G and H
        g: g_pq matrix
        h: h_pq matrix
    """
    # Get independent G and H tensors for a general tensor
    ind_G, ind_H, g, h = get_G_H_of_j(j, n)

    # Further reduce the mappings for tensors with internal symmetry.
    if symmetry is not None:
        ind_G = _get_symmetry_adapted_mappings(j, n, ind_G, h, symmetry)
        if not ind_G:
            return [], [], [], [], []

        g = get_g_matrix(j, n, ind_G)
        h = matrix_inverse(g)
        ind_H = get_H(h, ind_G)

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
        g: g_pq matrix
        h: h_pq matrix
    """
    # No isotropic rank-one mapping exists from a scalar natural tensor.
    if n == 1 and j == 0:
        return [], [], [], []

    # create G mapping operator
    if (n - j) % 2 == 0:
        all_G = get_G_even(j, n)
    else:
        all_G = get_G_odd(j, n)

    # WARNING, should not simplify G using the below function, as get_g_matrix() below
    # is set up to work with the original G tensors.
    # all_G = [simplify_linear_combination(g) for g in all_G]

    # Get numerical S tensors, embedding a random natura tensor X in space j to space n
    X = get_random_natural_tensor(j)
    all_num_S = [embed(G, X) for G in all_G]

    # Get linearly independent S tensors
    _, independent_indices = find_independent_tensors(all_num_S)

    # Get linearly independent G tensors
    ind_G = [all_G[i] for i in independent_indices]

    # Get g_pq matrix for independent G
    g = get_g_matrix(j, n, ind_G)

    # Get h_pq matrix
    h = matrix_inverse(g)

    # Get H tensors, corresponding to independent G
    ind_H = get_H(h, ind_G)

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

    out_j = {"G": [], "H": [], "S": []}

    if include_g:
        out_j["g_pq"] = {"symbolic": fraction_matrix(g), "numerical": float_matrix(g)}

    if include_h:
        out_j["h_pq"] = {"symbolic": fraction_matrix(h), "numerical": float_matrix(h)}

    # loop over seniority p
    for G_p, H_p, S_p in zip(G, H, S):
        lower = letter_index(j)
        upper = letter_index(n, upper_case=True)
        upper2 = letter_index(n, start=n, upper_case=True)

        # G
        out_j["G"].append(
            {
                "symbolic": str(G_p),
                "rule": (f"{upper}{lower},...{lower}->...{upper}"),
            },
        )
        if numerical:
            out_j["G"][-1]["numerical"] = evaluate_tensors(G_p, mode="G")

        # H
        out_j["H"].append(
            {"symbolic": str(H_p), "rule": f"{lower}{upper},...{upper}->...{lower}"}
        )
        if numerical:
            out_j["H"][-1]["numerical"] = evaluate_tensors(H_p, mode="H")

        # S
        out_j["S"].append(
            {
                "symbolic": str(S_p),
                "rule": f"{upper}{upper2},...{upper2}->...{upper}",
            }
        )
        if numerical:
            out_j["S"][-1]["numerical"] = evaluate_tensors(S_p, mode="S")

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
        mapping = sum(
            (
                coefficient * candidate
                for coefficient, candidate in zip(vector, mappings)
            ),
            LinearCombination(),
        )
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
    ``M[p, q] = (G_dual[p] \odot^(rank+weight) P G[q]) / (2*weight + 1)``.

    That form is not the one evaluated. A dual is a combination of all ``N``
    mappings, so contracting one costs ``N`` times a plain contraction, and the
    matrix costs ``N^3``. Expanding the dual moves the inverse Gram matrix
    outside the contraction,

        ``M = g^-1 O``,  ``O[p, q] = (G[p] \odot^(rank+weight) P G[q]) / (2w+1)``

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
        [get_g_pq(weight, rank, mapping, image) for image in permuted]
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
