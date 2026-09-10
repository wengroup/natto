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

from natto.EGH import (
    get_extraction_operators,
    get_G_even,
    get_G_odd,
    get_gram_matrix,
    get_S,
)
from natto.evaluate import embed, evaluate_tensors
from natto.matrix import float_matrix, fraction_matrix, matrix_inverse
from natto.ops import simplify_linear_combination
from natto.qr import find_independent_tensors
from natto.symbolic import LinearCombination
from natto.symmetrize import get_random_natural_tensor
from natto.symmetry_adapted import get_symmetry_adapted_mappings
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
        ind_G = get_symmetry_adapted_mappings(j, n, ind_G, h, symmetry)
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
