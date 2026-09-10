"""
Symbolic E, G, and H tensors.

Reference:

References:
1. [CS70] Irreducible Cartesian Tensors. II. General Formulation, http://dx.doi.org/10.1063/1.1665190
2. [AG82] Irreducible fourth-rank Cartesian tensors, https://doi.org/10.1103/PhysRevA.25.2647
"""

import itertools
from collections import Counter
from fractions import Fraction
from functools import reduce
from math import gcd

from natto.ops import multiply_2, simplify_linear_combination
from natto.symbolic import (
    CartesianTensor,
    Delta,
    Epsilon,
    LinearCombination,
    Scalar,
    TensorProduct,
)
from natto.symmetrize import get_permutations_2
from natto.utils import letter_index


def get_natural_projector(
    j: int, s_letters: str = None, verbose: int = 0
) -> LinearCombination:
    """
    Invariant tensors of rank j: E(j | j).

    References:
        Eq 19 and Eq 21 of [CS70].

    Args:
        j: rank of the projection operator
        s_letters: letters for the upper case indices, if None, use the default:
            A, B, C, etc.
        verbose: verbosity level for debugging

    Returns:
        Tensor operations with delta tensors. We use lower case letters `a`, `b`, `c`,
        etc. for indices `r`, and upper case letters `A`, `B`, `C`, etc. for indices `s`.
    """
    k = j // 2

    out = []
    c = Fraction(1, 1)  # c for t = 0
    for t in range(k + 1):
        if t > 0:
            c *= -Fraction(
                (j - 2 * t + 2) * (j - 2 * t + 1), 2 * t * (2 * j - 2 * t + 1)
            )

        # get all rules
        all_rules = get_E_rules(j, t, s_letters)

        # Total factor: c / len(all_rules), where len(all_rules) averages over all
        # the rules.
        factor = c / len(all_rules)

        # create tensor products of deltas for each rule
        delta_tensors = [
            create_delta_epsilon_tensors(
                rule["d_rs"] + rule["d_rr"] + rule["d_ss"], factor=factor
            )
            for rule in all_rules
        ]

        out.extend(delta_tensors)

        if verbose > 0:
            tmp = []
            tmp.extend(delta_tensors)
            print(
                f"@ debug E_j: j={j}, t={t}, c={c}, num terms: {len(tmp)}, Terms: ",
                "{LinearCombination(*tmp)}",
            )

    return LinearCombination(*out)


def get_G_even(j: int, n: int) -> list[LinearCombination]:
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

    E_s_letters, delta_rules = get_G_rules_even(j, n)

    all_G = []
    for si, rule in zip(E_s_letters, delta_rules):
        E_j = get_natural_projector(j, s_letters=si)
        f_q = create_delta_epsilon_tensors(rule)
        G = multiply_2(E_j, f_q)
        all_G.append(G)

    return all_G


def get_G_odd(j: int, n: int) -> list[LinearCombination]:
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

    E_s_letters, f_epsilon_rules, f_delta_rules = get_G_rules_odd(j, n)

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


def get_S(
    G: list[LinearCombination], H: list[LinearCombination], n: int
) -> list[LinearCombination]:
    r"""
    Get S tensors for a given G and H.

    S = G \odot^j H

    Args:
        G: G tensors
        H: H tensors. The order of H tensors should correspond to the order of G.

    Returns:
        S: S tensors
    """

    S = []
    for G_i, dual_i in zip(G, H):
        # Shift upper letters of H to distinguish those from G
        dual_i = shift_index_2(dual_i, n, letter_index(24, upper_case=True))

        S_i = multiply_2(G_i, dual_i)
        S_i = simplify_linear_combination(S_i)

        S.append(S_i)

    return S


def create_delta_epsilon_tensors(
    rule: list[str], epsilon: str = None, factor: int | Fraction = 1
) -> TensorProduct:
    """Create a TensorProduct of deltas and epsilons.

    Currently, we only support a single epsilon tensor in the product, because it is
    all needed to create the E, G, H tensors.

    Args:
        rule: Each string contains a pair of indices for a delta tensor.
        epsilon: A three letter string for the epsilon tensor.
        factor: additional factor to multiply with the tensor product

    Returns:
        List of TensorProduct objects.
    """

    tensors = [Delta(pair) for pair in rule]

    if epsilon is not None:
        e = Epsilon(epsilon)
        tensors.append(e)

    tp = TensorProduct(*tensors, factor=factor)

    return tp


def get_E_rules(j: int, t: int, s_letters: str = None) -> list[dict[str, list[str]]]:
    """
    Rules for E(j|j): d_{rs}^{j-2t} d_{rr}^t d_{ss}^t.

    This is in Eq. 19 of the paper.

    Args:
        j: rank of the projection operator
        t: number of d_rr and d_ss
        s_letters: letters for the upper case indices, if None, use the default:
            A, B, C, etc.

    Examples:
        get_E_rules(3, 1)
        [{'d_rs': ['cC'], 'd_rr': ['ab'], 'd_ss': ['AB']},
         {'d_rs': ['cB'], 'd_rr': ['ab'], 'd_ss': ['AC']},
         {'d_rs': ['cA'], 'd_rr': ['ab'], 'd_ss': ['BC']},
         {'d_rs': ['bC'], 'd_rr': ['ac'], 'd_ss': ['AB']},
         {'d_rs': ['bB'], 'd_rr': ['ac'], 'd_ss': ['AC']},
         {'d_rs': ['bA'], 'd_rr': ['ac'], 'd_ss': ['BC']},
         {'d_rs': ['aC'], 'd_rr': ['bc'], 'd_ss': ['AB']},
         {'d_rs': ['aB'], 'd_rr': ['bc'], 'd_ss': ['AC']},
         {'d_rs': ['aA'], 'd_rr': ['bc'], 'd_ss': ['BC']},
        ]

    Returns:
        Each dict {'d_rs': list_rs, 'd_rr': list_rr, 'd_ss': list_ss} contains the
        indices for constructing the deltas.

    """
    assert j >= 2 * t, f"j must be greater than or equal to 2*t, got j={j}, t={t}"

    r_letters = letter_index(j, upper_case=False)

    if s_letters is None:
        s_letters = letter_index(j, upper_case=True)

    perms = get_permutations_2(j, num_delta=t)

    # TODO, this depends on the order of the indices get_permutations_2 returns, where
    #   we put the remaining indices of t at the front, and the contracted indices at
    #   the end.
    start = j - 2 * t

    all_indices = []
    for p_r in perms:
        r_indices = [r_letters[p_r.index(i)] for i in range(j)]

        # indices for d_{rr}^t
        rr_pairs = [r_indices[i] + r_indices[i + 1] for i in range(start, j, 2)]

        # permute the remaining r indices that will be used for d_{rs}^{j-2t}
        r_remaining = r_indices[:start]
        r_remaining_perms = list(itertools.permutations(r_remaining))

        for p_s in perms:
            s_indices = [s_letters[p_s.index(i)] for i in range(j)]

            # indices for d_{ss}^t
            ss_pairs = [s_indices[i] + s_indices[i + 1] for i in range(start, j, 2)]

            # get the remaining s indices that will be used for d_{rs}^{j-2t}
            # no need to permute it, since it is to be combined with r_remaining
            s_remaining = s_indices[:start]

            # Create indices permutations for d_{rs}^{j-2t}
            for r_remaining_p in r_remaining_perms:
                rs_pairs = [f"{r}{s}" for r, s in zip(r_remaining_p, s_remaining)]

                # Sort it here to make it ordered according to the indices of r.
                # For example, ['bA', 'aB'] -> ['aB', 'bA']
                # But sort it or not does not matter for the creation of the delta
                # tensors.
                rs_pairs = sorted(rs_pairs)

                all_indices.append(
                    {"d_rs": rs_pairs, "d_rr": rr_pairs, "d_ss": ss_pairs}
                )

    return all_indices


def get_G_rules_even(j: int, n: int) -> tuple[list[str], list[list[str]]]:
    """
    Rules for G(n|j) for even n-j.

    Args:
        j:
        n:

    Returns:
        E_s_indices: s letters to use for E_j
        f_rules: rules to create deltas for for f_{n-j}^q
    """
    assert (n - j) % 2 == 0, f"n-j must be even, got n={n}, j={j}"

    letters = letter_index(n, upper_case=True)

    all_perms = get_permutations_2(n, num_delta=(n - j) // 2)

    # TODO, this depends on the order of the indices get_permutations_2 returns, where
    #   we put the remaining indices of t at the front, and the contracted indices at
    #   the end.
    start = j

    f_rules = []
    E_s_letters = []
    for perm in all_perms:  # each perm for a q in f_q
        indices = [letters[perm.index(i)] for i in range(n)]

        # indices for f_{n-j}^q
        delta_pairs = [indices[i] + indices[i + 1] for i in range(start, n, 2)]
        f_rules.append(delta_pairs)

        # s indices for E_j
        s_remaining = "".join(indices[:start])
        E_s_letters.append(s_remaining)

    return E_s_letters, f_rules


def get_G_rules_odd(j: int, n: int) -> tuple[list[str], list[str], list[list[str]]]:
    """
    Rules for G(n|j) for odd n-j.


    # NOTE,
    Upper case letter n+1 will be used as the index in epsilon to contract with E(j|j).
    In other words, it is the tau index.
    For example, if n = 3, then the letter D will be used as the tau index.

    Args:
        j:
        n:

    Returns:
        E_s_indices: s letters to use for E_j
        f_epsilon_rules: rules to create epsilons for f_{n-j}^q
        f_delta_rules: rules to create deltas for f_{n-j}^q
    """
    assert (n - j) % 2 == 1, f"n-j must be odd, got n={n}, j={j}"

    if j == 0:
        return get_G_rules_odd_j0(j, n)

    # All s letters
    letters = letter_index(n, upper_case=True)

    # Extra letter used in epsilon. See Table I of [AG82]
    tau_letter = letter_index(1, start=n, upper_case=True)

    all_perms = get_permutations_2(n, num_delta=(n - j - 1) // 2)

    # TODO, this depends on the order of the indices get_permutations_2 returns, where
    #   we put the remaining indices of t at the front, and the contracted indices at
    #   the end.
    start = j + 1

    f_delta_rules = []
    f_epsilon_rules = []
    E_s_letters = []
    for perm in all_perms:  # each perm for a q in f_q
        indices = [letters[perm.index(i)] for i in range(n)]

        # delta indices for f_{n-j}^q
        delta_pairs = [indices[i] + indices[i + 1] for i in range(start, n, 2)]

        # remaining indices for epsilon and E_j
        s_remaining = indices[:start]
        s_remaining_set = set(s_remaining)

        for comb in itertools.combinations(s_remaining, 2):
            f_delta_rules.append(delta_pairs)

            # choose two indices for epsilon
            f_epsilon_rules.append(tau_letter + "".join(sorted(comb)))

            # the remaining indices and also tau for E_j
            E_s_letters.append(
                "".join(sorted(s_remaining_set - set(comb))) + tau_letter
            )

    return E_s_letters, f_epsilon_rules, f_delta_rules


def get_G_rules_odd_j0(j, n):
    """
    For j = 0, and odd n, the rules for G(n|0) are different from the general case.

    Here we do a trivial contraction with epsilon tensor, instead of a double
    contraction in the general case.
    """
    assert j == 0, f"j must be 0, got {j}"
    assert n % 2 == 1, f"n must be odd, got {n}"
    assert n >= 3, f"n must be greater than or equal to 3, got {n}"

    # All s letters
    letters = letter_index(n, upper_case=True)

    all_perms = get_permutations_2(n, num_delta=(n - 3) // 2)

    # TODO, this depends on the order of the indices get_permutations_2 returns, where
    #   we put the remaining indices of t at the front, and the contracted indices at
    #   the end.
    start = 3

    f_delta_rules = []
    f_epsilon_rules = []
    E_s_letters = []

    for perm in all_perms:  # each perm for a q in f_q
        indices = [letters[perm.index(i)] for i in range(n)]

        # delta indices for f_{n-j}^q
        delta_pairs = [indices[i] + indices[i + 1] for i in range(start, n, 2)]
        f_delta_rules.append(delta_pairs)

        # remaining indices for epsilon (indices for E_j is empty)
        s_remaining = indices[:start]
        f_epsilon_rules.append("".join(sorted(s_remaining)))
        E_s_letters.append("")

    return E_s_letters, f_epsilon_rules, f_delta_rules


def shift_index(
    tensor: CartesianTensor | TensorProduct, shift: int, letters: str = None
) -> CartesianTensor | TensorProduct:
    """
    Shift the index of a tensor by a certain amount.

    For example, for T_ijk, and shift=1, the new tensor is T_jkl.

    Args:
        tensor: The tensor to shift the index.
        shift: The amount to shift the index.
        letters: The letters signifying the indices to shift. If None, shift all the
            indices. For example, if T_ijAB, and letters = 'AB', then the new tensor
            would be T_ijCD, where C and D are the new letters.

    Returns:
        The new tensor with the shifted index.
    """

    # The zero tensor has no meaningful indices to shift. In particular, a zero
    # TensorProduct stores a Zero component whose constructor does not accept the
    # general CartesianTensor arguments used below.
    if tensor.factor == 0:
        return tensor

    def _shift(t: CartesianTensor):
        if letters is None:
            indices = "".join([chr(ord(i) + shift) for i in t.indices])
        else:
            indices = "".join(
                [chr(ord(i) + shift) if i in letters else i for i in t.indices]
            )
        return t.__class__(indices, factor=t.factor, symbol=t.symbol)

    if isinstance(tensor, CartesianTensor):
        return _shift(tensor)

    elif isinstance(tensor, TensorProduct):
        components = [_shift(t) for t in tensor]
        return tensor.__class__(*components, factor=tensor.factor)

    else:
        raise ValueError(f"Unknown tensor type: {type(tensor)}")


def relabel_indices(
    tensor: CartesianTensor | TensorProduct, mapping: dict[str, str]
) -> CartesianTensor | TensorProduct:
    """Rename the indices of a tensor according to `mapping`.

    The substitution is simultaneous, so a mapping may permute letters among
    themselves. Applying it one letter at a time would chain the replacements,
    turning a transposition into a collapse.

    Args:
        tensor: The tensor whose indices to rename.
        mapping: Old index letter to new. Letters absent from it are left alone.

    Returns:
        The tensor with its indices renamed.
    """
    # A zero tensor carries no meaningful indices, and a zero TensorProduct holds
    # a component whose constructor does not take the arguments used below.
    if tensor.factor == 0:
        return tensor

    def _relabel(t: CartesianTensor):
        indices = "".join(mapping.get(index, index) for index in t.indices)

        return t.__class__(indices, factor=t.factor, symbol=t.symbol)

    if isinstance(tensor, CartesianTensor):
        return _relabel(tensor)

    if isinstance(tensor, TensorProduct):
        return tensor.__class__(*[_relabel(t) for t in tensor], factor=tensor.factor)

    raise ValueError(f"Unknown tensor type: {type(tensor)}")


def relabel_indices_2(
    tensor: LinearCombination, mapping: dict[str, str]
) -> LinearCombination:
    """Rename the indices of every term of a linear combination.

    Args:
        tensor: The linear combination whose indices to rename.
        mapping: Old index letter to new, applied simultaneously.

    Returns:
        The linear combination with its indices renamed.
    """
    return LinearCombination(*[relabel_indices(t, mapping) for t in tensor])


def shift_index_2(
    tensor: LinearCombination, shift: int, letters: str = None
) -> LinearCombination:
    """
    Shift all the index of a Tensors object by a certain amount.

    Returns:
        The new tensor with the shifted index.
    """
    components = [shift_index(t, shift, letters) for t in tensor]
    return LinearCombination(*components)


def contract_G(
    G1: LinearCombination, G2: LinearCombination, G1_indices: str, G2_indices: str
) -> LinearCombination:
    """
    Contract two G tensors.

    Args:
        G1: The first G tensor
        G2: The second G tensor

    Returns:
        The contracted tensor.
    """
    contraction_delta = [Delta(i + j) for i, j in zip(G1_indices, G2_indices)]
    contraction_delta = TensorProduct(*contraction_delta)
    prod = multiply_2(G1, G2, contraction_delta)
    simplified = simplify_linear_combination(prod)

    return simplified


def get_scalar_factor(t1: LinearCombination, t2: LinearCombination) -> Fraction | None:
    """
    Check if two tensors are scalar multiples of each other.

    Namely, t1 = c * t2, where c is a scalar.

    Args:
        t1:
        t2:

    Returns:
        The scalar factor to multiple with t2 to get t1. If None, t1 is not a scalar
        multiple of t2.
    """

    def to_dict(tensor: LinearCombination):
        d = {}
        for tp in tensor:
            k = " ".join([f"{t.symbol}_{t.indices}" for t in tp])
            d[k] = tp.factor
        return d

    t1_d = to_dict(t1)
    t2_d = to_dict(t2)

    if t1_d.keys() == t2_d.keys():
        factors = [t1_d[k] / t2_d[k] for k in t1_d.keys()]
        # Check all the factors are the same
        if len(set(factors)) == 1:
            return factors[0]
        else:
            raise ValueError(
                "Tensor 1 is not a scale multiple of tensor 2, although they have "
                "the same components Symbols."
            )

    else:
        return None


def get_gram_entry(
    j: int, n: int, G_p: LinearCombination, G_q: LinearCombination
) -> Fraction:
    r"""
    Compute one entry of the exact Gram matrix of symbolic mapping tensors.

    For rank-``n``, weight-``j`` mapping tensors, this evaluates

    ``g_pq = (G_p \odot^(n+j) G_q) / (2*j + 1)``.

    Every free index of ``G_p`` is paired with the corresponding free index of
    ``G_q``. Indices repeated within an individual symbolic term are internal dummy
    indices and are contracted independently. The resulting scalar is evaluated with
    :class:`fractions.Fraction`, so the result is exact.

    Args:
        j: Weight of the natural-tensor space.
        n: Rank of the Cartesian tensor space.
        G_p: First rank-``n + j`` symbolic mapping tensor.
        G_q: Second rank-``n + j`` symbolic mapping tensor.

    Returns:
        The exact Gram-matrix entry ``g_pq``.
    """

    def get_free_indices(tensor: LinearCombination) -> str:
        free_indices = None
        for term in tensor:
            if term.factor == 0:
                continue
            counts = Counter(term.indices)
            term_free_indices = "".join(
                sorted(index for index, count in counts.items() if count == 1)
            )
            if free_indices is None:
                free_indices = term_free_indices
            elif term_free_indices != free_indices:
                raise ValueError("All terms must have the same free indices")

        return free_indices or ""

    # Give the second tensor a disjoint index set before pairing corresponding free
    # indices. Repeated indices within either tensor are internal contraction indices.
    G_q = shift_index_2(G_q, n + j + 1)
    p_indices = get_free_indices(G_p)
    q_indices = get_free_indices(G_q)
    expected_rank = n + j
    if len(p_indices) != expected_rank or len(q_indices) != expected_rank:
        raise ValueError(
            f"Mapping tensors must have rank {expected_rank}, got "
            f"{len(p_indices)} and {len(q_indices)}"
        )

    contracted = contract_G(G_p, G_q, p_indices, q_indices)
    if any(term.indices for term in contracted):
        raise ValueError("Full contraction left unpaired indices")

    full_contraction = sum((term.factor for term in contracted), Fraction())

    return full_contraction / (2 * j + 1)


def get_gram_matrix(
    j: int, n: int, all_G: list[LinearCombination]
) -> list[list[Fraction]]:
    r"""Compute the exact Gram matrix of symbolic mapping tensors.

    Each entry is evaluated as

    ``g_pq = (G_p \odot^(n+j) G_q) / (2*j + 1)``

    using :func:`get_gram_entry`.

    Args:
        j: Weight of the natural-tensor space.
        n: Rank of the Cartesian tensor space.
        all_G: Rank-``n + j`` symbolic mapping tensors spanning the weight-``j``
            sector.

    Returns:
        Symmetric matrix whose entries are exact :class:`fractions.Fraction` values.
    """
    num = len(all_G)

    matrix = [[None] * num for _ in range(num)]
    for p in range(num):
        for q in range(num):
            matrix[p][q] = get_gram_entry(j, n, all_G[p], all_G[q])

    return matrix


def find_matrix_factorization(
    A: list[list[Fraction]],
) -> tuple[Fraction, list[list[int]]]:
    """
    For a matrix A consisting of Fraction, find the scalar factor c and integer
    matrix B such that A = c*B.
    """
    # Flatten and get nums/denoms
    nums = [f.numerator for row in A for f in row if f is not None]
    denoms = [f.denominator for row in A for f in row if f is not None]

    # Find GCD of nums and LCM of denoms
    def lcm(a, b):
        return abs(a * b) // gcd(a, b)

    num_gcd = reduce(gcd, [abs(n) for n in nums])
    denom_lcm = reduce(lcm, denoms)

    # Scalar factor
    c = Fraction(num_gcd, denom_lcm)

    # Integer matrix B = A/c
    B = [[int(f / c) if f is not None else None for f in row] for row in A]

    return c, B


if __name__ == "__main__":
    E3 = get_natural_projector(j=2, verbose=0)
