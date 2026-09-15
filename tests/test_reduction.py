import functools
from collections import defaultdict
from fractions import Fraction
from typing import NamedTuple, Optional

import numpy as np
import pytest

from natto.gram import get_gram_entry
from natto.intrinsic_symmetry import generate_permutations, impose_symmetry
from natto.mapping_tensors import get_extraction_operators
from natto.rational import matrix_inverse
from natto.reduction import get_independent_mappings, get_reduction


class TensorClass(NamedTuple):
    """One physical tensor class, i.e. one row of Table III."""

    # `Example` column, e.g. `photoelastic effect`
    example: str
    # `Rank n` column
    rank: int
    # `Symmetry` column, spelled the way `get_reduction` wants it; the table writes
    # it with
    # parentheses around the symmetric index groups, so its `(ij)kl` is `ijkl=jikl`
    # here. None when the indices are unconstrained.
    symmetry: Optional[str]
    # `N_ind` column, the number of independent components
    n_ind: int
    # `Weight m` columns, as {weight m: multiplicity N_m}; weights with N_m = 0 omitted
    multiplicity: dict[int, int]

    @property
    def test_id(self) -> str:
        """Identifier of this class, unique because two rows share an example."""
        return f"rank{self.rank}_{self.example.replace(' ', '_')}"


PHYSICAL_TENSOR_CLASSES = [
    TensorClass("pressure", 0, None, 1, {0: 1}),
    TensorClass("spontaneous polarization", 1, None, 3, {1: 1}),
    TensorClass("optical activity", 2, None, 9, {0: 1, 1: 1, 2: 1}),
    TensorClass("stress and strain", 2, "ij=ji", 6, {0: 1, 2: 1}),
    TensorClass("rate-of-rotation tensor", 2, "ij=-ji", 3, {1: 1}),
    TensorClass("optical mixing", 3, None, 27, {0: 1, 1: 3, 2: 2, 3: 1}),
    TensorClass("piezoelectric effect", 3, "ijk=ikj", 18, {1: 2, 2: 1, 3: 1}),
    TensorClass("Kleinman symmetry in SHG", 3, "ijk=ikj=jik", 10, {1: 1, 3: 1}),
    TensorClass("optical mixing", 4, None, 81, {0: 3, 1: 6, 2: 6, 3: 3, 4: 1}),
    # the j=1 sector of the photoelastic class is what the unpivoted-QR selection in
    # `natto.qr` used to get wrong, see tests/test_qr.py
    TensorClass(
        "photoelastic effect", 4, "ijkl=jikl", 54, {0: 2, 1: 3, 2: 4, 3: 2, 4: 1}
    ),
    TensorClass("Kerr effect", 4, "ijkl=jikl=ijlk", 36, {0: 2, 1: 1, 2: 3, 3: 1, 4: 1}),
    TensorClass(
        "third harmonic generation",
        4,
        "ijkl=ikjl=jikl",
        30,
        {0: 1, 1: 1, 2: 2, 3: 1, 4: 1},
    ),
    TensorClass("elasticity", 4, "ijkl=jikl=klij", 21, {0: 2, 2: 2, 4: 1}),
    TensorClass("Cauchy relations", 4, "ijkl=jikl=kjil=ljki", 15, {0: 1, 2: 1, 4: 1}),
]

#: The symbolic Gram matrix is exact, so a structural zero in it meets a float64
#: residue of order 1e-17 on the numerical side. `assert_allclose` defaults to
#: atol=0, which no residue can pass, so the comparison needs one.
GRAM_ATOL = 1e-12

# Classes `get_reduction` cannot handle yet, as {test_id: reason}. Kept out of the table
# above so that it stays a statement about the physics, not about the code.
NOT_SUPPORTED = {}


def get_tensor_class_params(
    tensor_classes: Optional[list[TensorClass]] = None,
) -> list:
    """Parametrize over tensor classes, xfailing the unsupported.

    Args:
        tensor_classes: classes to parametrize over, all of Table III by default.
    """
    if tensor_classes is None:
        tensor_classes = PHYSICAL_TENSOR_CLASSES

    params = []
    for tc in tensor_classes:
        marks = []
        reason = NOT_SUPPORTED.get(tc.test_id)
        if reason is not None:
            marks.append(
                pytest.mark.xfail(raises=AssertionError, strict=True, reason=reason)
            )
        params.append(pytest.param(tc, id=tc.test_id, marks=marks))

    return params


@functools.lru_cache(maxsize=None)
def get_reduction_cached(rank: int, symmetry: str) -> dict:
    """`get_reduction`, computed once per class; rank 4 costs seconds."""
    return get_reduction(rank, symmetry)


@functools.lru_cache(maxsize=None)
def get_orthonormal_cached(rank: int, symmetry: str) -> dict:
    """`get_reduction` in the orthonormal basis, computed once per class."""
    return get_reduction(rank, symmetry, basis="orthonormal")


def cycle_index_multiplicities(n: int, symmetry: Optional[str]) -> dict[int, int]:
    """The nonzero multiplicity of each weight in a rank-n tensor of a symmetry class.

    The character of the class, averaged over its signed permutations, is a Laurent
    polynomial in z = exp(it) with one factor z^-k + 1 + z^k per cycle of length k.
    Its overlap with the weight-ell character, the sum of z^m for |m| <= ell, under
    the SO(3) measure is c_0 - (c_1 + c_-1) / 2, with c_m the coefficient of z^m in
    their product. No mapping tensor is built, and the arithmetic is exact.

    References:
        B.2 of [Wen2026Refactor].
    """
    group = generate_permutations(symmetry) if symmetry else [(tuple(range(n)), 1)]

    character = defaultdict(Fraction)
    for permutation, sign in group:
        term = {0: Fraction(sign, len(group))}
        for length in cycle_lengths(permutation):
            product = defaultdict(Fraction)
            for power, value in term.items():
                for shift in (-length, 0, length):
                    product[power + shift] += value
            term = product
        for power, value in term.items():
            character[power] += value

    multiplicities = {}
    for ell in range(n + 1):
        c_minus, c_zero, c_plus = (
            sum((character.get(m - k, 0) for k in range(-ell, ell + 1)), Fraction(0))
            for m in (-1, 0, 1)
        )
        multiplicity = c_zero - (c_plus + c_minus) / 2
        assert multiplicity.denominator == 1
        if multiplicity:
            multiplicities[ell] = int(multiplicity)

    return multiplicities


def cycle_lengths(permutation: tuple[int, ...]) -> list[int]:
    """The lengths of the cycles of a permutation, fixed points included."""
    seen, lengths = set(), []
    for start in range(len(permutation)):
        length, slot = 0, start
        while slot not in seen:
            seen.add(slot)
            slot = permutation[slot]
            length += 1
        if length:
            lengths.append(length)

    return lengths


@pytest.mark.parametrize("tensor_class", get_tensor_class_params())
def test_weight_multiplicity(tensor_class: TensorClass):
    """Check the weight decomposition of each physical tensor class in Table III.

    Each weight-m sector appears N_m times, and the multiplicities account for all
    N_ind independent components: N_ind = sum_m N_m (2m + 1).

    The multiplicities are the ranks found when selecting independent mappings, so
    this is where a selection that found the wrong rank would show up.

    Args:
        tensor_class: physical tensor class to check
    """
    output = get_reduction_cached(tensor_class.rank, tensor_class.symmetry)

    found = {
        m: len(out_m["extraction"])
        for m, out_m in output.items()
        if len(out_m["extraction"]) > 0
    }

    assert found == tensor_class.multiplicity
    assert sum(N_m * (2 * m + 1) for m, N_m in found.items()) == tensor_class.n_ind


@pytest.mark.parametrize("tensor_class", get_tensor_class_params())
def test_multiplicity_from_cycle_index(tensor_class: TensorClass):
    """The channels found are the multiplicities the symmetry's character predicts.

    The prediction uses the symmetry group alone, so it checks the selection and the
    symmetry adaptation independently of the numbers typed into Table III.
    """
    output = get_reduction_cached(tensor_class.rank, tensor_class.symmetry)
    found = {ell: len(data["extraction"]) for ell, data in output.items()}

    assert found == cycle_index_multiplicities(tensor_class.rank, tensor_class.symmetry)


@pytest.mark.parametrize("tensor_class", get_tensor_class_params())
def test_qr_selection_spans_the_same_weights(tensor_class: TensorClass):
    """Algorithm 1 of the paper may keep other mappings, but spans the same spaces.

    Pivoted QR keeps the same number of mappings at every weight as the exact scan, and
    their extraction and embedding give the same projector onto each weight.
    """
    rank = tensor_class.rank
    exact = get_reduction_cached(rank, tensor_class.symmetry)
    qr = get_reduction(rank, tensor_class.symmetry, selection="qr")

    assert {w: len(d["extraction"]) for w, d in qr.items()} == {
        w: len(d["extraction"]) for w, d in exact.items()
    }
    for weight in exact:
        projectors = [
            sum(
                np.tensordot(g["numerical"], h["numerical"], axes=weight)
                for g, h in zip(data["embedding"], data["extraction"])
            )
            for data in (exact[weight], qr[weight])
        ]

        np.testing.assert_allclose(projectors[1], projectors[0], rtol=0, atol=1e-10)


@pytest.mark.parametrize("tensor_class", get_tensor_class_params())
def test_duality_is_exact(tensor_class: TensorClass):
    """Each dual contracts to one with its own mapping and to zero with the others.

    The contractions are exact Gram entries, so the identity holds as a fact rather
    than to within a tolerance.
    """
    for weight in tensor_class.multiplicity:
        G, gram = get_independent_mappings(
            weight, tensor_class.rank, tensor_class.symmetry
        )
        duals = get_extraction_operators(matrix_inverse(gram), G)

        for p, dual in enumerate(duals):
            entries = [get_gram_entry(dual, G_q) for G_q in G]

            assert entries == [Fraction(int(p == q)) for q in range(len(G))]


@pytest.mark.parametrize("tensor_class", get_tensor_class_params())
def test_weight_projector_is_basis_independent(tensor_class: TensorClass):
    """Summed over channels, both bases give the same projector onto each weight.

    The dual basis builds its decomposition operators symbolically and exactly; the
    orthonormal basis rotates the mappings by a numerical inverse square root. The
    projector onto a weight does not depend on the basis within it, so the two routes
    must agree.
    """
    rank = tensor_class.rank
    dual = get_reduction_cached(rank, tensor_class.symmetry)
    orthonormal = get_orthonormal_cached(rank, tensor_class.symmetry)

    assert sorted(dual) == sorted(orthonormal)
    for weight, data in dual.items():
        summed = sum(entry["numerical"] for entry in data["decomposition"])

        weight_axes = list(range(rank, rank + weight))
        expected = sum(
            np.tensordot(
                entry["numerical"], entry["numerical"], axes=(weight_axes,) * 2
            )
            for entry in orthonormal[weight]["embedding"]
        )

        np.testing.assert_allclose(summed, expected, rtol=0, atol=1e-12)


@pytest.mark.parametrize(
    "tensor_class",
    get_tensor_class_params(
        [tc for tc in PHYSICAL_TENSOR_CLASSES if tc.symmetry is not None]
    ),
)
def test_symbolic_symmetry_adapted_gram_matrix(tensor_class: TensorClass):
    """Check exact Gram matrices for every internally symmetric weight sector."""
    for weight in tensor_class.multiplicity:
        Q, gram = get_independent_mappings(
            weight, tensor_class.rank, tensor_class.symmetry
        )
        numerical_Q = np.stack([Q_p.expand().evaluate(("rank", "weight")) for Q_p in Q])
        flattened_Q = numerical_Q.reshape(len(Q), -1)
        numerical = flattened_Q @ flattened_Q.T / (2 * weight + 1)
        symbolic = np.array(
            [[float(value) for value in row] for row in gram], dtype=numerical.dtype
        )

        # a Gram entry that is exactly zero meets a float64 residue of ~1e-17,
        # and `assert_allclose` defaults to atol=0, which no residue can pass
        np.testing.assert_allclose(symbolic, numerical, atol=GRAM_ATOL)


def test_symmetry_rank_must_match_tensor_rank():
    """Reject a symmetry whose reference term has the wrong tensor rank."""
    with pytest.raises(ValueError):
        get_reduction(3, "ij", numerical=False)


@pytest.mark.parametrize("tensor_class", get_tensor_class_params())
def test_reduction_round_trip(tensor_class: TensorClass):
    """Check that extracting and embedding recovers the tensor it started from.

    Every weight and channel of `T` is extracted into its natural tensor and
    embedded back; the parts must sum to `T`. The decomposition operator is
    checked against the two-step route on the way, since it is their composition.

    Args:
        tensor_class: physical tensor class to check
    """
    rank = tensor_class.rank
    symmetry = tensor_class.symmetry

    T = np.random.default_rng(35).standard_normal((3,) * rank)

    # symmetrize the tensor if `symmetry` is not None
    if symmetry is not None:
        T = impose_symmetry(T, symmetry)

    output = get_reduction_cached(rank, symmetry)

    all_T_prime = []
    for j, out_j in output.items():
        for p, (extraction, embedding, decomposition) in enumerate(
            zip(out_j["extraction"], out_j["embedding"], out_j["decomposition"])
        ):
            # X = G~ . T
            X = np.einsum(extraction["rule"], extraction["numerical"], T)

            # T' = G . X
            T_p_1 = np.einsum(embedding["rule"], embedding["numerical"], X)

            # T' = S . T
            T_p_2 = np.einsum(decomposition["rule"], decomposition["numerical"], T)

            # T_p_1 and T_p_2 should be equal
            assert np.allclose(T_p_1, T_p_2, rtol=0, atol=1e-10), (
                f"T_p_1 and T_p_2 are not equal for j={j}, p={p}"
            )

            all_T_prime.append(T_p_1)

    sum_T_prime = np.sum(np.stack(all_T_prime), axis=0)

    assert np.allclose(sum_T_prime, T, rtol=0, atol=1e-10)
