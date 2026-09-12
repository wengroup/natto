from pytest import fixture

from natto.algebra import (
    contract_epsilon_delta,
    contract_two_epsilon,
    contract_with_delta,
    simplify_isotropic_product,
)
from natto.symbolic import (
    Delta,
    Epsilon,
    IsotropicProduct,
    IsotropicTensor,
    LinearCombination,
    Scalar,
    Zero,
)


@fixture
def T4():
    return IsotropicTensor("ijkl")


@fixture
def delta_ij():
    return Delta("ij")


@fixture
def delta_ik():
    return Delta("ik")


@fixture
def delta_ai():
    return Delta("ai")


@fixture
def epsilon_ijk():
    return Epsilon("ijk")


@fixture
def epsilon_kij():
    return Epsilon("kij")


@fixture
def epsilon_jik():
    return Epsilon("jik")


@fixture
def epsilon_ijl():
    return Epsilon("ijl")


@fixture
def epsilon_ilm():
    return Epsilon("ilm")


def test_cartesian_tensor():
    assert IsotropicTensor("ijjl") == IsotropicTensor("ikkl")


def test_delta():
    assert Delta("ij") == Delta("ji")
    assert Delta("ij") != Delta("ik")


def test_epsilon(epsilon_ijk, epsilon_kij, epsilon_jik):
    assert epsilon_ijk == epsilon_kij
    assert epsilon_ijk != epsilon_jik


def test_contract_with_delta(T4, delta_ij, delta_ik, delta_ai):
    assert contract_with_delta(delta_ij, T4) == IsotropicTensor("jjkl")
    assert contract_with_delta(delta_ik, T4) == IsotropicTensor("kjkl")
    assert contract_with_delta(delta_ai, T4) == IsotropicTensor("ajkl")

    assert contract_with_delta(delta_ij, delta_ik) == Delta("jk")


def test_contract_epsilon_delta(epsilon_ijk, delta_ij, delta_ai):
    assert contract_epsilon_delta(epsilon_ijk, delta_ij) == Zero()
    assert contract_epsilon_delta(epsilon_ijk, delta_ai) == Epsilon("ajk")


def test_contract_two_epsilon(epsilon_ijk, epsilon_ijl, epsilon_ilm):
    assert contract_two_epsilon(epsilon_ijl, epsilon_ijl) == Scalar(6)
    assert contract_two_epsilon(epsilon_ijk, epsilon_ijl) == Delta("kl", factor=2)
    assert contract_two_epsilon(epsilon_ijk, epsilon_ilm) == LinearCombination(
        IsotropicProduct(Delta("jl"), Delta("km")),
        IsotropicProduct(Delta("jm"), Delta("kl"), factor=-1),
    )


def test_simplify():
    d1 = Delta("ij", factor=2)
    d2 = Delta("jk", factor=2)
    e1 = Epsilon("ijk", factor=3)
    e2 = Epsilon("ikl", factor=3)
    e3 = Epsilon("ilm", factor=3)
    T1 = IsotropicTensor("ijkl", factor=4)

    tp = IsotropicProduct(d1, d2)
    tp_s = simplify_isotropic_product(tp)
    assert tp_s.to_str_list() == ["+4 δ_ik"]

    tp = IsotropicProduct(d1, e1)
    tp_s = simplify_isotropic_product(tp)
    assert len(tp_s) == 0

    tp = IsotropicProduct(d1, e2)
    tp_s = simplify_isotropic_product(tp)
    assert tp_s.to_str_list() == ["+6 ε_jkl"]

    tp = IsotropicProduct(d2, e2)
    tp_s = simplify_isotropic_product(tp)
    assert tp_s.to_str_list() == ["+6 ε_ijl"]

    tp = IsotropicProduct(e1, e2)
    tp_s = simplify_isotropic_product(tp)
    assert tp_s.to_str_list() == ["-18 δ_jl"]

    tp = IsotropicProduct(e1, e3)
    tp_s = simplify_isotropic_product(tp)
    assert tp_s.to_str_list() == ["+9 δ_jl δ_km", "-9 δ_jm δ_kl"]

    tp = IsotropicProduct(d1, e1, e3)
    tp_s = simplify_isotropic_product(tp)
    assert tp_s.to_str_list() == ["+18 δ_km δ_il", "-18 δ_kl δ_im"]

    tp = IsotropicProduct(d1, T1)
    tp_s = simplify_isotropic_product(tp)
    assert tp_s.to_str_list() == ["+8 T_jjkl"]

    tp = IsotropicProduct(d1, e1, e2, T1)
    tp_s = simplify_isotropic_product(tp)
    assert tp_s.to_str_list() == ["-144 T_ljkl"]

    tp = IsotropicProduct(d1, e1, e3, T1)
    tp_s = simplify_isotropic_product(tp)
    assert tp_s.to_str_list() == ["+72 T_ljml", "-72 T_mjll"]


# def test_symmetrize():
#     indices = "ijkl"  # symmetrizing indices
#     tensors = [
#         IsotropicTensor("".join(p), factor=Fraction(1, 24))
#         for p in itertools.permutations(indices)
#     ]
#     assert symmetrize(IsotropicTensor(indices)) == LinearCombination(*tensors)
#
#     indices = "akl"  # symmetrizing indices
#     tensors = []
#     for p in itertools.permutations(indices):
#         t = IsotropicProduct(
#             Epsilon(f"{p[0]}ij"),
#             IsotropicTensor(f"ij{p[1]}{p[2]}"),
#             factor=Fraction(1, 6),
#         )
#         tensors.append(t)
#     lin_comb = LinearCombination(*tensors)
#
#     t = IsotropicProduct(Epsilon("aij"), IsotropicTensor("ijkl"))
#     s = symmetrize(t)
#     assert s == lin_comb
