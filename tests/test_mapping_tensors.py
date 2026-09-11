from natto.gram import get_gram_matrix
from natto.mapping_tensors import get_mappings
from natto.natural_projector import get_natural_projector


def test_mappings_even_parity():
    # rank 0, weight 0
    all_G = get_mappings(ell=0, n=0)
    assert len(all_G) == 1
    assert set(all_G[0].to_str_list()) == {"+1"}

    # rank 1, weight 1
    all_G = get_mappings(ell=1, n=1)
    assert len(all_G) == 1
    assert set(all_G[0].to_str_list()) == {"+1 δ_aA"}

    # rank 2, weight 0
    all_G = get_mappings(ell=0, n=2)
    assert len(all_G) == 1
    assert set(all_G[0].to_str_list()) == {"+1 δ_AB"}

    # rank 2, weight 2
    all_G = get_mappings(ell=2, n=2)
    assert len(all_G) == 1
    assert set(all_G[0].to_str_list()) == {
        "-1/3 δ_ab δ_AB",
        "+1/2 δ_aB δ_bA",
        "+1/2 δ_aA δ_bB",
    }

    # rank 3, weight 1
    all_G = get_mappings(ell=1, n=3)
    assert len(all_G) == 3
    assert set(all_G[0].to_str_list()) == {"+1 δ_aA δ_BC"}
    assert set(all_G[1].to_str_list()) == {"+1 δ_aB δ_AC"}
    assert set(all_G[2].to_str_list()) == {"+1 δ_aC δ_AB"}

    # rank 3, weight 3
    all_G = get_mappings(ell=3, n=3)
    assert len(all_G) == 1
    assert set(all_G[0].to_str_list()) == set(get_natural_projector(3).to_str_list())

    # rank 4, weight 0
    all_G = get_mappings(ell=0, n=4)
    assert len(all_G) == 3
    assert set(all_G[0].to_str_list()) == {"+1 δ_AB δ_CD"}
    assert set(all_G[1].to_str_list()) == {"+1 δ_AC δ_BD"}
    assert set(all_G[2].to_str_list()) == {"+1 δ_AD δ_BC"}

    # rank 4, weight 2
    all_G = get_mappings(ell=2, n=4)
    assert len(all_G) == 6
    assert set(all_G[0].to_str_list()) == {
        "+1/2 δ_aA δ_bB δ_CD",
        "+1/2 δ_aB δ_bA δ_CD",
        "-1/3 δ_ab δ_AB δ_CD",
    }
    assert set(all_G[1].to_str_list()) == {
        "+1/2 δ_aA δ_bC δ_BD",
        "+1/2 δ_aC δ_bA δ_BD",
        "-1/3 δ_ab δ_AC δ_BD",
    }

    assert set(all_G[2].to_str_list()) == {
        "+1/2 δ_aA δ_bD δ_BC",
        "+1/2 δ_aD δ_bA δ_BC",
        "-1/3 δ_ab δ_AD δ_BC",
    }

    assert set(all_G[3].to_str_list()) == {
        "+1/2 δ_aB δ_bC δ_AD",
        "+1/2 δ_aC δ_bB δ_AD",
        "-1/3 δ_ab δ_BC δ_AD",
    }

    assert set(all_G[4].to_str_list()) == {
        "+1/2 δ_aB δ_bD δ_AC",
        "+1/2 δ_aD δ_bB δ_AC",
        "-1/3 δ_ab δ_BD δ_AC",
    }

    assert set(all_G[5].to_str_list()) == {
        "+1/2 δ_aC δ_bD δ_AB",
        "+1/2 δ_aD δ_bC δ_AB",
        "-1/3 δ_ab δ_CD δ_AB",
    }


def test_mappings_odd_parity():
    # n = 1, j = 0 not possible

    # # n = 2, j = 1
    # all_G = get_mappings(ell=1, n=2)
    # assert len(all_G) == 1
    # assert set(all_G[0].to_str_list()) == {"(1) δ_aC ε_CAB"}

    # TODO, seems we need to implement triple products os epsilon
    # n 3, j = 0
    # all_G = get_mappings(ell=0, n=3)
    # assert len(all_G) == 3
    # assert set(all_G[0].to_str_list()) == {"(1) ε_CAB δ_aC"}

    # n = 3, j = 2
    all_G = get_mappings(ell=2, n=3)
    assert len(all_G) == 3
    assert set(all_G[0].to_str_list()) == {
        "-1/3 δ_ab δ_CD ε_DAB",
        "+1/2 δ_aD δ_bC ε_DAB",
        "+1/2 δ_aC δ_bD ε_DAB",
    }
    assert set(all_G[1].to_str_list()) == {
        "-1/3 δ_ab δ_BD ε_DAC",
        "+1/2 δ_aD δ_bB ε_DAC",
        "+1/2 δ_aB δ_bD ε_DAC",
    }
    assert set(all_G[2].to_str_list()) == {
        "-1/3 δ_ab δ_AD ε_DBC",
        "+1/2 δ_aD δ_bA ε_DBC",
        "+1/2 δ_aA δ_bD ε_DBC",
    }

    # n = 4, j = 1
    all_G = get_mappings(ell=1, n=4)
    assert len(all_G) == 6
    # the deltas come before the epsilon, the order create_delta_epsilon_tensors
    # builds them in; a product's factors commute, so this is presentation only
    assert set(all_G[0].to_str_list()) == {"+1 δ_aE δ_CD ε_EAB"}
    assert set(all_G[1].to_str_list()) == {"+1 δ_aE δ_BD ε_EAC"}
    assert set(all_G[2].to_str_list()) == {"+1 δ_aE δ_BC ε_EAD"}
    assert set(all_G[3].to_str_list()) == {"+1 δ_aE δ_AD ε_EBC"}
    assert set(all_G[4].to_str_list()) == {"+1 δ_aE δ_AC ε_EBD"}
    assert set(all_G[5].to_str_list()) == {"+1 δ_aE δ_AB ε_ECD"}

    # n = 4, j = 3
    all_G = get_mappings(ell=3, n=4)
    assert len(all_G) == 6
    assert set(all_G[0].to_str_list()) == {
        "-1/15 δ_aC δ_bc δ_DE ε_EAB",
        "-1/15 δ_aD δ_bc δ_CE ε_EAB",
        "-1/15 δ_aE δ_bc δ_CD ε_EAB",
        "-1/15 δ_bC δ_ac δ_DE ε_EAB",
        "-1/15 δ_bD δ_ac δ_CE ε_EAB",
        "-1/15 δ_bE δ_ac δ_CD ε_EAB",
        "-1/15 δ_cC δ_ab δ_DE ε_EAB",
        "-1/15 δ_cD δ_ab δ_CE ε_EAB",
        "-1/15 δ_cE δ_ab δ_CD ε_EAB",
        "+1/6 δ_aC δ_bD δ_cE ε_EAB",
        "+1/6 δ_aC δ_bE δ_cD ε_EAB",
        "+1/6 δ_aD δ_bC δ_cE ε_EAB",
        "+1/6 δ_aD δ_bE δ_cC ε_EAB",
        "+1/6 δ_aE δ_bC δ_cD ε_EAB",
        "+1/6 δ_aE δ_bD δ_cC ε_EAB",
    }
    # ignore a couple of them
    assert set(all_G[5].to_str_list()) == {
        "-1/15 δ_aA δ_bc δ_BE ε_ECD",
        "-1/15 δ_aB δ_bc δ_AE ε_ECD",
        "-1/15 δ_aE δ_bc δ_AB ε_ECD",
        "-1/15 δ_bA δ_ac δ_BE ε_ECD",
        "-1/15 δ_bB δ_ac δ_AE ε_ECD",
        "-1/15 δ_bE δ_ac δ_AB ε_ECD",
        "-1/15 δ_cA δ_ab δ_BE ε_ECD",
        "-1/15 δ_cB δ_ab δ_AE ε_ECD",
        "-1/15 δ_cE δ_ab δ_AB ε_ECD",
        "+1/6 δ_aA δ_bB δ_cE ε_ECD",
        "+1/6 δ_aA δ_bE δ_cB ε_ECD",
        "+1/6 δ_aB δ_bA δ_cE ε_ECD",
        "+1/6 δ_aB δ_bE δ_cA ε_ECD",
        "+1/6 δ_aE δ_bA δ_cB ε_ECD",
        "+1/6 δ_aE δ_bB δ_cA ε_ECD",
    }


def test_g_matrix_ignores_symbolic_zero_terms():
    """Check that explicit zero terms do not affect the symbolic Gram matrix."""
    all_G = get_mappings(ell=2, n=4)
    mapping_with_zeros = all_G[0] + 0 * all_G[1]

    assert get_gram_matrix(2, 4, [mapping_with_zeros]) == get_gram_matrix(
        2, 4, [all_G[0]]
    )
