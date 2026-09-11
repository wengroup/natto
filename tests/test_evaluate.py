import numpy as np

from natto.evaluate import (
    _cached_delta_epsilon_contraction,
    _contract_delta_epsilon,
)


def test_large_delta_epsilon_contractions_bypass_cache():
    """Keep dense high-rank contraction results out of the bounded cache."""
    _cached_delta_epsilon_contraction.cache_clear()

    small_rule = "ab->ab"
    _contract_delta_epsilon(small_rule, num_delta=1, num_epsilon=0)
    _contract_delta_epsilon(small_rule, num_delta=1, num_epsilon=0)
    small_cache_info = _cached_delta_epsilon_contraction.cache_info()
    assert small_cache_info.currsize == 1
    assert small_cache_info.hits == 1

    large_rule = "ab,cd,ef,gh,ij->abcdefghij"
    first = _contract_delta_epsilon(large_rule, num_delta=5, num_epsilon=0)
    second = _contract_delta_epsilon(large_rule, num_delta=5, num_epsilon=0)
    assert first.shape == (3,) * 10
    np.testing.assert_allclose(first, second, atol=1e-12)
    assert _cached_delta_epsilon_contraction.cache_info() == small_cache_info
    _cached_delta_epsilon_contraction.cache_clear()
