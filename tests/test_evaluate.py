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


def test_cache_distinguishes_dtypes():
    """The dtype is part of the cache key, so one dtype cannot serve another."""
    _cached_delta_epsilon_contraction.cache_clear()
    try:
        float32_result = _contract_delta_epsilon(
            "ab->ab", num_delta=1, num_epsilon=0, dtype=np.float32
        )
        float64_result = _contract_delta_epsilon(
            "ab->ab", num_delta=1, num_epsilon=0, dtype=np.float64
        )

        assert float32_result.dtype == np.float32
        assert float64_result.dtype == np.float64
        assert _cached_delta_epsilon_contraction.cache_info().currsize == 2
    finally:
        _cached_delta_epsilon_contraction.cache_clear()
