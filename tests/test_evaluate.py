import torch

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
    torch.testing.assert_close(first, second)
    assert _cached_delta_epsilon_contraction.cache_info() == small_cache_info
    _cached_delta_epsilon_contraction.cache_clear()


def test_cache_uses_effective_default_dtype():
    """Include the effective PyTorch default dtype in the cache key."""
    original_dtype = torch.get_default_dtype()
    _cached_delta_epsilon_contraction.cache_clear()
    try:
        torch.set_default_dtype(torch.float32)
        float32_result = _contract_delta_epsilon("ab->ab", num_delta=1, num_epsilon=0)
        torch.set_default_dtype(torch.float64)
        float64_result = _contract_delta_epsilon("ab->ab", num_delta=1, num_epsilon=0)

        assert float32_result.dtype == torch.float32
        assert float64_result.dtype == torch.float64
        assert _cached_delta_epsilon_contraction.cache_info().currsize == 2
    finally:
        torch.set_default_dtype(original_dtype)
        _cached_delta_epsilon_contraction.cache_clear()
