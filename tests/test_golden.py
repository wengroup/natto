"""Tests for the golden-snapshot harness itself.

It must never create a snapshot by accident, must rewrite one only when asked, and must
say where a changed operator differs.
"""

import pytest

from tests.golden import REGOLD_ENV, assert_snapshot, compare, read_snapshot


def test_a_missing_snapshot_is_an_error(tmp_path):
    """A missing snapshot fails the test, so none is created as a side effect."""
    with pytest.raises(AssertionError, match="No snapshot at"):
        assert_snapshot("absent", {"a": 1}, directory=tmp_path)

    assert list(tmp_path.iterdir()) == []


def test_regold_rewrites_the_snapshot(tmp_path, monkeypatch):
    """Only with the environment variable set is a changed snapshot re-blessed."""
    monkeypatch.setenv(REGOLD_ENV, "1")
    assert_snapshot("regold", {"G": ["+1 δ_AB"]}, directory=tmp_path)
    monkeypatch.setenv(REGOLD_ENV, "0")

    with pytest.raises(AssertionError):
        assert_snapshot("regold", {"G": ["+2 δ_AB"]}, directory=tmp_path)
    assert read_snapshot("regold", directory=tmp_path) == {"G": ["+1 δ_AB"]}


def test_a_mismatch_names_its_path():
    """A mismatch says where it is, so a changed operator is easy to find."""
    expected = {"2": {"extraction": [["+1 δ_AB"]]}}
    actual = {"2": {"extraction": [["+2 δ_AB"]]}}

    (message,) = compare(expected, actual)

    assert ".2.extraction[0][0]" in message
