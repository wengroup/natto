"""Tests for the golden-snapshot harness itself.

The harness is what every later phase of the port is verified against, so its
own failure modes are worth pinning: that it refuses to write a snapshot by
accident, that it notices the kinds of change the identity tests cannot see, and
that it does not cry wolf over the last bits of a float.
"""

from fractions import Fraction

import pytest
import torch

from tests.golden import (
    DEFAULT_ATOL,
    REGOLD_ENV,
    assert_snapshot,
    compare,
    read_snapshot,
    to_snapshot,
    write_snapshot,
)


def test_to_snapshot_keeps_fractions_exact():
    """Exact quantities must not become floats on the way into a snapshot."""
    assert to_snapshot(Fraction(2, 15)) == "2/15"
    assert to_snapshot([Fraction(9), Fraction(-1, 30)]) == ["9", "-1/30"]


def test_to_snapshot_carries_tensor_shape():
    """Nesting carries shape, so a reshaped operator is a mismatch."""
    assert to_snapshot(torch.zeros(2, 3)) == [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    assert to_snapshot(torch.tensor(1.5)) == 1.5
    assert compare(to_snapshot(torch.zeros(2, 3)), to_snapshot(torch.zeros(3, 2))) != []


def test_to_snapshot_rejects_unknown_types():
    """An unknown type is an error, never a silent repr."""
    with pytest.raises(TypeError, match="No snapshot form"):
        to_snapshot(object())


def test_to_snapshot_keeps_bool_distinct_from_int():
    """`bool` is a subclass of `int`; the two must not merge."""
    assert to_snapshot({"a": True, "b": 1}) == {"a": True, "b": 1}
    assert compare({"a": True}, {"a": 1}) != []


def test_compare_accepts_float_noise_but_not_a_changed_basis():
    """Tolerance is set to see a change of basis and ignore last-bit drift."""
    assert compare([1.0], [1.0 + DEFAULT_ATOL / 10]) == []
    assert compare([1.0], [0.5]) != []


def test_compare_scales_tolerance_with_magnitude():
    """A fingerprint runs to 1e4, where an absolute tolerance means nothing."""
    large = 60745.368
    assert compare([large], [large * (1 + 1e-7)], rtol=1e-5) == []
    assert compare([large], [large * (1 + 1e-3)], rtol=1e-5) != []


def test_compare_reports_the_path_of_a_mismatch():
    """A mismatch names where it is, so a diff is readable without a debugger."""
    expected = {"2": {"G": [{"symbolic": "+1 d_AB"}]}}
    actual = {"2": {"G": [{"symbolic": "+2 d_AB"}]}}

    (message,) = compare(expected, actual)

    assert ".2.G[0].symbolic" in message


@pytest.mark.parametrize(
    "expected,actual,reason",
    [
        ({"a": 1}, {"a": 1, "b": 2}, "added"),
        ({"a": 1, "b": 2}, {"a": 1}, "removed"),
        ([1, 2], [1, 2, 3], "length"),
    ],
)
def test_compare_notices_structural_change(expected, actual, reason):
    """A changed multiplicity shows up as structure, not as numbers."""
    (message,) = compare(expected, actual)

    assert reason in message


def test_assert_snapshot_refuses_to_create_one(tmp_path):
    """A missing snapshot is an error, so none is created as a side effect."""
    with pytest.raises(AssertionError, match="No snapshot at"):
        assert_snapshot("absent", {"a": 1.0}, directory=tmp_path)

    assert list(tmp_path.iterdir()) == []


def test_assert_snapshot_round_trip(tmp_path):
    """What is written is what is read back and matched."""
    data = {"g": [Fraction(9), Fraction(6)], "G": torch.eye(2)}
    write_snapshot("round_trip", data, directory=tmp_path)

    assert read_snapshot("round_trip", directory=tmp_path) == to_snapshot(data)
    assert_snapshot("round_trip", data, directory=tmp_path)


def test_symbols_are_stored_unescaped(tmp_path):
    """Delta and epsilon must reach the file as themselves, not as escapes."""
    path = write_snapshot(
        "symbols", {"symbolic": ["+1 δ_AB ε_Cab"]}, directory=tmp_path
    )
    text = path.read_text(encoding="utf-8")

    assert "δ_AB" in text and "ε_Cab" in text
    assert "\\u03b4" not in text
    assert_snapshot("symbols", {"symbolic": ["+1 δ_AB ε_Cab"]}, directory=tmp_path)


def test_assert_snapshot_reports_a_changed_operator(tmp_path):
    """The failure message says what moved and how to re-bless it."""
    write_snapshot("changed", {"G": [1.0, 2.0]}, directory=tmp_path)

    with pytest.raises(AssertionError) as error:
        assert_snapshot("changed", {"G": [1.0, 3.0]}, directory=tmp_path)

    message = str(error.value)
    assert "Operator content changed" in message
    assert ".G[1]: 2.0 -> 3.0" in message
    assert REGOLD_ENV in message


def test_regold_rewrites_instead_of_failing(tmp_path, monkeypatch):
    """With the environment variable set, a changed operator is re-blessed."""
    write_snapshot("regold", {"G": [1.0]}, directory=tmp_path)
    monkeypatch.setenv(REGOLD_ENV, "1")

    assert_snapshot("regold", {"G": [2.0]}, directory=tmp_path)

    assert read_snapshot("regold", directory=tmp_path) == {"G": [2.0]}


def test_regold_is_off_for_an_empty_or_zero_value(tmp_path, monkeypatch):
    """`NATTO_REGOLD=0` must not silently re-bless."""
    write_snapshot("guard", {"G": [1.0]}, directory=tmp_path)

    for value in ("", "0"):
        monkeypatch.setenv(REGOLD_ENV, value)
        with pytest.raises(AssertionError):
            assert_snapshot("guard", {"G": [2.0]}, directory=tmp_path)
