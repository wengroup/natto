"""Golden snapshots of operator content.

The existing tests pin the *relations* the paper asserts: `test_mappings.py` checks
multiplicities, exact Gram matrices and reconstruction, and `test_orthonormal.py`
checks self-duality and orthonormality. None of them pins the *content* of the
operators, meaning the entries themselves.

That content is what the port must not change. Renaming a function, deleting a
redundant route, or replacing a float64 contraction with an exact rational one
should each leave every operator exactly where it was. A snapshot is the
cheapest statement of that, and it is the only one that catches a silent change
of basis, which the identity tests cannot see: a different basis of the same
weight-l mapping space satisfies every identity just as well, while changing
every stored operator downstream.

What is compared exactly: the shape of the tree, dictionary keys, symbolic
strings, einsum rules, and exact `Fraction` entries. Those carry the content, so
nothing about them is approximate.

Floats are compared with a tolerance, absolute and relative together, because a
genuine change is O(1) relative while float arithmetic legitimately drifts:
an eigendecomposition may differ in its last bits between BLAS builds, and
re-associating a sum of a few thousand float64 terms moves the total in its last
bits too. Both tolerances are far below any real change and far above that
noise.

Regenerating is deliberate: set `NATTO_REGOLD=1`. A missing snapshot is an error
rather than a silent write, so no snapshot ever comes into existence as a side
effect of an ordinary test run, and re-blessing a changed operator is always an
explicit act.
"""

from __future__ import annotations

import json
import math
import os
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

SNAPSHOT_DIR = Path(__file__).parent / "snapshots"

#: Snapshots too large to be worth committing. This directory is gitignored, so
#: a snapshot placed here guards the working copy of whoever generated it and
#: nothing else. A test using it is responsible for skipping when it is absent,
#: since the harness itself treats a missing snapshot as an error.
LOCAL_SNAPSHOT_DIR = SNAPSHOT_DIR / "local"

#: Setting this environment variable rewrites snapshots instead of checking them.
REGOLD_ENV = "NATTO_REGOLD"

#: Absolute tolerance for float comparison, which carries values near zero.
DEFAULT_ATOL = 1e-12

#: Relative tolerance for float comparison. Floats are compared as
#: `|expected - actual| <= atol + rtol * |expected|`, because the quantities
#: snapshotted here span many orders of magnitude: an operator entry is order
#: one, while a fingerprint of a rank-8 operator runs to 1e4. A flat absolute
#: tolerance is meaningless across that range.
#:
#: Everything is evaluated in float64, so the only drift is in the last bits:
#: a different BLAS, or a sum of a few thousand terms re-associated. Any real
#: change -- a different basis, a different coefficient -- is order one, so this
#: sits far from both.
DEFAULT_RTOL = 1e-9

#: Mismatches reported before the report is truncated.
MAX_REPORTED = 20


def regolding() -> bool:
    """Whether this run rewrites snapshots rather than checking them."""
    return os.environ.get(REGOLD_ENV, "") not in ("", "0")


def to_snapshot(value: Any) -> Any:
    """Convert a value into its canonical, JSON-representable snapshot form.

    Tensors become nested lists, so shape is carried by the nesting. `Fraction`
    becomes its exact string form, never a float, so exact quantities stay
    exact. Dictionary keys become strings, because JSON has no other kind; this
    means a snapshot cannot distinguish the key `0` from the key `"0"`, which
    costs nothing here since weights are the only integer keys in play.

    Anything else raises. Silently stringifying an unknown object would let it
    into a snapshot in a form that compares equal to itself forever, whatever it
    later becomes.

    Args:
        value: The object to canonicalize.

    Returns:
        A structure of dicts, lists, strings, floats, ints, bools and None.

    Raises:
        TypeError: If `value` contains a type with no defined snapshot form.
    """
    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, Fraction):
        return str(value)

    if isinstance(value, dict):
        return {str(key): to_snapshot(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [to_snapshot(item) for item in value]

    # bool before int: bool is a subclass of int, and the two should not merge.
    if isinstance(value, bool) or value is None:
        return value

    if isinstance(value, (int, float, str)):
        return value

    raise TypeError(
        f"No snapshot form defined for {type(value).__name__}. Convert it in the "
        "caller, or add a case here if it is a type snapshots should carry."
    )


def fingerprint(tensor: np.ndarray, tol: float = DEFAULT_ATOL) -> dict[str, Any]:
    """Reduce an evaluated operator to a handful of comparable numbers.

    Storing evaluated operators in full is not affordable: a rank-4 class runs to
    a few hundred kilobytes of floats, and several megabytes once indented. The
    symbolic form is snapshotted instead, exactly, and it is the real content --
    a change of basis rewrites it, legibly, and the evaluated array is a
    deterministic function of it. This fingerprint is the second line of
    defence, catching a change in how the symbolic form is evaluated.

    `moment` weights each entry by its flat position, so a permutation of the
    entries is caught too. The other statistics are all permutation invariant,
    and an index-ordering bug is exactly the kind of change worth seeing.

    Args:
        tensor: An evaluated operator.
        tol: Magnitude above which an entry counts as nonzero.

    Returns:
        Statistics comparable under the harness's float tolerance.
    """
    flat = tensor.ravel().astype(np.float64)
    positions = np.arange(1, flat.size + 1, dtype=np.float64)

    return {
        "shape": list(tensor.shape),
        "nonzero": int((np.abs(flat) > tol).sum()),
        "sum": float(flat.sum()),
        "abs_sum": float(np.abs(flat).sum()),
        "square_sum": float((flat * flat).sum()),
        "min": float(flat.min()),
        "max": float(flat.max()),
        "moment": float((flat * positions).sum()),
    }


def snapshot_path(name: str, directory: Path | None = None) -> Path:
    """Path of the snapshot file for `name`."""
    return (directory or SNAPSHOT_DIR) / f"{name}.json"


def write_snapshot(name: str, data: Any, directory: Path | None = None) -> Path:
    """Write `data` as the snapshot for `name` and return the path written."""
    path = snapshot_path(name, directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        # `ensure_ascii=False` keeps the delta and epsilon of a symbolic operator
        # readable. Escaped, every one of them becomes a six-character sequence,
        # which defeats the readability the file exists for. The encoding is
        # pinned on both sides so the bytes do not depend on the locale of
        # whoever runs the tests.
        json.dump(to_snapshot(data), f, indent=2, sort_keys=True, ensure_ascii=False)
        f.write("\n")

    return path


def read_snapshot(name: str, directory: Path | None = None) -> Any:
    """Read the stored snapshot for `name`."""
    with snapshot_path(name, directory).open(encoding="utf-8") as f:
        return json.load(f)


def compare(
    expected: Any,
    actual: Any,
    atol: float = DEFAULT_ATOL,
    rtol: float = DEFAULT_RTOL,
) -> list[str]:
    """Compare two canonical snapshot structures.

    Args:
        expected: The stored snapshot.
        actual: The canonical form of what was just computed.
        atol: Absolute tolerance for float comparison.
        rtol: Relative tolerance for float comparison.

    Returns:
        One message per mismatch, each naming the path at which it occurs. Empty
        if the two agree.
    """
    mismatches: list[str] = []
    _compare_into(expected, actual, "", atol, rtol, mismatches)

    return mismatches


def _compare_into(
    expected: Any, actual: Any, path: str, atol: float, rtol: float, out: list[str]
) -> None:
    """Collect mismatches between two canonical structures into `out`."""
    if len(out) >= MAX_REPORTED:
        return

    where = path or "<root>"

    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            out.append(f"{where}: expected an object, got {_describe(actual)}")
            return
        for key in sorted(set(expected) | set(actual)):
            if key not in expected:
                out.append(f"{where}.{key}: added")
            elif key not in actual:
                out.append(f"{where}.{key}: removed")
            else:
                _compare_into(
                    expected[key], actual[key], f"{path}.{key}", atol, rtol, out
                )
        return

    if isinstance(expected, list):
        if not isinstance(actual, list):
            out.append(f"{where}: expected a list, got {_describe(actual)}")
            return
        if len(expected) != len(actual):
            out.append(f"{where}: length {len(expected)} -> {len(actual)}")
            return
        for index, (want, got) in enumerate(zip(expected, actual)):
            _compare_into(want, got, f"{path}[{index}]", atol, rtol, out)
        return

    # `bool` is a subclass of `int` and `True == 1`, so booleans need their own
    # comparison or a flag would silently agree with the number one.
    if isinstance(expected, bool):
        if not isinstance(actual, bool) or expected != actual:
            out.append(f"{where}: {expected!r} -> {actual!r}")
        return

    # Floats compare with a tolerance; ints promote so that 1 and 1.0 agree,
    # since JSON does not preserve the difference reliably.
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if not isinstance(actual, (int, float)) or isinstance(actual, bool):
            out.append(f"{where}: expected a number, got {_describe(actual)}")
        elif not _numbers_close(float(expected), float(actual), atol, rtol):
            out.append(f"{where}: {expected!r} -> {actual!r}")
        return

    if expected != actual:
        out.append(f"{where}: {expected!r} -> {actual!r}")


def _numbers_close(expected: float, actual: float, atol: float, rtol: float) -> bool:
    """Whether two floats agree, treating two NaNs as agreeing."""
    if math.isnan(expected) and math.isnan(actual):
        return True

    return abs(expected - actual) <= atol + rtol * abs(expected)


def _describe(value: Any) -> str:
    """A short description of a value, for a mismatch message."""
    return f"{type(value).__name__} {value!r}"


def assert_snapshot(
    name: str,
    data: Any,
    atol: float = DEFAULT_ATOL,
    rtol: float = DEFAULT_RTOL,
    directory: Path | None = None,
) -> None:
    """Assert that `data` matches the stored snapshot `name`.

    With `NATTO_REGOLD=1` set, the snapshot is rewritten instead of checked.

    Args:
        name: Snapshot name, used as the file stem.
        data: The value to snapshot; canonicalized with `to_snapshot`.
        atol: Absolute tolerance for float comparison.
        rtol: Relative tolerance for float comparison.
        directory: Where snapshots live. Defaults to `tests/snapshots`.

    Raises:
        AssertionError: If the snapshot is missing, or if `data` differs from it.
    """
    path = snapshot_path(name, directory)

    if regolding():
        write_snapshot(name, data, directory)
        return

    if not path.exists():
        raise AssertionError(
            f"No snapshot at {path}. Snapshots are never written by an ordinary "
            f"test run, so that re-blessing an operator is always deliberate. "
            f"Create it with {REGOLD_ENV}=1 and commit the result after reading "
            f"the diff."
        )

    mismatches = compare(read_snapshot(name, directory), to_snapshot(data), atol, rtol)
    if not mismatches:
        return

    report = "\n".join(f"  {line}" for line in mismatches[:MAX_REPORTED])
    if len(mismatches) > MAX_REPORTED:
        report += f"\n  ... and {len(mismatches) - MAX_REPORTED} more"

    raise AssertionError(
        f"Operator content changed against {path.name}:\n{report}\n\n"
        f"Nothing in the port should change these values. If the change is "
        f"intended, re-bless with {REGOLD_ENV}=1 and review the file diff."
    )
