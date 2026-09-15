"""Golden snapshots of the published operators.

The other tests check relations the operators satisfy, such as duality, the round trip
and the multiplicities, and any valid basis satisfies them. A snapshot records the
operators themselves, so it also catches a change of basis. Everything stored is exact,
symbolic terms and rational Gram entries as strings, and is compared for equality.

Regenerating is deliberate: set `NATTO_REGOLD=1`. A missing snapshot is an error rather
than a silent write, so re-blessing an operator is always an explicit act.
"""

import json
import os
from fractions import Fraction
from pathlib import Path
from typing import Any

SNAPSHOT_DIR = Path(__file__).parent / "test_data"

#: Setting this environment variable rewrites snapshots instead of checking them.
REGOLD_ENV = "NATTO_REGOLD"

#: Mismatches reported before the report is truncated.
MAX_REPORTED = 20


def regolding() -> bool:
    """Whether this run rewrites snapshots rather than checking them."""
    return os.environ.get(REGOLD_ENV, "") not in ("", "0")


def to_snapshot(value: Any) -> Any:
    """Convert a value into its JSON form.

    A `Fraction` becomes its exact string, dictionary keys become strings, and tuples
    become lists. Anything else that JSON cannot hold raises, rather than being stored
    in a form that would compare equal to itself whatever it later becomes.

    Args:
        value: The object to convert.

    Returns:
        A structure of dicts, lists, strings, ints, bools and None.

    Raises:
        TypeError: If `value` contains a type with no snapshot form.
    """
    if isinstance(value, Fraction):
        return str(value)

    if isinstance(value, dict):
        return {str(key): to_snapshot(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [to_snapshot(item) for item in value]

    if value is None or isinstance(value, (bool, int, str)):
        return value

    raise TypeError(f"No snapshot form defined for {type(value).__name__}")


def snapshot_path(name: str, directory: Path | None = None) -> Path:
    """Path of the snapshot file for `name`."""
    return (directory or SNAPSHOT_DIR) / f"{name}.json"


def write_snapshot(name: str, data: Any, directory: Path | None = None) -> Path:
    """Write `data` as the snapshot for `name`, and return the path written."""
    path = snapshot_path(name, directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        # Unescaped, so the delta and epsilon of a symbolic operator stay readable.
        json.dump(to_snapshot(data), f, indent=2, sort_keys=True, ensure_ascii=False)
        f.write("\n")

    return path


def read_snapshot(name: str, directory: Path | None = None) -> Any:
    """Read the stored snapshot for `name`."""
    with snapshot_path(name, directory).open(encoding="utf-8") as f:
        return json.load(f)


def compare(expected: Any, actual: Any) -> list[str]:
    """Compare two snapshot structures exactly.

    Args:
        expected: The stored snapshot.
        actual: The JSON form of what was just computed.

    Returns:
        One message per mismatch, each naming the path at which it occurs. Empty if the
        two agree.
    """
    mismatches: list[str] = []
    _compare_into(expected, actual, "", mismatches)

    return mismatches


def assert_snapshot(name: str, data: Any, directory: Path | None = None) -> None:
    """Assert that `data` matches the stored snapshot `name`.

    With `NATTO_REGOLD=1` set, the snapshot is rewritten instead of checked.

    Args:
        name: Snapshot name, used as the file stem.
        data: The value to snapshot; converted with `to_snapshot`.
        directory: Where snapshots live. Defaults to `tests/test_data`.

    Raises:
        AssertionError: If the snapshot is missing, or if `data` differs from it.
    """
    path = snapshot_path(name, directory)

    if regolding():
        write_snapshot(name, data, directory)
        return

    if not path.exists():
        raise AssertionError(
            f"No snapshot at {path}. Create it with {REGOLD_ENV}=1 and commit the "
            f"result after reading it."
        )

    mismatches = compare(read_snapshot(name, directory), to_snapshot(data))
    if not mismatches:
        return

    report = "\n".join(f"  {line}" for line in mismatches[:MAX_REPORTED])
    if len(mismatches) > MAX_REPORTED:
        report += f"\n  ... and {len(mismatches) - MAX_REPORTED} more"

    raise AssertionError(
        f"Operator content changed against {path.name}:\n{report}\n\n"
        f"If the change is intended, re-bless with {REGOLD_ENV}=1 and review the diff."
    )


def _compare_into(expected: Any, actual: Any, path: str, out: list[str]) -> None:
    """Collect mismatches between two snapshot structures into `out`."""
    if len(out) >= MAX_REPORTED:
        return

    where = path or "<root>"

    if isinstance(expected, dict) and isinstance(actual, dict):
        for key in sorted(set(expected) | set(actual)):
            if key not in expected:
                out.append(f"{where}.{key}: added")
            elif key not in actual:
                out.append(f"{where}.{key}: removed")
            else:
                _compare_into(expected[key], actual[key], f"{path}.{key}", out)
        return

    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            out.append(f"{where}: length {len(expected)} -> {len(actual)}")
            return
        for index, (want, got) in enumerate(zip(expected, actual)):
            _compare_into(want, got, f"{path}[{index}]", out)
        return

    if expected != actual or type(expected) is not type(actual):
        out.append(f"{where}: {expected!r} -> {actual!r}")
