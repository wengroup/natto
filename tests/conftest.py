import numpy as np
import pytest

from natto.symmetrize import symmetrize_and_remove_trace


def pytest_configure(config):
    """Let `-m full` mean the whole suite rather than a marker of that name.

    The default run excludes the tests marked `slow` (see `pyproject.toml`), and
    `-m slow` selects only those. `full` is not a marker any test carries: it is
    the spelling for "no selection at all", which as a marker expression is the
    empty string.
    """
    if config.option.markexpr.strip() == "full":
        config.option.markexpr = ""


@pytest.fixture(scope="session")
def T0():
    return get_T(0)


@pytest.fixture(scope="session")
def T1():
    return get_T(1)


@pytest.fixture(scope="session")
def T2():
    return get_T(2)


@pytest.fixture(scope="session")
def T3():
    return get_T(3)


@pytest.fixture(scope="session")
def T4():
    return get_T(4)


@pytest.fixture(scope="session")
def NT0():
    return get_NT(0)


@pytest.fixture(scope="session")
def NT1():
    return get_NT(1)


@pytest.fixture(scope="session")
def NT2():
    return get_NT(2)


@pytest.fixture(scope="session")
def NT3():
    return get_NT(3)


@pytest.fixture(scope="session")
def NT4(T4):
    return get_NT(4)


def get_T(rank: int):
    """Create a tensor of rank `rank` for testing."""
    if rank == 0:
        return np.array(1.0)
    t = np.arange(3**rank).reshape([3] * rank).astype(np.float32)
    return t / t.mean()


def get_NT(rank: int):
    """Create a natural tensor of rank `rank` for testing."""
    t = get_T(rank)
    return symmetrize_and_remove_trace(t)
