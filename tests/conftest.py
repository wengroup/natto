def pytest_configure(config):
    """Let `-m full` mean the whole suite rather than a marker of that name.

    The default run excludes the tests marked `slow` (see `pyproject.toml`), and
    `-m slow` selects only those. `full` is not a marker any test carries: it is
    the spelling for "no selection at all", which as a marker expression is the
    empty string.
    """
    if config.option.markexpr.strip() == "full":
        config.option.markexpr = ""
