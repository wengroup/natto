# Dev guide

## Installation

Install `natto` from source, with the extras for tests and documentation:

```shell
git clone https://github.com/wengroup/natto.git
cd natto
pip install -e ".[test,doc]"
```

Run the tests with `pytest`.

### Build the docs

The documentation is built with [Jupyter Book 2](https://jupyterbook.org). Its pages
are notebooks, so pass `--execute` to run their code and show the results. From the
`docs` directory, start a live preview with

```shell
jupyter-book start --execute
```

or build the static site into `docs/_build/html` with

```shell
jupyter-book build --html --execute
```

A push to `main` builds the documentation and deploys it to GitHub Pages, at
https://wengroup.github.io/natto/.

## Code style

- Python code is linted and formatted with [Ruff](https://docs.astral.sh/ruff/), and
  YAML files with [prettier](https://prettier.io).
- Docstrings follow the
  [Google style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings).
- The documentation is written in [MyST Markdown](https://mystmd.org).

[pre-commit](https://pre-commit.com) runs all the checks, and fixes what it can:

```shell
pip install pre-commit
pre-commit run --all-files --show-diff-on-failure
```

To run them on every commit, install them as a git hook with `pre-commit install`, and
remove it with `pre-commit uninstall`. GitHub Actions runs the same checks, and the
tests, on every push and pull request.
