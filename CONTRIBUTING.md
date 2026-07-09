# Contributing

First read the overall project contributing guidelines. These are all
included in the qiskit documentation:

https://github.com/Qiskit/qiskit/blob/main/CONTRIBUTING.md

## Contributing to Qiskit IonQ Provider

### Getting started

All contributions are welcome.

If you've noticed a bug or have a feature request, we encourage to open an issue in [this repo's issues tracker](https://github.com/qiskit-partners/qiskit-ionq/issues), whether or not you plan to address it yourself.

If you intend to contribute code, please still start the contribution process by opening a new issue or making a comment on an existing issue briefly explaining what you intend to address and how. This helps us understand your intent/approach and provide support and commentary before you take the time to actually write code, and ensures multiple people aren't accidentally working on the same thing.

### Setting up a development environment

This project uses [uv](https://docs.astral.sh/uv/) to manage its development
environment. After [installing uv](https://docs.astral.sh/uv/getting-started/installation/),
run the following from the root of the repository:

```bash
uv sync
```

This creates a virtual environment in `.venv`, installs `qiskit-ionq` in
editable mode, and installs all development dependencies (the `test` and
`docs` [dependency groups](https://docs.astral.sh/uv/concepts/projects/dependencies/#dependency-groups)
defined in `pyproject.toml`). Prefix commands with `uv run` to execute them
inside this environment.

We also use [pre-commit](https://pre-commit.com/) hooks (ruff, mypy, and
friends) to catch issues before they land. Enable them once per clone with:

```bash
uv run pre-commit install
```

### Making a pull request

When you're ready to make a pull request, please make sure the following is true:

1. The code matches the project's code style
2. The documentation, _including any docstrings for changed methods_, has been updated
3. If appropriate for your change, that new tests have been added to address any new functionality, or that existing tests have been updated as appropriate
4. All of the tests (new and old) still pass!
5. You have added notes in the pull request that explains what has changed and links to the relevant issues in the issues tracker

### Running the tests

This package uses the [pytest](https://docs.pytest.org/en/stable/) test runner:

```bash
uv run pytest [pytest-args]
```

### Linting and formatting

Continuous integration runs [pylint](https://pylint.readthedocs.io/) as well
as [ruff](https://docs.astral.sh/ruff/) (via pre-commit) over the codebase.
To run the same checks locally:

```bash
uv run pylint -rn qiskit_ionq test
uv run ruff check
uv run ruff format
```

### Building the documentation

To build the API reference documentation with Sphinx and view it locally:

```bash
uv run make html
open build/html/index.html
```
