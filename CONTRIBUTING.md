# Contributing

First read the overall project contributing guidelines. These are all
included in the qiskit documentation:

https://qiskit.org/documentation/contributing_to_qiskit.html

## Contributing to Qiskit IonQ Provider

### Getting started

All contributions are welcome.

Please start the contribution process by opening a new issue or making a comment on an existing issue briefly explaining what you intend to address and how in [this repo's issues tracker](https://github.com/qiskit-community/qiskit-ionq-provider/issues). This helps us understand your intent and guide your approach before you take the time to write any code, and ensures multiple people aren't accidentally working on the same thing.

### Making a pull request

When you're ready to make a pull request, please make sure the following is true:

1. The code matches the project's code style
2. The documentation, including any docstrings for changed methods, has been updated
3. If appropriate for your change, that new tests have been added to address any new functionality, or that existing tests have been updated as appropriate
4. All of the tests (new and old) still pass!

### Running the tests

This package uses the [pytest](https://docs.pytest.org/en/stable/) test runner.

To use pytest directly, just run:

```bash
pytest [pytest-args]
```

Alternatively, you may also use setuptools integration by running tests through `setup.py`, e.g.:

```bash
python setup.py test --addopts="[pytest-args]"
```
