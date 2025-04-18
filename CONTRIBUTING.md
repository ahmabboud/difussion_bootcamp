# Contributing to bootcamp_template

Thanks for your interest in contributing to the bootcamp_template!

To submit PRs, please fill out the PR template along with the PR. If the PR fixes an issue, don't forget to link the PR to the issue!

## Pre-commit hooks

Once the python virtual environment is setup, you can run pre-commit hooks using:

```bash
pre-commit run --all-files
```

## Coding guidelines

For docstrings we use [numpy format](https://numpydoc.readthedocs.io/en/latest/format.html).

For code style, we recommend the [PEP 8 style guide](https://peps.python.org/pep-0008/).
Pre-commit hooks apply [black](https://black.readthedocs.io/en/stable/the_black_code_style/current_style.html) code formatting.

We use [ruff](https://docs.astral.sh/ruff/) for code formatting and static code
analysis. Ruff checks various rules including [flake8](https://docs.astral.sh/ruff/faq/#how-does-ruff-compare-to-flake8). The pre-commit hooks show errors which you need to fix before submitting a PR.
