# Contributing to PrivacyGuard

Thanks for your interest in contributing! Here's how to get started.

## Development Setup

```bash
git clone https://github.com/laythayache/privacyguard.git
cd privacyguard
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest                          # all tests
pytest tests/test_anonymizer.py # single file
pytest -k "test_gaussian"       # by name pattern
pytest --cov=privacyguard       # with coverage
```

## Code Style

We use [Ruff](https://docs.astral.sh/ruff/) for linting and formatting:

```bash
ruff check src/ tests/    # lint
ruff format src/ tests/   # format
```

## Type Checking

```bash
mypy src/privacyguard/
```

## Pull Request Guidelines

1. Fork the repository and create a feature branch from `main`.
2. Add tests for any new functionality.
3. Ensure all tests pass and linting is clean.
4. Write a clear PR description explaining *what* and *why*.
5. Keep PRs focused — one feature or fix per PR.

## Reporting Issues

Open a GitHub issue with:
- A clear title and description
- Steps to reproduce (if applicable)
- Python version, OS, and `pip list` output
