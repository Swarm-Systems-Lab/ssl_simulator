# Contributing

We welcome contributions to ssl_simulator! This guide outlines how to contribute effectively following the **SSL Golden Path** standards.

## Before You Start

ssl_simulator follows the Swarm Systems Lab (SSL) Golden Path, which means:

- ✅ All code must pass linting, type checking, and tests
- ✅ Pre-commit hooks enforce code quality
- ✅ CI/CD validates all changes automatically
- ✅ Documentation is required for all public APIs
- ✅ Changes should include tests

## Development Setup

### Quick Start

Follow these exact steps to set up your development environment:

```bash
# Fork and clone the repository
git clone https://gitea.lyapunov.local/Swarm-Systems-Lab/ssl-simulator
cd ssl_simulator

# Setup environment (installs uv if needed, creates .venv, installs deps)
just setup

# Verify everything works
just test
```

**What just setup does**:
1. Locks dependencies (`uv lock`)
2. Creates virtual environment in `.venv/`3. Installs development dependencies (dev, lint, tests, type-checking, pre-commit)
4. Installs the package in editable mode

**Available just tasks** - run `just --list` to see all:
- `just test` - Run full test suite (what CI runs)
- `just test-fast` - Quick parallel tests (skip slow tests)
- `just test-multi-py` - Test on Python 3.10-3.14
- `just lint` - Check code style
- `just typecheck` - Check types
- `just security` - Run security scans
- `just check-all` - Full CI simulation (run before pushing!)
- `just build` - Build package
- `just docs` - Start doc server at http://localhost:8000


See the [Golden Path](golden-path.md#just-tasks-reference) for a complete reference of all tasks.

### What `just setup` Does

1. Installs `uv` package manager (if not already installed)
2. Creates a virtual environment in `.venv/`
3. Syncs development dependencies from `pyproject.toml`
4. Installs the package in editable mode (`pip install -e .`)
5. Initializes pre-commit hooks for quality checks

### Manual Setup (Without `just`)

If you prefer not to use `just`:

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create and sync environment
uv venv
source .venv/bin/activate
uv sync --group dev

# Install package in editable mode
uv run pip install -e .

# Setup pre-commit hooks
uv run pre-commit install
```


### DevContainer Option

For a zero-configuration setup:

1. Install Docker and VS Code with Dev Containers extension
2. Open the project in VS Code
3. Click "Reopen in Container"
4. Wait for container to build
5. Run `just setup` in the integrated terminal

This gives you a complete development environment with all system dependencies pre-installed.


## Contribution Workflow

### 1. Create an Issue

Before making changes:

1. Check if an issue already exists
2. If not, create one describing:
   - The problem or feature
   - Proposed solution
   - Any breaking changes
3. Wait for discussion/approval (for major changes)

### 2. Fork and Branch

```bash
# Fork on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/ssl_simulator
cd ssl_simulator

# Add upstream remote
git remote add upstream https://gitea.lyapunov.local/Swarm-Systems-Lab/ssl-simulator

# Create feature branch
git checkout -b feature/your-feature-name
# Or for bug fixes:
git checkout -b fix/issue-description
```

### 3. Make Your Changes

```bash
# Edit code in src/ssl_simulator/
# Add tests in tests/
# Update docs if needed

# Run quality checks frequently during development
just test-fast     # Quick test run (parallel, skip slow)
just test-one test_my_feature  # Run specific test
just lint          # Check code style
just typecheck     # Check types

# Before committing, run the full suite
just test          # Full test suite with coverage
just check-all     # Everything (lint, security, test, typecheck, pre-commit)
```

**Pro tip**: Use `just test-fast` during development for rapid iteration, then `just check-all` before pushing to catch everything CI will check.

### 4. Commit Your Changes

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```bash
# Stage changes
git add .

# Commit with conventional commit message
git commit -m "feat: add new feature"
# Or: fix:, docs:, style:, refactor:, test:, chore:
```

Pre-commit hooks will automatically:
- Format code with `ruff format`
- Check linting with `ruff check`
- Validate commit message format
- Run other quality checks

If hooks fail, fix the issues and commit again.

### 5. Push and Create Pull Request

```bash
# Push to your fork
git push origin feature/your-feature-name

# Create Pull Request on GitHub
# Fill in the PR template with:
# - Description of changes
# - Related issue number
# - Testing performed
# - Breaking changes (if any)
```

### 6. Code Review

- CI will automatically run all quality checks
- Maintainers will review your code
- Address any feedback
- Update your branch if needed:
  ```bash
  git fetch upstream
  git rebase upstream/main
  git push --force-with-lease
  ```

## Code Quality Standards

### Linting and Formatting

We use [Ruff](https://github.com/astral-sh/ruff) - extremely fast Python linter and formatter:

```bash
# Check for issues
uv run ruff check .

# Auto-fix issues
uv run ruff check --fix .

# Format code
uv run ruff format .

# Or use just
just lint
```

**Configuration**: See `[tool.ruff]` in `pyproject.toml`

**Standards**:
- Line length: 100 characters
- Target Python: 3.10+
- Quote style: double quotes
- Docstring style: Google/NumPy

### Type Checking

We use `ty` (wrapper around `mypy`) for static type analysis:

```bash
# Type check entire project
uv run ty check

# Or use just
just typecheck
```

**Requirements**:
- All public functions must have type hints
- Use `typing` module for complex types
- Avoid `Any` where possible

Example:

```python
from typing import List, Optional, Dict

def process_items(
    items: List[str],
    config: Optional[Dict[str, int]] = None
) -> Dict[str, int]:
    """Process items according to configuration.

    Args:
        items: List of items to process
        config: Optional configuration dictionary

    Returns:
        Dictionary mapping items to counts
    """
    ...
```

### Security Scanning

We use [Semgrep](https://semgrep.dev/) for security analysis:

```bash
# Run security scan
just security

# Or directly
uv run semgrep --config=auto src/
```

This catches:
- SQL injection vulnerabilities
- XSS risks
- Unsafe deserialization
- Common security anti-patterns

## Testing

### Writing Tests

All changes **must** include tests. We use `pytest`:

```python
# tests/test_feature.py
import pytest
from ssl_simulator import my_function

def test_my_function_basic():
    """Test basic functionality."""
    result = my_function(input_data)
    assert result == expected_output

def test_my_function_edge_case():
    """Test edge case handling."""
    with pytest.raises(ValueError):
        my_function(invalid_input)

def test_my_function_with_fixture(sample_data):
    """Test using fixture."""
    result = my_function(sample_data)
    assert len(result) > 0
```

### Running Tests

```bash
# Run all tests
just test

# Or with uv
uv run pytest

# Run specific test file
uv run pytest tests/test_feature.py

# Run specific test function
uv run pytest tests/test_feature.py::test_my_function_basic

# Run with coverage
uv run pytest --cov=ssl_simulator --cov-report=html

# Run in parallel (faster)
uv run pytest -n auto
```

### Test Guidelines

- ✅ Test one thing per test function
- ✅ Use descriptive test names
- ✅ Include docstrings explaining what's tested
- ✅ Test both success and failure cases
- ✅ Use fixtures for common setup
- ✅ Keep tests fast (< 1 second each)
- ✅ Use parametrize for multiple inputs:

```python
@pytest.mark.parametrize("input,expected", [
    (1, 2),
    (2, 4),
    (3, 6),
])
def test_double(input, expected):
    assert double(input) == expected
```


## Documentation

### Docstrings

All public APIs require docstrings using Google or NumPy style:

```python
def calculate_score(
    items: List[str],
    weights: Dict[str, float],
    normalize: bool = True
) -> float:
    """Calculate weighted score for items.

    This function computes a weighted sum of items based on
    provided weights. Optionally normalizes the result.

    Args:
        items: List of item identifiers
        weights: Dictionary mapping items to weights
        normalize: Whether to normalize the final score

    Returns:
        Calculated score as a float

    Raises:
        ValueError: If items list is empty
        KeyError: If an item is not in weights dict

    Examples:
        >>> items = ["a", "b", "c"]
        >>> weights = {"a": 1.0, "b": 2.0, "c": 3.0}
        >>> calculate_score(items, weights)
        6.0
    """
    ...
```

### Documentation Files

- Update `docs/` markdown files when adding features
- Add examples to `docs/examples.md`
- Update `docs/api.md` if adding new modules
- Keep `README.md` up to date

### Building Docs Locally

```bash
# Start doc server with hot-reload
just docs

# Or directly
uv sync --group docs
uv run mkdocs serve
```

Visit [http://localhost:8000](http://localhost:8000)


## Commit Message Guidelines

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Code style (formatting, no logic change)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks (deps, build, etc.)
- `perf`: Performance improvement
- `ci`: CI/CD changes

**Examples**:

```
feat(api): add batch processing endpoint

Implement batch processing for multiple items at once.
This improves throughput for large workloads.

Closes #123
```

```
fix(parser): handle edge case with empty input

Previously the parser would crash on empty strings.
Now it returns an empty result.
```

```
docs: update installation instructions

Add troubleshooting section for common issues.
```

## Pull Request Guidelines

### PR Requirements

Before submitting:

- ✅ All tests pass (`just test`)
- ✅ Code is linted and formatted (`just lint`)
- ✅ Types are checked (`just typecheck`)
- ✅ No security issues (`just security`)
- ✅ Documentation is updated
- ✅ Changelog is updated (if applicable)

### PR Template

Your PR should include:

**Description**
- What does this PR do?
- Why is this change needed?

**Related Issue**
- Fixes #123
- Related to #456

**Type of Change**
- [ ] Bug fix (non-breaking)
- [ ] New feature (non-breaking)
- [ ] Breaking change
- [ ] Documentation update

**Testing**
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Manual testing performed

**Checklist**
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex code
- [ ] Documentation updated
- [ ] No new warnings generated
- [ ] Tests added that prove fix/feature works
- [ ] Dependent changes merged

### CI Checks

All PRs must pass:

1. **Linting**: Ruff checks code style
2. **Type Checking**: Mypy validates types
3. **Tests**: pytest runs all tests
4. **Security**: Semgrep scans for vulnerabilities
5. **Coverage**: Test coverage must not decrease
6. **Build**: Package builds successfully

If CI fails, check the logs and fix issues locally:

```bash
# Run full CI simulation locally
just ci-local

# This runs the same checks as CI
```

## Code Style Guidelines

### Python Standards

Follow PEP 8 with these additions:

```python
# Good: Clear, typed, documented
def process_data(
    input_file: Path,
    output_format: str = "json"
) -> Dict[str, Any]:
    """Process data from input file.

    Args:
        input_file: Path to input file
        output_format: Output format (json/yaml/xml)

    Returns:
        Processed data as dictionary
    """
    ...

# Bad: No types, no docs, unclear naming
def proc(f, fmt="json"):
    ...
```

### Project Structure

Follow the `src/` layout:

```
src/ssl_simulator/
├── __init__.py          # Package initialization
├── core.py              # Core functionality
├── utils.py             # Utility functions
└── subpackage/
    ├── __init__.py
    └── module.py

tests/
├── test_core.py         # Mirror src structure
├── test_utils.py
└── subpackage/
    └── test_module.py
```

### Imports

Organize imports per PEP 8:

```python
# Standard library
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional

# Third-party
import numpy as np
import pandas as pd

# Local
from ssl_simulator.core import CoreClass
from ssl_simulator.utils import helper_function
```

Ruff will automatically sort and organize imports.

## Architecture Guidelines

### Adding New Features

1. **Plan**: Discuss design in an issue first
2. **API Design**: Keep APIs simple and consistent
3. **Implementation**: Break into small, testable units
4. **Testing**: Test at multiple levels (unit, integration)
5. **Documentation**: Document behavior and examples

### Code Organization

- Keep modules focused and cohesive
- Use clear, descriptive names
- Avoid deep inheritance hierarchies
- Prefer composition over inheritance
- Keep functions small and single-purpose

### Performance

- Profile before optimizing
- Use appropriate data structures
- Avoid premature optimization
- Document performance characteristics
- Add benchmarks for critical paths

## Release Process

Releases are automated via CI/CD:

1. **Version Bump**: Create git tag
   ```bash
   git tag v1.0.0
   git push --tags
   ```

2. **CI Builds**: Automatically triggered
   - Runs all quality checks
   - Builds distribution packages
   - Publishes to PyPI
   - Deploys documentation

3. **Changelog**: Update `CHANGELOG.md` (if present)

Version follows [Semantic Versioning](https://semver.org/):
- `MAJOR.MINOR.PATCH`
- MAJOR: Breaking changes
- MINOR: New features (backward compatible)
- PATCH: Bug fixes (backward compatible)

## Getting Help

### Resources

- 📖 [Golden Path Documentation](golden-path.md)
- 🔧 [Usage Guide](usage.md)
- 🐛 [Troubleshooting](troubleshooting.md)
- 💬 [GitHub Discussions](https://gitea.lyapunov.local/Swarm-Systems-Lab/ssl-simulator/discussions)
- 🐛 [Issue Tracker](https://gitea.lyapunov.local/Swarm-Systems-Lab/ssl-simulator/issues)

### Community

- **Questions**: Use GitHub Discussions
- **Bugs**: Open an issue with reproduction steps
- **Features**: Open an issue with use case and proposal
- **Security**: Email jesbauti20@gmail.com (do not open public issue)

## Code of Conduct

We are committed to providing a welcoming and inclusive environment. We follow the [Python Community Code of Conduct](https://www.python.org/psf/conduct/).

**Expected Behavior**:
- Be respectful and professional
- Welcome newcomers
- Accept constructive criticism gracefully
- Focus on what's best for the community

**Unacceptable Behavior**:
- Harassment or discrimination
- Trolling or insulting comments
- Public or private harassment
- Publishing others' private information

## License

By contributing to ssl_simulator, you agree that your contributions will be licensed under the **MIT** license.

---

**Thank you for contributing!** 🎉 Your efforts help make ssl_simulator better for everyone.
