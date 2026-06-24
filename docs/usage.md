# Usage Guide

This guide covers how to use ssl_simulator effectively, from installation to common workflows.

## Installation

### For End Users

Install the latest stable version from PyPI:

```bash
pip install ssl_simulator
```

Or with `uv` (faster):

```bash
uv pip install ssl_simulator
```

### For Developers

Clone and set up the development environment:

```bash
# Clone the repository
git clone https://github.com/Swarm-Systems-Lab/ssl-simulator
cd ssl_simulator

# One-command setup
just setup

# Verify installation
uv run python -c "import ssl_simulator; print(ssl_simulator.__version__)"
```

## Basic Usage

### Importing the Package

```python
import ssl_simulator

# Access package version
print(ssl_simulator.__version__)

# Import specific modules
from ssl_simulator.core import CoreClass
from ssl_simulator.utils import helper_function
```

### Common Workflows

#### Example 1: Basic Usage

```python
from ssl_simulator import example_function

# Use the function
result = example_function(input_data)
print(result)
```

#### Example 2: Advanced Usage

```python
from ssl_simulator.core import CoreClass

# Initialize with configuration
processor = CoreClass(
    param1="value1",
    param2=42
)

# Process data
output = processor.process(data)
```

## Development Workflows

### Running Tests

ssl_simulator uses `just` for task automation. Here are the most common test commands:

```bash
# Run all tests with coverage (what CI runs)
just test

# Quick test run (parallel, skips slow tests)
just test-fast

# Run a specific test
just test-one test_my_feature

# Test across all supported Python versions
just test-multi-py
```

**Direct pytest usage** (if you need more control):

```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_core.py

# Run specific test function
uv run pytest tests/test_core.py::test_specific_function

# Run with verbose output
uv run pytest -vv

# Run with coverage report
uv run pytest --cov=ssl_simulator --cov-report=html

# Run in parallel (faster)
uv run pytest -n auto

# Run only failed tests from last run
uv run pytest --lf

# Drop into debugger on failure
uv run pytest --pdb
```

### Code Quality Checks

Ensure your code meets quality standards before committing:

```bash
# Run linting (ruff check + format check)
just lint

# Run type checking (mypy)
just typecheck

# Run security scanning (semgrep)
just security

# Run pre-commit hooks manually
just pre-commit

# Run everything (what CI does)
just check-all
```

**Manual ruff usage** (for auto-fixes):

```bash
# Auto-fix linting issues
uv run ruff check --fix .

# Format code
uv run ruff format .

# Check specific file
uv run ruff check src/ssl_simulator/myfile.py
```

### Building the Package

```bash
# Build sdist and wheel
just build

# Output appears in dist/
ls -lh dist/
```


### Working with Documentation

```bash
# Start local documentation server (http://localhost:8000)
just docs

# Build static documentation
just docs-build

# Validate documentation
just validate-docs
```

Documentation is written in Markdown in the `docs/` directory. The API reference is automatically generated from docstrings using `mkdocstrings`.


## Environment Management

### Using uv (Recommended)

`uv` is a fast Python package manager written in Rust. It's the recommended way to manage dependencies:

```bash
# Create/sync environment (handled by just setup)
uv sync

# Install additional packages
uv add package-name

# Install specific extras
uv sync --extra dev
uv sync --extra tests

# Install all extras
uv sync --all-extras

# Run commands in the environment
uv run python script.py
uv run pytest

# Update dependencies
uv lock --upgrade
```

### Virtual Environment Activation

While `uv run` is convenient, you can also activate the virtual environment:

```bash
# Activate (Linux/macOS)
source .venv/bin/activate

# Activate (Windows)
.venv\Scripts\activate

# Now you can run commands directly
python script.py
pytest

# Deactivate when done
deactivate
```

### Dependency Management

Dependencies are defined in `pyproject.toml`:

- **Runtime dependencies**: `[project.dependencies]`
- **Optional extras**: `[project.optional-dependencies]`
  - `dev`: Development tools (tox, ruff, etc.)
  - `tests`: Testing tools (pytest, pytest-cov, etc.)
  - `lint`: Linting tools (ruff, semgrep)
  - `type-checking`: Type checking tools (ty/mypy)
  - `pre-commit`: Pre-commit hooks

  - `docs`: Documentation tools (mkdocs, mkdocstrings)


  - `examples`: Example dependencies (matplotlib, IPython, etc.)


```bash
# Lock dependencies (creates/updates uv.lock)
uv lock

# Update to latest compatible versions
uv lock --upgrade

# Add a new dependency
uv add package-name

# Add a dev dependency
uv add --group dev package-name
```

## Configuration

### Tool Configuration

All tool configuration is in `pyproject.toml`:

- **Ruff**: `[tool.ruff]` - linting and formatting
- **Pytest**: `[tool.pytest.ini_options]` - test configuration
- **Coverage**: `[tool.coverage.*]` - coverage settings
- **Tox**: `[tool.tox.*]` - test automation
- **Hatch**: `[tool.hatch.*]` - build configuration

### Environment Variables

Some tasks use environment variables:

- `UV_PUBLISH_TOKEN`: PyPI token for `just publish`
- `UV_PUBLISH_URL`: PyPI URL for `just publish`
- `TWINE_USERNAME`: PyPI username for `just publish-ci`
- `TWINE_PASSWORD`: PyPI password for `just publish-ci`

## Troubleshooting

### Common Issues

**Tests fail locally but pass in CI**
```bash
# Clean environment and rebuild
just clean
rm -rf .venv uv.lock
just setup
just test
```

**Linting failures**
```bash
# Auto-fix most issues
uv run ruff check --fix .
uv run ruff format .
```

**Type checking errors**
```bash
# Run type checker with verbose output
uv run ty check --verbose
```

**Import errors**
```bash
# Ensure package is installed in editable mode
uv sync

# Or manually
uv pip install -e .
```

**Slow tests**
```bash
# Run tests in parallel
just test-fast

# Or directly with pytest
uv run pytest -n auto
```

### Getting Help

- 📖 **Documentation**: Check [Golden Path](golden-path.md) and [Contributing](contributing.md)
- 🐛 **Issues**: Search or create an issue at [https://github.com/Swarm-Systems-Lab/ssl-simulator/issues](https://github.com/Swarm-Systems-Lab/ssl-simulator/issues)
- 💬 **Discussions**: Ask questions at [https://github.com/Swarm-Systems-Lab/ssl-simulator/discussions](https://github.com/Swarm-Systems-Lab/ssl-simulator/discussions)
- 📧 **Email**: Contact jesbauti20@gmail.com

## Next Steps

- Review the [API Reference](api.md) for detailed module documentation
- Check out [Contributing Guidelines](contributing.md) to contribute
- Explore the [Golden Path](golden-path.md) to understand the development philosophy

## Quick Reference

### Essential Commands

| Command | Purpose |
|---------|----------|
| `just setup` | Initial setup (run once after cloning) |
| `just test` | Run full test suite |
| `just test-fast` | Quick test run (parallel, skip slow) |
| `just lint` | Check code style |
| `just typecheck` | Check types |
| `just security` | Run security scans |
| `just check-all` | Full CI simulation |
| `just build` | Build package |

| `just docs` | Start doc server |

| `just clean` | Remove build artifacts |

### Quick Tips

- 🔧 **List all commands**: `just --list` or `just`
- 🧪 **Run single test**: `just test-one <test_name>`
- 🔍 **Verbose pytest**: `uv run pytest -vv`
- 🐛 **Debug test**: `uv run pytest --pdb`
- ⚡ **Parallel tests**: `uv run pytest -n auto`
- 🔄 **Re-run failed**: `uv run pytest --lf`
