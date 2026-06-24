# Golden Path

This section outlines the recommended "golden path" for developing, building, and contributing to ssl_simulator. It covers the toolchain choices, development environment setup, architectural decisions, and step-by-step tutorials to ensure a standardized, hassle-free development experience.

## The SSL Toolchain

We have carefully selected tools that prioritize speed, reproducibility, and developer experience.

- **Dependency Management (`uv`)**: We use [uv](https://github.com/astral-sh/uv) as our primary package manager. It is written in Rust and is significantly faster than `pip` or `pip-tools`. It handles virtual environment creation and dependency resolution with a pinning strategy that ensures everyone on the team has the exact same environment.
- **Task Runner (`just`)**: Instead of `make`, we use [just](https://github.com/casey/just). It provides a cleaner syntax, better error messages, and is specifically designed for project-specific tasks. See our commands with `just --list`.
- **Build System (`hatch`)**: We use [Hatch](https://hatch.pypa.io/) as our build backend. It follows modern Python packaging standards (PEP 517/660) and integrates [hatch-vcs](https://github.com/ofek/hatch-vcs) to automatically derive version numbers from current Git tags.
- **Automation (`Gitea Workflows`)**: CI/CD is managed via Gitea Workflows (fully compatible with GitHub Actions). These pipelines automatically run linting, tests, and security scans on every push, ensuring the main branch is always stable.
- **Development Environment (`DevContainer`)**: For a "zero-config" setup, we provide a `.devcontainer` configuration. If you use VS Code, opening the project in a container automatically installs all system dependencies (C++ compilers, Eigen3, etc.) and project tools.
- **Documentation (`MkDocs` & `mkdocstrings`)**: Beautiful documentation is generated from Markdown using [MkDocs](https://www.mkdocs.org/) with the Material theme. Reference documentation is automatically pulled from your Python source code via `mkdocstrings`.
- **Quality Assurance (`Ruff` & `Ty`)**:
    - [Ruff](https://github.com/astral-sh/ruff) replaces several tools (Flake8, isort, Black) with a single, lightning-fast Rust-based engine for linting and formatting.
    - `Ty` (a wrapper around `mypy`) ensures strict static type checking to catch bugs before they reach production.
- **Security (`Semgrep`)**:
    - [Semgrep](https://semgrep.dev/) scans for common security vulnerabilities and anti-patterns.
- **Local Enforcement (`pre-commit`)**: We use Git hooks via [pre-commit](https://pre-commit.com/) to run a subset of checks (formatting, linting) before you even commit your code, preventing small errors from reaching the CI.

## Quick Golden Path (clone → dev install → test → docs)

Follow these copy-paste commands for a minimal developer setup (Linux):

```bash
# Clone
git clone https://github.com/jesusBV20/ssl_simulator.git
cd ssl_simulator

# Install system prerequisites (see prerequisites page for details)
sudo apt update && sudo apt install -y build-essential cmake python3.10-dev python3.10-venv libeigen3-dev ninja-build

# One-command setup (locks deps, creates .venv, installs everything)
just setup

# Run quality checks
just lint
just security
just typecheck
just test

# Build docs locally (when docs are enabled)
just docs-build
```

## Just Tasks Reference

All project tasks are managed via `just` - a command runner designed for project-specific workflows. Run `just --list` or simply `just` to see all available commands.

### Setup & Environment

**`just setup`** - One-command development environment setup
- Locks dependencies with `uv lock`
- Creates/updates `.venv` virtual environment
- Installs dev, lint, tests, type-checking, and pre-commit dependencies
- **Use this first** after cloning the project

```bash
just setup
```

**`just sync`** - Sync all optional dependency groups
- Installs ALL extras (dev, lint, tests, type-checking, pre-commit, docs, examples, etc.)
- Useful when you need the complete environment

### Testing

**`just test`** - Run full test suite with coverage
- Executes all tests via `tox`
- Generates coverage reports (terminal + XML)
- **This is what CI runs** - make sure it passes before pushing

**`just test-fast`** - Quick test run
- Runs tests in parallel with `pytest-xdist`
- Skips slow tests (marked with `@pytest.mark.slow`)
- Perfect for rapid iteration during development

**`just test-one TEST`** - Run specific test
- Runs a single test matching the pattern
- Example: `just test-one test_parser` runs all tests with "test_parser" in the name

**`just test-multi-py`** - Test across Python versions
- Runs tests on Python 3.10, 3.11, 3.12, 3.13, and 3.14
- Each version gets its own isolated tox environment
- **Use before releases** to ensure compatibility
- Note: Requires all Python versions installed (use `pyenv` or system packages)

### Code Quality

**`just lint`** - Run linting and formatting checks
- Checks code style with `ruff check`
- Verifies formatting with `ruff format --check`
- **Must pass** before committing

**`just typecheck`** - Run static type analysis
- Uses `ty` (mypy wrapper) to validate type hints
- Catches type errors before runtime
- All public APIs must be fully typed

**`just security`** - Run security scans
- Executes `semgrep` with CI rules + custom `.semgrep.yml`
- Detects vulnerabilities, anti-patterns, and security issues
- Zero findings required for PR approval

**`just pre-commit`** - Run pre-commit hooks manually
- Same hooks that run on `git commit`
- Useful for checking staged changes before committing

**`just check-all`** - Full CI simulation
- Runs lint, security, test, typecheck, and pre-commit
- **Run this before pushing** to ensure CI will pass
- Saves time by catching issues locally

### Building & Releasing

**`just build`** - Build package (sdist + wheel)
- Creates distributable packages in `dist/`
- Uses modern PEP 517 build standards
- Version automatically derived from git tags

**`just build-release`** - Build multiplatform wheels
- Uses `cibuildwheel` for Linux/macOS/Windows wheels
- Only needed for projects with compiled extensions

**`just publish`** - Publish to PyPI with `uv`
- Requires `UV_PUBLISH_TOKEN` and `UV_PUBLISH_URL` env vars
- Modern uv-based publishing

**`just publish-ci`** - Publish to PyPI with `twine`
- Requires `TWINE_USERNAME` and `TWINE_PASSWORD` env vars
- Traditional twine-based publishing (CI-friendly)

### Documentation


**`just docs`** - Start documentation server
- Serves docs at `http://localhost:8000`
- Live-reload on file changes
- Perfect for writing and previewing docs

**`just docs-build`** - Build static documentation
- Generates static HTML in `site/`
- Same output as deployed docs
- Validates all links and references

**`just validate-docs`** - Validate documentation structure
- Checks for broken links, missing images, etc.
- Runs automatically in CI


### Utilities

**`just clean`** - Remove build artifacts
- Deletes `dist/`, `build/`, `.pytest_cache`, etc.
- Cleans up generated files
- Run when switching branches or after failed builds

**`just clean-docs`** - Remove documentation build artifacts
- Deletes `site/` directory
- Useful when docs build is stale

**`just list`** - List all tox environments
- Shows available tox test environments
- Includes Python version-specific envs

**`just example`** - Run basic usage example
- Executes `examples/basic_usage.py`
- Quick smoke test to verify package works

**`just act`** - Test CI workflows locally
- Uses `act` to run GitHub Actions/Gitea Workflows locally
- Catches CI issues before pushing
- Requires Docker

## Typical Development Workflows

### First-time Setup

```bash
# Clone and setup
git clone https://github.com/Swarm-Systems-Lab/ssl-simulator
cd ssl_simulator
just setup

# Verify everything works
just test
```

### Daily Development Loop

```bash
# 1. Pull latest changes
git pull

# 2. Update dependencies if needed
just setup

# 3. Create feature branch
git checkout -b feature/my-feature

# 4. Make changes to src/ssl_simulator/
# ...

# 5. Run tests frequently
just test-fast

# 6. Run specific test you're working on
just test-one test_my_feature

# 7. Before committing, run quality checks
just lint
just typecheck

# 8. Commit (pre-commit hooks run automatically)
git add .
git commit -m "feat: add my feature"

# 9. Before pushing, run full CI simulation
just check-all

# 10. Push
git push origin feature/my-feature
```

### Before Releasing

```bash
# 1. Ensure all tests pass on all Python versions
just test-multi-py

# 2. Run full quality suite
just check-all

# 3. Build package locally to verify
just build

# 4. Test the built package
cd dist/
pip install ssl_simulator-*.whl
python -c "import ssl_simulator; print(ssl_simulator.__version__)"

# 5. Tag and push (CI handles the rest)
git tag v1.0.0
git push --tags
```

### Debugging Test Failures

```bash
# Run failed test in isolation
just test-one test_specific_failure

# Run with verbose output
uv run pytest tests/test_file.py::test_function -vv

# Drop into debugger on failure
uv run pytest tests/test_file.py --pdb

# Run with print statements visible
uv run pytest tests/test_file.py -s
```

### Working on Documentation

```bash
# Start doc server
just docs

# Edit docs in docs/
# Browser auto-refreshes at http://localhost:8000

# Validate before committing
just docs-build
just validate-docs
```
