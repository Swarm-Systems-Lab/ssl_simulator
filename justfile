# ============================================================================
# Setup & Environment
# ============================================================================

# Setup the development environment (dev deps only by default)
setup:
    #!/usr/bin/env bash
    set -euo pipefail
    if ! command -v ssl-pydev >/dev/null 2>&1; then
        uv tool install ssl_pydev
    fi
    if ! command -v ssl-pydev >/dev/null 2>&1; then
        echo "error: ssl-pydev was installed but is not on PATH." >&2
        echo "check https://github.com/Swarm-Systems-Lab/ssl_pydev#install" >&2
        exit 1
    fi
    uv lock
    ssl-pydev setup-env

# Sync all dependency groups
sync:
    uv sync --frozen --all-extras

# Build and install the package in development mode
build:
    uv build

# ============================================================================
# Development & Code Quality
# ============================================================================

# Run linting and formatting checks
lint:
    uv run ruff format .
    uv run ruff check . --fix

# Run type checking
typecheck:
    uv run ty check src/ssl_simulator

# Run pre-commit hooks on all files
pre-commit:
    uv run pre-commit run --all-files --show-diff-on-failure

# Run security scans with semgrep
security:
    ssl-pydev security

# ============================================================================
# Testing
# ============================================================================

# Run all tests with tox (recommended)
test:
    uv run tox -e tests

# Run tests in parallel, skip slow tests
test-fast:
    uv run tox -e tests-fast

# Run specific test by name
test-one TEST:
    uv run pytest tests/ -v -k "{{TEST}}"

# Run tests across multiple Python versions
test-multi-py:
    uv run tox -e py312,py313,py314

# List all available tox environments
list:
    uv run tox list

# ============================================================================
# Publishing
# ============================================================================

# Publish artifacts with uv (requires UV_PUBLISH_* env vars)
publish:
    ssl-pydev publish

# Publish artifacts with twine (CI-friendly; requires TWINE_* env vars)
publish-ci:
    ssl-pydev publish-ci

# ============================================================================
# Project Generation & CI Testing
# ============================================================================

# Generate a test project locally for manual inspection
generate OUTPUT="./.test_projects/just_project":
    uv run copier copy --vcs-ref=HEAD --defaults . {{OUTPUT}}
    git -C {{OUTPUT}} init
    git -C {{OUTPUT}} add .
    git -C {{OUTPUT}} commit -m "Initial commit"
    @echo "- Test project generated at: {{OUTPUT}}"
    @echo ""
    @echo "To test it:"
    @echo "  cd {{OUTPUT}}"
    @echo "  just setup"
    @echo "  just test"

# Test GitHub Actions workflows locally with act
act:
    ssl-pydev act

# ============================================================================
# Utilities
# ============================================================================

# Clean all build and test artifacts
clean:
    rm -rf build dist *.egg-info .pytest_cache .ruff_cache __pycache__ .venv cov.xml .coverage .tox .mypy_cache .test_projects
    rm -rf tests/.pytest_cache
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# ============================================================================
# Composite Commands
# ============================================================================

# Full CI simulation (do this before pushing!)
check-all: lint security typecheck pre-commit test
    @echo " - All checks passed!"

# Show help information
help:
    @echo "Development Commands"
    @echo "========================================"
    @echo ""
    @echo "Setup & Environment:"
    @echo "  just setup        - Install dependencies with uv"
    @echo "  just sync         - Sync all dependency groups"
    @echo "  just build        - Build package in development mode"
    @echo ""
    @echo "Development & Code Quality:"
    @echo "  just lint         - Run ruff formatting and linting"
    @echo "  just typecheck    - Run type checking with ty"
    @echo "  just pre-commit   - Run pre-commit hooks on all files"
    @echo "  just security     - Run security scans with semgrep"
    @echo ""
    @echo "Testing:"
    @echo "  just test         - Run all tests with tox (recommended)"
    @echo "  just test-fast    - Run tests in parallel, skip slow tests"
    @echo "  just test-one TEST - Run specific test by name"
    @echo "  just test-multi-py - Run tests across Python 3.12-3.14"
    @echo "  just list         - List all tox environments"
    @echo ""
    @echo "Publishing:"
    @echo "  just publish      - Publish with uv"
    @echo "  just publish-ci   - Publish with twine (CI-friendly)"
    @echo ""
    @echo "Project Generation & CI:"
    @echo "  just generate     - Generate test project for manual inspection"
    @echo "  just act          - Test CI workflows locally"
    @echo ""
    @echo "Utilities:"
    @echo "  just clean        - Clean build and test artifacts"
    @echo "  just check-all    - Run full CI simulation"
    @echo "  just help         - Show this help message"
    @echo ""
    @echo "Note: These commands are for TEMPLATE DEVELOPMENT only."
    @echo "      Generated projects get their own justfile."
