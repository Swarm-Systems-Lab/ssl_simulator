# ssl_simulator

Swarm Systems Lab Python Simulator

⚠️ **Warning:** This project is a work in progress

## Installation

```bash
pip install ssl_simulator
```

## Usage

```python
import ssl_simulator

# Example usage
result = ssl_simulator.hello()
print(result)
```

## Development

This project follows the **SSL Golden Path** for streamlined Python development.

### Quick Setup

```bash
# Clone the repository
git clone https://github.com/Swarm-Systems-Lab/ssl-simulator
cd ssl_simulator

# One-command setup (installs uv, creates .venv, installs dependencies)
just setup

# Verify everything works
just test
```

### Common Commands

All project tasks are managed via `just`. Run `just --list` to see all available commands.

**Essential commands:**
```bash
just setup          # Initial environment setup
just test           # Run full test suite (what CI runs)
just test-fast      # Quick parallel tests (skip slow tests)
just lint           # Check code style
just typecheck      # Check types
just security       # Run security scans
just check-all      # Full CI simulation (run before pushing!)
just build          # Build package

just docs           # Start documentation server

```

**Testing workflows:**
```bash
just test-one test_name      # Run specific test
just test-multi-py           # Test on Python 3.10-3.14
uv run pytest -vv            # Verbose output
uv run pytest --pdb          # Debug on failure
```

### Development Tools

- **Package Manager**: [`uv`](https://github.com/astral-sh/uv) - Fast Rust-based Python package manager
- **Task Runner**: [`just`](https://github.com/casey/just) - Command runner for project tasks
- **Linting**: [`ruff`](https://github.com/astral-sh/ruff) - Fast linter and formatter
- **Type Checking**: `ty` (mypy wrapper) - Static type analysis
- **Testing**: `pytest` with coverage and parallel execution
- **Security**: `semgrep` - Security vulnerability scanning

### Project Structure

```
ssl_simulator/
├── src/ssl_simulator/     # Source code (importable package)
├── tests/                      # Test files (mirrors src structure)

├── docs/                       # Documentation (MkDocs)


├── examples/                   # Usage examples

├── pyproject.toml              # Project metadata and dependencies
├── justfile                    # Task definitions
└── uv.lock                     # Locked dependencies
```


## Documentation


Full documentation is available at [https://github.com/Swarm-Systems-Lab/ssl-simulator/blob/main/docs](https://github.com/Swarm-Systems-Lab/ssl-simulator/blob/main/docs)


**Build locally:**
```bash
just docs        # Start server at http://localhost:8000 with live reload
just docs-build  # Build static documentation
```



## Examples

See the [examples/](examples/) directory for usage examples.

**Run the basic example:**
```bash
just example
# Or directly:
uv run python examples/basic_usage.py
```


## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](docs/contributing.md) for guidelines.

**Before submitting a PR:**
```bash
just check-all  # Runs lint, security, test, typecheck, pre-commit
```

## License

MIT
