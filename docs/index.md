# ssl_simulator

Welcome to the documentation for **ssl_simulator**!

Swarm Systems Lab Python Simulator

## Overview

This project follows the **Swarm Systems Lab (SSL) Golden Path** - a standardized, battle-tested approach to Python project development that emphasizes:

- **⚡ Speed**: Lightning-fast tooling with `uv` and `ruff`
- **🔒 Quality**: Automated linting, testing, and security scanning
- **📦 Reproducibility**: Consistent environments across local dev, CI, and production
- **📚 Documentation**: Auto-generated API docs and comprehensive guides
- **🚀 Developer Experience**: One-command setup, clear task runners, and DevContainer support

## Quick Start

Get up and running in minutes:

```bash
# Clone the repository
git clone https://gitea.lyapunov.local/Swarm-Systems-Lab/ssl-simulator
cd ssl_simulator

# One-command setup (installs uv if needed, creates .venv, installs deps)
just setup

# Run tests to verify everything works
just test

# Start coding!
uv run python -c "import ssl_simulator; print('Ready to go!')"
```

**Next steps:**
- 📖 See all available commands: `just --list`
- 🚀 Learn the workflows: [Usage Guide](usage.md)
- 🛠️ Understand the toolchain: [Golden Path](golden-path.md)


## Documentation Structure

- **[Usage Guide](usage.md)**: Practical usage instructions, common workflows, and just tasks reference
- **[Golden Path](golden-path.md)**: Development approach, toolchain explanations, and complete just tasks documentation
- **[Contributing](contributing.md)**: Guidelines for contributing code, tests, and documentation
- **[API Reference](api.md)**: Complete API documentation auto-generated from docstrings
- **[Troubleshooting](troubleshooting.md)**: Solutions to common issues


## Key Features

✨ **Modern Python Tooling**
:   Built with `uv`, `just`, `ruff`, and `hatch` for a streamlined developer experience

🧪 **Quality Assurance**
:   Pre-configured testing with `pytest`, type checking with `mypy`, and security scanning with `semgrep`

📦 **Standardized Structure**
:   Follows Python best practices with `src/` layout and PEP 517/660 compliance

🔄 **CI/CD Ready**
:   Gitea Workflows (GitHub Actions compatible) for automated testing and deployment


🐳 **DevContainer Support**
:   One-click development environment with all dependencies pre-configured



📚 **Beautiful Documentation**
:   Auto-generated docs with MkDocs Material theme and `mkdocstrings`


## Project Philosophy

ssl_simulator embraces the SSL Golden Path philosophy:

1. **Consistency Over Configuration**: Standardized tooling reduces cognitive load
2. **Automation Over Repetition**: Let CI handle quality checks automatically
3. **Developer Experience First**: Fast tools, clear documentation, easy onboarding
4. **Reproducibility Everywhere**: Identical environments from dev to production

## Common Tasks

Get productive immediately with these essential commands:

```bash
# Development
just setup           # Initial environment setup
just test            # Run full test suite
just test-fast       # Quick parallel tests
just lint            # Check code style
just typecheck       # Check types
just check-all       # Full CI simulation

# Building
just build           # Build package

# Documentation
just docs            # Start doc server (http://localhost:8000)
just docs-build      # Build static docs

```

**Pro tip**: Run `just --list` to see all available commands, or check the [Usage Guide](usage.md) for detailed workflows.

## Next Steps

- 🚀 Read the [Usage Guide](usage.md) for practical examples and workflows
- 🛠️ Explore the [Golden Path](golden-path.md) to understand the development philosophy
- 🤝 Review [Contributing Guidelines](contributing.md) before making changes
- 📚 Browse the [API Reference](api.md) for detailed documentation

## Support & Community

- 🐛 **Issues**: Report bugs or request features on [https://gitea.lyapunov.local/Swarm-Systems-Lab/ssl-simulator](https://gitea.lyapunov.local/Swarm-Systems-Lab/ssl-simulator/issues)
- 📧 **Contact**: jesbauti20@gmail.com
- 📄 **License**: MIT

---

**Ready to dive in?** Start with the [Golden Path](golden-path.md) to set up your development environment!
s
