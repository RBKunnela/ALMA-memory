# Contributing to ALMA

Thank you for your interest in contributing to ALMA! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Pull Request Process](#pull-request-process)
- [Issue Guidelines](#issue-guidelines)
- [Community](#community)

---

## Code of Conduct

We are committed to providing a welcoming and inclusive environment. Please:

- Be respectful and constructive in discussions
- Welcome newcomers and help them get started
- Focus on what is best for the community
- Show empathy towards other community members

---

## Getting Started

### Good First Issues

New to ALMA? Look for issues labeled [`good-first-issue`](https://github.com/RBKunnela/ALMA-memory/labels/good-first-issue). These are specifically selected to be approachable for newcomers.

### Understanding the Codebase

```
alma/
├── core.py              # Main ALMA class and interface
├── types.py             # Data types (Heuristic, Outcome, etc.)
├── config/              # Configuration loading
├── storage/             # Storage backends (SQLite, Azure, File)
├── retrieval/           # Memory retrieval and embeddings
├── learning/            # Learning protocols
├── extraction/          # LLM-powered fact extraction (NEW)
├── graph/               # Graph memory with Neo4j (NEW)
├── mcp/                 # MCP server for Claude integration
├── progress/            # Work item tracking
├── session/             # Session handoff management
├── domains/             # Domain-specific memory schemas
├── harness/             # Agent harness pattern
├── confidence/          # Forward-looking confidence engine
└── initializer/         # Session initialization
```

### Key Concepts

1. **Memory Types**: ALMA has 5 memory types - Heuristics, Outcomes, Preferences, Domain Knowledge, Anti-patterns
2. **Scoped Learning**: Agents can only learn within their defined domains
3. **The Harness Pattern**: Setting → Context → Agent → Memory Schema

---

## How to Contribute

### Types of Contributions

| Type | Description | Difficulty |
|------|-------------|------------|
| Documentation | Fix typos, improve explanations, add examples | Easy |
| Bug Reports | Report issues with clear reproduction steps | Easy |
| Bug Fixes | Fix reported issues | Medium |
| Tests | Add test coverage for existing features | Medium |
| Features | Implement new functionality | Hard |
| Integrations | Add new storage backends, LLM providers | Hard |

### What We Need Most Right Now

1. **Documentation improvements** - Examples, tutorials, explanations
2. **Test coverage** - We need more tests for edge cases
3. **Storage backends** - PostgreSQL, MongoDB, Pinecone integrations
4. **LLM providers** - Ollama, Groq, local models for extraction
5. **Language SDKs** - TypeScript/JavaScript SDK

---

## Development Setup

### Prerequisites

- Python 3.10+
- Git
- (Optional) Docker for testing with Neo4j

### Installation

```bash
# Clone the repository
git clone https://github.com/RBKunnela/ALMA-memory.git
cd ALMA-memory

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pip install pre-commit
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=alma --cov-report=html

# Run specific test file
pytest tests/test_storage.py

# Run tests matching a pattern
pytest -k "test_retrieval"
```

### Code Style

We use:
- **Black** for code formatting
- **isort** for import sorting
- **flake8** for linting
- **mypy** for type checking

```bash
# Format code
black alma/ tests/

# Sort imports
isort alma/ tests/

# Run linter
flake8 alma/ tests/

# Type check
mypy alma/
```

---

## Pull Request Process

### Before You Start

1. Check existing issues and PRs to avoid duplicate work
2. For large changes, open an issue first to discuss the approach
3. Fork the repository and create a branch from `main`

### Branch Naming

Use descriptive branch names:
- `feature/add-postgres-storage`
- `fix/retrieval-cache-bug`
- `docs/improve-quickstart`
- `test/add-learning-tests`

### Making Changes

1. Write clear, documented code
2. Add tests for new functionality
3. Update documentation if needed
4. Ensure all tests pass
5. Run code formatters

### Submitting PR

1. Fill out the PR template completely
2. Link to any related issues
3. Provide a clear description of changes
4. Include screenshots for UI changes
5. Request review from maintainers

### PR Review

- Maintainers will review within 48 hours (usually faster)
- Address feedback promptly
- Be open to suggestions
- Once approved, maintainer will merge

---

## Issue Guidelines

### Bug Reports

Please include:
- ALMA version (`pip show alma-memory`)
- Python version
- Operating system
- Minimal code to reproduce
- Expected vs actual behavior
- Full error traceback

### Feature Requests

Please include:
- Clear description of the feature
- Use case - why is this needed?
- Proposed implementation (optional)
- Willingness to implement (optional)

### Questions

For questions, please use:
- GitHub Discussions (preferred)
- Issues labeled `question`

---

## Community

### Getting Help

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: Questions and general discussion
- **Email**: renata@jurevo.io (maintainer)

### Recognition

Contributors are recognized in:
- README.md Contributors section
- Release notes
- Social media shoutouts

### Becoming a Maintainer

Active contributors may be invited to become maintainers. This includes:
- Triage access to issues
- Merge access for PRs
- Input on project direction

---

## License

By contributing to ALMA, you agree that your contributions will be licensed under the MIT License.

---

## Thank You!

Every contribution matters, whether it is fixing a typo or implementing a major feature. We appreciate your time and effort in making ALMA better!

Questions? Open an issue or reach out to [@RBKunnela](https://github.com/RBKunnela).
