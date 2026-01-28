# Contributing to ALMA

Thank you for your interest in contributing to ALMA! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Code Style Guidelines](#code-style-guidelines)
- [Testing Requirements](#testing-requirements)
- [Pull Request Process](#pull-request-process)
- [Issue Guidelines](#issue-guidelines)
- [Release Process](#release-process)
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
├── exceptions.py        # Exception hierarchy (NEW in v0.4.0)
├── config/              # Configuration loading
├── storage/             # Storage backends (SQLite, PostgreSQL, Azure, File)
├── retrieval/           # Memory retrieval and embeddings
├── learning/            # Learning protocols
├── extraction/          # LLM-powered fact extraction
├── graph/               # Graph memory with Neo4j
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
3. **The Harness Pattern**: Setting -> Context -> Agent -> Memory Schema

### Architecture Principles

- **Pluggable backends**: Storage and embedding providers are interchangeable
- **Clean abstractions**: All backends implement abstract base classes
- **No side effects**: Pure functions where possible
- **Type safety**: Full type hints throughout

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
3. **Storage backends** - MongoDB, Pinecone, Qdrant integrations
4. **LLM providers** - Ollama, Groq, local models for extraction
5. **Language SDKs** - TypeScript/JavaScript SDK

### Priority Areas (from v0.4.0 Roadmap)

- Multi-agent memory sharing
- Memory consolidation engine
- Event system / webhooks
- TypeScript SDK

---

## Development Setup

### Prerequisites

- Python 3.10+
- Git
- (Optional) Docker for testing with Neo4j/PostgreSQL
- (Optional) Node.js for TypeScript SDK development

### Installation

```bash
# Clone the repository
git clone https://github.com/RBKunnela/ALMA-memory.git
cd ALMA-memory

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode with all extras
pip install -e ".[dev,local,postgres,azure]"

# Install pre-commit hooks
pip install pre-commit
pre-commit install
```

### Running the Full Test Suite

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=alma --cov-report=html

# Run specific test file
pytest tests/test_storage.py

# Run tests matching a pattern
pytest -k "test_retrieval"

# Run integration tests (requires Docker)
pytest tests/integration/ --run-integration
```

### Setting Up Test Databases

```bash
# PostgreSQL with pgvector
docker run -d \
  --name alma-postgres \
  -e POSTGRES_PASSWORD=test \
  -e POSTGRES_DB=alma_test \
  -p 5432:5432 \
  pgvector/pgvector:pg16

# Neo4j for graph memory
docker run -d \
  --name alma-neo4j \
  -e NEO4J_AUTH=neo4j/testpassword \
  -p 7474:7474 -p 7687:7687 \
  neo4j:5
```

---

## Code Style Guidelines

### Python Style

We use the following tools for code quality:

| Tool | Purpose | Config |
|------|---------|--------|
| **Black** | Code formatting | `pyproject.toml` |
| **isort** | Import sorting | `pyproject.toml` |
| **flake8** | Linting | `.flake8` |
| **mypy** | Type checking | `pyproject.toml` |

### Running Code Quality Checks

```bash
# Format code
black alma/ tests/

# Sort imports
isort alma/ tests/

# Run linter
flake8 alma/ tests/

# Type check
mypy alma/

# Run all checks (same as pre-commit)
pre-commit run --all-files
```

### Code Style Rules

1. **Type hints required**: All function signatures must have type hints
   ```python
   # Good
   def retrieve(self, task: str, agent: str, top_k: int = 5) -> MemorySlice:

   # Bad
   def retrieve(self, task, agent, top_k=5):
   ```

2. **Docstrings required**: All public functions/classes need docstrings
   ```python
   def learn(self, agent: str, task: str, outcome: str) -> bool:
       """
       Learn from a task outcome.

       Args:
           agent: The agent that performed the task
           task: Description of the task
           outcome: "success" or "failure"

       Returns:
           True if learning was successful

       Raises:
           ScopeViolationError: If agent cannot learn this type
       """
   ```

3. **Use the exception hierarchy**: Never raise generic `Exception`
   ```python
   # Good
   from alma.exceptions import ValidationError, StorageError
   raise ValidationError("agent name cannot be empty")

   # Bad
   raise Exception("agent name cannot be empty")
   ```

4. **Timezone-aware datetimes**: Always use UTC
   ```python
   # Good
   from datetime import datetime, timezone
   now = datetime.now(timezone.utc)

   # Bad (deprecated in Python 3.12+)
   now = datetime.utcnow()
   ```

5. **No `eval()`**: Security risk, use `json.loads()` or `ast.literal_eval()`

---

## Testing Requirements

### Test Coverage Requirements

- **Minimum coverage**: 80% for new code
- **Critical paths**: 100% coverage for:
  - Storage backends (CRUD operations)
  - Security-sensitive code (input validation)
  - Exception handling paths

### Test Structure

```
tests/
├── unit/                    # Fast, isolated tests
│   ├── test_core.py
│   ├── test_types.py
│   └── test_scoring.py
├── integration/             # Tests with external dependencies
│   ├── test_postgresql.py
│   ├── test_neo4j.py
│   └── test_azure.py
├── e2e/                     # End-to-end scenarios
│   └── test_full_workflow.py
└── fixtures/                # Shared test fixtures
    ├── conftest.py
    └── sample_data.py
```

### Writing Good Tests

```python
import pytest
from alma import ALMA
from alma.exceptions import ValidationError

class TestALMARetrieval:
    """Tests for ALMA.retrieve() method."""

    @pytest.fixture
    def alma_instance(self, tmp_path):
        """Create a fresh ALMA instance for testing."""
        config = {
            "project_id": "test",
            "storage": "sqlite",
            "storage_dir": str(tmp_path),
        }
        return ALMA.from_dict(config)

    def test_retrieve_returns_memory_slice(self, alma_instance):
        """retrieve() should return a MemorySlice object."""
        result = alma_instance.retrieve(task="test task", agent="test-agent")
        assert isinstance(result, MemorySlice)

    def test_retrieve_empty_task_raises_validation_error(self, alma_instance):
        """retrieve() should raise ValidationError for empty task."""
        with pytest.raises(ValidationError, match="task cannot be empty"):
            alma_instance.retrieve(task="", agent="test-agent")

    @pytest.mark.parametrize("top_k,expected", [
        (1, 1),
        (5, 5),
        (100, 100),
    ])
    def test_retrieve_respects_top_k(self, alma_instance, top_k, expected):
        """retrieve() should return at most top_k results."""
        result = alma_instance.retrieve(task="test", agent="agent", top_k=top_k)
        assert len(result.heuristics) <= expected
```

### Integration Test Markers

```python
@pytest.mark.integration
def test_postgresql_connection():
    """Test that requires PostgreSQL."""
    pass

@pytest.mark.slow
def test_large_dataset():
    """Test that takes >30 seconds."""
    pass

@pytest.mark.azure
def test_cosmos_db():
    """Test that requires Azure credentials."""
    pass
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

### PR Checklist

Before submitting your PR, ensure:

- [ ] Code passes all tests (`pytest`)
- [ ] Code passes linting (`pre-commit run --all-files`)
- [ ] New code has type hints
- [ ] New code has docstrings
- [ ] Tests added for new functionality
- [ ] Documentation updated if needed
- [ ] CHANGELOG.md updated for user-facing changes
- [ ] No security vulnerabilities introduced

### PR Template

```markdown
## Summary
Brief description of changes.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
How was this tested?

## Checklist
- [ ] Tests pass
- [ ] Linting passes
- [ ] Documentation updated
- [ ] CHANGELOG updated

## Related Issues
Fixes #123
```

### Review Process

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
- Storage backend being used
- Minimal code to reproduce
- Expected vs actual behavior
- Full error traceback

### Feature Requests

Please include:
- Clear description of the feature
- Use case - why is this needed?
- Proposed implementation (optional)
- Willingness to implement (optional)

### Security Issues

**Do not open public issues for security vulnerabilities.**

Instead, email security@jurevo.io with:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

---

## Release Process

### Version Numbering

We follow [Semantic Versioning](https://semver.org/):
- **MAJOR** (1.0.0): Breaking changes
- **MINOR** (0.1.0): New features, backwards compatible
- **PATCH** (0.0.1): Bug fixes, backwards compatible

### Release Checklist

1. Update version in `pyproject.toml`
2. Update CHANGELOG.md with release date
3. Create release PR
4. After merge, tag release: `git tag v0.4.0`
5. Push tag: `git push origin v0.4.0`
6. GitHub Actions builds and publishes to PyPI

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
