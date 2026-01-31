# Contributing

We welcome contributions to ALMA! Here's how to get started.

## Quick Links

- [GitHub Repository](https://github.com/RBKunnela/ALMA-memory)
- [Issue Tracker](https://github.com/RBKunnela/ALMA-memory/issues)
- [Good First Issues](https://github.com/RBKunnela/ALMA-memory/labels/good%20first%20issue)

## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/RBKunnela/ALMA-memory.git
cd ALMA-memory
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install development dependencies:
```bash
pip install -e ".[dev,all]"
```

4. Run tests:
```bash
pytest
```

## What We Need Most

- **Documentation improvements** - Clarify concepts, add examples
- **Test coverage** - Edge cases, integration tests
- **LLM provider integrations** - Ollama, Groq, local models
- **Frontend dashboard** - Memory visualization

## Pull Request Process

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make your changes
4. Run tests: `pytest`
5. Run linting: `ruff check .`
6. Commit with conventional commits: `feat: add new feature`
7. Push and create a PR

## Code Style

- Use [Ruff](https://github.com/astral-sh/ruff) for linting
- Type hints required for all public APIs
- Docstrings for all public functions/classes
- Tests for all new functionality

## Questions?

Open an issue or start a discussion on GitHub!
