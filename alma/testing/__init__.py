"""
ALMA Testing Module.

Provides reusable test utilities for ALMA integrations:

- MockStorage: In-memory storage backend for isolated testing
- MockEmbedder: Deterministic fake embedding provider
- Factory functions: Create test data with sensible defaults

Example usage:
    >>> from alma.testing import MockStorage, create_test_heuristic
    >>>
    >>> def test_my_integration():
    ...     storage = MockStorage()
    ...     heuristic = create_test_heuristic(agent="test-agent")
    ...     storage.save_heuristic(heuristic)
    ...     found = storage.get_heuristics("test-project", agent="test-agent")
    ...     assert len(found) == 1

The module is designed for:
- Unit testing ALMA integrations
- Testing agent hooks without real storage
- Creating test fixtures with minimal boilerplate
- Isolated testing without external dependencies
"""

from alma.testing.factories import (
    create_test_anti_pattern,
    create_test_heuristic,
    create_test_knowledge,
    create_test_outcome,
    create_test_preference,
)
from alma.testing.mocks import MockEmbedder, MockStorage

__all__ = [
    # Mocks
    "MockStorage",
    "MockEmbedder",
    # Factories
    "create_test_heuristic",
    "create_test_outcome",
    "create_test_preference",
    "create_test_knowledge",
    "create_test_anti_pattern",
]
