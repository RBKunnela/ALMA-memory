# Testing Guide

ALMA provides a dedicated testing module (`alma.testing`) with mock implementations and factory functions for writing isolated, fast tests without external dependencies.

## Quick Start

```python
from alma.testing import (
    MockStorage,
    MockEmbedder,
    create_test_heuristic,
    create_test_outcome,
    create_test_preference,
    create_test_knowledge,
    create_test_anti_pattern,
)

def test_my_agent_integration():
    # Create isolated storage
    storage = MockStorage()

    # Create test data with sensible defaults
    heuristic = create_test_heuristic(agent="my-agent", confidence=0.9)
    storage.save_heuristic(heuristic)

    # Query and assert
    found = storage.get_heuristics("test-project", agent="my-agent")
    assert len(found) == 1
    assert found[0].confidence == 0.9
```

---

## MockStorage

In-memory implementation of `StorageBackend` for fast, isolated testing. No database setup required.

### Basic Usage

```python
from alma.testing import MockStorage, create_test_heuristic

storage = MockStorage()

# Save memories
heuristic = create_test_heuristic(agent="helena")
storage.save_heuristic(heuristic)

# Query memories
results = storage.get_heuristics(
    project_id="test-project",
    agent="helena",
    top_k=5
)

# Clean up between tests
storage.clear()
```

### Supported Operations

MockStorage implements the full `StorageBackend` interface:

**Write Operations:**
- `save_heuristic(heuristic)` - Save a heuristic
- `save_outcome(outcome)` - Save an outcome
- `save_user_preference(preference)` - Save a user preference
- `save_domain_knowledge(knowledge)` - Save domain knowledge
- `save_anti_pattern(anti_pattern)` - Save an anti-pattern

**Read Operations:**
- `get_heuristics(project_id, agent=None, ...)` - Get heuristics with filtering
- `get_outcomes(project_id, agent=None, task_type=None, ...)` - Get outcomes
- `get_user_preferences(user_id, category=None)` - Get user preferences
- `get_domain_knowledge(project_id, agent=None, domain=None, ...)` - Get knowledge
- `get_anti_patterns(project_id, agent=None, ...)` - Get anti-patterns

**Update Operations:**
- `update_heuristic(heuristic_id, updates)` - Update heuristic fields
- `increment_heuristic_occurrence(heuristic_id, success)` - Track usage
- `update_heuristic_confidence(heuristic_id, new_confidence)` - Update confidence
- `update_knowledge_confidence(knowledge_id, new_confidence)` - Update knowledge confidence

**Delete Operations:**
- `delete_heuristic(heuristic_id)` - Delete a heuristic
- `delete_outcome(outcome_id)` - Delete an outcome
- `delete_domain_knowledge(knowledge_id)` - Delete knowledge
- `delete_anti_pattern(anti_pattern_id)` - Delete an anti-pattern
- `delete_outcomes_older_than(project_id, older_than, agent=None)` - Prune old outcomes
- `delete_low_confidence_heuristics(project_id, below_confidence, agent=None)` - Prune weak heuristics

**Statistics:**
- `get_stats(project_id, agent=None)` - Get memory counts

### Mock-Specific Features

```python
storage = MockStorage()

# Check counts (useful for assertions)
assert storage.heuristic_count == 0
assert storage.outcome_count == 0
assert storage.preference_count == 0
assert storage.knowledge_count == 0
assert storage.anti_pattern_count == 0

# Clear all data
storage.clear()

# Create from config (ignores config, returns fresh instance)
storage = MockStorage.from_config({"some": "config"})
```

---

## MockEmbedder

Deterministic fake embedding provider for testing. Generates consistent embeddings based on text hash.

### Basic Usage

```python
from alma.testing import MockEmbedder

embedder = MockEmbedder(dimension=384)

# Generate single embedding
embedding = embedder.encode("test text")
assert len(embedding) == 384

# Generate batch embeddings
embeddings = embedder.encode_batch(["text one", "text two"])
assert len(embeddings) == 2

# Embeddings are deterministic - same text = same embedding
emb1 = embedder.encode("hello")
emb2 = embedder.encode("hello")
assert emb1 == emb2
```

### Properties

- **Deterministic**: Same input always produces same embedding
- **Configurable dimension**: Match your production embedding size
- **No external dependencies**: No API calls or model loading
- **Fast**: Hash-based computation

### When to Use

Use MockEmbedder when:
- Testing retrieval logic without real embeddings
- Unit testing storage backends
- Integration testing without network calls
- CI/CD pipelines where model loading is slow

Do NOT use MockEmbedder when:
- Testing actual semantic similarity
- Validating retrieval quality
- End-to-end testing

---

## Factory Functions

Factory functions create test data with sensible defaults. All fields can be overridden.

### create_test_heuristic

```python
from alma.testing import create_test_heuristic

# With all defaults
heuristic = create_test_heuristic()

# Override specific fields
heuristic = create_test_heuristic(
    id="my-id",
    agent="helena",
    project_id="my-project",
    condition="when testing forms",
    strategy="use incremental validation",
    confidence=0.95,
    occurrence_count=20,
    success_count=18,
)
```

**Default values:**
- `id`: Auto-generated UUID
- `agent`: "test-agent"
- `project_id`: "test-project"
- `condition`: "test condition"
- `strategy`: "test strategy"
- `confidence`: 0.85
- `occurrence_count`: 10
- `success_count`: 8
- `last_validated`: Now
- `created_at`: 7 days ago

### create_test_outcome

```python
from alma.testing import create_test_outcome

# Success outcome
outcome = create_test_outcome(success=True)

# Failure outcome with error
outcome = create_test_outcome(
    success=False,
    error_message="Validation failed",
    task_type="api_test",
    strategy_used="retry with backoff",
)
```

**Default values:**
- `id`: Auto-generated UUID
- `agent`: "test-agent"
- `project_id`: "test-project"
- `task_type`: "test_task"
- `task_description`: "Test task description"
- `success`: True
- `strategy_used`: "test strategy"
- `duration_ms`: 500
- `timestamp`: Now

### create_test_preference

```python
from alma.testing import create_test_preference

preference = create_test_preference(
    user_id="user-123",
    category="communication",
    preference="No emojis in responses",
)
```

**Default values:**
- `id`: Auto-generated UUID
- `user_id`: "test-user"
- `category`: "code_style"
- `preference`: "Test preference value"
- `source`: "explicit_instruction"
- `confidence`: 1.0
- `timestamp`: Now

### create_test_knowledge

```python
from alma.testing import create_test_knowledge

knowledge = create_test_knowledge(
    domain="authentication",
    fact="JWT tokens expire in 24 hours",
    source="code_analysis",
)
```

**Default values:**
- `id`: Auto-generated UUID
- `agent`: "test-agent"
- `project_id`: "test-project"
- `domain`: "test_domain"
- `fact`: "Test domain fact"
- `source`: "test_source"
- `confidence`: 1.0
- `last_verified`: Now

### create_test_anti_pattern

```python
from alma.testing import create_test_anti_pattern

anti_pattern = create_test_anti_pattern(
    pattern="Using sleep() for async waits",
    why_bad="Causes flaky tests and slow execution",
    better_alternative="Use explicit waits or polling",
)
```

**Default values:**
- `id`: Auto-generated UUID
- `agent`: "test-agent"
- `project_id`: "test-project"
- `pattern`: "Test anti-pattern"
- `why_bad`: "This is why it's bad"
- `better_alternative`: "Do this instead"
- `occurrence_count`: 3
- `last_seen`: Now
- `created_at`: 3 days ago

---

## Example Test Patterns

### Testing Agent Memory Retrieval

```python
import pytest
from alma.testing import MockStorage, create_test_heuristic, create_test_knowledge

@pytest.fixture
def storage():
    """Create fresh storage for each test."""
    return MockStorage()

def test_agent_retrieves_own_memories(storage):
    # Setup: Create memories for specific agent
    h1 = create_test_heuristic(agent="helena", confidence=0.9)
    h2 = create_test_heuristic(agent="victor", confidence=0.8)
    storage.save_heuristic(h1)
    storage.save_heuristic(h2)

    # Act: Retrieve only helena's memories
    results = storage.get_heuristics("test-project", agent="helena")

    # Assert
    assert len(results) == 1
    assert results[0].agent == "helena"

def test_confidence_filtering(storage):
    # Setup
    storage.save_heuristic(create_test_heuristic(confidence=0.3))
    storage.save_heuristic(create_test_heuristic(confidence=0.7))
    storage.save_heuristic(create_test_heuristic(confidence=0.9))

    # Act
    results = storage.get_heuristics(
        "test-project",
        min_confidence=0.5,
        top_k=10
    )

    # Assert: Only high-confidence heuristics returned
    assert len(results) == 2
    assert all(h.confidence >= 0.5 for h in results)
```

### Testing Memory Updates

```python
def test_heuristic_occurrence_tracking(storage):
    # Setup
    heuristic = create_test_heuristic(
        occurrence_count=10,
        success_count=8
    )
    storage.save_heuristic(heuristic)

    # Act: Record successful usage
    storage.increment_heuristic_occurrence(heuristic.id, success=True)

    # Assert
    updated = storage.get_heuristics("test-project")[0]
    assert updated.occurrence_count == 11
    assert updated.success_count == 9

def test_confidence_update(storage):
    heuristic = create_test_heuristic(confidence=0.5)
    storage.save_heuristic(heuristic)

    storage.update_heuristic_confidence(heuristic.id, 0.9)

    updated = storage.get_heuristics("test-project")[0]
    assert updated.confidence == 0.9
```

### Testing Memory Cleanup

```python
from datetime import datetime, timedelta, timezone

def test_prune_old_outcomes(storage):
    # Setup: Create old and new outcomes
    old_outcome = create_test_outcome(
        timestamp=datetime.now(timezone.utc) - timedelta(days=90)
    )
    new_outcome = create_test_outcome(
        timestamp=datetime.now(timezone.utc)
    )
    storage.save_outcome(old_outcome)
    storage.save_outcome(new_outcome)

    # Act: Delete outcomes older than 30 days
    cutoff = datetime.now(timezone.utc) - timedelta(days=30)
    deleted_count = storage.delete_outcomes_older_than("test-project", cutoff)

    # Assert
    assert deleted_count == 1
    assert storage.outcome_count == 1

def test_prune_low_confidence_heuristics(storage):
    storage.save_heuristic(create_test_heuristic(confidence=0.2))
    storage.save_heuristic(create_test_heuristic(confidence=0.8))

    deleted = storage.delete_low_confidence_heuristics(
        "test-project",
        below_confidence=0.5
    )

    assert deleted == 1
    assert storage.heuristic_count == 1
```

### Testing with Embeddings

```python
from alma.testing import MockEmbedder

def test_embedding_provider_integration():
    embedder = MockEmbedder(dimension=384)

    # Test determinism
    text = "Test authentication flow"
    emb1 = embedder.encode(text)
    emb2 = embedder.encode(text)
    assert emb1 == emb2

    # Test dimension
    assert len(emb1) == 384
    assert embedder.dimension == 384

def test_batch_embeddings():
    embedder = MockEmbedder()
    texts = ["First text", "Second text", "Third text"]

    embeddings = embedder.encode_batch(texts)

    assert len(embeddings) == 3
    assert all(len(e) == 384 for e in embeddings)
```

### Testing Statistics

```python
def test_memory_statistics(storage):
    # Setup various memory types
    storage.save_heuristic(create_test_heuristic(agent="helena"))
    storage.save_heuristic(create_test_heuristic(agent="helena"))
    storage.save_outcome(create_test_outcome(agent="helena"))
    storage.save_domain_knowledge(create_test_knowledge(agent="helena"))

    # Get stats
    stats = storage.get_stats("test-project", agent="helena")

    assert stats["heuristics"] == 2
    assert stats["outcomes"] == 1
    assert stats["domain_knowledge"] == 1
    assert stats["total_count"] > 0
```

---

## Best Practices

### 1. Use Fixtures for Isolation

```python
@pytest.fixture
def storage():
    """Fresh storage for each test."""
    s = MockStorage()
    yield s
    s.clear()  # Cleanup after test
```

### 2. Create Minimal Test Data

```python
# Good: Only specify relevant fields
heuristic = create_test_heuristic(confidence=0.9)

# Avoid: Over-specifying irrelevant fields
heuristic = create_test_heuristic(
    id="some-id",
    agent="some-agent",
    project_id="some-project",
    condition="some condition",
    strategy="some strategy",
    confidence=0.9,
    occurrence_count=10,
    success_count=8,
    # ... etc
)
```

### 3. Test Edge Cases

```python
def test_empty_results():
    storage = MockStorage()
    results = storage.get_heuristics("nonexistent-project")
    assert results == []

def test_delete_nonexistent():
    storage = MockStorage()
    result = storage.delete_heuristic("nonexistent-id")
    assert result is False
```

### 4. Use Meaningful Test Names

```python
# Good
def test_agent_only_retrieves_own_heuristics():
    pass

def test_low_confidence_heuristics_excluded_by_default():
    pass

# Avoid
def test_heuristics():
    pass

def test_get():
    pass
```

---

## Integration with pytest

### conftest.py Example

```python
# tests/conftest.py
import pytest
from alma.testing import MockStorage, MockEmbedder

@pytest.fixture
def storage():
    """Provide fresh MockStorage for each test."""
    return MockStorage()

@pytest.fixture
def embedder():
    """Provide MockEmbedder for each test."""
    return MockEmbedder(dimension=384)

@pytest.fixture
def populated_storage(storage):
    """Storage with pre-populated test data."""
    from alma.testing import (
        create_test_heuristic,
        create_test_outcome,
        create_test_knowledge,
    )

    storage.save_heuristic(create_test_heuristic(agent="helena"))
    storage.save_outcome(create_test_outcome(agent="helena"))
    storage.save_domain_knowledge(create_test_knowledge(agent="helena"))

    return storage
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=alma

# Run specific test file
pytest tests/test_storage.py

# Run tests matching pattern
pytest -k "heuristic"
```
