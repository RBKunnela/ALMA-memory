"""
Unit tests for Qdrant storage backend.

These tests verify the Qdrant storage implementation without requiring
an actual Qdrant database. For integration tests with a real database,
see tests/integration/test_qdrant_integration.py
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

# Skip all tests if qdrant-client is not installed
qdrant_available = False
try:
    from qdrant_client import QdrantClient  # noqa: F401

    qdrant_available = True
except ImportError:
    pass

pytestmark = pytest.mark.skipif(
    not qdrant_available,
    reason="qdrant-client not installed. Install with: pip install 'alma-memory[qdrant]'",
)


@pytest.fixture
def mock_client():
    """Create a mock Qdrant client."""
    client = MagicMock()

    # Mock collection operations
    client.get_collection.return_value = MagicMock()
    client.create_collection.return_value = None
    client.upsert.return_value = None
    client.search.return_value = []
    client.scroll.return_value = ([], None)
    client.retrieve.return_value = []
    client.delete.return_value = None
    client.set_payload.return_value = None
    client.count.return_value = MagicMock(count=0)

    return client


@pytest.fixture
def storage_with_mock_client(mock_client):
    """Create QdrantStorage with mocked client."""
    from alma.storage.qdrant import QdrantStorage

    with patch.object(QdrantStorage, "__init__", lambda self, **kwargs: None):
        storage = QdrantStorage()
        storage._client = mock_client
        storage.url = "http://localhost:6333"
        storage.api_key = None
        storage.collection_prefix = "alma_"
        storage.embedding_dim = 384
        storage.timeout = 30

    return storage


class TestQdrantStorageInit:
    """Tests for storage initialization."""

    def test_from_config_basic(self):
        """Test from_config with basic config."""
        from alma.storage.qdrant import QdrantStorage

        config = {
            "qdrant": {
                "url": "http://localhost:6333",
                "collection_prefix": "test_",
            },
            "embedding_dim": 384,
        }

        with patch.object(
            QdrantStorage, "__init__", return_value=None
        ) as mock_init:
            QdrantStorage.from_config(config)
            mock_init.assert_called_once()
            call_kwargs = mock_init.call_args[1]
            assert call_kwargs["url"] == "http://localhost:6333"
            assert call_kwargs["collection_prefix"] == "test_"
            assert call_kwargs["embedding_dim"] == 384

    def test_from_config_with_env_vars(self):
        """Test from_config with environment variable expansion."""
        import os

        from alma.storage.qdrant import QdrantStorage

        os.environ["TEST_QDRANT_URL"] = "http://qdrant.example.com:6333"
        os.environ["TEST_QDRANT_API_KEY"] = "secret123"

        config = {
            "qdrant": {
                "url": "${TEST_QDRANT_URL}",
                "api_key": "${TEST_QDRANT_API_KEY}",
            },
        }

        with patch.object(
            QdrantStorage, "__init__", return_value=None
        ) as mock_init:
            QdrantStorage.from_config(config)
            call_kwargs = mock_init.call_args[1]
            assert call_kwargs["url"] == "http://qdrant.example.com:6333"
            assert call_kwargs["api_key"] == "secret123"

        # Cleanup
        del os.environ["TEST_QDRANT_URL"]
        del os.environ["TEST_QDRANT_API_KEY"]

    def test_from_config_defaults(self):
        """Test from_config with minimal config uses defaults."""
        from alma.storage.qdrant import QdrantStorage

        config = {}

        with patch.object(
            QdrantStorage, "__init__", return_value=None
        ) as mock_init:
            QdrantStorage.from_config(config)
            call_kwargs = mock_init.call_args[1]
            assert call_kwargs["url"] == "http://localhost:6333"
            assert call_kwargs["api_key"] is None
            assert call_kwargs["collection_prefix"] == "alma_"


class TestQdrantStorageHelpers:
    """Tests for helper methods."""

    def test_collection_name(self, storage_with_mock_client):
        """Test collection name generation."""
        storage = storage_with_mock_client
        assert storage._collection_name("heuristics") == "alma_heuristics"
        assert storage._collection_name("outcomes") == "alma_outcomes"

    def test_datetime_to_str(self, storage_with_mock_client):
        """Test datetime to string conversion."""
        storage = storage_with_mock_client
        dt = datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
        result = storage._datetime_to_str(dt)
        assert "2024-01-15" in result
        assert storage._datetime_to_str(None) is None

    def test_str_to_datetime(self, storage_with_mock_client):
        """Test string to datetime conversion."""
        storage = storage_with_mock_client
        result = storage._str_to_datetime("2024-01-15T10:30:00+00:00")
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15
        assert storage._str_to_datetime(None) is None
        assert storage._str_to_datetime("invalid") is None

    def test_get_dummy_vector(self, storage_with_mock_client):
        """Test dummy vector generation."""
        storage = storage_with_mock_client
        vector = storage._get_dummy_vector()
        assert len(vector) == 384
        assert all(v == 0.0 for v in vector)


class TestQdrantStorageWriteOps:
    """Tests for write operations."""

    def test_save_heuristic(self, storage_with_mock_client, mock_client):
        """Test saving a heuristic."""
        from alma.types import Heuristic

        storage = storage_with_mock_client

        heuristic = Heuristic(
            id="test-h-1",
            agent="Helena",
            project_id="test-project",
            condition="form with multiple fields",
            strategy="test happy path first",
            confidence=0.85,
            occurrence_count=10,
            success_count=8,
            last_validated=datetime.now(timezone.utc),
            created_at=datetime.now(timezone.utc),
            metadata={"tags": ["forms"]},
        )

        result = storage.save_heuristic(heuristic)
        assert result == "test-h-1"
        mock_client.upsert.assert_called_once()

    def test_save_heuristic_with_embedding(self, storage_with_mock_client, mock_client):
        """Test saving a heuristic with embedding."""
        from alma.types import Heuristic

        storage = storage_with_mock_client
        embedding = [0.1] * 384

        heuristic = Heuristic(
            id="test-h-2",
            agent="Helena",
            project_id="test-project",
            condition="API endpoint",
            strategy="validate schema first",
            confidence=0.9,
            occurrence_count=5,
            success_count=5,
            last_validated=datetime.now(timezone.utc),
            created_at=datetime.now(timezone.utc),
            embedding=embedding,
        )

        result = storage.save_heuristic(heuristic)
        assert result == "test-h-2"

        # Verify the embedding was passed
        call_args = mock_client.upsert.call_args
        points = call_args[1]["points"]
        assert len(points) == 1
        assert points[0].vector == embedding

    def test_save_outcome(self, storage_with_mock_client, mock_client):
        """Test saving an outcome."""
        from alma.types import Outcome

        storage = storage_with_mock_client

        outcome = Outcome(
            id="test-o-1",
            agent="Victor",
            project_id="test-project",
            task_type="api_testing",
            task_description="Test login endpoint",
            success=True,
            strategy_used="happy_path",
            duration_ms=150,
            timestamp=datetime.now(timezone.utc),
            metadata={},
        )

        result = storage.save_outcome(outcome)
        assert result == "test-o-1"
        mock_client.upsert.assert_called_once()

    def test_save_user_preference(self, storage_with_mock_client, mock_client):
        """Test saving a user preference."""
        from alma.types import UserPreference

        storage = storage_with_mock_client

        preference = UserPreference(
            id="test-p-1",
            user_id="user-123",
            category="communication",
            preference="No emojis in documentation",
            source="explicit_instruction",
            confidence=1.0,
            timestamp=datetime.now(timezone.utc),
        )

        result = storage.save_user_preference(preference)
        assert result == "test-p-1"
        mock_client.upsert.assert_called_once()

    def test_save_domain_knowledge(self, storage_with_mock_client, mock_client):
        """Test saving domain knowledge."""
        from alma.types import DomainKnowledge

        storage = storage_with_mock_client

        knowledge = DomainKnowledge(
            id="test-dk-1",
            agent="Derek",
            project_id="test-project",
            domain="authentication",
            fact="Login uses JWT with 24h expiry",
            source="code_analysis",
            confidence=0.95,
            last_verified=datetime.now(timezone.utc),
        )

        result = storage.save_domain_knowledge(knowledge)
        assert result == "test-dk-1"
        mock_client.upsert.assert_called_once()

    def test_save_anti_pattern(self, storage_with_mock_client, mock_client):
        """Test saving an anti-pattern."""
        from alma.types import AntiPattern

        storage = storage_with_mock_client

        anti_pattern = AntiPattern(
            id="test-ap-1",
            agent="Helena",
            project_id="test-project",
            pattern="Using fixed sleep() for async waits",
            why_bad="Causes flaky tests",
            better_alternative="Use explicit waits with conditions",
            occurrence_count=3,
            last_seen=datetime.now(timezone.utc),
            created_at=datetime.now(timezone.utc),
        )

        result = storage.save_anti_pattern(anti_pattern)
        assert result == "test-ap-1"
        mock_client.upsert.assert_called_once()


class TestQdrantStorageBatchOps:
    """Tests for batch write operations."""

    def test_save_heuristics_batch(self, storage_with_mock_client, mock_client):
        """Test batch saving heuristics."""
        from alma.types import Heuristic

        storage = storage_with_mock_client

        heuristics = [
            Heuristic(
                id=f"test-h-{i}",
                agent="Helena",
                project_id="test-project",
                condition=f"condition {i}",
                strategy=f"strategy {i}",
                confidence=0.8,
                occurrence_count=5,
                success_count=4,
                last_validated=datetime.now(timezone.utc),
                created_at=datetime.now(timezone.utc),
            )
            for i in range(3)
        ]

        result = storage.save_heuristics(heuristics)
        assert len(result) == 3
        assert result == ["test-h-0", "test-h-1", "test-h-2"]
        mock_client.upsert.assert_called_once()

        # Verify all points were included
        call_args = mock_client.upsert.call_args
        points = call_args[1]["points"]
        assert len(points) == 3

    def test_save_outcomes_batch(self, storage_with_mock_client, mock_client):
        """Test batch saving outcomes."""
        from alma.types import Outcome

        storage = storage_with_mock_client

        outcomes = [
            Outcome(
                id=f"test-o-{i}",
                agent="Victor",
                project_id="test-project",
                task_type="testing",
                task_description=f"Task {i}",
                success=True,
                strategy_used="default",
            )
            for i in range(2)
        ]

        result = storage.save_outcomes(outcomes)
        assert len(result) == 2
        mock_client.upsert.assert_called_once()

    def test_save_empty_batch(self, storage_with_mock_client, mock_client):
        """Test batch saving empty list."""
        storage = storage_with_mock_client

        result = storage.save_heuristics([])
        assert result == []
        mock_client.upsert.assert_not_called()


class TestQdrantStorageReadOps:
    """Tests for read operations."""

    def test_get_heuristics_basic(self, storage_with_mock_client, mock_client):
        """Test basic heuristic retrieval."""
        storage = storage_with_mock_client

        # Mock returns empty list
        mock_client.scroll.return_value = ([], None)

        results = storage.get_heuristics(
            project_id="test-project",
            agent="Helena",
            top_k=5,
        )

        assert results == []
        mock_client.scroll.assert_called_once()

    def test_get_heuristics_with_embedding(self, storage_with_mock_client, mock_client):
        """Test heuristic retrieval with vector search."""
        storage = storage_with_mock_client

        # Mock search returns empty list
        mock_client.search.return_value = []

        embedding = [0.1] * 384
        results = storage.get_heuristics(
            project_id="test-project",
            embedding=embedding,
            top_k=5,
        )

        assert results == []
        mock_client.search.assert_called_once()

        # Verify embedding was passed
        call_args = mock_client.search.call_args
        assert call_args[1]["query_vector"] == embedding

    def test_get_heuristics_with_results(self, storage_with_mock_client, mock_client):
        """Test heuristic retrieval with actual results."""
        storage = storage_with_mock_client

        # Create mock point
        mock_point = MagicMock()
        mock_point.payload = {
            "id": "h-123",
            "agent": "Helena",
            "project_id": "test-project",
            "condition": "test condition",
            "strategy": "test strategy",
            "confidence": 0.9,
            "occurrence_count": 5,
            "success_count": 4,
            "last_validated": datetime.now(timezone.utc).isoformat(),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "metadata": {"key": "value"},
        }
        mock_point.vector = None

        mock_client.scroll.return_value = ([mock_point], None)

        results = storage.get_heuristics(
            project_id="test-project",
            top_k=5,
        )

        assert len(results) == 1
        assert results[0].id == "h-123"
        assert results[0].agent == "Helena"
        assert results[0].confidence == 0.9

    def test_get_outcomes_with_filters(self, storage_with_mock_client, mock_client):
        """Test outcome retrieval with filters."""
        storage = storage_with_mock_client
        mock_client.scroll.return_value = ([], None)

        results = storage.get_outcomes(
            project_id="test-project",
            agent="Victor",
            task_type="api_testing",
            success_only=True,
            top_k=10,
        )

        assert results == []
        mock_client.scroll.assert_called_once()

        # Verify filter was built correctly
        call_args = mock_client.scroll.call_args
        scroll_filter = call_args[1]["scroll_filter"]
        assert scroll_filter is not None

    def test_get_user_preferences(self, storage_with_mock_client, mock_client):
        """Test user preference retrieval."""
        storage = storage_with_mock_client

        mock_point = MagicMock()
        mock_point.payload = {
            "id": "p-123",
            "user_id": "user-1",
            "category": "communication",
            "preference": "Be concise",
            "source": "explicit",
            "confidence": 1.0,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": {},
        }
        mock_client.scroll.return_value = ([mock_point], None)

        results = storage.get_user_preferences(
            user_id="user-1",
            category="communication",
        )

        assert len(results) == 1
        assert results[0].preference == "Be concise"

    def test_get_domain_knowledge(self, storage_with_mock_client, mock_client):
        """Test domain knowledge retrieval."""
        storage = storage_with_mock_client
        mock_client.scroll.return_value = ([], None)

        results = storage.get_domain_knowledge(
            project_id="test-project",
            domain="authentication",
            top_k=5,
        )

        assert results == []

    def test_get_anti_patterns(self, storage_with_mock_client, mock_client):
        """Test anti-pattern retrieval."""
        storage = storage_with_mock_client
        mock_client.scroll.return_value = ([], None)

        results = storage.get_anti_patterns(
            project_id="test-project",
            agent="Helena",
            top_k=5,
        )

        assert results == []


class TestQdrantStorageMultiAgent:
    """Tests for multi-agent memory sharing."""

    def test_get_heuristics_for_agents(self, storage_with_mock_client, mock_client):
        """Test getting heuristics from multiple agents."""
        storage = storage_with_mock_client
        mock_client.scroll.return_value = ([], None)

        results = storage.get_heuristics_for_agents(
            project_id="test-project",
            agents=["Helena", "Victor"],
            top_k=5,
        )

        assert results == []
        mock_client.scroll.assert_called_once()

    def test_get_heuristics_for_agents_empty_list(self, storage_with_mock_client, mock_client):
        """Test getting heuristics with empty agent list."""
        storage = storage_with_mock_client

        results = storage.get_heuristics_for_agents(
            project_id="test-project",
            agents=[],
            top_k=5,
        )

        assert results == []
        mock_client.scroll.assert_not_called()

    def test_get_outcomes_for_agents(self, storage_with_mock_client, mock_client):
        """Test getting outcomes from multiple agents."""
        storage = storage_with_mock_client
        mock_client.scroll.return_value = ([], None)

        results = storage.get_outcomes_for_agents(
            project_id="test-project",
            agents=["Helena", "Victor"],
            task_type="testing",
            success_only=True,
            top_k=5,
        )

        assert results == []

    def test_get_domain_knowledge_for_agents(self, storage_with_mock_client, mock_client):
        """Test getting domain knowledge from multiple agents."""
        storage = storage_with_mock_client
        mock_client.scroll.return_value = ([], None)

        results = storage.get_domain_knowledge_for_agents(
            project_id="test-project",
            agents=["Derek", "Helena"],
            domain="api",
            top_k=5,
        )

        assert results == []

    def test_get_anti_patterns_for_agents(self, storage_with_mock_client, mock_client):
        """Test getting anti-patterns from multiple agents."""
        storage = storage_with_mock_client
        mock_client.scroll.return_value = ([], None)

        results = storage.get_anti_patterns_for_agents(
            project_id="test-project",
            agents=["Helena", "Victor"],
            top_k=5,
        )

        assert results == []


class TestQdrantStorageUpdateOps:
    """Tests for update operations."""

    def test_update_heuristic(self, storage_with_mock_client, mock_client):
        """Test updating a heuristic."""
        storage = storage_with_mock_client

        result = storage.update_heuristic(
            heuristic_id="h-123",
            updates={"confidence": 0.95, "strategy": "new strategy"},
        )

        assert result is True
        mock_client.set_payload.assert_called_once()

    def test_update_heuristic_empty_updates(self, storage_with_mock_client, mock_client):
        """Test updating with empty updates."""
        storage = storage_with_mock_client

        result = storage.update_heuristic(
            heuristic_id="h-123",
            updates={},
        )

        assert result is False
        mock_client.set_payload.assert_not_called()

    def test_increment_heuristic_occurrence(self, storage_with_mock_client, mock_client):
        """Test incrementing heuristic occurrence count."""
        storage = storage_with_mock_client

        # Mock retrieve to return existing heuristic
        mock_point = MagicMock()
        mock_point.payload = {
            "occurrence_count": 5,
            "success_count": 4,
        }
        mock_client.retrieve.return_value = [mock_point]

        result = storage.increment_heuristic_occurrence(
            heuristic_id="h-123",
            success=True,
        )

        assert result is True
        mock_client.retrieve.assert_called_once()
        mock_client.set_payload.assert_called_once()

        # Verify incremented values
        call_args = mock_client.set_payload.call_args
        payload = call_args[1]["payload"]
        assert payload["occurrence_count"] == 6
        assert payload["success_count"] == 5

    def test_increment_heuristic_occurrence_not_found(self, storage_with_mock_client, mock_client):
        """Test incrementing non-existent heuristic."""
        storage = storage_with_mock_client
        mock_client.retrieve.return_value = []

        result = storage.increment_heuristic_occurrence(
            heuristic_id="non-existent",
            success=True,
        )

        assert result is False

    def test_update_heuristic_confidence(self, storage_with_mock_client, mock_client):
        """Test updating heuristic confidence."""
        storage = storage_with_mock_client

        result = storage.update_heuristic_confidence(
            heuristic_id="h-123",
            new_confidence=0.99,
        )

        assert result is True
        mock_client.set_payload.assert_called_once()

        call_args = mock_client.set_payload.call_args
        assert call_args[1]["payload"]["confidence"] == 0.99

    def test_update_knowledge_confidence(self, storage_with_mock_client, mock_client):
        """Test updating knowledge confidence."""
        storage = storage_with_mock_client

        result = storage.update_knowledge_confidence(
            knowledge_id="dk-123",
            new_confidence=0.85,
        )

        assert result is True


class TestQdrantStorageDeleteOps:
    """Tests for delete operations."""

    def test_delete_heuristic(self, storage_with_mock_client, mock_client):
        """Test deleting a heuristic."""
        storage = storage_with_mock_client

        result = storage.delete_heuristic("h-123")

        assert result is True
        mock_client.delete.assert_called_once()

    def test_delete_outcome(self, storage_with_mock_client, mock_client):
        """Test deleting an outcome."""
        storage = storage_with_mock_client

        result = storage.delete_outcome("o-123")

        assert result is True
        mock_client.delete.assert_called_once()

    def test_delete_domain_knowledge(self, storage_with_mock_client, mock_client):
        """Test deleting domain knowledge."""
        storage = storage_with_mock_client

        result = storage.delete_domain_knowledge("dk-123")

        assert result is True

    def test_delete_anti_pattern(self, storage_with_mock_client, mock_client):
        """Test deleting an anti-pattern."""
        storage = storage_with_mock_client

        result = storage.delete_anti_pattern("ap-123")

        assert result is True

    def test_delete_outcomes_older_than(self, storage_with_mock_client, mock_client):
        """Test deleting old outcomes."""
        storage = storage_with_mock_client
        mock_client.count.return_value = MagicMock(count=5)

        result = storage.delete_outcomes_older_than(
            project_id="test-project",
            older_than=datetime.now(timezone.utc),
            agent="Victor",
        )

        assert result == 5
        mock_client.delete.assert_called_once()

    def test_delete_low_confidence_heuristics(self, storage_with_mock_client, mock_client):
        """Test deleting low-confidence heuristics."""
        storage = storage_with_mock_client
        mock_client.count.return_value = MagicMock(count=3)

        result = storage.delete_low_confidence_heuristics(
            project_id="test-project",
            below_confidence=0.5,
        )

        assert result == 3
        mock_client.delete.assert_called_once()


class TestQdrantStorageStats:
    """Tests for statistics."""

    def test_get_stats(self, storage_with_mock_client, mock_client):
        """Test getting storage statistics."""
        storage = storage_with_mock_client
        mock_client.count.return_value = MagicMock(count=10)

        stats = storage.get_stats(project_id="test-project")

        assert stats["project_id"] == "test-project"
        assert stats["storage_type"] == "qdrant"
        assert stats["url"] == "http://localhost:6333"
        assert "heuristics_count" in stats
        assert "total_count" in stats

    def test_get_stats_with_agent(self, storage_with_mock_client, mock_client):
        """Test getting storage statistics filtered by agent."""
        storage = storage_with_mock_client
        mock_client.count.return_value = MagicMock(count=5)

        stats = storage.get_stats(
            project_id="test-project",
            agent="Helena",
        )

        assert stats["agent"] == "Helena"


class TestQdrantStoragePointConversion:
    """Tests for point to object conversion."""

    def test_point_to_heuristic(self, storage_with_mock_client):
        """Test converting a Qdrant point to Heuristic."""
        storage = storage_with_mock_client

        mock_point = MagicMock()
        mock_point.payload = {
            "id": "h-123",
            "agent": "Helena",
            "project_id": "test",
            "condition": "test condition",
            "strategy": "test strategy",
            "confidence": 0.9,
            "occurrence_count": 5,
            "success_count": 4,
            "last_validated": "2024-01-15T10:30:00+00:00",
            "created_at": "2024-01-01T00:00:00+00:00",
            "metadata": {"key": "value"},
        }
        mock_point.vector = [0.1] * 384

        heuristic = storage._point_to_heuristic(mock_point)

        assert heuristic.id == "h-123"
        assert heuristic.agent == "Helena"
        assert heuristic.confidence == 0.9
        assert heuristic.metadata == {"key": "value"}

    def test_point_to_outcome(self, storage_with_mock_client):
        """Test converting a Qdrant point to Outcome."""
        storage = storage_with_mock_client

        mock_point = MagicMock()
        mock_point.payload = {
            "id": "o-123",
            "agent": "Victor",
            "project_id": "test",
            "task_type": "api_testing",
            "task_description": "Test endpoint",
            "success": True,
            "strategy_used": "happy_path",
            "duration_ms": 100,
            "error_message": None,
            "user_feedback": None,
            "timestamp": "2024-01-15T10:30:00+00:00",
            "metadata": {},
        }
        mock_point.vector = None

        outcome = storage._point_to_outcome(mock_point)

        assert outcome.id == "o-123"
        assert outcome.agent == "Victor"
        assert outcome.success is True
        assert outcome.duration_ms == 100

    def test_point_to_preference(self, storage_with_mock_client):
        """Test converting a Qdrant point to UserPreference."""
        storage = storage_with_mock_client

        mock_point = MagicMock()
        mock_point.payload = {
            "id": "p-123",
            "user_id": "user-1",
            "category": "communication",
            "preference": "Be concise",
            "source": "explicit",
            "confidence": 1.0,
            "timestamp": "2024-01-15T10:30:00+00:00",
            "metadata": {},
        }

        preference = storage._point_to_preference(mock_point)

        assert preference.id == "p-123"
        assert preference.preference == "Be concise"

    def test_point_to_domain_knowledge(self, storage_with_mock_client):
        """Test converting a Qdrant point to DomainKnowledge."""
        storage = storage_with_mock_client

        mock_point = MagicMock()
        mock_point.payload = {
            "id": "dk-123",
            "agent": "Derek",
            "project_id": "test",
            "domain": "auth",
            "fact": "Uses JWT",
            "source": "code",
            "confidence": 0.95,
            "last_verified": "2024-01-15T10:30:00+00:00",
            "metadata": {},
        }
        mock_point.vector = None

        knowledge = storage._point_to_domain_knowledge(mock_point)

        assert knowledge.id == "dk-123"
        assert knowledge.fact == "Uses JWT"

    def test_point_to_anti_pattern(self, storage_with_mock_client):
        """Test converting a Qdrant point to AntiPattern."""
        storage = storage_with_mock_client

        mock_point = MagicMock()
        mock_point.payload = {
            "id": "ap-123",
            "agent": "Helena",
            "project_id": "test",
            "pattern": "Using sleep()",
            "why_bad": "Flaky tests",
            "better_alternative": "Use explicit waits",
            "occurrence_count": 3,
            "last_seen": "2024-01-15T10:30:00+00:00",
            "created_at": "2024-01-01T00:00:00+00:00",
            "metadata": {},
        }
        mock_point.vector = None

        anti_pattern = storage._point_to_anti_pattern(mock_point)

        assert anti_pattern.id == "ap-123"
        assert anti_pattern.pattern == "Using sleep()"


class TestQdrantStorageFilterBuilding:
    """Tests for filter building."""

    def test_build_filter_empty(self, storage_with_mock_client):
        """Test building filter with no conditions."""
        storage = storage_with_mock_client

        result = storage._build_filter()
        assert result is None

    def test_build_filter_with_project_id(self, storage_with_mock_client):
        """Test building filter with project_id."""
        storage = storage_with_mock_client

        result = storage._build_filter(project_id="test-project")
        assert result is not None
        assert len(result.must) == 1

    def test_build_filter_with_multiple_agents(self, storage_with_mock_client):
        """Test building filter with multiple agents."""
        storage = storage_with_mock_client

        result = storage._build_filter(
            project_id="test-project",
            agents=["Helena", "Victor"],
        )

        assert result is not None
        assert len(result.must) == 2

    def test_build_filter_with_all_params(self, storage_with_mock_client):
        """Test building filter with all parameters."""
        storage = storage_with_mock_client

        result = storage._build_filter(
            project_id="test-project",
            agent="Helena",
            task_type="testing",
            domain="auth",
            success_only=True,
            min_confidence=0.5,
        )

        assert result is not None
        # Should have conditions for: project_id, agent, task_type, domain, success, confidence
        assert len(result.must) == 6


class TestQdrantStorageClose:
    """Tests for connection closing."""

    def test_close(self, storage_with_mock_client, mock_client):
        """Test closing the client connection."""
        storage = storage_with_mock_client

        storage.close()
        mock_client.close.assert_called_once()
