"""
Unit tests for Pinecone storage backend.

These tests verify the Pinecone storage implementation without requiring
an actual Pinecone account. For integration tests with a real Pinecone index,
see tests/integration/test_pinecone_integration.py
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

# Skip all tests if pinecone is not installed
pinecone_available = False
try:
    from pinecone import Pinecone  # noqa: F401

    pinecone_available = True
except ImportError:
    pass

pytestmark = pytest.mark.skipif(
    not pinecone_available,
    reason="pinecone not installed. Install with: pip install 'alma-memory[pinecone]'",
)


@pytest.fixture
def mock_pinecone_client():
    """Create a mock Pinecone client."""
    client = MagicMock()
    index = MagicMock()

    # Mock list_indexes to return empty list (index doesn't exist)
    client.list_indexes.return_value = []

    # Mock Index
    client.Index.return_value = index

    # Mock index methods
    index.upsert.return_value = {"upserted_count": 1}
    index.query.return_value = {"matches": []}
    index.fetch.return_value = {"vectors": {}}
    index.delete.return_value = {}
    index.describe_index_stats.return_value = {
        "namespaces": {},
        "total_vector_count": 0,
    }

    return client, index


@pytest.fixture
def storage_with_mock_client(mock_pinecone_client):
    """Create PineconeStorage with mocked client."""
    from alma.storage.pinecone import PineconeStorage

    client, index = mock_pinecone_client

    with patch("alma.storage.pinecone.Pinecone", return_value=client):
        storage = PineconeStorage(api_key="test-api-key", index_name="test-index")
        storage._index = index

    return storage, index


class TestPineconeStorageInit:
    """Tests for storage initialization."""

    def test_from_config_basic(self):
        """Test from_config with basic config."""
        from alma.storage.pinecone import PineconeStorage

        config = {
            "pinecone": {
                "api_key": "test-api-key",
                "index_name": "test-index",
            },
            "embedding_dim": 384,
        }

        with patch.object(PineconeStorage, "__init__", return_value=None) as mock_init:
            PineconeStorage.from_config(config)
            mock_init.assert_called_once()
            call_kwargs = mock_init.call_args[1]
            assert call_kwargs["api_key"] == "test-api-key"
            assert call_kwargs["index_name"] == "test-index"
            assert call_kwargs["embedding_dim"] == 384

    def test_from_config_with_env_vars(self):
        """Test from_config with environment variable expansion."""
        import os

        from alma.storage.pinecone import PineconeStorage

        os.environ["TEST_PINECONE_KEY"] = "env-api-key"
        os.environ["TEST_PINECONE_INDEX"] = "env-index"

        config = {
            "pinecone": {
                "api_key": "${TEST_PINECONE_KEY}",
                "index_name": "${TEST_PINECONE_INDEX}",
            },
        }

        with patch.object(PineconeStorage, "__init__", return_value=None) as mock_init:
            PineconeStorage.from_config(config)
            call_kwargs = mock_init.call_args[1]
            assert call_kwargs["api_key"] == "env-api-key"
            assert call_kwargs["index_name"] == "env-index"

        # Cleanup
        del os.environ["TEST_PINECONE_KEY"]
        del os.environ["TEST_PINECONE_INDEX"]

    def test_from_config_defaults(self):
        """Test from_config uses defaults when not specified."""
        from alma.storage.pinecone import PineconeStorage

        config = {
            "pinecone": {
                "api_key": "test-key",
            },
        }

        with patch.object(PineconeStorage, "__init__", return_value=None) as mock_init:
            PineconeStorage.from_config(config)
            call_kwargs = mock_init.call_args[1]
            assert call_kwargs["index_name"] == "alma-memory"
            assert call_kwargs["cloud"] == "aws"
            assert call_kwargs["region"] == "us-east-1"


class TestPineconeStorageMetadata:
    """Tests for metadata conversion."""

    def test_metadata_to_pinecone_heuristic(self, storage_with_mock_client):
        """Test converting Heuristic to Pinecone metadata."""
        from alma.storage.pinecone import NAMESPACE_HEURISTICS
        from alma.types import Heuristic

        storage, _ = storage_with_mock_client

        heuristic = Heuristic(
            id="h-123",
            agent="Helena",
            project_id="test-project",
            condition="form with fields",
            strategy="test happy path",
            confidence=0.85,
            occurrence_count=10,
            success_count=8,
            last_validated=datetime(2024, 1, 1, tzinfo=timezone.utc),
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            metadata={"tags": ["forms"]},
        )

        metadata = storage._metadata_to_pinecone(heuristic, NAMESPACE_HEURISTICS)

        assert metadata["agent"] == "Helena"
        assert metadata["project_id"] == "test-project"
        assert metadata["condition"] == "form with fields"
        assert metadata["strategy"] == "test happy path"
        assert metadata["confidence"] == 0.85
        assert metadata["occurrence_count"] == 10
        assert metadata["success_count"] == 8
        assert "metadata_json" in metadata

    def test_metadata_to_heuristic(self, storage_with_mock_client):
        """Test converting Pinecone metadata to Heuristic."""
        storage, _ = storage_with_mock_client

        metadata = {
            "agent": "Helena",
            "project_id": "test-project",
            "condition": "form condition",
            "strategy": "test strategy",
            "confidence": 0.9,
            "occurrence_count": 5,
            "success_count": 4,
            "last_validated": "2024-01-01T00:00:00+00:00",
            "created_at": "2024-01-01T00:00:00+00:00",
            "metadata_json": '{"key": "value"}',
        }

        heuristic = storage._metadata_to_heuristic("h-123", metadata)

        assert heuristic.id == "h-123"
        assert heuristic.agent == "Helena"
        assert heuristic.project_id == "test-project"
        assert heuristic.confidence == 0.9
        assert heuristic.occurrence_count == 5
        assert heuristic.metadata == {"key": "value"}

    def test_metadata_to_outcome(self, storage_with_mock_client):
        """Test converting Pinecone metadata to Outcome."""
        storage, _ = storage_with_mock_client

        metadata = {
            "agent": "Victor",
            "project_id": "test-project",
            "task_type": "api_testing",
            "task_description": "Test endpoint",
            "success": True,
            "strategy_used": "happy_path",
            "duration_ms": 100,
            "error_message": "",
            "user_feedback": "",
            "timestamp": "2024-01-01T00:00:00+00:00",
            "metadata_json": "{}",
        }

        outcome = storage._metadata_to_outcome("o-123", metadata)

        assert outcome.id == "o-123"
        assert outcome.agent == "Victor"
        assert outcome.task_type == "api_testing"
        assert outcome.success is True
        assert outcome.duration_ms == 100

    def test_metadata_to_preference(self, storage_with_mock_client):
        """Test converting Pinecone metadata to UserPreference."""
        storage, _ = storage_with_mock_client

        metadata = {
            "user_id": "user-123",
            "category": "code_style",
            "preference": "Use tabs",
            "source": "explicit",
            "confidence": 1.0,
            "timestamp": "2024-01-01T00:00:00+00:00",
            "metadata_json": "{}",
        }

        preference = storage._metadata_to_preference("p-123", metadata)

        assert preference.id == "p-123"
        assert preference.user_id == "user-123"
        assert preference.category == "code_style"
        assert preference.preference == "Use tabs"

    def test_metadata_to_domain_knowledge(self, storage_with_mock_client):
        """Test converting Pinecone metadata to DomainKnowledge."""
        storage, _ = storage_with_mock_client

        metadata = {
            "agent": "Daphne",
            "project_id": "test-project",
            "domain": "auth",
            "fact": "Uses JWT tokens",
            "source": "code_analysis",
            "confidence": 0.95,
            "last_verified": "2024-01-01T00:00:00+00:00",
            "metadata_json": "{}",
        }

        knowledge = storage._metadata_to_domain_knowledge("d-123", metadata)

        assert knowledge.id == "d-123"
        assert knowledge.agent == "Daphne"
        assert knowledge.domain == "auth"
        assert knowledge.fact == "Uses JWT tokens"

    def test_metadata_to_anti_pattern(self, storage_with_mock_client):
        """Test converting Pinecone metadata to AntiPattern."""
        storage, _ = storage_with_mock_client

        metadata = {
            "agent": "Helena",
            "project_id": "test-project",
            "pattern": "Using sleep",
            "why_bad": "Flaky tests",
            "better_alternative": "Use explicit waits",
            "occurrence_count": 3,
            "last_seen": "2024-01-01T00:00:00+00:00",
            "created_at": "2024-01-01T00:00:00+00:00",
            "metadata_json": "{}",
        }

        anti_pattern = storage._metadata_to_anti_pattern("a-123", metadata)

        assert anti_pattern.id == "a-123"
        assert anti_pattern.pattern == "Using sleep"
        assert anti_pattern.why_bad == "Flaky tests"
        assert anti_pattern.occurrence_count == 3


class TestPineconeStorageWriteOps:
    """Tests for write operations."""

    def test_save_heuristic(self, storage_with_mock_client):
        """Test saving a heuristic."""
        from alma.storage.pinecone import NAMESPACE_HEURISTICS
        from alma.types import Heuristic

        storage, index = storage_with_mock_client

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
            embedding=[0.1] * 384,
            metadata={"tags": ["forms"]},
        )

        result = storage.save_heuristic(heuristic)

        assert result == "test-h-1"
        index.upsert.assert_called_once()
        call_args = index.upsert.call_args
        assert call_args[1]["namespace"] == NAMESPACE_HEURISTICS
        assert call_args[1]["vectors"][0]["id"] == "test-h-1"

    def test_save_heuristic_without_embedding(self, storage_with_mock_client):
        """Test saving a heuristic without an embedding uses zero vector."""
        from alma.types import Heuristic

        storage, index = storage_with_mock_client

        heuristic = Heuristic(
            id="test-h-2",
            agent="Helena",
            project_id="test-project",
            condition="test condition",
            strategy="test strategy",
            confidence=0.5,
            occurrence_count=1,
            success_count=1,
            last_validated=datetime.now(timezone.utc),
            created_at=datetime.now(timezone.utc),
            embedding=None,  # No embedding
        )

        result = storage.save_heuristic(heuristic)

        assert result == "test-h-2"
        call_args = index.upsert.call_args
        vector_values = call_args[1]["vectors"][0]["values"]
        # Should be zero vector
        assert all(v == 0.0 for v in vector_values)
        assert len(vector_values) == 384

    def test_save_outcome(self, storage_with_mock_client):
        """Test saving an outcome."""
        from alma.storage.pinecone import NAMESPACE_OUTCOMES
        from alma.types import Outcome

        storage, index = storage_with_mock_client

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
        call_args = index.upsert.call_args
        assert call_args[1]["namespace"] == NAMESPACE_OUTCOMES

    def test_save_user_preference(self, storage_with_mock_client):
        """Test saving a user preference."""
        from alma.storage.pinecone import NAMESPACE_PREFERENCES
        from alma.types import UserPreference

        storage, index = storage_with_mock_client

        preference = UserPreference(
            id="test-p-1",
            user_id="user-123",
            category="code_style",
            preference="Use 4-space indentation",
            source="explicit",
            confidence=1.0,
            timestamp=datetime.now(timezone.utc),
        )

        result = storage.save_user_preference(preference)

        assert result == "test-p-1"
        call_args = index.upsert.call_args
        assert call_args[1]["namespace"] == NAMESPACE_PREFERENCES

    def test_save_domain_knowledge(self, storage_with_mock_client):
        """Test saving domain knowledge."""
        from alma.storage.pinecone import NAMESPACE_DOMAIN_KNOWLEDGE
        from alma.types import DomainKnowledge

        storage, index = storage_with_mock_client

        knowledge = DomainKnowledge(
            id="test-d-1",
            agent="Daphne",
            project_id="test-project",
            domain="authentication",
            fact="Uses JWT with 24h expiry",
            source="code_analysis",
            confidence=0.95,
            last_verified=datetime.now(timezone.utc),
        )

        result = storage.save_domain_knowledge(knowledge)

        assert result == "test-d-1"
        call_args = index.upsert.call_args
        assert call_args[1]["namespace"] == NAMESPACE_DOMAIN_KNOWLEDGE

    def test_save_anti_pattern(self, storage_with_mock_client):
        """Test saving an anti-pattern."""
        from alma.storage.pinecone import NAMESPACE_ANTI_PATTERNS
        from alma.types import AntiPattern

        storage, index = storage_with_mock_client

        anti_pattern = AntiPattern(
            id="test-a-1",
            agent="Helena",
            project_id="test-project",
            pattern="Using fixed sleep()",
            why_bad="Causes flaky tests",
            better_alternative="Use explicit waits",
            occurrence_count=3,
            last_seen=datetime.now(timezone.utc),
            created_at=datetime.now(timezone.utc),
        )

        result = storage.save_anti_pattern(anti_pattern)

        assert result == "test-a-1"
        call_args = index.upsert.call_args
        assert call_args[1]["namespace"] == NAMESPACE_ANTI_PATTERNS


class TestPineconeStorageBatchOps:
    """Tests for batch write operations."""

    def test_save_heuristics_batch(self, storage_with_mock_client):
        """Test batch saving heuristics."""
        from alma.types import Heuristic

        storage, index = storage_with_mock_client

        heuristics = [
            Heuristic(
                id=f"test-h-{i}",
                agent="Helena",
                project_id="test-project",
                condition=f"condition {i}",
                strategy=f"strategy {i}",
                confidence=0.5,
                occurrence_count=1,
                success_count=1,
                last_validated=datetime.now(timezone.utc),
                created_at=datetime.now(timezone.utc),
            )
            for i in range(5)
        ]

        result = storage.save_heuristics(heuristics)

        assert len(result) == 5
        assert result == [f"test-h-{i}" for i in range(5)]
        # Should call upsert once (batch is less than 100)
        assert index.upsert.call_count == 1

    def test_save_heuristics_empty(self, storage_with_mock_client):
        """Test batch saving empty list."""
        storage, index = storage_with_mock_client

        result = storage.save_heuristics([])

        assert result == []
        index.upsert.assert_not_called()


class TestPineconeStorageReadOps:
    """Tests for read operations."""

    def test_get_heuristics_basic(self, storage_with_mock_client):
        """Test basic heuristic retrieval."""
        storage, index = storage_with_mock_client

        index.query.return_value = {"matches": []}

        results = storage.get_heuristics(
            project_id="test-project",
            agent="Helena",
            top_k=5,
        )

        assert results == []
        index.query.assert_called_once()
        call_args = index.query.call_args[1]
        assert call_args["namespace"] == "heuristics"
        assert call_args["top_k"] == 5
        assert call_args["include_metadata"] is True

    def test_get_heuristics_with_embedding(self, storage_with_mock_client):
        """Test heuristic retrieval with vector search."""
        storage, index = storage_with_mock_client

        index.query.return_value = {
            "matches": [
                {
                    "id": "h-1",
                    "score": 0.95,
                    "metadata": {
                        "agent": "Helena",
                        "project_id": "test-project",
                        "condition": "test",
                        "strategy": "strategy",
                        "confidence": 0.8,
                        "occurrence_count": 5,
                        "success_count": 4,
                        "last_validated": "2024-01-01T00:00:00+00:00",
                        "created_at": "2024-01-01T00:00:00+00:00",
                        "metadata_json": "{}",
                    },
                }
            ]
        }

        embedding = [0.1] * 384
        results = storage.get_heuristics(
            project_id="test-project",
            embedding=embedding,
            top_k=5,
        )

        assert len(results) == 1
        assert results[0].id == "h-1"
        assert results[0].agent == "Helena"

        # Verify the embedding was used
        call_args = index.query.call_args[1]
        assert call_args["vector"] == embedding

    def test_get_heuristics_with_filter(self, storage_with_mock_client):
        """Test heuristic retrieval with metadata filters."""
        storage, index = storage_with_mock_client

        index.query.return_value = {"matches": []}

        storage.get_heuristics(
            project_id="test-project",
            agent="Helena",
            min_confidence=0.5,
            top_k=5,
        )

        call_args = index.query.call_args[1]
        filter_dict = call_args["filter"]
        # Should have project_id, agent, and confidence filters
        assert "$and" in filter_dict or "project_id" in filter_dict

    def test_get_outcomes_success_only(self, storage_with_mock_client):
        """Test outcome retrieval with success_only filter."""
        storage, index = storage_with_mock_client

        index.query.return_value = {"matches": []}

        storage.get_outcomes(
            project_id="test-project",
            success_only=True,
        )

        call_args = index.query.call_args[1]
        # Should have success filter in the query
        assert "filter" in call_args

    def test_get_user_preferences(self, storage_with_mock_client):
        """Test user preference retrieval."""
        storage, index = storage_with_mock_client

        index.query.return_value = {
            "matches": [
                {
                    "id": "p-1",
                    "score": 1.0,
                    "metadata": {
                        "user_id": "user-123",
                        "category": "code_style",
                        "preference": "Use tabs",
                        "source": "explicit",
                        "confidence": 1.0,
                        "timestamp": "2024-01-01T00:00:00+00:00",
                        "metadata_json": "{}",
                    },
                }
            ]
        }

        results = storage.get_user_preferences(
            user_id="user-123",
            category="code_style",
        )

        assert len(results) == 1
        assert results[0].user_id == "user-123"
        assert results[0].category == "code_style"


class TestPineconeStorageMultiAgent:
    """Tests for multi-agent memory sharing."""

    def test_get_heuristics_for_agents(self, storage_with_mock_client):
        """Test getting heuristics from multiple agents."""
        storage, index = storage_with_mock_client

        index.query.return_value = {
            "matches": [
                {
                    "id": "h-1",
                    "metadata": {
                        "agent": "Helena",
                        "project_id": "test-project",
                        "condition": "test",
                        "strategy": "strategy",
                        "confidence": 0.8,
                        "occurrence_count": 5,
                        "success_count": 4,
                        "last_validated": "2024-01-01T00:00:00+00:00",
                        "created_at": "2024-01-01T00:00:00+00:00",
                        "metadata_json": "{}",
                    },
                },
                {
                    "id": "h-2",
                    "metadata": {
                        "agent": "Victor",
                        "project_id": "test-project",
                        "condition": "test2",
                        "strategy": "strategy2",
                        "confidence": 0.7,
                        "occurrence_count": 3,
                        "success_count": 2,
                        "last_validated": "2024-01-01T00:00:00+00:00",
                        "created_at": "2024-01-01T00:00:00+00:00",
                        "metadata_json": "{}",
                    },
                },
            ]
        }

        results = storage.get_heuristics_for_agents(
            project_id="test-project",
            agents=["Helena", "Victor"],
            top_k=5,
        )

        assert len(results) == 2
        # Verify $in filter was used
        call_args = index.query.call_args[1]
        filter_dict = call_args["filter"]
        assert "$and" in filter_dict
        # Check that agent filter uses $in
        agent_filter = next((f for f in filter_dict["$and"] if "agent" in f), None)
        assert agent_filter is not None
        assert "$in" in agent_filter["agent"]

    def test_get_heuristics_for_agents_empty(self, storage_with_mock_client):
        """Test getting heuristics with empty agent list."""
        storage, index = storage_with_mock_client

        results = storage.get_heuristics_for_agents(
            project_id="test-project",
            agents=[],
            top_k=5,
        )

        assert results == []
        index.query.assert_not_called()


class TestPineconeStorageUpdateOps:
    """Tests for update operations."""

    def test_update_heuristic(self, storage_with_mock_client):
        """Test updating a heuristic."""
        storage, index = storage_with_mock_client

        # Mock fetch to return existing record
        index.fetch.return_value = {
            "vectors": {
                "h-1": {
                    "values": [0.1] * 384,
                    "metadata": {
                        "agent": "Helena",
                        "project_id": "test-project",
                        "condition": "old condition",
                        "strategy": "old strategy",
                        "confidence": 0.5,
                        "occurrence_count": 1,
                        "success_count": 1,
                        "last_validated": "2024-01-01T00:00:00+00:00",
                        "created_at": "2024-01-01T00:00:00+00:00",
                        "metadata_json": "{}",
                    },
                }
            }
        }

        result = storage.update_heuristic(
            "h-1", {"confidence": 0.9, "strategy": "new strategy"}
        )

        assert result is True
        # Should fetch then upsert
        index.fetch.assert_called_once()
        index.upsert.assert_called_once()

        # Verify updated metadata
        call_args = index.upsert.call_args[1]
        updated_metadata = call_args["vectors"][0]["metadata"]
        assert updated_metadata["confidence"] == 0.9
        assert updated_metadata["strategy"] == "new strategy"

    def test_update_heuristic_not_found(self, storage_with_mock_client):
        """Test updating non-existent heuristic."""
        storage, index = storage_with_mock_client

        index.fetch.return_value = {"vectors": {}}

        result = storage.update_heuristic("nonexistent", {"confidence": 0.9})

        assert result is False
        index.upsert.assert_not_called()

    def test_increment_heuristic_occurrence(self, storage_with_mock_client):
        """Test incrementing heuristic occurrence count."""
        storage, index = storage_with_mock_client

        index.fetch.return_value = {
            "vectors": {
                "h-1": {
                    "values": [0.1] * 384,
                    "metadata": {
                        "occurrence_count": 5,
                        "success_count": 4,
                        "last_validated": "2024-01-01T00:00:00+00:00",
                    },
                }
            }
        }

        result = storage.increment_heuristic_occurrence("h-1", success=True)

        assert result is True
        call_args = index.upsert.call_args[1]
        updated_metadata = call_args["vectors"][0]["metadata"]
        assert updated_metadata["occurrence_count"] == 6
        assert updated_metadata["success_count"] == 5

    def test_update_heuristic_confidence(self, storage_with_mock_client):
        """Test updating heuristic confidence."""
        storage, index = storage_with_mock_client

        index.fetch.return_value = {
            "vectors": {
                "h-1": {
                    "values": [0.1] * 384,
                    "metadata": {"confidence": 0.5},
                }
            }
        }

        result = storage.update_heuristic_confidence("h-1", 0.95)

        assert result is True


class TestPineconeStorageDeleteOps:
    """Tests for delete operations."""

    def test_delete_heuristic(self, storage_with_mock_client):
        """Test deleting a heuristic."""
        storage, index = storage_with_mock_client

        result = storage.delete_heuristic("h-1")

        assert result is True
        index.delete.assert_called_once_with(ids=["h-1"], namespace="heuristics")

    def test_delete_outcome(self, storage_with_mock_client):
        """Test deleting an outcome."""
        storage, index = storage_with_mock_client

        result = storage.delete_outcome("o-1")

        assert result is True
        index.delete.assert_called_once_with(ids=["o-1"], namespace="outcomes")

    def test_delete_domain_knowledge(self, storage_with_mock_client):
        """Test deleting domain knowledge."""
        storage, index = storage_with_mock_client

        result = storage.delete_domain_knowledge("d-1")

        assert result is True
        index.delete.assert_called_once_with(ids=["d-1"], namespace="domain_knowledge")

    def test_delete_anti_pattern(self, storage_with_mock_client):
        """Test deleting an anti-pattern."""
        storage, index = storage_with_mock_client

        result = storage.delete_anti_pattern("a-1")

        assert result is True
        index.delete.assert_called_once_with(ids=["a-1"], namespace="anti_patterns")

    def test_delete_outcomes_older_than(self, storage_with_mock_client):
        """Test deleting old outcomes."""
        storage, index = storage_with_mock_client

        # Mock query to return old outcomes
        index.query.return_value = {
            "matches": [
                {"id": "o-1", "metadata": {"timestamp": "2023-01-01T00:00:00+00:00"}},
                {"id": "o-2", "metadata": {"timestamp": "2023-06-01T00:00:00+00:00"}},
                {"id": "o-3", "metadata": {"timestamp": "2024-06-01T00:00:00+00:00"}},
            ]
        }

        cutoff = datetime(2024, 1, 1, tzinfo=timezone.utc)
        result = storage.delete_outcomes_older_than("test-project", cutoff)

        # Should delete o-1 and o-2 (before cutoff)
        assert result == 2
        # Delete should be called
        index.delete.assert_called()

    def test_delete_low_confidence_heuristics(self, storage_with_mock_client):
        """Test deleting low-confidence heuristics."""
        storage, index = storage_with_mock_client

        index.query.return_value = {
            "matches": [
                {"id": "h-1", "metadata": {"confidence": 0.2}},
                {"id": "h-2", "metadata": {"confidence": 0.4}},
                {"id": "h-3", "metadata": {"confidence": 0.8}},
            ]
        }

        result = storage.delete_low_confidence_heuristics(
            "test-project", below_confidence=0.5
        )

        # Should delete h-1 and h-2 (below 0.5)
        assert result == 2


class TestPineconeStorageStats:
    """Tests for statistics."""

    def test_get_stats(self, storage_with_mock_client):
        """Test getting storage statistics."""
        storage, index = storage_with_mock_client

        index.describe_index_stats.return_value = {
            "namespaces": {
                "heuristics": {"vector_count": 10},
                "outcomes": {"vector_count": 50},
                "domain_knowledge": {"vector_count": 5},
                "anti_patterns": {"vector_count": 3},
                "preferences": {"vector_count": 8},
            },
            "total_vector_count": 76,
        }

        stats = storage.get_stats(project_id="test-project")

        assert stats["project_id"] == "test-project"
        assert stats["storage_type"] == "pinecone"
        assert stats["heuristics_count"] == 10
        assert stats["outcomes_count"] == 50
        assert stats["domain_knowledge_count"] == 5
        assert stats["anti_patterns_count"] == 3
        assert stats["preferences_count"] == 8
        assert stats["total_count"] == 76

    def test_get_stats_empty_index(self, storage_with_mock_client):
        """Test getting stats from empty index."""
        storage, index = storage_with_mock_client

        index.describe_index_stats.return_value = {
            "namespaces": {},
            "total_vector_count": 0,
        }

        stats = storage.get_stats(project_id="test-project")

        assert stats["heuristics_count"] == 0
        assert stats["total_count"] == 0


class TestPineconeStorageFilters:
    """Tests for filter building."""

    def test_build_filter_single_condition(self, storage_with_mock_client):
        """Test building filter with single condition."""
        storage, _ = storage_with_mock_client

        filter_dict = storage._build_filter(project_id="test-project")

        assert filter_dict == {"project_id": {"$eq": "test-project"}}

    def test_build_filter_multiple_conditions(self, storage_with_mock_client):
        """Test building filter with multiple conditions."""
        storage, _ = storage_with_mock_client

        filter_dict = storage._build_filter(
            project_id="test-project",
            agent="Helena",
            min_confidence=0.5,
        )

        assert "$and" in filter_dict
        conditions = filter_dict["$and"]
        assert len(conditions) == 3

    def test_build_filter_empty(self, storage_with_mock_client):
        """Test building empty filter."""
        storage, _ = storage_with_mock_client

        filter_dict = storage._build_filter()

        assert filter_dict == {}


class TestPineconeStorageHelpers:
    """Tests for helper methods."""

    def test_get_zero_vector(self, storage_with_mock_client):
        """Test zero vector generation."""
        storage, _ = storage_with_mock_client

        zero_vector = storage._get_zero_vector()

        assert len(zero_vector) == 384
        assert all(v == 0.0 for v in zero_vector)

    def test_parse_datetime_valid(self, storage_with_mock_client):
        """Test parsing valid datetime string."""
        storage, _ = storage_with_mock_client

        result = storage._parse_datetime("2024-01-01T00:00:00+00:00")

        assert result is not None
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 1

    def test_parse_datetime_none(self, storage_with_mock_client):
        """Test parsing None datetime."""
        storage, _ = storage_with_mock_client

        result = storage._parse_datetime(None)

        assert result is None

    def test_parse_datetime_empty(self, storage_with_mock_client):
        """Test parsing empty datetime string."""
        storage, _ = storage_with_mock_client

        result = storage._parse_datetime("")

        assert result is None

    def test_parse_datetime_invalid(self, storage_with_mock_client):
        """Test parsing invalid datetime string."""
        storage, _ = storage_with_mock_client

        result = storage._parse_datetime("not-a-date")

        assert result is None

    def test_close(self, storage_with_mock_client):
        """Test close method (no-op for Pinecone)."""
        storage, _ = storage_with_mock_client

        # Should not raise
        storage.close()
