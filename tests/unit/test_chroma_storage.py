"""
Unit tests for Chroma storage backend.

These tests verify the Chroma storage implementation using mocked ChromaDB
client. For integration tests with a real Chroma instance, see
tests/integration/test_chroma_integration.py
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

# Skip all tests if chromadb is not installed
chromadb_available = False
try:
    import chromadb  # noqa: F401

    chromadb_available = True
except ImportError:
    pass

pytestmark = pytest.mark.skipif(
    not chromadb_available,
    reason="chromadb not installed. Install with: pip install 'alma-memory[chroma]'",
)


@pytest.fixture
def mock_collection():
    """Create a mock Chroma collection."""
    collection = MagicMock()
    collection.get.return_value = {
        "ids": [],
        "metadatas": [],
        "documents": [],
        "embeddings": None,
    }
    collection.query.return_value = {
        "ids": [[]],
        "metadatas": [[]],
        "documents": [[]],
        "embeddings": None,
    }
    return collection


@pytest.fixture
def mock_client(mock_collection):
    """Create a mock Chroma client."""
    client = MagicMock()
    client.get_or_create_collection.return_value = mock_collection
    return client


@pytest.fixture
def storage_with_mock_client(mock_client, mock_collection):
    """Create ChromaStorage with mocked client."""
    from alma.storage.chroma import ChromaStorage

    with patch.object(ChromaStorage, "__init__", lambda self, **kwargs: None):
        storage = ChromaStorage()
        storage._client = mock_client
        storage._mode = "ephemeral"
        storage.embedding_dim = 384
        storage._collection_metadata = {"hnsw:space": "cosine"}

        # Set up collections
        storage._heuristics = mock_collection
        storage._outcomes = mock_collection
        storage._preferences = mock_collection
        storage._domain_knowledge = mock_collection
        storage._anti_patterns = mock_collection

    return storage


class TestChromaStorageInit:
    """Tests for storage initialization."""

    def test_from_config_persistent_mode(self):
        """Test from_config with persistent directory."""
        from alma.storage.chroma import ChromaStorage

        config = {
            "chroma": {
                "persist_directory": "/tmp/chroma_test",
            },
            "embedding_dim": 384,
        }

        with patch.object(
            ChromaStorage, "__init__", return_value=None
        ) as mock_init:
            ChromaStorage.from_config(config)
            mock_init.assert_called_once()
            call_kwargs = mock_init.call_args[1]
            assert call_kwargs["persist_directory"] == "/tmp/chroma_test"
            assert call_kwargs["host"] is None
            assert call_kwargs["port"] is None

    def test_from_config_client_server_mode(self):
        """Test from_config with host/port."""
        from alma.storage.chroma import ChromaStorage

        config = {
            "chroma": {
                "host": "localhost",
                "port": 8000,
            },
            "embedding_dim": 768,
        }

        with patch.object(
            ChromaStorage, "__init__", return_value=None
        ) as mock_init:
            ChromaStorage.from_config(config)
            mock_init.assert_called_once()
            call_kwargs = mock_init.call_args[1]
            assert call_kwargs["host"] == "localhost"
            assert call_kwargs["port"] == 8000
            assert call_kwargs["persist_directory"] is None

    def test_from_config_with_env_vars(self):
        """Test from_config with environment variable expansion."""
        import os

        from alma.storage.chroma import ChromaStorage

        os.environ["TEST_CHROMA_HOST"] = "chroma.example.com"
        os.environ["TEST_CHROMA_PORT"] = "9000"

        config = {
            "chroma": {
                "host": "${TEST_CHROMA_HOST}",
                "port": "${TEST_CHROMA_PORT}",
            },
        }

        with patch.object(
            ChromaStorage, "__init__", return_value=None
        ) as mock_init:
            ChromaStorage.from_config(config)
            call_kwargs = mock_init.call_args[1]
            assert call_kwargs["host"] == "chroma.example.com"
            assert call_kwargs["port"] == 9000

        # Cleanup
        del os.environ["TEST_CHROMA_HOST"]
        del os.environ["TEST_CHROMA_PORT"]


class TestChromaStorageWriteOps:
    """Tests for write operations."""

    def test_save_heuristic(self, storage_with_mock_client, mock_collection):
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
            embedding=[0.1] * 384,
            metadata={"tags": ["forms"]},
        )

        result = storage.save_heuristic(heuristic)
        assert result == "test-h-1"
        mock_collection.upsert.assert_called_once()

    def test_save_heuristic_without_embedding(self, storage_with_mock_client, mock_collection):
        """Test saving a heuristic without embedding."""
        from alma.types import Heuristic

        storage = storage_with_mock_client

        heuristic = Heuristic(
            id="test-h-2",
            agent="Helena",
            project_id="test-project",
            condition="simple condition",
            strategy="simple strategy",
            confidence=0.5,
            occurrence_count=5,
            success_count=3,
            last_validated=datetime.now(timezone.utc),
            created_at=datetime.now(timezone.utc),
            metadata={},
        )

        result = storage.save_heuristic(heuristic)
        assert result == "test-h-2"
        mock_collection.upsert.assert_called_once()
        # Verify no embeddings were passed
        call_kwargs = mock_collection.upsert.call_args[1]
        assert "embeddings" not in call_kwargs

    def test_save_outcome(self, storage_with_mock_client, mock_collection):
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
        mock_collection.upsert.assert_called_once()

    def test_save_user_preference(self, storage_with_mock_client, mock_collection):
        """Test saving a user preference."""
        from alma.types import UserPreference

        storage = storage_with_mock_client

        preference = UserPreference(
            id="test-p-1",
            user_id="user123",
            category="communication",
            preference="No emojis in documentation",
            source="explicit_instruction",
            confidence=1.0,
            timestamp=datetime.now(timezone.utc),
            metadata={},
        )

        result = storage.save_user_preference(preference)
        assert result == "test-p-1"
        mock_collection.upsert.assert_called_once()

    def test_save_domain_knowledge(self, storage_with_mock_client, mock_collection):
        """Test saving domain knowledge."""
        from alma.types import DomainKnowledge

        storage = storage_with_mock_client

        knowledge = DomainKnowledge(
            id="test-dk-1",
            agent="Helena",
            project_id="test-project",
            domain="authentication",
            fact="Login endpoint uses JWT with 24h expiry",
            source="code_analysis",
            confidence=0.95,
            last_verified=datetime.now(timezone.utc),
            metadata={},
        )

        result = storage.save_domain_knowledge(knowledge)
        assert result == "test-dk-1"
        mock_collection.upsert.assert_called_once()

    def test_save_anti_pattern(self, storage_with_mock_client, mock_collection):
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
            metadata={},
        )

        result = storage.save_anti_pattern(anti_pattern)
        assert result == "test-ap-1"
        mock_collection.upsert.assert_called_once()


class TestChromaStorageBatchOps:
    """Tests for batch write operations."""

    def test_save_heuristics_batch(self, storage_with_mock_client, mock_collection):
        """Test saving multiple heuristics in a batch."""
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
                metadata={},
            )
            for i in range(3)
        ]

        result = storage.save_heuristics(heuristics)
        assert len(result) == 3
        mock_collection.upsert.assert_called_once()

    def test_save_outcomes_batch(self, storage_with_mock_client, mock_collection):
        """Test saving multiple outcomes in a batch."""
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
                strategy_used="strategy",
                timestamp=datetime.now(timezone.utc),
                metadata={},
            )
            for i in range(3)
        ]

        result = storage.save_outcomes(outcomes)
        assert len(result) == 3
        mock_collection.upsert.assert_called_once()


class TestChromaStorageReadOps:
    """Tests for read operations."""

    def test_get_heuristics_basic(self, storage_with_mock_client, mock_collection):
        """Test basic heuristic retrieval."""
        storage = storage_with_mock_client

        # Setup mock response
        mock_collection.get.return_value = {
            "ids": ["h-1", "h-2"],
            "metadatas": [
                {
                    "agent": "Helena",
                    "project_id": "test-project",
                    "condition": "condition 1",
                    "strategy": "strategy 1",
                    "confidence": 0.9,
                    "occurrence_count": 5,
                    "success_count": 4,
                    "last_validated": datetime.now(timezone.utc).isoformat(),
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "extra_metadata": "{}",
                },
                {
                    "agent": "Helena",
                    "project_id": "test-project",
                    "condition": "condition 2",
                    "strategy": "strategy 2",
                    "confidence": 0.8,
                    "occurrence_count": 3,
                    "success_count": 2,
                    "last_validated": datetime.now(timezone.utc).isoformat(),
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "extra_metadata": "{}",
                },
            ],
            "documents": ["condition 1\nstrategy 1", "condition 2\nstrategy 2"],
            "embeddings": None,
        }

        results = storage.get_heuristics(
            project_id="test-project",
            agent="Helena",
            top_k=5,
        )

        assert len(results) == 2
        assert results[0].id == "h-1"
        assert results[0].confidence == 0.9
        mock_collection.get.assert_called_once()

    def test_get_heuristics_with_embedding(self, storage_with_mock_client, mock_collection):
        """Test heuristic retrieval with vector search."""
        storage = storage_with_mock_client

        mock_collection.query.return_value = {
            "ids": [["h-1"]],
            "metadatas": [[
                {
                    "agent": "Helena",
                    "project_id": "test-project",
                    "condition": "condition 1",
                    "strategy": "strategy 1",
                    "confidence": 0.9,
                    "occurrence_count": 5,
                    "success_count": 4,
                    "last_validated": datetime.now(timezone.utc).isoformat(),
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "extra_metadata": "{}",
                }
            ]],
            "documents": [["condition 1\nstrategy 1"]],
            "embeddings": [[[0.1] * 384]],
        }

        embedding = [0.1] * 384
        results = storage.get_heuristics(
            project_id="test-project",
            embedding=embedding,
            top_k=5,
        )

        assert len(results) == 1
        mock_collection.query.assert_called_once()
        # Verify query_embeddings was used
        call_kwargs = mock_collection.query.call_args[1]
        assert "query_embeddings" in call_kwargs

    def test_get_outcomes(self, storage_with_mock_client, mock_collection):
        """Test outcome retrieval."""
        storage = storage_with_mock_client

        mock_collection.get.return_value = {
            "ids": ["o-1"],
            "metadatas": [{
                "agent": "Victor",
                "project_id": "test-project",
                "task_type": "api_testing",
                "success": True,
                "strategy_used": "happy_path",
                "duration_ms": 100,
                "error_message": "",
                "user_feedback": "",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "extra_metadata": "{}",
            }],
            "documents": ["Test endpoint"],
            "embeddings": None,
        }

        results = storage.get_outcomes(
            project_id="test-project",
            agent="Victor",
        )

        assert len(results) == 1
        assert results[0].success

    def test_get_user_preferences(self, storage_with_mock_client, mock_collection):
        """Test user preference retrieval."""
        storage = storage_with_mock_client

        mock_collection.get.return_value = {
            "ids": ["p-1"],
            "metadatas": [{
                "user_id": "user123",
                "category": "communication",
                "source": "explicit",
                "confidence": 1.0,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "extra_metadata": "{}",
            }],
            "documents": ["No emojis"],
        }

        results = storage.get_user_preferences(user_id="user123")

        assert len(results) == 1
        assert results[0].preference == "No emojis"

    def test_get_domain_knowledge(self, storage_with_mock_client, mock_collection):
        """Test domain knowledge retrieval."""
        storage = storage_with_mock_client

        mock_collection.get.return_value = {
            "ids": ["dk-1"],
            "metadatas": [{
                "agent": "Helena",
                "project_id": "test-project",
                "domain": "auth",
                "source": "code_analysis",
                "confidence": 0.95,
                "last_verified": datetime.now(timezone.utc).isoformat(),
                "extra_metadata": "{}",
            }],
            "documents": ["JWT uses 24h expiry"],
            "embeddings": None,
        }

        results = storage.get_domain_knowledge(
            project_id="test-project",
            domain="auth",
        )

        assert len(results) == 1
        assert results[0].fact == "JWT uses 24h expiry"

    def test_get_anti_patterns(self, storage_with_mock_client, mock_collection):
        """Test anti-pattern retrieval."""
        storage = storage_with_mock_client

        mock_collection.get.return_value = {
            "ids": ["ap-1"],
            "metadatas": [{
                "agent": "Helena",
                "project_id": "test-project",
                "why_bad": "Causes flaky tests",
                "better_alternative": "Use explicit waits",
                "occurrence_count": 3,
                "last_seen": datetime.now(timezone.utc).isoformat(),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "extra_metadata": "{}",
            }],
            "documents": ["Using fixed sleep()"],
            "embeddings": None,
        }

        results = storage.get_anti_patterns(project_id="test-project")

        assert len(results) == 1
        assert "sleep" in results[0].pattern


class TestChromaStorageMultiAgent:
    """Tests for multi-agent memory sharing."""

    def test_get_heuristics_for_agents(self, storage_with_mock_client, mock_collection):
        """Test getting heuristics from multiple agents."""
        storage = storage_with_mock_client

        mock_collection.get.return_value = {
            "ids": ["h-1", "h-2"],
            "metadatas": [
                {
                    "agent": "Helena",
                    "project_id": "test-project",
                    "condition": "condition 1",
                    "strategy": "strategy 1",
                    "confidence": 0.9,
                    "occurrence_count": 5,
                    "success_count": 4,
                    "last_validated": datetime.now(timezone.utc).isoformat(),
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "extra_metadata": "{}",
                },
                {
                    "agent": "Victor",
                    "project_id": "test-project",
                    "condition": "condition 2",
                    "strategy": "strategy 2",
                    "confidence": 0.8,
                    "occurrence_count": 3,
                    "success_count": 2,
                    "last_validated": datetime.now(timezone.utc).isoformat(),
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "extra_metadata": "{}",
                },
            ],
            "documents": ["condition 1\nstrategy 1", "condition 2\nstrategy 2"],
            "embeddings": None,
        }

        results = storage.get_heuristics_for_agents(
            project_id="test-project",
            agents=["Helena", "Victor"],
        )

        assert len(results) == 2

    def test_get_heuristics_for_agents_empty(self, storage_with_mock_client):
        """Test getting heuristics for empty agent list."""
        storage = storage_with_mock_client

        results = storage.get_heuristics_for_agents(
            project_id="test-project",
            agents=[],
        )

        assert results == []


class TestChromaStorageUpdateOps:
    """Tests for update operations."""

    def test_update_heuristic(self, storage_with_mock_client, mock_collection):
        """Test updating a heuristic."""
        storage = storage_with_mock_client

        mock_collection.get.return_value = {
            "ids": ["h-1"],
            "metadatas": [{
                "agent": "Helena",
                "project_id": "test-project",
                "condition": "old condition",
                "strategy": "old strategy",
                "confidence": 0.5,
                "occurrence_count": 3,
                "success_count": 2,
                "last_validated": datetime.now(timezone.utc).isoformat(),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "extra_metadata": "{}",
            }],
            "documents": ["old condition\nold strategy"],
            "embeddings": None,
        }

        result = storage.update_heuristic("h-1", {"confidence": 0.9})

        assert result
        mock_collection.upsert.assert_called_once()

    def test_update_heuristic_not_found(self, storage_with_mock_client, mock_collection):
        """Test updating a non-existent heuristic."""
        storage = storage_with_mock_client

        mock_collection.get.return_value = {
            "ids": [],
            "metadatas": [],
            "documents": [],
            "embeddings": None,
        }

        result = storage.update_heuristic("non-existent", {"confidence": 0.9})

        assert not result

    def test_increment_heuristic_occurrence(self, storage_with_mock_client, mock_collection):
        """Test incrementing heuristic occurrence count."""
        storage = storage_with_mock_client

        mock_collection.get.return_value = {
            "ids": ["h-1"],
            "metadatas": [{
                "agent": "Helena",
                "project_id": "test-project",
                "condition": "condition",
                "strategy": "strategy",
                "confidence": 0.8,
                "occurrence_count": 5,
                "success_count": 4,
                "last_validated": datetime.now(timezone.utc).isoformat(),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "extra_metadata": "{}",
            }],
            "documents": ["condition\nstrategy"],
            "embeddings": None,
        }

        result = storage.increment_heuristic_occurrence("h-1", success=True)

        assert result
        call_kwargs = mock_collection.upsert.call_args[1]
        meta = call_kwargs["metadatas"][0]
        assert meta["occurrence_count"] == 6
        assert meta["success_count"] == 5

    def test_update_heuristic_confidence(self, storage_with_mock_client, mock_collection):
        """Test updating heuristic confidence."""
        storage = storage_with_mock_client

        mock_collection.get.return_value = {
            "ids": ["h-1"],
            "metadatas": [{
                "agent": "Helena",
                "project_id": "test-project",
                "condition": "condition",
                "strategy": "strategy",
                "confidence": 0.5,
                "occurrence_count": 3,
                "success_count": 2,
                "last_validated": datetime.now(timezone.utc).isoformat(),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "extra_metadata": "{}",
            }],
            "documents": ["condition\nstrategy"],
            "embeddings": None,
        }

        result = storage.update_heuristic_confidence("h-1", 0.95)

        assert result

    def test_update_knowledge_confidence(self, storage_with_mock_client, mock_collection):
        """Test updating domain knowledge confidence."""
        storage = storage_with_mock_client

        mock_collection.get.return_value = {
            "ids": ["dk-1"],
            "metadatas": [{
                "agent": "Helena",
                "project_id": "test-project",
                "domain": "auth",
                "source": "code_analysis",
                "confidence": 0.5,
                "last_verified": datetime.now(timezone.utc).isoformat(),
                "extra_metadata": "{}",
            }],
            "documents": ["Some fact"],
            "embeddings": None,
        }

        result = storage.update_knowledge_confidence("dk-1", 0.95)

        assert result


class TestChromaStorageDeleteOps:
    """Tests for delete operations."""

    def test_delete_heuristic(self, storage_with_mock_client, mock_collection):
        """Test deleting a heuristic."""
        storage = storage_with_mock_client

        mock_collection.get.return_value = {
            "ids": ["h-1"],
            "metadatas": [{}],
            "documents": [""],
        }

        result = storage.delete_heuristic("h-1")

        assert result
        mock_collection.delete.assert_called_once_with(ids=["h-1"])

    def test_delete_heuristic_not_found(self, storage_with_mock_client, mock_collection):
        """Test deleting a non-existent heuristic."""
        storage = storage_with_mock_client

        mock_collection.get.return_value = {
            "ids": [],
            "metadatas": [],
            "documents": [],
        }

        result = storage.delete_heuristic("non-existent")

        assert not result

    def test_delete_outcome(self, storage_with_mock_client, mock_collection):
        """Test deleting an outcome."""
        storage = storage_with_mock_client

        mock_collection.get.return_value = {
            "ids": ["o-1"],
            "metadatas": [{}],
            "documents": [""],
        }

        result = storage.delete_outcome("o-1")

        assert result
        mock_collection.delete.assert_called_once()

    def test_delete_domain_knowledge(self, storage_with_mock_client, mock_collection):
        """Test deleting domain knowledge."""
        storage = storage_with_mock_client

        mock_collection.get.return_value = {
            "ids": ["dk-1"],
            "metadatas": [{}],
            "documents": [""],
        }

        result = storage.delete_domain_knowledge("dk-1")

        assert result

    def test_delete_anti_pattern(self, storage_with_mock_client, mock_collection):
        """Test deleting an anti-pattern."""
        storage = storage_with_mock_client

        mock_collection.get.return_value = {
            "ids": ["ap-1"],
            "metadatas": [{}],
            "documents": [""],
        }

        result = storage.delete_anti_pattern("ap-1")

        assert result

    def test_delete_outcomes_older_than(self, storage_with_mock_client, mock_collection):
        """Test deleting old outcomes."""
        storage = storage_with_mock_client

        old_time = datetime(2023, 1, 1, tzinfo=timezone.utc)
        new_time = datetime(2024, 6, 1, tzinfo=timezone.utc)
        cutoff = datetime(2024, 1, 1, tzinfo=timezone.utc)

        mock_collection.get.return_value = {
            "ids": ["o-1", "o-2"],
            "metadatas": [
                {"timestamp": old_time.isoformat()},
                {"timestamp": new_time.isoformat()},
            ],
            "documents": ["old", "new"],
        }

        result = storage.delete_outcomes_older_than(
            project_id="test-project",
            older_than=cutoff,
        )

        assert result == 1
        mock_collection.delete.assert_called_once_with(ids=["o-1"])

    def test_delete_low_confidence_heuristics(self, storage_with_mock_client, mock_collection):
        """Test deleting low-confidence heuristics."""
        storage = storage_with_mock_client

        mock_collection.get.return_value = {
            "ids": ["h-1", "h-2", "h-3"],
            "metadatas": [
                {"confidence": 0.2},
                {"confidence": 0.8},
                {"confidence": 0.1},
            ],
            "documents": ["", "", ""],
        }

        result = storage.delete_low_confidence_heuristics(
            project_id="test-project",
            below_confidence=0.5,
        )

        assert result == 2
        mock_collection.delete.assert_called_once()
        deleted_ids = mock_collection.delete.call_args[1]["ids"]
        assert "h-1" in deleted_ids
        assert "h-3" in deleted_ids
        assert "h-2" not in deleted_ids


class TestChromaStorageStats:
    """Tests for statistics."""

    def test_get_stats(self, storage_with_mock_client, mock_collection):
        """Test getting storage statistics."""
        storage = storage_with_mock_client

        mock_collection.get.return_value = {
            "ids": ["1", "2", "3"],
            "metadatas": [{}, {}, {}],
            "documents": ["", "", ""],
        }

        stats = storage.get_stats(project_id="test-project")

        assert stats["project_id"] == "test-project"
        assert stats["storage_type"] == "chroma"
        assert stats["mode"] == "ephemeral"
        assert "heuristics_count" in stats
        assert "total_count" in stats


class TestChromaStorageFilters:
    """Tests for filter building."""

    def test_build_where_filter_simple(self, storage_with_mock_client):
        """Test building a simple where filter."""
        storage = storage_with_mock_client

        filter_result = storage._build_where_filter(project_id="test-project")

        assert filter_result == {"project_id": {"$eq": "test-project"}}

    def test_build_where_filter_multiple_conditions(self, storage_with_mock_client):
        """Test building a filter with multiple conditions."""
        storage = storage_with_mock_client

        filter_result = storage._build_where_filter(
            project_id="test-project",
            agent="Helena",
            min_confidence=0.5,
        )

        assert "$and" in filter_result
        conditions = filter_result["$and"]
        assert len(conditions) == 3

    def test_build_where_filter_none(self, storage_with_mock_client):
        """Test building a filter with no conditions."""
        storage = storage_with_mock_client

        filter_result = storage._build_where_filter()

        assert filter_result is None

    def test_build_agents_filter(self, storage_with_mock_client):
        """Test building a filter for multiple agents."""
        storage = storage_with_mock_client

        filter_result = storage._build_agents_filter(
            project_id="test-project",
            agents=["Helena", "Victor"],
        )

        assert "$and" in filter_result


class TestChromaStorageHelpers:
    """Tests for helper methods."""

    def test_datetime_to_str(self, storage_with_mock_client):
        """Test datetime to string conversion."""
        storage = storage_with_mock_client

        dt = datetime(2024, 1, 15, 12, 30, 0, tzinfo=timezone.utc)
        result = storage._datetime_to_str(dt)

        assert "2024-01-15" in result
        assert result is not None

    def test_datetime_to_str_none(self, storage_with_mock_client):
        """Test datetime to string conversion with None."""
        storage = storage_with_mock_client

        result = storage._datetime_to_str(None)

        assert result is None

    def test_str_to_datetime(self, storage_with_mock_client):
        """Test string to datetime conversion."""
        storage = storage_with_mock_client

        result = storage._str_to_datetime("2024-01-15T12:30:00+00:00")

        assert result is not None
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15

    def test_str_to_datetime_none(self, storage_with_mock_client):
        """Test string to datetime conversion with None."""
        storage = storage_with_mock_client

        result = storage._str_to_datetime(None)

        assert result is None

    def test_str_to_datetime_invalid(self, storage_with_mock_client):
        """Test string to datetime conversion with invalid string."""
        storage = storage_with_mock_client

        result = storage._str_to_datetime("not-a-date")

        # Should return current time as fallback
        assert result is not None


class TestChromaStorageResultConversion:
    """Tests for result conversion methods."""

    def test_results_to_heuristics_empty(self, storage_with_mock_client):
        """Test converting empty results to heuristics."""
        storage = storage_with_mock_client

        results = {"ids": [[]], "metadatas": [[]], "documents": [[]]}
        heuristics = storage._results_to_heuristics(results)

        assert heuristics == []

    def test_results_to_outcomes_empty(self, storage_with_mock_client):
        """Test converting empty results to outcomes."""
        storage = storage_with_mock_client

        results = {"ids": [[]], "metadatas": [[]], "documents": [[]]}
        outcomes = storage._results_to_outcomes(results)

        assert outcomes == []

    def test_results_to_heuristics_with_data(self, storage_with_mock_client):
        """Test converting results with data to heuristics."""
        storage = storage_with_mock_client

        results = {
            "ids": [["h-1"]],
            "metadatas": [[{
                "agent": "Helena",
                "project_id": "test",
                "condition": "cond",
                "strategy": "strat",
                "confidence": 0.9,
                "occurrence_count": 5,
                "success_count": 4,
                "last_validated": datetime.now(timezone.utc).isoformat(),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "extra_metadata": '{"key": "value"}',
            }]],
            "documents": [["cond\nstrat"]],
            "embeddings": None,
        }

        heuristics = storage._results_to_heuristics(results)

        assert len(heuristics) == 1
        assert heuristics[0].id == "h-1"
        assert heuristics[0].metadata == {"key": "value"}
