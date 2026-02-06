"""
Unit tests for PostgreSQL storage backend.

These tests verify the PostgreSQL storage implementation without requiring
an actual PostgreSQL database. For integration tests with a real database,
see tests/integration/test_postgresql_integration.py
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock, Mock, patch

import pytest

# Skip all tests if psycopg is not installed
psycopg_available = False
try:
    from psycopg_pool import ConnectionPool  # noqa: F401

    psycopg_available = True
except ImportError:
    pass

pytestmark = pytest.mark.skipif(
    not psycopg_available,
    reason="psycopg not installed. Install with: pip install 'alma-memory[postgres]'",
)


@pytest.fixture
def mock_pool():
    """Create a mock connection pool."""
    pool = MagicMock()
    connection = MagicMock()
    cursor = MagicMock()

    # Setup context manager chain
    pool.connection.return_value.__enter__ = Mock(return_value=connection)
    pool.connection.return_value.__exit__ = Mock(return_value=False)
    connection.execute.return_value = cursor
    cursor.fetchall.return_value = []
    cursor.fetchone.return_value = {"count": 0}
    cursor.rowcount = 0

    return pool


@pytest.fixture
def storage_with_mock_pool(mock_pool):
    """Create PostgreSQLStorage with mocked pool."""
    from alma.storage.postgresql import PostgreSQLStorage

    with patch.object(PostgreSQLStorage, "__init__", lambda self, **kwargs: None):
        storage = PostgreSQLStorage()
        storage._pool = mock_pool
        storage._pgvector_available = True
        storage.embedding_dim = 384
        storage.schema = "public"

    return storage


class TestPostgreSQLStorageInit:
    """Tests for storage initialization."""

    def test_from_config_basic(self):
        """Test from_config with basic config."""
        from alma.storage.postgresql import PostgreSQLStorage

        config = {
            "postgres": {
                "host": "localhost",
                "port": 5432,
                "database": "test_db",
                "user": "test_user",
                "password": "test_pass",
            },
            "embedding_dim": 384,
        }

        with patch.object(
            PostgreSQLStorage, "__init__", return_value=None
        ) as mock_init:
            PostgreSQLStorage.from_config(config)
            # Verify from_config calls __init__ with correct params
            mock_init.assert_called_once()
            call_kwargs = mock_init.call_args[1]
            assert call_kwargs["host"] == "localhost"
            assert call_kwargs["port"] == 5432
            assert call_kwargs["database"] == "test_db"
            assert call_kwargs["user"] == "test_user"

    def test_from_config_with_env_vars(self):
        """Test from_config with environment variable expansion."""
        import os

        from alma.storage.postgresql import PostgreSQLStorage

        os.environ["TEST_PG_HOST"] = "db.example.com"
        os.environ["TEST_PG_PASS"] = "secret123"

        config = {
            "postgres": {
                "host": "${TEST_PG_HOST}",
                "port": 5432,
                "database": "alma",
                "user": "admin",
                "password": "${TEST_PG_PASS}",
            },
        }

        with patch.object(
            PostgreSQLStorage, "__init__", return_value=None
        ) as mock_init:
            PostgreSQLStorage.from_config(config)
            call_kwargs = mock_init.call_args[1]
            assert call_kwargs["host"] == "db.example.com"
            assert call_kwargs["password"] == "secret123"

        # Cleanup
        del os.environ["TEST_PG_HOST"]
        del os.environ["TEST_PG_PASS"]


class TestPostgreSQLStorageEmbeddings:
    """Tests for embedding handling."""

    def test_embedding_to_db_pgvector(self, storage_with_mock_pool):
        """Test embedding conversion for pgvector format."""
        storage = storage_with_mock_pool
        storage._pgvector_available = True

        embedding = [0.1, 0.2, 0.3]
        result = storage._embedding_to_db(embedding)

        assert result == "[0.1,0.2,0.3]"

    def test_embedding_to_db_bytes(self, storage_with_mock_pool):
        """Test embedding conversion for bytes format (fallback)."""
        storage = storage_with_mock_pool
        storage._pgvector_available = False

        embedding = [0.1, 0.2, 0.3]
        result = storage._embedding_to_db(embedding)

        # Should be bytes
        assert isinstance(result, bytes)

    def test_embedding_to_db_none(self, storage_with_mock_pool):
        """Test embedding conversion with None."""
        storage = storage_with_mock_pool
        result = storage._embedding_to_db(None)
        assert result is None

    def test_embedding_from_db_pgvector(self, storage_with_mock_pool):
        """Test embedding parsing from pgvector format."""
        storage = storage_with_mock_pool
        storage._pgvector_available = True

        db_value = "[0.1,0.2,0.3]"
        result = storage._embedding_from_db(db_value)

        assert result == [0.1, 0.2, 0.3]

    def test_cosine_similarity(self, storage_with_mock_pool):
        """Test cosine similarity calculation."""
        storage = storage_with_mock_pool

        a = [1.0, 0.0, 0.0]
        b = [1.0, 0.0, 0.0]
        similarity = storage._cosine_similarity(a, b)
        assert abs(similarity - 1.0) < 0.001

        c = [0.0, 1.0, 0.0]
        similarity_orth = storage._cosine_similarity(a, c)
        assert abs(similarity_orth - 0.0) < 0.001


class TestPostgreSQLStorageWriteOps:
    """Tests for write operations."""

    def test_save_heuristic(self, storage_with_mock_pool, mock_pool):
        """Test saving a heuristic."""
        from alma.types import Heuristic

        storage = storage_with_mock_pool

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

    def test_save_outcome(self, storage_with_mock_pool, mock_pool):
        """Test saving an outcome."""
        from alma.types import Outcome

        storage = storage_with_mock_pool

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


class TestPostgreSQLStorageReadOps:
    """Tests for read operations."""

    def test_get_heuristics_basic(self, storage_with_mock_pool, mock_pool):
        """Test basic heuristic retrieval."""
        storage = storage_with_mock_pool

        # Mock returns empty list
        connection = mock_pool.connection.return_value.__enter__.return_value
        cursor = connection.execute.return_value
        cursor.fetchall.return_value = []

        results = storage.get_heuristics(
            project_id="test-project",
            agent="Helena",
            top_k=5,
        )

        assert results == []
        connection.execute.assert_called_once()

    def test_get_heuristics_with_embedding(self, storage_with_mock_pool, mock_pool):
        """Test heuristic retrieval with vector search."""
        storage = storage_with_mock_pool
        storage._pgvector_available = True

        connection = mock_pool.connection.return_value.__enter__.return_value
        cursor = connection.execute.return_value
        cursor.fetchall.return_value = []

        embedding = [0.1] * 384
        results = storage.get_heuristics(
            project_id="test-project",
            embedding=embedding,
            top_k=5,
        )

        assert results == []
        # Should use vector search query
        call_args = connection.execute.call_args
        query = call_args[0][0]
        assert "<=>" in query  # pgvector cosine distance operator


class TestPostgreSQLStorageStats:
    """Tests for statistics."""

    def test_get_stats(self, storage_with_mock_pool, mock_pool):
        """Test getting storage statistics."""
        storage = storage_with_mock_pool

        connection = mock_pool.connection.return_value.__enter__.return_value
        cursor = connection.execute.return_value
        cursor.fetchone.return_value = {"count": 5}

        stats = storage.get_stats(project_id="test-project")

        assert stats["project_id"] == "test-project"
        assert stats["storage_type"] == "postgresql"
        assert stats["pgvector_available"]


class TestPostgreSQLStorageRowConversion:
    """Tests for database row conversion."""

    def test_row_to_heuristic(self, storage_with_mock_pool):
        """Test converting a database row to Heuristic."""
        storage = storage_with_mock_pool

        row = {
            "id": "h-123",
            "agent": "Helena",
            "project_id": "test",
            "condition": "test condition",
            "strategy": "test strategy",
            "confidence": 0.9,
            "occurrence_count": 5,
            "success_count": 4,
            "last_validated": datetime.now(timezone.utc),
            "created_at": datetime.now(timezone.utc),
            "metadata": {"key": "value"},
            "embedding": None,
        }

        heuristic = storage._row_to_heuristic(row)

        assert heuristic.id == "h-123"
        assert heuristic.agent == "Helena"
        assert heuristic.confidence == 0.9
        assert heuristic.metadata == {"key": "value"}

    def test_row_to_outcome(self, storage_with_mock_pool):
        """Test converting a database row to Outcome."""
        storage = storage_with_mock_pool

        row = {
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
            "timestamp": datetime.now(timezone.utc),
            "metadata": {},
            "embedding": None,
        }

        outcome = storage._row_to_outcome(row)

        assert outcome.id == "o-123"
        assert outcome.agent == "Victor"
        assert outcome.success
        assert outcome.duration_ms == 100
