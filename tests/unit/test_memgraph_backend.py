"""
Unit tests for Memgraph Backend.

Tests the MemgraphBackend implementation of the GraphBackend interface.
These tests verify the configuration and structure of the backend without
requiring a live Memgraph instance.
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from alma.graph import GraphBackend, create_graph_backend
from alma.graph.store import Entity, Relationship

# Check if neo4j package is available (required for Memgraph backend)
try:
    import neo4j

    HAS_NEO4J = True
except ImportError:
    HAS_NEO4J = False

pytestmark = pytest.mark.skipif(
    not HAS_NEO4J, reason="neo4j package not installed (required for Memgraph backend)"
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_entity() -> Entity:
    """Create a sample entity for testing."""
    return Entity(
        id="entity-1",
        name="Test Entity",
        entity_type="concept",
        properties={"key": "value"},
        created_at=datetime.now(timezone.utc),
    )


@pytest.fixture
def sample_entity_2() -> Entity:
    """Create a second sample entity for testing."""
    return Entity(
        id="entity-2",
        name="Related Entity",
        entity_type="person",
        properties={"role": "developer"},
        created_at=datetime.now(timezone.utc),
    )


@pytest.fixture
def sample_relationship(sample_entity, sample_entity_2) -> Relationship:
    """Create a sample relationship for testing."""
    return Relationship(
        id="rel-1",
        source_id=sample_entity.id,
        target_id=sample_entity_2.id,
        relation_type="RELATED_TO",
        properties={"strength": 0.9},
        confidence=0.95,
        created_at=datetime.now(timezone.utc),
    )


@pytest.fixture
def memgraph_backend():
    """Create a MemgraphBackend instance for testing."""
    from alma.graph.backends.memgraph import MemgraphBackend

    return MemgraphBackend(
        uri="bolt://localhost:7687",
        username="",
        password="",
    )


# =============================================================================
# MemgraphBackend Configuration Tests
# =============================================================================


class TestMemgraphBackendConfiguration:
    """Tests for MemgraphBackend configuration and initialization."""

    def test_default_configuration(self):
        """Test default configuration values."""
        from alma.graph.backends.memgraph import MemgraphBackend

        backend = MemgraphBackend()

        assert backend.uri == "bolt://localhost:7687"
        assert backend.username == ""
        assert backend.password == ""
        assert backend.database == "memgraph"
        assert backend._driver is None

    def test_custom_configuration(self):
        """Test custom configuration values."""
        from alma.graph.backends.memgraph import MemgraphBackend

        backend = MemgraphBackend(
            uri="bolt://custom-host:7688",
            username="custom_user",
            password="custom_pass",
            database="custom_db",
        )

        assert backend.uri == "bolt://custom-host:7688"
        assert backend.username == "custom_user"
        assert backend.password == "custom_pass"
        assert backend.database == "custom_db"

    def test_driver_is_lazy_initialized(self):
        """Test that driver is not created until first use."""
        from alma.graph.backends.memgraph import MemgraphBackend

        backend = MemgraphBackend()
        assert backend._driver is None

    def test_is_graph_backend(self, memgraph_backend):
        """Test that MemgraphBackend is a GraphBackend."""
        assert isinstance(memgraph_backend, GraphBackend)

    def test_has_required_methods(self, memgraph_backend):
        """Test that all required abstract methods are implemented."""
        required_methods = [
            "add_entity",
            "add_relationship",
            "get_entity",
            "get_entities",
            "get_relationships",
            "search_entities",
            "delete_entity",
            "delete_relationship",
            "close",
        ]

        for method in required_methods:
            assert hasattr(memgraph_backend, method), f"Missing method: {method}"
            assert callable(getattr(memgraph_backend, method)), (
                f"Method not callable: {method}"
            )


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestCreateGraphBackendMemgraph:
    """Tests for the create_graph_backend factory function with Memgraph."""

    def test_create_memgraph_backend(self):
        """Test creating a Memgraph backend via factory."""
        from alma.graph.backends.memgraph import MemgraphBackend

        backend = create_graph_backend(
            "memgraph",
            uri="bolt://localhost:7687",
            username="",
            password="",
        )

        assert isinstance(backend, MemgraphBackend)
        assert backend.uri == "bolt://localhost:7687"

    def test_create_memgraph_backend_with_auth(self):
        """Test creating a Memgraph backend with authentication."""
        from alma.graph.backends.memgraph import MemgraphBackend

        backend = create_graph_backend(
            "memgraph",
            uri="bolt://localhost:7687",
            username="memgraph_user",
            password="memgraph_pass",
        )

        assert isinstance(backend, MemgraphBackend)
        assert backend.username == "memgraph_user"
        assert backend.password == "memgraph_pass"


# =============================================================================
# Mocked Driver Tests
# =============================================================================


class TestMemgraphBackendWithMockedDriver:
    """Tests for MemgraphBackend with mocked neo4j driver."""

    @pytest.fixture
    def mock_driver(self):
        """Create a mock neo4j driver."""
        with patch("neo4j.GraphDatabase") as mock_gd:
            mock_driver = MagicMock()
            mock_gd.driver.return_value = mock_driver
            yield mock_driver

    @pytest.fixture
    def backend_with_mock(self, mock_driver):
        """Create a MemgraphBackend with mocked driver."""
        from alma.graph.backends.memgraph import MemgraphBackend

        backend = MemgraphBackend(
            uri="bolt://localhost:7687",
            username="",
            password="",
        )
        backend._driver = mock_driver
        return backend

    def test_run_query_uses_session(self, backend_with_mock, mock_driver):
        """Test that _run_query uses driver session."""
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_result.__iter__ = lambda self: iter([])
        mock_session.run.return_value = mock_result
        mock_driver.session.return_value.__enter__ = MagicMock(
            return_value=mock_session
        )
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)

        backend_with_mock._run_query("MATCH (n) RETURN n")

        mock_driver.session.assert_called_once()

    def test_close_closes_driver(self, backend_with_mock, mock_driver):
        """Test that close() closes the driver."""
        backend_with_mock.close()

        mock_driver.close.assert_called_once()
        assert backend_with_mock._driver is None

    def test_close_when_no_driver(self, memgraph_backend):
        """Test that close() handles no driver gracefully."""
        # Should not raise
        memgraph_backend.close()
        assert memgraph_backend._driver is None

    def test_add_entity_query_structure(
        self, backend_with_mock, mock_driver, sample_entity
    ):
        """Test that add_entity constructs correct query."""
        mock_session = MagicMock()
        mock_record = MagicMock()
        mock_record.__getitem__ = lambda self, key: sample_entity.id
        mock_record.keys.return_value = ["id"]
        mock_result = MagicMock()
        mock_result.__iter__ = lambda self: iter([mock_record])
        mock_session.run.return_value = mock_result
        mock_driver.session.return_value.__enter__ = MagicMock(
            return_value=mock_session
        )
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)

        # Mock dict conversion
        def record_to_dict(rec):
            return {"id": sample_entity.id}

        with patch.object(backend_with_mock, "_run_query") as mock_run:
            mock_run.return_value = [{"id": sample_entity.id}]
            result = backend_with_mock.add_entity(sample_entity)

        assert result == sample_entity.id
        mock_run.assert_called_once()
        # Verify query contains MERGE
        call_args = mock_run.call_args
        assert "MERGE" in call_args[0][0]
        assert "Entity" in call_args[0][0]

    def test_add_relationship_query_structure(
        self, backend_with_mock, sample_relationship
    ):
        """Test that add_relationship constructs correct query."""
        with patch.object(backend_with_mock, "_run_query") as mock_run:
            mock_run.return_value = [{"id": sample_relationship.id}]
            result = backend_with_mock.add_relationship(sample_relationship)

        assert result == sample_relationship.id
        mock_run.assert_called_once()
        # Verify query contains MERGE and relationship type
        call_args = mock_run.call_args
        assert "MERGE" in call_args[0][0]
        assert "RELATED_TO" in call_args[0][0]

    def test_get_entity_returns_none_when_not_found(self, backend_with_mock):
        """Test that get_entity returns None when entity not found."""
        with patch.object(backend_with_mock, "_run_query") as mock_run:
            mock_run.return_value = []
            result = backend_with_mock.get_entity("nonexistent")

        assert result is None

    def test_get_entity_returns_entity(self, backend_with_mock, sample_entity):
        """Test that get_entity returns entity when found."""
        with patch.object(backend_with_mock, "_run_query") as mock_run:
            mock_run.return_value = [
                {
                    "id": sample_entity.id,
                    "name": sample_entity.name,
                    "entity_type": sample_entity.entity_type,
                    "properties": '{"key": "value"}',
                    "created_at": sample_entity.created_at.isoformat(),
                    "project_id": None,
                    "agent": None,
                }
            ]
            result = backend_with_mock.get_entity(sample_entity.id)

        assert result is not None
        assert result.id == sample_entity.id
        assert result.name == sample_entity.name
        assert result.entity_type == sample_entity.entity_type

    def test_delete_entity_returns_true_when_deleted(self, backend_with_mock):
        """Test that delete_entity returns True when entity is deleted."""
        with patch.object(backend_with_mock, "_run_query") as mock_run:
            mock_run.return_value = [{"deleted": 1}]
            result = backend_with_mock.delete_entity("entity-1")

        assert result is True

    def test_delete_entity_returns_false_when_not_found(self, backend_with_mock):
        """Test that delete_entity returns False when entity not found."""
        with patch.object(backend_with_mock, "_run_query") as mock_run:
            mock_run.return_value = [{"deleted": 0}]
            result = backend_with_mock.delete_entity("nonexistent")

        assert result is False

    def test_delete_relationship_returns_true_when_deleted(self, backend_with_mock):
        """Test that delete_relationship returns True when deleted."""
        with patch.object(backend_with_mock, "_run_query") as mock_run:
            mock_run.return_value = [{"deleted": 1}]
            result = backend_with_mock.delete_relationship("rel-1")

        assert result is True

    def test_delete_relationship_returns_false_when_not_found(self, backend_with_mock):
        """Test that delete_relationship returns False when not found."""
        with patch.object(backend_with_mock, "_run_query") as mock_run:
            mock_run.return_value = [{"deleted": 0}]
            result = backend_with_mock.delete_relationship("nonexistent")

        assert result is False

    def test_search_entities_returns_matches(self, backend_with_mock, sample_entity):
        """Test that search_entities returns matching entities."""
        with patch.object(backend_with_mock, "_run_query") as mock_run:
            mock_run.return_value = [
                {
                    "id": sample_entity.id,
                    "name": sample_entity.name,
                    "entity_type": sample_entity.entity_type,
                    "properties": '{"key": "value"}',
                    "created_at": sample_entity.created_at.isoformat(),
                    "project_id": None,
                    "agent": None,
                }
            ]
            results = backend_with_mock.search_entities("Test")

        assert len(results) == 1
        assert results[0].name == sample_entity.name

    def test_search_entities_returns_empty_when_no_matches(self, backend_with_mock):
        """Test that search_entities returns empty list when no matches."""
        with patch.object(backend_with_mock, "_run_query") as mock_run:
            mock_run.return_value = []
            results = backend_with_mock.search_entities("NonExistent")

        assert len(results) == 0

    def test_get_entities_with_filters(self, backend_with_mock):
        """Test get_entities with various filters."""
        with patch.object(backend_with_mock, "_run_query") as mock_run:
            mock_run.return_value = []
            backend_with_mock.get_entities(
                entity_type="person", project_id="proj-1", agent="agent-1", limit=50
            )

        # Verify query contains WHERE clause with filters
        call_args = mock_run.call_args
        query = call_args[0][0]
        # Parameters are passed as second positional argument
        params = call_args[0][1] if len(call_args[0]) > 1 else call_args[1]

        assert "entity_type" in query
        assert "project_id" in query
        assert "agent" in query
        assert params["limit"] == 50

    def test_get_relationships_query_structure(self, backend_with_mock):
        """Test that get_relationships constructs correct query."""
        with patch.object(backend_with_mock, "_run_query") as mock_run:
            mock_run.return_value = []
            backend_with_mock.get_relationships("entity-1")

        call_args = mock_run.call_args
        query = call_args[0][0]
        # Parameters are passed as second positional argument
        params = call_args[0][1] if len(call_args[0]) > 1 else call_args[1]

        assert "MATCH" in query
        assert params["entity_id"] == "entity-1"


# =============================================================================
# Context Manager Tests
# =============================================================================


class TestMemgraphBackendContextManager:
    """Tests for context manager functionality."""

    def test_context_manager_calls_close(self):
        """Test that context manager calls close on exit."""
        from alma.graph.backends.memgraph import MemgraphBackend

        backend = MemgraphBackend()
        backend.close = MagicMock()

        with backend:
            pass

        backend.close.assert_called_once()


# =============================================================================
# Compatibility Methods Tests
# =============================================================================


class TestMemgraphBackendCompatibilityMethods:
    """Tests for GraphStore API compatibility methods."""

    def test_find_entities_by_name(self, memgraph_backend):
        """Test find_entities delegates to search_entities when name provided."""
        with patch.object(memgraph_backend, "search_entities") as mock_search:
            mock_search.return_value = []
            memgraph_backend.find_entities(name="Test", limit=5)

        mock_search.assert_called_once_with(query="Test", top_k=5)

    def test_find_entities_by_type(self, memgraph_backend):
        """Test find_entities delegates to get_entities when type provided."""
        with patch.object(memgraph_backend, "get_entities") as mock_get:
            mock_get.return_value = []
            memgraph_backend.find_entities(entity_type="person", limit=5)

        mock_get.assert_called_once_with(entity_type="person", limit=5)

    def test_get_relationships_directional_outgoing(self, memgraph_backend):
        """Test get_relationships_directional with outgoing direction."""
        with patch.object(memgraph_backend, "_run_query") as mock_run:
            mock_run.return_value = []
            memgraph_backend.get_relationships_directional(
                "entity-1", direction="outgoing"
            )

        call_args = mock_run.call_args
        query = call_args[0][0]
        assert "->" in query

    def test_get_relationships_directional_incoming(self, memgraph_backend):
        """Test get_relationships_directional with incoming direction."""
        with patch.object(memgraph_backend, "_run_query") as mock_run:
            mock_run.return_value = []
            memgraph_backend.get_relationships_directional(
                "entity-1", direction="incoming"
            )

        call_args = mock_run.call_args
        query = call_args[0][0]
        assert "<-" in query

    def test_get_relationships_directional_with_type_filter(self, memgraph_backend):
        """Test get_relationships_directional with relation type filter."""
        with patch.object(memgraph_backend, "_run_query") as mock_run:
            mock_run.return_value = []
            memgraph_backend.get_relationships_directional(
                "entity-1", direction="both", relation_type="KNOWS"
            )

        call_args = mock_run.call_args
        query = call_args[0][0]
        assert "KNOWS" in query


# =============================================================================
# Driver Initialization Tests
# =============================================================================


class TestMemgraphDriverInitialization:
    """Tests for driver initialization logic."""

    def test_get_driver_without_auth(self):
        """Test driver initialization without authentication."""
        from alma.graph.backends.memgraph import MemgraphBackend

        with patch("neo4j.GraphDatabase") as mock_gd:
            mock_driver = MagicMock()
            mock_gd.driver.return_value = mock_driver

            backend = MemgraphBackend(uri="bolt://localhost:7687")
            driver = backend._get_driver()

            # Should call driver without auth tuple
            mock_gd.driver.assert_called_once_with("bolt://localhost:7687")
            assert driver is mock_driver

    def test_get_driver_with_auth(self):
        """Test driver initialization with authentication."""
        from alma.graph.backends.memgraph import MemgraphBackend

        with patch("neo4j.GraphDatabase") as mock_gd:
            mock_driver = MagicMock()
            mock_gd.driver.return_value = mock_driver

            backend = MemgraphBackend(
                uri="bolt://localhost:7687",
                username="user",
                password="pass",
            )
            driver = backend._get_driver()

            # Should call driver with auth tuple
            mock_gd.driver.assert_called_once_with(
                "bolt://localhost:7687", auth=("user", "pass")
            )
            assert driver is mock_driver

    def test_get_driver_caches_driver(self):
        """Test that _get_driver caches the driver instance."""
        from alma.graph.backends.memgraph import MemgraphBackend

        with patch("neo4j.GraphDatabase") as mock_gd:
            mock_driver = MagicMock()
            mock_gd.driver.return_value = mock_driver

            backend = MemgraphBackend()

            # Call twice
            driver1 = backend._get_driver()
            driver2 = backend._get_driver()

            # Should only create driver once
            assert mock_gd.driver.call_count == 1
            assert driver1 is driver2

    def test_get_driver_raises_import_error_without_neo4j(self):
        """Test that _get_driver raises ImportError if neo4j not installed."""
        from alma.graph.backends.memgraph import MemgraphBackend

        backend = MemgraphBackend()

        with patch.dict("sys.modules", {"neo4j": None}):
            with patch("builtins.__import__", side_effect=ImportError("No module")):
                # Reset driver to force re-initialization
                backend._driver = None
                with pytest.raises(ImportError, match="neo4j package required"):
                    backend._get_driver()
