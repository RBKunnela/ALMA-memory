"""
Unit tests for Kuzu Graph Database Backend.

Tests the KuzuBackend implementation of the GraphBackend interface.
Tests are skipped if kuzu package is not installed.
"""

# Check if kuzu is available
import importlib.util
import os
import tempfile
from datetime import datetime, timezone

import pytest

from alma.graph import (
    BackendGraphStore,
    Entity,
    GraphBackend,
    GraphQuery,
    Relationship,
    create_graph_backend,
)

KUZU_AVAILABLE = importlib.util.find_spec("kuzu") is not None

pytestmark = pytest.mark.skipif(not KUZU_AVAILABLE, reason="kuzu package not installed")


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
def kuzu_backend():
    """Create a Kuzu backend in in-memory mode for testing."""
    from alma.graph.backends.kuzu import KuzuBackend

    backend = KuzuBackend()  # In-memory mode
    yield backend
    backend.close()


@pytest.fixture
def persistent_kuzu_backend():
    """Create a Kuzu backend with persistent storage for testing."""
    from alma.graph.backends.kuzu import KuzuBackend

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_kuzu_db")
        backend = KuzuBackend(database_path=db_path)
        yield backend
        backend.close()


@pytest.fixture
def populated_backend(
    kuzu_backend, sample_entity, sample_entity_2, sample_relationship
):
    """Create a backend with pre-populated data."""
    kuzu_backend.add_entity(sample_entity)
    kuzu_backend.add_entity(sample_entity_2)
    kuzu_backend.add_relationship(sample_relationship)
    return kuzu_backend


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestCreateKuzuBackend:
    """Tests for creating Kuzu backend via factory function."""

    def test_create_kuzu_backend_in_memory(self):
        """Test creating a Kuzu backend in in-memory mode."""
        backend = create_graph_backend("kuzu")
        try:
            assert backend is not None
            # Verify it's a KuzuBackend
            from alma.graph.backends.kuzu import KuzuBackend

            assert isinstance(backend, KuzuBackend)
        finally:
            backend.close()

    def test_create_kuzu_backend_with_path(self):
        """Test creating a Kuzu backend with a database path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_db")
            backend = create_graph_backend("kuzu", database_path=db_path)
            try:
                assert backend is not None
                assert backend.database_path == db_path
            finally:
                backend.close()


# =============================================================================
# KuzuBackend Entity Tests
# =============================================================================


class TestKuzuBackendEntity:
    """Tests for entity operations in KuzuBackend."""

    def test_add_entity(self, kuzu_backend, sample_entity):
        """Test adding an entity."""
        result = kuzu_backend.add_entity(sample_entity)
        assert result == sample_entity.id

    def test_get_entity(self, kuzu_backend, sample_entity):
        """Test retrieving an entity by ID."""
        kuzu_backend.add_entity(sample_entity)
        entity = kuzu_backend.get_entity(sample_entity.id)

        assert entity is not None
        assert entity.id == sample_entity.id
        assert entity.name == sample_entity.name
        assert entity.entity_type == sample_entity.entity_type

    def test_get_entity_not_found(self, kuzu_backend):
        """Test retrieving a non-existent entity."""
        entity = kuzu_backend.get_entity("non-existent-id")
        assert entity is None

    def test_get_entities_all(self, populated_backend):
        """Test getting all entities."""
        entities = populated_backend.get_entities()
        assert len(entities) == 2

    def test_get_entities_by_type(self, populated_backend):
        """Test filtering entities by type."""
        entities = populated_backend.get_entities(entity_type="person")
        assert len(entities) == 1
        assert entities[0].entity_type == "person"

    def test_get_entities_with_limit(self, populated_backend):
        """Test limiting entity results."""
        entities = populated_backend.get_entities(limit=1)
        assert len(entities) == 1

    def test_search_entities(self, populated_backend):
        """Test searching entities by name."""
        results = populated_backend.search_entities("Test")
        assert len(results) >= 1
        assert any("Test" in e.name for e in results)

    def test_search_entities_case_insensitive(self, populated_backend):
        """Test that search is case-insensitive."""
        results = populated_backend.search_entities("test")
        assert len(results) >= 1

    def test_search_entities_not_found(self, populated_backend):
        """Test searching for non-existent entity."""
        results = populated_backend.search_entities("NonExistent12345")
        assert len(results) == 0

    def test_delete_entity(self, populated_backend):
        """Test deleting an entity."""
        result = populated_backend.delete_entity("entity-1")
        assert result is True
        assert populated_backend.get_entity("entity-1") is None

    def test_delete_entity_not_found(self, kuzu_backend):
        """Test deleting a non-existent entity."""
        result = kuzu_backend.delete_entity("non-existent")
        assert result is False

    def test_delete_entity_removes_relationships(self, populated_backend):
        """Test that deleting an entity removes its relationships."""
        populated_backend.delete_entity("entity-1")

        # Relationship should be removed
        rels = populated_backend.get_relationships("entity-2")
        assert len(rels) == 0


class TestKuzuBackendRelationship:
    """Tests for relationship operations in KuzuBackend."""

    def test_add_relationship(self, populated_backend, sample_relationship):
        """Test that relationship is added (already done in populated_backend)."""
        rels = populated_backend.get_relationships("entity-1")
        assert len(rels) == 1

    def test_get_relationships_both_directions(self, populated_backend):
        """Test getting relationships in both directions."""
        # From source
        rels_from_source = populated_backend.get_relationships("entity-1")
        assert len(rels_from_source) == 1

        # From target
        rels_from_target = populated_backend.get_relationships("entity-2")
        assert len(rels_from_target) == 1

    def test_get_relationships_directional_outgoing(self, populated_backend):
        """Test getting only outgoing relationships."""
        rels = populated_backend.get_relationships_directional(
            "entity-1", direction="outgoing"
        )
        assert len(rels) == 1

        rels = populated_backend.get_relationships_directional(
            "entity-2", direction="outgoing"
        )
        assert len(rels) == 0

    def test_get_relationships_directional_incoming(self, populated_backend):
        """Test getting only incoming relationships."""
        rels = populated_backend.get_relationships_directional(
            "entity-2", direction="incoming"
        )
        assert len(rels) == 1

        rels = populated_backend.get_relationships_directional(
            "entity-1", direction="incoming"
        )
        assert len(rels) == 0

    def test_get_relationships_filter_by_type(self, populated_backend):
        """Test filtering relationships by type."""
        # Add another relationship with different type
        rel2 = Relationship(
            id="rel-2",
            source_id="entity-1",
            target_id="entity-2",
            relation_type="WORKS_WITH",
        )
        populated_backend.add_relationship(rel2)

        rels = populated_backend.get_relationships_directional(
            "entity-1", relation_type="RELATED_TO"
        )
        assert len(rels) == 1
        assert rels[0].relation_type == "RELATED_TO"

    def test_delete_relationship(self, populated_backend):
        """Test deleting a specific relationship."""
        result = populated_backend.delete_relationship("rel-1")
        assert result is True

        # Relationship should be gone
        rels = populated_backend.get_relationships("entity-1")
        assert len(rels) == 0

    def test_delete_relationship_not_found(self, kuzu_backend):
        """Test deleting a non-existent relationship."""
        result = kuzu_backend.delete_relationship("non-existent")
        assert result is False


# =============================================================================
# Persistence Tests
# =============================================================================


class TestKuzuBackendPersistence:
    """Tests for persistent storage functionality."""

    def test_data_persists_after_close(self):
        """Test that data persists after closing and reopening the database."""
        from alma.graph.backends.kuzu import KuzuBackend

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "persist_test_db")

            # Create and populate database
            backend1 = KuzuBackend(database_path=db_path)
            entity = Entity(id="persist-1", name="Persist Test", entity_type="concept")
            backend1.add_entity(entity)
            backend1.close()

            # Reopen and verify data
            backend2 = KuzuBackend(database_path=db_path)
            try:
                retrieved = backend2.get_entity("persist-1")
                assert retrieved is not None
                assert retrieved.name == "Persist Test"
            finally:
                backend2.close()

    def test_in_memory_mode_data_lost_on_close(self):
        """Test that in-memory mode uses :memory: and data is ephemeral."""
        from alma.graph.backends.kuzu import KuzuBackend

        backend = KuzuBackend()  # In-memory mode (uses :memory:)
        entity = Entity(id="temp-1", name="Temporary", entity_type="concept")
        backend.add_entity(entity)

        # Data exists while open
        assert backend.get_entity("temp-1") is not None

        # After close, the in-memory database is destroyed
        backend.close()
        # Note: Cannot verify data is lost as we can't reopen the same in-memory db
        # but the behavior is that :memory: databases are ephemeral


# =============================================================================
# Context Manager Tests
# =============================================================================


class TestKuzuBackendContextManager:
    """Tests for context manager functionality."""

    def test_context_manager(self):
        """Test using backend as context manager."""
        from alma.graph.backends.kuzu import KuzuBackend

        with KuzuBackend() as backend:
            entity = Entity(id="e1", name="Test", entity_type="concept")
            backend.add_entity(entity)
            assert backend.get_entity("e1") is not None

    def test_clear(self, populated_backend):
        """Test clearing all data."""
        populated_backend.clear()

        entities = populated_backend.get_entities()
        assert len(entities) == 0


# =============================================================================
# BackendGraphStore Integration Tests
# =============================================================================


class TestKuzuBackendGraphStore:
    """Tests for the BackendGraphStore adapter with Kuzu."""

    @pytest.fixture
    def backend_store(self, populated_backend) -> BackendGraphStore:
        """Create a BackendGraphStore wrapping a KuzuBackend."""
        return BackendGraphStore(populated_backend)

    def test_backend_property(self, backend_store, populated_backend):
        """Test accessing the underlying backend."""
        assert backend_store.backend is populated_backend

    def test_add_entity(self, backend_store):
        """Test adding entity through adapter."""
        entity = Entity(id="new-entity", name="New", entity_type="concept")
        result = backend_store.add_entity(entity)
        assert result == "new-entity"
        assert backend_store.get_entity("new-entity") is not None

    def test_find_entities(self, backend_store):
        """Test finding entities through adapter."""
        entities = backend_store.find_entities(name="Test")
        assert len(entities) >= 1

    def test_traverse(self, backend_store):
        """Test graph traversal through adapter."""
        result = backend_store.traverse("entity-1", max_hops=2)

        assert len(result.entities) == 2
        assert len(result.relationships) == 1
        assert result.query_time_ms >= 0

    def test_query(self, backend_store):
        """Test executing a graph query through adapter."""
        query = GraphQuery(entities=["Test"], max_hops=1, limit=10)
        result = backend_store.query(query)

        assert len(result.entities) >= 1


# =============================================================================
# GraphBackend Interface Compliance Tests
# =============================================================================


class TestKuzuGraphBackendInterface:
    """Tests to verify KuzuBackend implements GraphBackend interface."""

    def test_is_graph_backend(self, kuzu_backend):
        """Test that KuzuBackend is a GraphBackend."""
        assert isinstance(kuzu_backend, GraphBackend)

    def test_has_required_methods(self, kuzu_backend):
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
            assert hasattr(kuzu_backend, method), f"Missing method: {method}"
            assert callable(getattr(kuzu_backend, method)), (
                f"Method not callable: {method}"
            )


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestKuzuEdgeCases:
    """Tests for edge cases and error handling."""

    def test_add_duplicate_entity(self, kuzu_backend, sample_entity):
        """Test adding the same entity twice updates it."""
        kuzu_backend.add_entity(sample_entity)

        # Modify and add again
        updated_entity = Entity(
            id=sample_entity.id,
            name="Updated Name",
            entity_type=sample_entity.entity_type,
            properties=sample_entity.properties,
            created_at=sample_entity.created_at,
        )
        kuzu_backend.add_entity(updated_entity)

        entity = kuzu_backend.get_entity(sample_entity.id)
        assert entity.name == "Updated Name"

    def test_empty_search(self, kuzu_backend):
        """Test searching an empty backend."""
        results = kuzu_backend.search_entities("anything")
        assert len(results) == 0

    def test_get_relationships_empty(self, kuzu_backend):
        """Test getting relationships for entity with none."""
        rels = kuzu_backend.get_relationships("no-entity")
        assert len(rels) == 0

    def test_traverse_isolated_entity(self, kuzu_backend, sample_entity):
        """Test traversing from an entity with no relationships."""
        kuzu_backend.add_entity(sample_entity)

        store = BackendGraphStore(kuzu_backend)
        result = store.traverse(sample_entity.id)

        assert len(result.entities) == 1
        assert len(result.relationships) == 0
        assert len(result.paths) == 0

    def test_entity_with_project_id_and_agent(self, kuzu_backend):
        """Test entity with project_id and agent in properties."""
        entity = Entity(
            id="project-entity",
            name="Project Entity",
            entity_type="concept",
            properties={
                "project_id": "proj-123",
                "agent": "test-agent",
                "extra": "data",
            },
        )
        kuzu_backend.add_entity(entity)

        retrieved = kuzu_backend.get_entity("project-entity")
        assert retrieved is not None
        assert retrieved.properties.get("project_id") == "proj-123"
        assert retrieved.properties.get("agent") == "test-agent"
        assert retrieved.properties.get("extra") == "data"

    def test_filter_entities_by_project_and_agent(self, kuzu_backend):
        """Test filtering entities by project_id and agent."""
        entity1 = Entity(
            id="e1",
            name="Entity 1",
            entity_type="concept",
            properties={"project_id": "proj-1", "agent": "agent-a"},
        )
        entity2 = Entity(
            id="e2",
            name="Entity 2",
            entity_type="concept",
            properties={"project_id": "proj-2", "agent": "agent-b"},
        )
        kuzu_backend.add_entity(entity1)
        kuzu_backend.add_entity(entity2)

        # Filter by project_id
        results = kuzu_backend.get_entities(project_id="proj-1")
        assert len(results) == 1
        assert results[0].id == "e1"

        # Filter by agent
        results = kuzu_backend.get_entities(agent="agent-b")
        assert len(results) == 1
        assert results[0].id == "e2"
