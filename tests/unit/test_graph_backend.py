"""
Unit tests for Graph Database Backend Abstraction.

Tests the GraphBackend abstract interface, InMemoryBackend implementation,
and BackendGraphStore adapter.
"""

from datetime import datetime, timezone

import pytest

from alma.graph import (
    BackendGraphStore,
    Entity,
    GraphBackend,
    GraphQuery,
    Relationship,
    create_graph_backend,
    create_graph_store,
)
from alma.graph.backends.memory import InMemoryBackend

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
def memory_backend() -> InMemoryBackend:
    """Create an in-memory backend for testing."""
    return InMemoryBackend()


@pytest.fixture
def populated_backend(
    memory_backend, sample_entity, sample_entity_2, sample_relationship
) -> InMemoryBackend:
    """Create a backend with pre-populated data."""
    memory_backend.add_entity(sample_entity)
    memory_backend.add_entity(sample_entity_2)
    memory_backend.add_relationship(sample_relationship)
    return memory_backend


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestCreateGraphBackend:
    """Tests for the create_graph_backend factory function."""

    def test_create_memory_backend(self):
        """Test creating an in-memory backend."""
        backend = create_graph_backend("memory")
        assert isinstance(backend, InMemoryBackend)

    def test_create_unknown_backend_raises_error(self):
        """Test that unknown backend type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown graph backend"):
            create_graph_backend("unknown_backend")

    def test_create_neo4j_backend_with_mock(self):
        """Test Neo4j backend creation (mocked to avoid dependency)."""
        from alma.graph.backends.neo4j import Neo4jBackend

        backend = Neo4jBackend(
            uri="bolt://localhost:7687",
            username="neo4j",
            password="password",
        )
        # Verify configuration is stored correctly
        assert backend.uri == "bolt://localhost:7687"
        assert backend.username == "neo4j"
        assert backend.password == "password"
        assert backend.database == "neo4j"
        # Driver should be None until first use (lazy init)
        assert backend._driver is None


# =============================================================================
# InMemoryBackend Tests
# =============================================================================


class TestInMemoryBackendEntity:
    """Tests for entity operations in InMemoryBackend."""

    def test_add_entity(self, memory_backend, sample_entity):
        """Test adding an entity."""
        result = memory_backend.add_entity(sample_entity)
        assert result == sample_entity.id

    def test_get_entity(self, memory_backend, sample_entity):
        """Test retrieving an entity by ID."""
        memory_backend.add_entity(sample_entity)
        entity = memory_backend.get_entity(sample_entity.id)

        assert entity is not None
        assert entity.id == sample_entity.id
        assert entity.name == sample_entity.name
        assert entity.entity_type == sample_entity.entity_type

    def test_get_entity_not_found(self, memory_backend):
        """Test retrieving a non-existent entity."""
        entity = memory_backend.get_entity("non-existent-id")
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

    def test_delete_entity_not_found(self, memory_backend):
        """Test deleting a non-existent entity."""
        result = memory_backend.delete_entity("non-existent")
        assert result is False

    def test_delete_entity_removes_relationships(self, populated_backend):
        """Test that deleting an entity removes its relationships."""
        populated_backend.delete_entity("entity-1")

        # Relationship should be removed
        rels = populated_backend.get_relationships("entity-2")
        assert len(rels) == 0


class TestInMemoryBackendRelationship:
    """Tests for relationship operations in InMemoryBackend."""

    def test_add_relationship(self, populated_backend, sample_relationship):
        """Test that relationship is added (already done in populated_backend)."""
        rels = populated_backend.get_relationships("entity-1")
        assert len(rels) == 1

    def test_add_relationship_creates_indices(
        self, memory_backend, sample_entity, sample_entity_2
    ):
        """Test that adding relationship updates both incoming and outgoing indices."""
        memory_backend.add_entity(sample_entity)
        memory_backend.add_entity(sample_entity_2)

        rel = Relationship(
            id="new-rel",
            source_id=sample_entity.id,
            target_id=sample_entity_2.id,
            relation_type="KNOWS",
        )
        memory_backend.add_relationship(rel)

        # Check outgoing from source
        assert "new-rel" in memory_backend._outgoing.get(sample_entity.id, [])
        # Check incoming to target
        assert "new-rel" in memory_backend._incoming.get(sample_entity_2.id, [])

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

    def test_delete_relationship_not_found(self, memory_backend):
        """Test deleting a non-existent relationship."""
        result = memory_backend.delete_relationship("non-existent")
        assert result is False


class TestInMemoryBackendContextManager:
    """Tests for context manager functionality."""

    def test_context_manager(self):
        """Test using backend as context manager."""
        with InMemoryBackend() as backend:
            entity = Entity(id="e1", name="Test", entity_type="concept")
            backend.add_entity(entity)
            assert backend.get_entity("e1") is not None

    def test_clear(self, populated_backend):
        """Test clearing all data."""
        populated_backend.clear()

        assert len(populated_backend._entities) == 0
        assert len(populated_backend._relationships) == 0
        assert len(populated_backend._outgoing) == 0
        assert len(populated_backend._incoming) == 0


# =============================================================================
# BackendGraphStore Adapter Tests
# =============================================================================


class TestBackendGraphStore:
    """Tests for the BackendGraphStore adapter."""

    @pytest.fixture
    def backend_store(self, populated_backend) -> BackendGraphStore:
        """Create a BackendGraphStore wrapping an InMemoryBackend."""
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

    def test_add_relationship(self, backend_store):
        """Test adding relationship through adapter."""
        rel = Relationship(
            id="new-rel",
            source_id="entity-1",
            target_id="entity-2",
            relation_type="NEW_REL",
        )
        result = backend_store.add_relationship(rel)
        assert result == "new-rel"

    def test_find_entities(self, backend_store):
        """Test finding entities through adapter."""
        entities = backend_store.find_entities(name="Test")
        assert len(entities) >= 1

    def test_find_entities_by_type(self, backend_store):
        """Test finding entities by type through adapter."""
        entities = backend_store.find_entities(entity_type="person")
        assert len(entities) == 1
        assert entities[0].entity_type == "person"

    def test_get_relationships(self, backend_store):
        """Test getting relationships through adapter."""
        rels = backend_store.get_relationships("entity-1")
        assert len(rels) == 1

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

    def test_delete_entity(self, backend_store):
        """Test deleting entity through adapter."""
        result = backend_store.delete_entity("entity-1")
        assert result is True
        assert backend_store.get_entity("entity-1") is None

    def test_close(self, backend_store):
        """Test closing the adapter closes the backend."""
        backend_store.close()
        # Should not raise

    def test_context_manager(self, populated_backend):
        """Test using adapter as context manager."""
        with BackendGraphStore(populated_backend) as store:
            entity = store.get_entity("entity-1")
            assert entity is not None


# =============================================================================
# Integration with create_graph_store Factory
# =============================================================================


class TestCreateGraphStoreWithBackend:
    """Tests for creating GraphStore using backend provider."""

    def test_create_store_with_backend(self, populated_backend):
        """Test creating a GraphStore with backend provider."""
        store = create_graph_store("backend", backend=populated_backend)

        assert isinstance(store, BackendGraphStore)
        assert store.backend is populated_backend

    def test_create_store_backend_without_backend_arg_raises(self):
        """Test that backend provider requires backend argument."""
        with pytest.raises(ValueError, match="'backend' argument required"):
            create_graph_store("backend")

    def test_create_memory_store_still_works(self):
        """Test that memory store creation still works."""
        from alma.graph.store import InMemoryGraphStore

        store = create_graph_store("memory")
        assert isinstance(store, InMemoryGraphStore)


# =============================================================================
# GraphBackend Interface Compliance Tests
# =============================================================================


class TestGraphBackendInterface:
    """Tests to verify InMemoryBackend implements GraphBackend interface."""

    def test_is_graph_backend(self, memory_backend):
        """Test that InMemoryBackend is a GraphBackend."""
        assert isinstance(memory_backend, GraphBackend)

    def test_has_required_methods(self, memory_backend):
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
            assert hasattr(memory_backend, method), f"Missing method: {method}"
            assert callable(
                getattr(memory_backend, method)
            ), f"Method not callable: {method}"


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_add_duplicate_entity(self, memory_backend, sample_entity):
        """Test adding the same entity twice updates it."""
        memory_backend.add_entity(sample_entity)

        # Modify and add again
        sample_entity.name = "Updated Name"
        memory_backend.add_entity(sample_entity)

        entity = memory_backend.get_entity(sample_entity.id)
        assert entity.name == "Updated Name"

    def test_relationship_to_nonexistent_entity(self, memory_backend):
        """Test that relationships can reference non-existent entities."""
        # This is allowed - the backend doesn't enforce referential integrity
        rel = Relationship(
            id="orphan-rel",
            source_id="nonexistent-1",
            target_id="nonexistent-2",
            relation_type="GHOST",
        )
        result = memory_backend.add_relationship(rel)
        assert result == "orphan-rel"

    def test_empty_search(self, memory_backend):
        """Test searching an empty backend."""
        results = memory_backend.search_entities("anything")
        assert len(results) == 0

    def test_get_relationships_empty(self, memory_backend):
        """Test getting relationships for entity with none."""
        rels = memory_backend.get_relationships("no-entity")
        assert len(rels) == 0

    def test_traverse_isolated_entity(self, memory_backend, sample_entity):
        """Test traversing from an entity with no relationships."""
        memory_backend.add_entity(sample_entity)

        store = BackendGraphStore(memory_backend)
        result = store.traverse(sample_entity.id)

        assert len(result.entities) == 1
        assert len(result.relationships) == 0
        assert len(result.paths) == 0
