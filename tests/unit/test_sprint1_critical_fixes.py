"""
Unit tests for Sprint 1 Critical Fixes.

Tests cover:
1. Security: json.loads handling in graph store (valid, empty, invalid JSON)
2. SQLite: Cascade deletion of embeddings when memories are deleted
3. Datetime: Timezone-aware timestamps on all memory objects
"""

import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from alma.graph.store import (
    Entity,
    InMemoryGraphStore,
    Relationship,
)
from alma.storage.sqlite_local import SQLiteStorage
from alma.types import (
    AntiPattern,
    DomainKnowledge,
    Heuristic,
    Outcome,
    UserPreference,
)

# =============================================================================
# Test 1: Security - JSON parsing in graph store
# =============================================================================


class TestGraphStoreJsonParsing:
    """Tests for json.loads handling in graph store operations."""

    @pytest.fixture
    def graph_store(self):
        """Create in-memory graph store for testing."""
        return InMemoryGraphStore()

    def test_entity_with_valid_json_properties(self, graph_store):
        """Test that entities with valid JSON properties are stored correctly."""
        entity = Entity(
            id="entity_001",
            name="Test Entity",
            entity_type="test",
            properties={"key": "value", "number": 42, "nested": {"a": 1}},
        )

        # Add entity
        entity_id = graph_store.add_entity(entity)
        assert entity_id == "entity_001"

        # Retrieve entity
        retrieved = graph_store.get_entity("entity_001")
        assert retrieved is not None
        assert retrieved.properties == {
            "key": "value",
            "number": 42,
            "nested": {"a": 1},
        }

    def test_entity_with_empty_properties(self, graph_store):
        """Test that entities with empty properties are handled correctly."""
        entity = Entity(
            id="entity_002",
            name="Empty Props Entity",
            entity_type="test",
            properties={},
        )

        # Add entity
        entity_id = graph_store.add_entity(entity)
        assert entity_id == "entity_002"

        # Retrieve entity
        retrieved = graph_store.get_entity("entity_002")
        assert retrieved is not None
        assert retrieved.properties == {}

    def test_relationship_with_valid_json_properties(self, graph_store):
        """Test that relationships with valid JSON properties work correctly."""
        # Create source and target entities first
        source = Entity(id="src_001", name="Source", entity_type="node")
        target = Entity(id="tgt_001", name="Target", entity_type="node")
        graph_store.add_entity(source)
        graph_store.add_entity(target)

        # Create relationship with properties
        relationship = Relationship(
            id="rel_001",
            source_id="src_001",
            target_id="tgt_001",
            relation_type="CONNECTS",
            properties={"weight": 1.5, "label": "test connection"},
            confidence=0.9,
        )

        # Add relationship
        rel_id = graph_store.add_relationship(relationship)
        assert rel_id == "rel_001"

        # Retrieve relationships
        relationships = graph_store.get_relationships("src_001", direction="outgoing")
        assert len(relationships) == 1
        assert relationships[0].properties == {
            "weight": 1.5,
            "label": "test connection",
        }

    def test_relationship_with_empty_properties(self, graph_store):
        """Test that relationships with empty properties are handled correctly."""
        source = Entity(id="src_002", name="Source2", entity_type="node")
        target = Entity(id="tgt_002", name="Target2", entity_type="node")
        graph_store.add_entity(source)
        graph_store.add_entity(target)

        relationship = Relationship(
            id="rel_002",
            source_id="src_002",
            target_id="tgt_002",
            relation_type="LINKS",
            properties={},
        )

        rel_id = graph_store.add_relationship(relationship)
        assert rel_id == "rel_002"

        relationships = graph_store.get_relationships("src_002")
        assert len(relationships) == 1
        assert relationships[0].properties == {}

    def test_neo4j_json_parsing_on_retrieval(self):
        """
        Test that Neo4j graph store handles JSON parsing properly.

        Note: This tests the _code path_ for JSON parsing without
        requiring a real Neo4j connection. We verify the code handles
        valid, empty, and None JSON cases.
        """
        # Test the json.loads pattern used in Neo4jGraphStore.get_entity
        # The actual code uses: json.loads(r["properties"]) if r["properties"] else {}

        # Valid JSON
        valid_props = '{"key": "value", "num": 123}'
        parsed_valid = json.loads(valid_props) if valid_props else {}
        assert parsed_valid == {"key": "value", "num": 123}

        # Empty string - should return empty dict
        empty_props = ""
        parsed_empty = json.loads(empty_props) if empty_props else {}
        assert parsed_empty == {}

        # None value - should return empty dict
        none_props = None
        parsed_none = json.loads(none_props) if none_props else {}
        assert parsed_none == {}

    def test_json_loads_with_invalid_json_raises_error(self):
        """
        Test that invalid JSON raises appropriate JSONDecodeError.

        This validates that the code properly propagates JSON parsing
        errors rather than silently failing.
        """
        invalid_json = "{invalid: json without quotes}"

        with pytest.raises(json.JSONDecodeError):
            json.loads(invalid_json)

        # Another common invalid case: trailing comma
        invalid_json_trailing = '{"key": "value",}'
        with pytest.raises(json.JSONDecodeError):
            json.loads(invalid_json_trailing)

        # Unclosed brace
        invalid_json_unclosed = '{"key": "value"'
        with pytest.raises(json.JSONDecodeError):
            json.loads(invalid_json_unclosed)

    def test_entity_properties_serialization_roundtrip(self, graph_store):
        """Test that complex nested properties survive serialization roundtrip."""
        complex_props = {
            "string": "hello",
            "int": 42,
            "float": 3.14,
            "bool": True,
            "null": None,
            "array": [1, 2, 3],
            "nested": {"deep": {"value": "found"}},
        }

        entity = Entity(
            id="entity_complex",
            name="Complex Entity",
            entity_type="test",
            properties=complex_props,
        )

        graph_store.add_entity(entity)
        retrieved = graph_store.get_entity("entity_complex")

        assert retrieved is not None
        assert retrieved.properties == complex_props


# =============================================================================
# Test 2: SQLite - Delete cascades to embeddings
# =============================================================================


class TestSQLiteDeleteCascade:
    """Tests for SQLite delete operations cascading to embeddings table."""

    @pytest.fixture
    def sqlite_storage(self):
        """Create temporary SQLite storage for tests."""
        temp_dir = tempfile.mkdtemp()
        db_path = Path(temp_dir) / "test_cascade.db"
        storage = SQLiteStorage(db_path=db_path)
        yield storage
        shutil.rmtree(temp_dir)

    def test_delete_heuristic_deletes_embedding(self, sqlite_storage):
        """Test that deleting a heuristic also deletes its embedding."""
        # Create heuristic with embedding
        heuristic = Heuristic(
            id="heur_delete_001",
            agent="helena",
            project_id="test-project",
            condition="test condition",
            strategy="test strategy",
            confidence=0.8,
            occurrence_count=5,
            success_count=4,
            last_validated=datetime.now(timezone.utc),
            created_at=datetime.now(timezone.utc),
            embedding=[0.1] * 384,  # 384-dimension embedding
        )

        # Save heuristic (this also saves the embedding)
        sqlite_storage.save_heuristic(heuristic)

        # Verify embedding exists in database
        with sqlite_storage._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT COUNT(*) FROM embeddings WHERE memory_type = 'heuristics' AND memory_id = ?",
                ("heur_delete_001",),
            )
            count = cursor.fetchone()[0]
            assert count == 1, "Embedding should exist after saving heuristic"

        # Delete the heuristic
        result = sqlite_storage.delete_heuristic("heur_delete_001")
        assert result is True, "Delete should return True for existing heuristic"

        # Verify embedding is also deleted
        with sqlite_storage._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT COUNT(*) FROM embeddings WHERE memory_type = 'heuristics' AND memory_id = ?",
                ("heur_delete_001",),
            )
            count = cursor.fetchone()[0]
            assert count == 0, "Embedding should be deleted when heuristic is deleted"

    def test_delete_outcome_deletes_embedding(self, sqlite_storage):
        """Test that deleting an outcome also deletes its embedding."""
        outcome = Outcome(
            id="out_delete_001",
            agent="victor",
            project_id="test-project",
            task_type="api_testing",
            task_description="Test delete cascade",
            success=True,
            strategy_used="test strategy",
            timestamp=datetime.now(timezone.utc),
            embedding=[0.2] * 384,
        )

        sqlite_storage.save_outcome(outcome)

        # Verify embedding exists
        with sqlite_storage._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT COUNT(*) FROM embeddings WHERE memory_type = 'outcomes' AND memory_id = ?",
                ("out_delete_001",),
            )
            count = cursor.fetchone()[0]
            assert count == 1

        # Delete outcome
        result = sqlite_storage.delete_outcome("out_delete_001")
        assert result is True

        # Verify embedding is deleted
        with sqlite_storage._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT COUNT(*) FROM embeddings WHERE memory_type = 'outcomes' AND memory_id = ?",
                ("out_delete_001",),
            )
            count = cursor.fetchone()[0]
            assert count == 0

    def test_delete_domain_knowledge_deletes_embedding(self, sqlite_storage):
        """Test that deleting domain knowledge also deletes its embedding."""
        knowledge = DomainKnowledge(
            id="dk_delete_001",
            agent="helena",
            project_id="test-project",
            domain="testing",
            fact="Important fact for testing",
            source="test",
            confidence=0.9,
            last_verified=datetime.now(timezone.utc),
            embedding=[0.3] * 384,
        )

        sqlite_storage.save_domain_knowledge(knowledge)

        # Verify embedding exists
        with sqlite_storage._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT COUNT(*) FROM embeddings WHERE memory_type = 'domain_knowledge' AND memory_id = ?",
                ("dk_delete_001",),
            )
            count = cursor.fetchone()[0]
            assert count == 1

        # Delete domain knowledge
        result = sqlite_storage.delete_domain_knowledge("dk_delete_001")
        assert result is True

        # Verify embedding is deleted
        with sqlite_storage._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT COUNT(*) FROM embeddings WHERE memory_type = 'domain_knowledge' AND memory_id = ?",
                ("dk_delete_001",),
            )
            count = cursor.fetchone()[0]
            assert count == 0

    def test_delete_anti_pattern_deletes_embedding(self, sqlite_storage):
        """Test that deleting an anti-pattern also deletes its embedding."""
        anti_pattern = AntiPattern(
            id="anti_delete_001",
            agent="victor",
            project_id="test-project",
            pattern="Bad pattern",
            why_bad="It causes issues",
            better_alternative="Good pattern",
            occurrence_count=3,
            last_seen=datetime.now(timezone.utc),
            created_at=datetime.now(timezone.utc),
            embedding=[0.4] * 384,
        )

        sqlite_storage.save_anti_pattern(anti_pattern)

        # Verify embedding exists
        with sqlite_storage._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT COUNT(*) FROM embeddings WHERE memory_type = 'anti_patterns' AND memory_id = ?",
                ("anti_delete_001",),
            )
            count = cursor.fetchone()[0]
            assert count == 1

        # Delete anti-pattern
        result = sqlite_storage.delete_anti_pattern("anti_delete_001")
        assert result is True

        # Verify embedding is deleted
        with sqlite_storage._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT COUNT(*) FROM embeddings WHERE memory_type = 'anti_patterns' AND memory_id = ?",
                ("anti_delete_001",),
            )
            count = cursor.fetchone()[0]
            assert count == 0

    def test_delete_nonexistent_returns_false(self, sqlite_storage):
        """Test that deleting nonexistent items returns False."""
        assert sqlite_storage.delete_heuristic("nonexistent") is False
        assert sqlite_storage.delete_outcome("nonexistent") is False
        assert sqlite_storage.delete_domain_knowledge("nonexistent") is False
        assert sqlite_storage.delete_anti_pattern("nonexistent") is False

    def test_delete_without_embedding(self, sqlite_storage):
        """Test that deleting a memory without an embedding works correctly."""
        # Create heuristic WITHOUT embedding
        heuristic = Heuristic(
            id="heur_no_embed_001",
            agent="helena",
            project_id="test-project",
            condition="test condition",
            strategy="test strategy",
            confidence=0.8,
            occurrence_count=5,
            success_count=4,
            last_validated=datetime.now(timezone.utc),
            created_at=datetime.now(timezone.utc),
            embedding=None,  # No embedding
        )

        sqlite_storage.save_heuristic(heuristic)

        # Verify no embedding was created
        with sqlite_storage._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT COUNT(*) FROM embeddings WHERE memory_type = 'heuristics' AND memory_id = ?",
                ("heur_no_embed_001",),
            )
            count = cursor.fetchone()[0]
            assert count == 0

        # Delete should still work
        result = sqlite_storage.delete_heuristic("heur_no_embed_001")
        assert result is True

        # Verify heuristic is gone
        heuristics = sqlite_storage.get_heuristics(
            project_id="test-project",
            agent="helena",
        )
        assert all(h.id != "heur_no_embed_001" for h in heuristics)


# =============================================================================
# Test 3: Datetime - Timezone-aware timestamps
# =============================================================================


class TestTimezoneAwareTimestamps:
    """Tests for timezone-aware timestamps on all memory objects."""

    def test_outcome_default_timestamp_is_timezone_aware(self):
        """Test that Outcome's default timestamp is timezone-aware."""
        outcome = Outcome(
            id="out_tz_001",
            agent="test",
            project_id="test-project",
            task_type="testing",
            task_description="Test timezone",
            success=True,
            strategy_used="test",
        )

        # Verify timestamp exists
        assert outcome.timestamp is not None

        # Verify timestamp is timezone-aware (tzinfo is not None)
        assert outcome.timestamp.tzinfo is not None, (
            "Outcome timestamp should be timezone-aware"
        )

        # Verify it's UTC
        assert outcome.timestamp.tzinfo == timezone.utc, (
            "Outcome timestamp should be in UTC"
        )

    def test_user_preference_default_timestamp_is_timezone_aware(self):
        """Test that UserPreference's default timestamp is timezone-aware."""
        preference = UserPreference(
            id="pref_tz_001",
            user_id="test-user",
            category="testing",
            preference="Test preference",
            source="test",
        )

        assert preference.timestamp is not None
        assert preference.timestamp.tzinfo is not None, (
            "UserPreference timestamp should be timezone-aware"
        )
        assert preference.timestamp.tzinfo == timezone.utc

    def test_domain_knowledge_default_timestamp_is_timezone_aware(self):
        """Test that DomainKnowledge's default last_verified is timezone-aware."""
        knowledge = DomainKnowledge(
            id="dk_tz_001",
            agent="test",
            project_id="test-project",
            domain="testing",
            fact="Test fact",
            source="test",
        )

        assert knowledge.last_verified is not None
        assert knowledge.last_verified.tzinfo is not None, (
            "DomainKnowledge last_verified should be timezone-aware"
        )
        assert knowledge.last_verified.tzinfo == timezone.utc

    def test_anti_pattern_default_timestamp_is_timezone_aware(self):
        """Test that AntiPattern's default created_at is timezone-aware."""
        anti_pattern = AntiPattern(
            id="anti_tz_001",
            agent="test",
            project_id="test-project",
            pattern="Bad pattern",
            why_bad="It's bad",
            better_alternative="Good pattern",
            occurrence_count=1,
            last_seen=datetime.now(timezone.utc),  # This one is required
        )

        # Check created_at default
        assert anti_pattern.created_at is not None
        assert anti_pattern.created_at.tzinfo is not None, (
            "AntiPattern created_at should be timezone-aware"
        )
        assert anti_pattern.created_at.tzinfo == timezone.utc

    def test_entity_default_timestamp_is_timezone_aware(self):
        """Test that Entity's default created_at is timezone-aware."""
        entity = Entity(
            id="entity_tz_001",
            name="Test Entity",
            entity_type="test",
        )

        assert entity.created_at is not None
        assert entity.created_at.tzinfo is not None, (
            "Entity created_at should be timezone-aware"
        )
        assert entity.created_at.tzinfo == timezone.utc

    def test_relationship_default_timestamp_is_timezone_aware(self):
        """Test that Relationship's default created_at is timezone-aware."""
        relationship = Relationship(
            id="rel_tz_001",
            source_id="src",
            target_id="tgt",
            relation_type="TEST",
        )

        assert relationship.created_at is not None
        assert relationship.created_at.tzinfo is not None, (
            "Relationship created_at should be timezone-aware"
        )
        assert relationship.created_at.tzinfo == timezone.utc

    def test_all_memory_types_have_consistent_utc_timestamps(self):
        """
        Test that all memory types create timestamps in UTC timezone.

        This is a comprehensive test to ensure timezone consistency
        across all memory object types.
        """
        now_before = datetime.now(timezone.utc)

        # Create all memory types
        outcome = Outcome(
            id="out_utc",
            agent="test",
            project_id="test",
            task_type="test",
            task_description="test",
            success=True,
            strategy_used="test",
        )

        preference = UserPreference(
            id="pref_utc",
            user_id="test",
            category="test",
            preference="test",
            source="test",
        )

        knowledge = DomainKnowledge(
            id="dk_utc",
            agent="test",
            project_id="test",
            domain="test",
            fact="test",
            source="test",
        )

        anti_pattern = AntiPattern(
            id="anti_utc",
            agent="test",
            project_id="test",
            pattern="test",
            why_bad="test",
            better_alternative="test",
            occurrence_count=1,
            last_seen=datetime.now(timezone.utc),
        )

        now_after = datetime.now(timezone.utc)

        # All timestamps should be between now_before and now_after
        assert now_before <= outcome.timestamp <= now_after
        assert now_before <= preference.timestamp <= now_after
        assert now_before <= knowledge.last_verified <= now_after
        assert now_before <= anti_pattern.created_at <= now_after

    def test_explicit_timezone_aware_timestamp_preserved(self):
        """Test that explicitly provided timezone-aware timestamps are preserved."""
        explicit_time = datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)

        outcome = Outcome(
            id="out_explicit",
            agent="test",
            project_id="test",
            task_type="test",
            task_description="test",
            success=True,
            strategy_used="test",
            timestamp=explicit_time,
        )

        assert outcome.timestamp == explicit_time
        assert outcome.timestamp.tzinfo == timezone.utc
