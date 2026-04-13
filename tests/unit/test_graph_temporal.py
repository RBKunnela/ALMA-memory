"""
Unit tests for Graph Temporal Validity.

Tests the valid_from/valid_to fields on Relationship and
temporal filtering via get_relationships_as_of.
"""

from datetime import datetime, timedelta, timezone

import pytest

from alma.graph.backends.memory import InMemoryBackend
from alma.graph.store import (
    Entity,
    InMemoryGraphStore,
    Relationship,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def now() -> datetime:
    """Current UTC timestamp for test reference."""
    return datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def entity_alice() -> Entity:
    """Sample person entity."""
    return Entity(
        id="alice-1",
        name="Alice",
        entity_type="person",
    )


@pytest.fixture
def entity_bob() -> Entity:
    """Sample person entity."""
    return Entity(
        id="bob-1",
        name="Bob",
        entity_type="person",
    )


@pytest.fixture
def entity_project() -> Entity:
    """Sample project entity."""
    return Entity(
        id="proj-1",
        name="ALMA",
        entity_type="project",
    )


# =============================================================================
# Test: Relationship Temporal Fields
# =============================================================================


class TestRelationshipTemporalFields:
    """Tests for valid_from and valid_to on Relationship dataclass."""

    def test_default_values_are_none(self):
        """Temporal fields should default to None."""
        rel = Relationship(
            id="r1",
            source_id="a",
            target_id="b",
            relation_type="KNOWS",
        )
        assert rel.valid_from is None
        assert rel.valid_to is None

    def test_set_valid_from(self, now):
        """valid_from should be settable."""
        rel = Relationship(
            id="r1",
            source_id="a",
            target_id="b",
            relation_type="WORKS_AT",
            valid_from=now,
        )
        assert rel.valid_from == now
        assert rel.valid_to is None

    def test_set_both_temporal_fields(self, now):
        """Both valid_from and valid_to should be settable."""
        end = now + timedelta(days=365)
        rel = Relationship(
            id="r1",
            source_id="a",
            target_id="b",
            relation_type="WORKS_AT",
            valid_from=now,
            valid_to=end,
        )
        assert rel.valid_from == now
        assert rel.valid_to == end

    def test_temporal_fields_with_all_other_fields(self, now):
        """Temporal fields should work alongside all other Relationship fields."""
        rel = Relationship(
            id="r1",
            source_id="a",
            target_id="b",
            relation_type="MANAGES",
            properties={"role": "lead"},
            confidence=0.95,
            created_at=now,
            valid_from=now - timedelta(days=30),
            valid_to=now + timedelta(days=30),
        )
        assert rel.confidence == 0.95
        assert rel.properties == {"role": "lead"}
        assert rel.valid_from < now
        assert rel.valid_to > now


# =============================================================================
# Test: InMemoryGraphStore Temporal Filtering
# =============================================================================


class TestInMemoryGraphStoreTemporalFiltering:
    """Tests for get_relationships_as_of on InMemoryGraphStore."""

    def test_no_temporal_fields_always_matches(self, entity_alice, entity_bob, now):
        """Relationships with no temporal fields match any as_of date."""
        store = InMemoryGraphStore()
        store.add_entity(entity_alice)
        store.add_entity(entity_bob)

        rel = Relationship(
            id="r1",
            source_id=entity_alice.id,
            target_id=entity_bob.id,
            relation_type="KNOWS",
        )
        store.add_relationship(rel)

        result = store.get_relationships_as_of(entity_alice.id, now)
        assert len(result) == 1
        assert result[0].id == "r1"

    def test_valid_from_before_as_of(self, entity_alice, entity_project, now):
        """Relationship with valid_from before as_of should be included."""
        store = InMemoryGraphStore()
        store.add_entity(entity_alice)
        store.add_entity(entity_project)

        rel = Relationship(
            id="r1",
            source_id=entity_alice.id,
            target_id=entity_project.id,
            relation_type="WORKS_ON",
            valid_from=now - timedelta(days=30),
        )
        store.add_relationship(rel)

        result = store.get_relationships_as_of(entity_alice.id, now)
        assert len(result) == 1

    def test_valid_from_after_as_of_excluded(self, entity_alice, entity_project, now):
        """Relationship with valid_from after as_of should be excluded."""
        store = InMemoryGraphStore()
        store.add_entity(entity_alice)
        store.add_entity(entity_project)

        rel = Relationship(
            id="r1",
            source_id=entity_alice.id,
            target_id=entity_project.id,
            relation_type="WORKS_ON",
            valid_from=now + timedelta(days=30),
        )
        store.add_relationship(rel)

        result = store.get_relationships_as_of(entity_alice.id, now)
        assert len(result) == 0

    def test_expired_relationship_excluded(self, entity_alice, entity_bob, now):
        """Relationship with valid_to before as_of should be excluded."""
        store = InMemoryGraphStore()
        store.add_entity(entity_alice)
        store.add_entity(entity_bob)

        rel = Relationship(
            id="r1",
            source_id=entity_alice.id,
            target_id=entity_bob.id,
            relation_type="MANAGES",
            valid_from=now - timedelta(days=365),
            valid_to=now - timedelta(days=30),
        )
        store.add_relationship(rel)

        result = store.get_relationships_as_of(entity_alice.id, now)
        assert len(result) == 0

    def test_mixed_temporal_relationships(
        self, entity_alice, entity_bob, entity_project, now
    ):
        """Only currently valid relationships should be returned."""
        store = InMemoryGraphStore()
        store.add_entity(entity_alice)
        store.add_entity(entity_bob)
        store.add_entity(entity_project)

        # Current relationship (no end)
        current_rel = Relationship(
            id="r1",
            source_id=entity_alice.id,
            target_id=entity_project.id,
            relation_type="WORKS_ON",
            valid_from=now - timedelta(days=60),
        )
        # Expired relationship
        expired_rel = Relationship(
            id="r2",
            source_id=entity_alice.id,
            target_id=entity_bob.id,
            relation_type="MANAGED_BY",
            valid_from=now - timedelta(days=365),
            valid_to=now - timedelta(days=100),
        )
        # Future relationship
        future_rel = Relationship(
            id="r3",
            source_id=entity_alice.id,
            target_id=entity_bob.id,
            relation_type="MENTORS",
            valid_from=now + timedelta(days=30),
        )

        store.add_relationship(current_rel)
        store.add_relationship(expired_rel)
        store.add_relationship(future_rel)

        result = store.get_relationships_as_of(entity_alice.id, now)
        assert len(result) == 1
        assert result[0].id == "r1"

    def test_query_past_date(self, entity_alice, entity_bob, now):
        """Querying a past date should find relationships valid at that time."""
        store = InMemoryGraphStore()
        store.add_entity(entity_alice)
        store.add_entity(entity_bob)

        rel = Relationship(
            id="r1",
            source_id=entity_alice.id,
            target_id=entity_bob.id,
            relation_type="COLLEAGUES",
            valid_from=now - timedelta(days=365),
            valid_to=now - timedelta(days=100),
        )
        store.add_relationship(rel)

        # Query at a time when it was valid
        past_date = now - timedelta(days=200)
        result = store.get_relationships_as_of(entity_alice.id, past_date)
        assert len(result) == 1
        assert result[0].id == "r1"


# =============================================================================
# Test: InMemoryBackend Temporal Filtering
# =============================================================================


class TestInMemoryBackendTemporalFiltering:
    """Tests for get_relationships_as_of on InMemoryBackend."""

    def test_backend_temporal_filter(self, entity_alice, entity_bob, now):
        """Backend should support temporal filtering."""
        backend = InMemoryBackend()
        backend.add_entity(entity_alice)
        backend.add_entity(entity_bob)

        current = Relationship(
            id="r1",
            source_id=entity_alice.id,
            target_id=entity_bob.id,
            relation_type="KNOWS",
            valid_from=now - timedelta(days=30),
        )
        expired = Relationship(
            id="r2",
            source_id=entity_alice.id,
            target_id=entity_bob.id,
            relation_type="WORKS_WITH",
            valid_from=now - timedelta(days=365),
            valid_to=now - timedelta(days=100),
        )

        backend.add_relationship(current)
        backend.add_relationship(expired)

        result = backend.get_relationships_as_of(entity_alice.id, now)
        assert len(result) == 1
        assert result[0].id == "r1"

    def test_backend_no_temporal_fields_match_all(self, entity_alice, entity_bob, now):
        """Relationships with no temporal fields should always match."""
        backend = InMemoryBackend()
        backend.add_entity(entity_alice)
        backend.add_entity(entity_bob)

        rel = Relationship(
            id="r1",
            source_id=entity_alice.id,
            target_id=entity_bob.id,
            relation_type="KNOWS",
        )
        backend.add_relationship(rel)

        result = backend.get_relationships_as_of(entity_alice.id, now)
        assert len(result) == 1

    def test_backend_exact_boundary(self, entity_alice, entity_bob, now):
        """Relationship valid_from/valid_to equal to as_of should be included."""
        backend = InMemoryBackend()
        backend.add_entity(entity_alice)
        backend.add_entity(entity_bob)

        rel = Relationship(
            id="r1",
            source_id=entity_alice.id,
            target_id=entity_bob.id,
            relation_type="KNOWS",
            valid_from=now,
            valid_to=now,
        )
        backend.add_relationship(rel)

        result = backend.get_relationships_as_of(entity_alice.id, now)
        assert len(result) == 1
