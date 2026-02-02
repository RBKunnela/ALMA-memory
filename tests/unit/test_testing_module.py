"""
Tests for the alma.testing module.

Verifies that MockStorage, MockEmbedder, and factory functions work correctly.
"""

from datetime import datetime, timedelta, timezone

from alma.testing import (
    MockEmbedder,
    MockStorage,
    create_test_anti_pattern,
    create_test_heuristic,
    create_test_knowledge,
    create_test_outcome,
    create_test_preference,
)
from alma.types import (
    AntiPattern,
    DomainKnowledge,
    Heuristic,
    Outcome,
    UserPreference,
)

# =============================================================================
# MockStorage Tests
# =============================================================================


class TestMockStorage:
    """Tests for MockStorage implementation."""

    def test_init_creates_empty_storage(self):
        """MockStorage should initialize with empty collections."""
        storage = MockStorage()
        assert storage.heuristic_count == 0
        assert storage.outcome_count == 0
        assert storage.preference_count == 0
        assert storage.knowledge_count == 0
        assert storage.anti_pattern_count == 0

    def test_save_and_get_heuristic(self):
        """Should save and retrieve heuristics."""
        storage = MockStorage()
        heuristic = create_test_heuristic(agent="helena", project_id="my-project")

        storage.save_heuristic(heuristic)

        found = storage.get_heuristics("my-project", agent="helena")
        assert len(found) == 1
        assert found[0].id == heuristic.id
        assert found[0].agent == "helena"

    def test_save_and_get_outcome(self):
        """Should save and retrieve outcomes."""
        storage = MockStorage()
        outcome = create_test_outcome(agent="victor", project_id="my-project")

        storage.save_outcome(outcome)

        found = storage.get_outcomes("my-project", agent="victor")
        assert len(found) == 1
        assert found[0].id == outcome.id

    def test_save_and_get_preference(self):
        """Should save and retrieve user preferences."""
        storage = MockStorage()
        pref = create_test_preference(user_id="user-123", category="code_style")

        storage.save_user_preference(pref)

        found = storage.get_user_preferences("user-123", category="code_style")
        assert len(found) == 1
        assert found[0].preference == pref.preference

    def test_save_and_get_domain_knowledge(self):
        """Should save and retrieve domain knowledge."""
        storage = MockStorage()
        knowledge = create_test_knowledge(domain="authentication")

        storage.save_domain_knowledge(knowledge)

        found = storage.get_domain_knowledge("test-project", domain="authentication")
        assert len(found) == 1
        assert found[0].domain == "authentication"

    def test_save_and_get_anti_pattern(self):
        """Should save and retrieve anti-patterns."""
        storage = MockStorage()
        anti_pattern = create_test_anti_pattern(pattern="Using sleep()")

        storage.save_anti_pattern(anti_pattern)

        found = storage.get_anti_patterns("test-project")
        assert len(found) == 1
        assert "sleep" in found[0].pattern

    def test_filter_by_project_id(self):
        """Should filter items by project_id."""
        storage = MockStorage()
        h1 = create_test_heuristic(project_id="project-a")
        h2 = create_test_heuristic(project_id="project-b")

        storage.save_heuristic(h1)
        storage.save_heuristic(h2)

        found_a = storage.get_heuristics("project-a")
        found_b = storage.get_heuristics("project-b")

        assert len(found_a) == 1
        assert len(found_b) == 1
        assert found_a[0].project_id == "project-a"

    def test_filter_by_agent(self):
        """Should filter items by agent name."""
        storage = MockStorage()
        h1 = create_test_heuristic(agent="helena")
        h2 = create_test_heuristic(agent="victor")

        storage.save_heuristic(h1)
        storage.save_heuristic(h2)

        found = storage.get_heuristics("test-project", agent="helena")
        assert len(found) == 1
        assert found[0].agent == "helena"

    def test_filter_by_min_confidence(self):
        """Should filter heuristics by minimum confidence."""
        storage = MockStorage()
        h1 = create_test_heuristic(confidence=0.9)
        h2 = create_test_heuristic(confidence=0.5)

        storage.save_heuristic(h1)
        storage.save_heuristic(h2)

        found = storage.get_heuristics("test-project", min_confidence=0.7)
        assert len(found) == 1
        assert found[0].confidence == 0.9

    def test_top_k_limit(self):
        """Should respect top_k limit."""
        storage = MockStorage()
        for i in range(10):
            h = create_test_heuristic(confidence=i / 10)
            storage.save_heuristic(h)

        found = storage.get_heuristics("test-project", top_k=3)
        assert len(found) == 3

    def test_update_heuristic(self):
        """Should update heuristic fields."""
        storage = MockStorage()
        heuristic = create_test_heuristic(confidence=0.5)
        storage.save_heuristic(heuristic)

        result = storage.update_heuristic(heuristic.id, {"confidence": 0.9})
        assert result is True

        found = storage.get_heuristics("test-project")
        assert found[0].confidence == 0.9

    def test_update_heuristic_not_found(self):
        """Should return False when updating non-existent heuristic."""
        storage = MockStorage()
        result = storage.update_heuristic("non-existent-id", {"confidence": 0.9})
        assert result is False

    def test_increment_heuristic_occurrence(self):
        """Should increment occurrence counts."""
        storage = MockStorage()
        heuristic = create_test_heuristic(occurrence_count=5, success_count=4)
        storage.save_heuristic(heuristic)

        storage.increment_heuristic_occurrence(heuristic.id, success=True)

        found = storage.get_heuristics("test-project")
        assert found[0].occurrence_count == 6
        assert found[0].success_count == 5

    def test_delete_heuristic(self):
        """Should delete heuristic by ID."""
        storage = MockStorage()
        heuristic = create_test_heuristic()
        storage.save_heuristic(heuristic)

        result = storage.delete_heuristic(heuristic.id)
        assert result is True
        assert storage.heuristic_count == 0

    def test_delete_heuristic_not_found(self):
        """Should return False when deleting non-existent heuristic."""
        storage = MockStorage()
        result = storage.delete_heuristic("non-existent-id")
        assert result is False

    def test_delete_outcomes_older_than(self):
        """Should delete outcomes older than specified date."""
        storage = MockStorage()
        now = datetime.now(timezone.utc)
        old_outcome = create_test_outcome(timestamp=now - timedelta(days=30))
        new_outcome = create_test_outcome(timestamp=now)

        storage.save_outcome(old_outcome)
        storage.save_outcome(new_outcome)

        deleted = storage.delete_outcomes_older_than(
            "test-project", older_than=now - timedelta(days=7)
        )
        assert deleted == 1
        assert storage.outcome_count == 1

    def test_delete_low_confidence_heuristics(self):
        """Should delete heuristics below confidence threshold."""
        storage = MockStorage()
        high_conf = create_test_heuristic(confidence=0.9)
        low_conf = create_test_heuristic(confidence=0.3)

        storage.save_heuristic(high_conf)
        storage.save_heuristic(low_conf)

        deleted = storage.delete_low_confidence_heuristics(
            "test-project", below_confidence=0.5
        )
        assert deleted == 1
        assert storage.heuristic_count == 1

    def test_get_stats(self):
        """Should return accurate statistics."""
        storage = MockStorage()
        storage.save_heuristic(create_test_heuristic())
        storage.save_outcome(create_test_outcome())
        storage.save_user_preference(create_test_preference())
        storage.save_domain_knowledge(create_test_knowledge())
        storage.save_anti_pattern(create_test_anti_pattern())

        stats = storage.get_stats("test-project")
        assert stats["heuristics"] == 1
        assert stats["outcomes"] == 1
        assert stats["domain_knowledge"] == 1
        assert stats["anti_patterns"] == 1
        assert stats["total_count"] > 0

    def test_clear(self):
        """Should clear all stored data."""
        storage = MockStorage()
        storage.save_heuristic(create_test_heuristic())
        storage.save_outcome(create_test_outcome())

        storage.clear()

        assert storage.heuristic_count == 0
        assert storage.outcome_count == 0

    def test_from_config(self):
        """Should create instance from config (ignores config for mock)."""
        storage = MockStorage.from_config({"some": "config"})
        assert isinstance(storage, MockStorage)


# =============================================================================
# MockEmbedder Tests
# =============================================================================


class TestMockEmbedder:
    """Tests for MockEmbedder re-exported from alma.testing."""

    def test_encode_returns_list_of_floats(self):
        """MockEmbedder should return deterministic embeddings."""
        embedder = MockEmbedder(dimension=384)
        embedding = embedder.encode("test text")

        assert isinstance(embedding, list)
        assert len(embedding) == 384
        assert all(isinstance(v, float) for v in embedding)

    def test_encode_is_deterministic(self):
        """Same text should produce same embedding."""
        embedder = MockEmbedder()
        embedding1 = embedder.encode("hello world")
        embedding2 = embedder.encode("hello world")

        assert embedding1 == embedding2

    def test_different_text_produces_different_embedding(self):
        """Different text should produce different embeddings."""
        embedder = MockEmbedder()
        embedding1 = embedder.encode("hello")
        embedding2 = embedder.encode("world")

        assert embedding1 != embedding2

    def test_encode_batch(self):
        """Should encode multiple texts at once."""
        embedder = MockEmbedder()
        embeddings = embedder.encode_batch(["text1", "text2", "text3"])

        assert len(embeddings) == 3
        assert all(len(e) == embedder.dimension for e in embeddings)

    def test_dimension_property(self):
        """Should return configured dimension."""
        embedder = MockEmbedder(dimension=512)
        assert embedder.dimension == 512


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestCreateTestHeuristic:
    """Tests for create_test_heuristic factory."""

    def test_creates_valid_heuristic(self):
        """Should create a valid Heuristic instance."""
        heuristic = create_test_heuristic()
        assert isinstance(heuristic, Heuristic)
        assert heuristic.id is not None
        assert heuristic.agent == "test-agent"
        assert heuristic.project_id == "test-project"

    def test_override_parameters(self):
        """Should allow overriding any parameter."""
        heuristic = create_test_heuristic(
            agent="custom-agent",
            confidence=0.99,
            condition="custom condition",
        )
        assert heuristic.agent == "custom-agent"
        assert heuristic.confidence == 0.99
        assert heuristic.condition == "custom condition"

    def test_auto_generates_id(self):
        """Should auto-generate unique IDs."""
        h1 = create_test_heuristic()
        h2 = create_test_heuristic()
        assert h1.id != h2.id

    def test_can_specify_id(self):
        """Should use provided ID when specified."""
        heuristic = create_test_heuristic(id="my-custom-id")
        assert heuristic.id == "my-custom-id"


class TestCreateTestOutcome:
    """Tests for create_test_outcome factory."""

    def test_creates_valid_outcome(self):
        """Should create a valid Outcome instance."""
        outcome = create_test_outcome()
        assert isinstance(outcome, Outcome)
        assert outcome.success is True
        assert outcome.duration_ms == 500

    def test_create_failed_outcome(self):
        """Should create a failed outcome with error message."""
        outcome = create_test_outcome(
            success=False, error_message="Something went wrong"
        )
        assert outcome.success is False
        assert outcome.error_message == "Something went wrong"


class TestCreateTestPreference:
    """Tests for create_test_preference factory."""

    def test_creates_valid_preference(self):
        """Should create a valid UserPreference instance."""
        pref = create_test_preference()
        assert isinstance(pref, UserPreference)
        assert pref.user_id == "test-user"
        assert pref.confidence == 1.0

    def test_override_category(self):
        """Should allow overriding category."""
        pref = create_test_preference(
            category="communication", preference="No emojis please"
        )
        assert pref.category == "communication"
        assert pref.preference == "No emojis please"


class TestCreateTestKnowledge:
    """Tests for create_test_knowledge factory."""

    def test_creates_valid_knowledge(self):
        """Should create a valid DomainKnowledge instance."""
        knowledge = create_test_knowledge()
        assert isinstance(knowledge, DomainKnowledge)
        assert knowledge.domain == "test_domain"

    def test_override_domain_and_fact(self):
        """Should allow overriding domain and fact."""
        knowledge = create_test_knowledge(
            domain="authentication", fact="JWT tokens are used"
        )
        assert knowledge.domain == "authentication"
        assert "JWT" in knowledge.fact


class TestCreateTestAntiPattern:
    """Tests for create_test_anti_pattern factory."""

    def test_creates_valid_anti_pattern(self):
        """Should create a valid AntiPattern instance."""
        ap = create_test_anti_pattern()
        assert isinstance(ap, AntiPattern)
        assert ap.occurrence_count == 3

    def test_override_pattern_details(self):
        """Should allow overriding pattern details."""
        ap = create_test_anti_pattern(
            pattern="Hardcoding secrets",
            why_bad="Security risk",
            better_alternative="Use environment variables",
        )
        assert "Hardcoding" in ap.pattern
        assert "Security" in ap.why_bad
        assert "environment" in ap.better_alternative


# =============================================================================
# Integration Tests
# =============================================================================


class TestTestingModuleIntegration:
    """Integration tests showing typical usage patterns."""

    def test_example_usage_from_docstring(self):
        """
        Test the example usage shown in the module docstring.

        This ensures the documented example actually works.
        """
        from alma.testing import MockStorage, create_test_heuristic

        storage = MockStorage()
        heuristic = create_test_heuristic(agent="test-agent")
        storage.save_heuristic(heuristic)

        found = storage.get_heuristics("test-project", agent="test-agent")
        assert len(found) == 1

    def test_complex_workflow(self):
        """Test a more complex workflow using multiple factories."""
        storage = MockStorage()

        # Create test data
        h1 = create_test_heuristic(agent="helena", confidence=0.9)
        h2 = create_test_heuristic(agent="victor", confidence=0.8)
        o1 = create_test_outcome(agent="helena", success=True)
        o2 = create_test_outcome(agent="helena", success=False)
        ap = create_test_anti_pattern(agent="helena")

        # Save all data
        storage.save_heuristic(h1)
        storage.save_heuristic(h2)
        storage.save_outcome(o1)
        storage.save_outcome(o2)
        storage.save_anti_pattern(ap)

        # Query and verify
        helena_heuristics = storage.get_heuristics("test-project", agent="helena")
        assert len(helena_heuristics) == 1

        all_outcomes = storage.get_outcomes("test-project")
        assert len(all_outcomes) == 2

        successful_outcomes = storage.get_outcomes(
            "test-project", agent="helena", success_only=True
        )
        assert len(successful_outcomes) == 1

        stats = storage.get_stats("test-project")
        assert stats["heuristics"] == 2
        assert stats["outcomes"] == 2
        assert stats["anti_patterns"] == 1

    def test_isolated_test_with_clear(self):
        """Test that clear() enables isolated tests."""
        storage = MockStorage()

        # First test
        storage.save_heuristic(create_test_heuristic())
        assert storage.heuristic_count == 1

        # Clear for next test
        storage.clear()

        # Second test starts fresh
        assert storage.heuristic_count == 0
        storage.save_heuristic(create_test_heuristic())
        assert storage.heuristic_count == 1
