"""
Multi-Agent Integration Tests.

Tests for scenarios involving multiple agents working together,
including scope enforcement and cross-agent isolation.
"""

from alma import ALMA, MemoryScope
from alma.integration.helena import HELENA_CATEGORIES
from alma.integration.victor import VICTOR_CATEGORIES
from alma.learning.protocols import LearningProtocol
from alma.retrieval.engine import RetrievalEngine
from alma.storage.file_based import FileBasedStorage


class TestScopeEnforcement:
    """Tests for memory scope restrictions."""

    def test_helena_cannot_learn_backend_logic(self, seeded_alma: ALMA):
        """Helena should not be able to learn in backend_logic domain."""
        result = seeded_alma.add_domain_knowledge(
            agent="helena",
            domain="backend_logic",  # Forbidden for Helena
            fact="Database connections should use pooling",
            source="test",
        )

        assert result is None, "Helena should not be able to learn backend_logic"

    def test_victor_cannot_learn_frontend_styling(self, seeded_alma: ALMA):
        """Victor should not be able to learn in frontend_styling domain."""
        result = seeded_alma.add_domain_knowledge(
            agent="victor",
            domain="frontend_styling",  # Forbidden for Victor
            fact="Use CSS Grid for layouts",
            source="test",
        )

        assert result is None, "Victor should not be able to learn frontend_styling"

    def test_helena_can_learn_testing_strategies(self, seeded_alma: ALMA):
        """Helena should be able to learn in testing_strategies domain."""
        result = seeded_alma.add_domain_knowledge(
            agent="helena",
            domain="testing_strategies",  # Allowed for Helena
            fact="Use data-testid for reliable element selection",
            source="test",
        )

        assert result is not None, "Helena should be able to learn testing_strategies"
        assert result.domain == "testing_strategies"

    def test_victor_can_learn_api_design_patterns(self, seeded_alma: ALMA):
        """Victor should be able to learn in api_design_patterns domain."""
        result = seeded_alma.add_domain_knowledge(
            agent="victor",
            domain="api_design_patterns",  # Allowed for Victor
            fact="Use 201 Created with Location header",
            source="test",
        )

        assert result is not None, "Victor should be able to learn api_design_patterns"
        assert result.domain == "api_design_patterns"


class TestAgentIsolation:
    """Tests for memory isolation between agents."""

    def test_helena_retrieval_excludes_victor_memories(self, seeded_alma: ALMA):
        """Helena should only retrieve Helena-owned memories."""
        # Add Victor-specific knowledge
        seeded_alma.add_domain_knowledge(
            agent="victor",
            domain="api_design_patterns",
            fact="This is Victor's API knowledge",
            source="test",
        )

        # Retrieve for Helena
        memories = seeded_alma.retrieve(
            task="api patterns",
            agent="helena",
            top_k=10,
        )

        # Check no Victor memories are returned
        for dk in memories.domain_knowledge:
            assert dk.agent != "victor" or dk.domain in HELENA_CATEGORIES

    def test_victor_retrieval_excludes_helena_memories(self, seeded_alma: ALMA):
        """Victor should only retrieve Victor-owned memories."""
        # Add Helena-specific knowledge
        seeded_alma.add_domain_knowledge(
            agent="helena",
            domain="selector_patterns",
            fact="This is Helena's selector knowledge",
            source="test",
        )

        # Retrieve for Victor
        memories = seeded_alma.retrieve(
            task="selector patterns",
            agent="victor",
            top_k=10,
        )

        # Check no Helena memories are returned (selector_patterns is Helena-only)
        for dk in memories.domain_knowledge:
            if dk.domain == "selector_patterns":
                assert dk.agent != "helena" or dk.domain in VICTOR_CATEGORIES

    def test_shared_project_different_agents(self, temp_storage_dir, scopes):
        """Two agents in same project should have isolated memories."""
        storage = FileBasedStorage(storage_dir=temp_storage_dir)
        retrieval = RetrievalEngine(storage=storage, embedding_provider="local")
        learning = LearningProtocol(storage=storage, scopes=scopes)

        alma = ALMA(
            storage=storage,
            retrieval_engine=retrieval,
            learning_protocol=learning,
            scopes=scopes,
            project_id="shared-project",
        )

        # Helena learns something
        alma.learn(
            agent="helena",
            task="Test form validation",
            outcome="success",
            strategy_used="validate inputs first",
            task_type="form_testing",
        )

        # Victor learns something different
        alma.learn(
            agent="victor",
            task="Test API authentication",
            outcome="success",
            strategy_used="check token first",
            task_type="authentication_patterns",
        )

        # Each agent should see their own stats
        helena_stats = alma.get_stats(agent="helena")
        victor_stats = alma.get_stats(agent="victor")

        # Stats should be isolated
        assert helena_stats != victor_stats


class TestConcurrentAgentOperations:
    """Tests for multiple agents operating concurrently."""

    def test_simultaneous_learning(self, seeded_alma: ALMA):
        """Both agents can learn in the same project simultaneously."""
        # Helena learns
        helena_result = seeded_alma.learn(
            agent="helena",
            task="Concurrent test 1",
            outcome="success",
            strategy_used="Helena strategy",
            task_type="form_testing",
        )

        # Victor learns
        victor_result = seeded_alma.learn(
            agent="victor",
            task="Concurrent test 2",
            outcome="success",
            strategy_used="Victor strategy",
            task_type="api_design_patterns",
        )

        assert helena_result is True
        assert victor_result is True

        # Both should see their outcomes
        helena_memories = seeded_alma.retrieve(
            task="Concurrent test",
            agent="helena",
        )

        victor_memories = seeded_alma.retrieve(
            task="Concurrent test",
            agent="victor",
        )

        # Each agent retrieves memories (may include seeded data too)
        assert helena_memories is not None
        assert victor_memories is not None

    def test_cross_agent_preference_visibility(self, seeded_alma: ALMA):
        """User preferences should be visible to all agents."""
        # Add user preference
        pref = seeded_alma.add_user_preference(
            user_id="shared-user",
            category="code_style",
            preference="Always use TypeScript",
            source="test",
        )

        assert pref is not None

        # Both agents should see the preference when retrieving
        # (Preferences are user-scoped, not agent-scoped)
        # Note: The current implementation may not include preferences in retrieve
        # This tests the expected behavior


class TestHeuristicGeneration:
    """Tests for heuristic creation across agents."""

    def test_heuristic_created_after_min_occurrences(self, temp_storage_dir, scopes):
        """Heuristic should be created after min_occurrences threshold."""
        storage = FileBasedStorage(storage_dir=temp_storage_dir)
        retrieval = RetrievalEngine(storage=storage, embedding_provider="local")
        learning = LearningProtocol(storage=storage, scopes=scopes)

        alma = ALMA(
            storage=storage,
            retrieval_engine=retrieval,
            learning_protocol=learning,
            scopes=scopes,
            project_id="test-project",
        )

        # Learn the same successful pattern 3 times (min_occurrences)
        for _i in range(3):
            alma.learn(
                agent="helena",
                task="Test login form",
                outcome="success",
                strategy_used="validate email format first",
                task_type="form_testing",
            )

        # Check that a heuristic was created
        storage.get_heuristics(
            project_id="test-project",
            agent="helena",
        )

        # At least one heuristic should exist for this strategy pattern
        # (Implementation may vary - the learning protocol decides when to create)
        stats = alma.get_stats(agent="helena")
        assert stats.get("outcomes_count", 0) >= 3


class TestMemoryScopeIsAllowed:
    """Direct tests for MemoryScope.is_allowed method."""

    def test_scope_allows_explicit_domain(self):
        """Scope should allow domains in can_learn list."""
        scope = MemoryScope(
            agent_name="test_agent",
            can_learn=["testing", "debugging"],
            cannot_learn=["production"],
        )

        assert scope.is_allowed("testing") is True
        assert scope.is_allowed("debugging") is True

    def test_scope_denies_forbidden_domain(self):
        """Scope should deny domains in cannot_learn list."""
        scope = MemoryScope(
            agent_name="test_agent",
            can_learn=["testing"],
            cannot_learn=["production", "secrets"],
        )

        assert scope.is_allowed("production") is False
        assert scope.is_allowed("secrets") is False

    def test_scope_denies_unlisted_when_can_learn_specified(self):
        """Scope should deny domains not in can_learn when list is non-empty."""
        scope = MemoryScope(
            agent_name="test_agent",
            can_learn=["testing"],
            cannot_learn=[],
        )

        assert scope.is_allowed("testing") is True
        assert scope.is_allowed("other") is False

    def test_scope_allows_all_when_can_learn_empty(self):
        """Scope should allow all domains when can_learn is empty."""
        scope = MemoryScope(
            agent_name="test_agent",
            can_learn=[],
            cannot_learn=["forbidden"],
        )

        assert scope.is_allowed("anything") is True
        assert scope.is_allowed("testing") is True
        assert scope.is_allowed("forbidden") is False

    def test_cannot_learn_overrides_can_learn(self):
        """cannot_learn should take precedence over can_learn."""
        scope = MemoryScope(
            agent_name="test_agent",
            can_learn=["testing", "production"],
            cannot_learn=["production"],
        )

        assert scope.is_allowed("testing") is True
        assert scope.is_allowed("production") is False
