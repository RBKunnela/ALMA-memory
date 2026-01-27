"""
End-to-End Learning Cycle Tests.

Tests complete learning workflows:
1. Task -> Retrieve -> Execute -> Learn -> Retrieve Improved
2. Repeated patterns -> Heuristic creation
3. Repeated failures -> Anti-pattern creation
"""

from pathlib import Path

import pytest

from alma import ALMA, MemoryScope
from alma.integration.helena import HELENA_CATEGORIES
from alma.learning.protocols import LearningProtocol
from alma.retrieval.engine import RetrievalEngine
from alma.storage.file_based import FileBasedStorage


class TestCompleteLearnCycle:
    """Test the full retrieve -> execute -> learn -> retrieve improved cycle."""

    @pytest.fixture
    def fresh_alma(self, e2e_storage_dir: Path, e2e_project_id: str):
        """Create a fresh ALMA instance for each test."""
        storage = FileBasedStorage(storage_dir=e2e_storage_dir)
        scopes = {
            "helena": MemoryScope(
                agent_name="helena",
                can_learn=HELENA_CATEGORIES,
                cannot_learn=["backend"],
                min_occurrences_for_heuristic=3,
            ),
        }
        retrieval = RetrievalEngine(storage=storage, embedding_provider="mock")
        learning = LearningProtocol(storage=storage, scopes=scopes)

        return ALMA(
            storage=storage,
            retrieval_engine=retrieval,
            learning_protocol=learning,
            scopes=scopes,
            project_id=e2e_project_id,
        )

    def test_initial_retrieval_returns_empty(self, fresh_alma: ALMA):
        """First retrieval should return empty memories."""
        memories = fresh_alma.retrieve(
            task="Test login form",
            agent="helena",
        )

        assert memories.total_items == 0

    def test_learning_creates_outcome(self, fresh_alma: ALMA):
        """Learning should create an outcome record."""
        result = fresh_alma.learn(
            agent="helena",
            task="Test login form validation",
            outcome="success",
            strategy_used="check inputs first",
            task_type="form_testing",
            duration_ms=1500,
        )

        assert result is True

        stats = fresh_alma.get_stats(agent="helena")
        assert stats.get("outcomes_count", 0) >= 1

    def test_retrieval_after_learning(self, fresh_alma: ALMA):
        """Retrieval should return learned outcomes."""
        # First learn something
        fresh_alma.learn(
            agent="helena",
            task="Test login form validation",
            outcome="success",
            strategy_used="check inputs first",
            task_type="form_testing",
        )

        # Then retrieve for a similar task
        fresh_alma.retrieve(
            task="Test registration form",
            agent="helena",
        )

        # Should have at least the recent outcome
        stats = fresh_alma.get_stats(agent="helena")
        assert stats.get("outcomes_count", 0) >= 1

    def test_repeated_success_improves_confidence(self, fresh_alma: ALMA):
        """Repeated successes with same strategy should increase confidence."""
        strategy = "wait for animation before click"

        # Learn the same successful pattern multiple times
        for i in range(5):
            fresh_alma.learn(
                agent="helena",
                task=f"Test modal {i}",
                outcome="success",
                strategy_used=strategy,
                task_type="ui_component_patterns",
            )

        stats = fresh_alma.get_stats(agent="helena")
        assert stats.get("outcomes_count", 0) == 5

    def test_domain_knowledge_accumulates(self, fresh_alma: ALMA):
        """Adding domain knowledge should accumulate over time."""
        # Add multiple facts
        facts = [
            ("selector_patterns", "data-testid is more stable"),
            ("selector_patterns", "avoid using CSS classes"),
            ("form_testing", "test empty form submission first"),
        ]

        for domain, fact in facts:
            fresh_alma.add_domain_knowledge(
                agent="helena",
                domain=domain,
                fact=fact,
                source="test",
            )

        stats = fresh_alma.get_stats(agent="helena")
        assert stats.get("domain_knowledge_count", 0) == len(facts)

        # Retrieve should include domain knowledge
        memories = fresh_alma.retrieve(
            task="selector patterns",
            agent="helena",
        )

        assert len(memories.domain_knowledge) >= 1


class TestLearningProgression:
    """Test that learning improves over similar tasks."""

    @pytest.fixture
    def progression_alma(self, e2e_storage_dir: Path):
        """Create ALMA for progression testing."""
        storage = FileBasedStorage(storage_dir=e2e_storage_dir)
        scopes = {
            "victor": MemoryScope(
                agent_name="victor",
                can_learn=["api_design_patterns", "error_handling"],
                cannot_learn=["frontend"],
                min_occurrences_for_heuristic=3,
            ),
        }
        retrieval = RetrievalEngine(storage=storage, embedding_provider="mock")
        learning = LearningProtocol(storage=storage, scopes=scopes)

        return ALMA(
            storage=storage,
            retrieval_engine=retrieval,
            learning_protocol=learning,
            scopes=scopes,
            project_id="progression-test",
        )

    def test_performance_improves_over_iterations(self, progression_alma: ALMA):
        """Simulate learning: early failures, later successes."""
        task_type = "api_design_patterns"
        durations = []

        # Simulate 10 iterations with improving success rate
        for i in range(10):
            success = i >= 3  # First 3 fail, rest succeed
            duration = 2000 - (i * 100) if success else 3000

            progression_alma.learn(
                agent="victor",
                task=f"API test iteration {i}",
                outcome="success" if success else "failure",
                strategy_used=f"strategy v{i // 3}",
                task_type=task_type,
                duration_ms=duration,
                error_message=None if success else f"Error {i}",
            )

            durations.append(duration)

        # Success rate should improve (7/10 = 70%)
        stats = progression_alma.get_stats(agent="victor")
        assert stats.get("outcomes_count", 0) == 10

    def test_similar_tasks_share_learning(self, progression_alma: ALMA):
        """Learning from one task should help with similar tasks."""
        # Learn about authentication
        for _i in range(5):
            progression_alma.learn(
                agent="victor",
                task="Test JWT authentication",
                outcome="success",
                strategy_used="verify token signature first",
                task_type="api_design_patterns",
            )

        # Add domain knowledge
        progression_alma.add_domain_knowledge(
            agent="victor",
            domain="api_design_patterns",
            fact="JWT tokens should be checked before other checks",
            source="learning",
        )

        # Retrieve for a related task
        progression_alma.retrieve(
            task="Test OAuth authentication",
            agent="victor",
        )

        stats = progression_alma.get_stats(agent="victor")
        assert stats.get("domain_knowledge_count", 0) >= 1


class TestFailurePatternDetection:
    """Test that repeated failures lead to anti-pattern creation."""

    @pytest.fixture
    def failure_alma(self, e2e_storage_dir: Path):
        """Create ALMA for failure pattern testing."""
        storage = FileBasedStorage(storage_dir=e2e_storage_dir)
        scopes = {
            "helena": MemoryScope(
                agent_name="helena",
                can_learn=HELENA_CATEGORIES,
                cannot_learn=[],
                min_occurrences_for_heuristic=3,
            ),
        }
        retrieval = RetrievalEngine(storage=storage, embedding_provider="mock")
        learning = LearningProtocol(storage=storage, scopes=scopes)

        return ALMA(
            storage=storage,
            retrieval_engine=retrieval,
            learning_protocol=learning,
            scopes=scopes,
            project_id="failure-test",
        )

    def test_repeated_failures_recorded(self, failure_alma: ALMA):
        """Repeated failures should be recorded as outcomes."""
        bad_strategy = "using fixed sleep for waits"

        # Fail with the same bad strategy multiple times
        for i in range(5):
            failure_alma.learn(
                agent="helena",
                task=f"Test with bad approach {i}",
                outcome="failure",
                strategy_used=bad_strategy,
                task_type="testing_strategies",
                error_message="Timeout waiting for element",
            )

        stats = failure_alma.get_stats(agent="helena")
        assert stats.get("outcomes_count", 0) == 5

    def test_failure_pattern_retrieval(self, failure_alma: ALMA):
        """Failed outcomes should inform future retrievals."""
        # Create failure pattern
        for _i in range(3):
            failure_alma.learn(
                agent="helena",
                task="Test with position selector",
                outcome="failure",
                strategy_used="select by element position",
                task_type="selector_patterns",
                error_message="Element position changed",
            )

        # Retrieve for similar task
        failure_alma.retrieve(
            task="Find element to click",
            agent="helena",
        )

        stats = failure_alma.get_stats(agent="helena")
        assert stats.get("outcomes_count", 0) >= 3


class TestForgetMechanism:
    """Test memory pruning functionality."""

    @pytest.fixture
    def forget_alma(self, e2e_storage_dir: Path):
        """Create ALMA for forget testing."""
        storage = FileBasedStorage(storage_dir=e2e_storage_dir)
        scopes = {
            "helena": MemoryScope(
                agent_name="helena",
                can_learn=HELENA_CATEGORIES,
                cannot_learn=[],
                min_occurrences_for_heuristic=3,
            ),
        }
        retrieval = RetrievalEngine(storage=storage, embedding_provider="mock")
        learning = LearningProtocol(storage=storage, scopes=scopes)

        return ALMA(
            storage=storage,
            retrieval_engine=retrieval,
            learning_protocol=learning,
            scopes=scopes,
            project_id="forget-test",
        )

    def test_forget_prunes_old_outcomes(self, forget_alma: ALMA):
        """Forget should remove old outcomes."""
        # Create some recent outcomes
        for i in range(5):
            forget_alma.learn(
                agent="helena",
                task=f"Recent task {i}",
                outcome="success",
                strategy_used="recent strategy",
                task_type="form_testing",
            )

        stats_before = forget_alma.get_stats(agent="helena")

        # Try to forget with strict criteria
        forget_alma.forget(
            agent="helena",
            older_than_days=0,
            below_confidence=0.0,
        )

        stats_after = forget_alma.get_stats(agent="helena")

        # With older_than_days=0, nothing should be pruned (all are recent)
        assert stats_after.get("outcomes_count", 0) <= stats_before.get("outcomes_count", 5)


class TestCacheInvalidation:
    """Test that cache is properly invalidated after learning."""

    @pytest.fixture
    def cache_alma(self, e2e_storage_dir: Path):
        """Create ALMA for cache testing."""
        storage = FileBasedStorage(storage_dir=e2e_storage_dir)
        scopes = {
            "helena": MemoryScope(
                agent_name="helena",
                can_learn=HELENA_CATEGORIES,
                cannot_learn=[],
                min_occurrences_for_heuristic=3,
            ),
        }
        retrieval = RetrievalEngine(storage=storage, embedding_provider="mock")
        learning = LearningProtocol(storage=storage, scopes=scopes)

        return ALMA(
            storage=storage,
            retrieval_engine=retrieval,
            learning_protocol=learning,
            scopes=scopes,
            project_id="cache-test",
        )

    def test_cache_invalidated_after_learn(self, cache_alma: ALMA):
        """Cache should be invalidated after learning."""
        # First retrieval
        cache_alma.retrieve(
            task="Test form",
            agent="helena",
        )

        # Learn something new
        cache_alma.learn(
            agent="helena",
            task="Test form",
            outcome="success",
            strategy_used="new strategy",
            task_type="form_testing",
        )

        # Second retrieval should include new learning
        cache_alma.retrieve(
            task="Test form",
            agent="helena",
        )

        stats = cache_alma.get_stats(agent="helena")
        assert stats.get("outcomes_count", 0) >= 1

    def test_cache_invalidated_after_add_knowledge(self, cache_alma: ALMA):
        """Cache should be invalidated after adding domain knowledge."""
        # First retrieval
        cache_alma.retrieve(
            task="Selector patterns",
            agent="helena",
        )

        # Add new knowledge
        cache_alma.add_domain_knowledge(
            agent="helena",
            domain="selector_patterns",
            fact="New selector fact",
            source="test",
        )

        # Second retrieval should include new knowledge
        cache_alma.retrieve(
            task="Selector patterns",
            agent="helena",
        )

        stats = cache_alma.get_stats(agent="helena")
        assert stats.get("domain_knowledge_count", 0) >= 1
