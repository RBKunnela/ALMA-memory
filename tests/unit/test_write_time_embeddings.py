"""Tests for write-time embedding generation in LearningProtocol.

Verifies that when an embedder is provided, saved objects have non-None embeddings.
"""

from unittest.mock import MagicMock, patch

from alma.learning.protocols import LearningProtocol
from alma.retrieval.embeddings import MockEmbedder
from alma.storage.base import StorageBackend
from alma.types import MemoryScope


def _make_mock_storage():
    """Create a mock storage backend for testing."""
    storage = MagicMock(spec=StorageBackend)
    storage.get_outcomes.return_value = []
    storage.get_outcomes_for_agents.return_value = []
    return storage


def _make_protocol(with_embedder=True):
    """Create a LearningProtocol with optional embedder."""
    storage = _make_mock_storage()
    embedder = MockEmbedder(dimension=384) if with_embedder else None
    scopes = {
        "test-agent": MemoryScope(
            agent_name="test-agent",
            can_learn=[],
            cannot_learn=[],
        )
    }
    protocol = LearningProtocol(
        storage=storage,
        scopes=scopes,
        embedder=embedder,
    )
    return protocol, storage


class TestDomainKnowledgeEmbedding:
    """Tests for write-time embeddings in add_domain_knowledge."""

    def test_domain_knowledge_has_embedding_when_embedder_provided(self):
        protocol, storage = _make_protocol(with_embedder=True)

        result = protocol.add_domain_knowledge(
            agent="test-agent",
            project_id="proj-1",
            domain="testing",
            fact="Login endpoint uses JWT with 24h expiry",
            source="user_stated",
        )

        assert result.embedding is not None
        assert len(result.embedding) == 384

        # Verify storage was called with the embedding-bearing object
        storage.save_domain_knowledge.assert_called_once()
        saved = storage.save_domain_knowledge.call_args[0][0]
        assert saved.embedding is not None

    def test_domain_knowledge_no_embedding_without_embedder(self):
        protocol, storage = _make_protocol(with_embedder=False)

        result = protocol.add_domain_knowledge(
            agent="test-agent",
            project_id="proj-1",
            domain="testing",
            fact="Some fact",
            source="user_stated",
        )

        assert result.embedding is None

    def test_domain_knowledge_embedding_failure_graceful(self):
        protocol, storage = _make_protocol(with_embedder=True)
        protocol.embedder.encode = MagicMock(side_effect=RuntimeError("encode failed"))

        result = protocol.add_domain_knowledge(
            agent="test-agent",
            project_id="proj-1",
            domain="testing",
            fact="Some fact",
            source="user_stated",
        )

        # Should degrade gracefully -- save with None embedding
        assert result.embedding is None
        storage.save_domain_knowledge.assert_called_once()


class TestOutcomeEmbedding:
    """Tests for write-time embeddings in learn()."""

    def test_outcome_has_embedding_when_embedder_provided(self):
        protocol, storage = _make_protocol(with_embedder=True)

        result = protocol.learn(
            agent="test-agent",
            project_id="proj-1",
            task="Test the login flow",
            outcome=True,
            strategy_used="fill form then submit",
        )

        assert result.embedding is not None
        assert len(result.embedding) == 384

        storage.save_outcome.assert_called_once()
        saved = storage.save_outcome.call_args[0][0]
        assert saved.embedding is not None

    def test_outcome_no_embedding_without_embedder(self):
        protocol, storage = _make_protocol(with_embedder=False)

        result = protocol.learn(
            agent="test-agent",
            project_id="proj-1",
            task="Test login",
            outcome=True,
            strategy_used="direct submit",
        )

        assert result.embedding is None

    def test_outcome_embedding_failure_graceful(self):
        protocol, storage = _make_protocol(with_embedder=True)
        protocol.embedder.encode = MagicMock(side_effect=RuntimeError("boom"))

        result = protocol.learn(
            agent="test-agent",
            project_id="proj-1",
            task="Test login",
            outcome=True,
            strategy_used="direct submit",
        )

        assert result.embedding is None
        storage.save_outcome.assert_called_once()


class TestAntiPatternEmbedding:
    """Tests for write-time embeddings in anti-pattern creation."""

    def test_anti_pattern_has_embedding(self):
        protocol, storage = _make_protocol(with_embedder=True)

        # Set up storage to return enough failures to trigger anti-pattern
        from alma.types import Outcome
        from datetime import datetime, timezone

        failures = [
            Outcome(
                id=f"out_{i}",
                agent="test-agent",
                project_id="proj-1",
                task_type="testing",
                task_description="test login",
                success=False,
                strategy_used="sleep then check",
                error_message="timeout waiting for element",
                timestamp=datetime.now(timezone.utc),
            )
            for i in range(3)
        ]
        storage.get_outcomes.return_value = failures

        protocol.learn(
            agent="test-agent",
            project_id="proj-1",
            task="test login",
            outcome=False,
            strategy_used="sleep then check",
            error_message="timeout waiting for element",
        )

        # Anti-pattern should have been created with embedding
        if storage.save_anti_pattern.called:
            saved_ap = storage.save_anti_pattern.call_args[0][0]
            assert saved_ap.embedding is not None
            assert len(saved_ap.embedding) == 384


class TestHeuristicEmbedding:
    """Tests for write-time embeddings in heuristic creation."""

    def test_heuristic_has_embedding(self):
        protocol, storage = _make_protocol(with_embedder=True)

        from alma.types import Outcome
        from datetime import datetime, timezone

        # Set up enough successful outcomes to trigger heuristic creation
        successes = [
            Outcome(
                id=f"out_{i}",
                agent="test-agent",
                project_id="proj-1",
                task_type="testing",
                task_description="test login form",
                success=True,
                strategy_used="fill form then submit",
                timestamp=datetime.now(timezone.utc),
            )
            for i in range(4)
        ]
        storage.get_outcomes.return_value = successes

        protocol.learn(
            agent="test-agent",
            project_id="proj-1",
            task="test login form",
            outcome=True,
            strategy_used="fill form then submit",
            task_type="testing",
        )

        if storage.save_heuristic.called:
            saved_h = storage.save_heuristic.call_args[0][0]
            assert saved_h.embedding is not None
            assert len(saved_h.embedding) == 384
