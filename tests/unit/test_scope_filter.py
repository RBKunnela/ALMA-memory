"""
Unit tests for scope_filter passthrough in retrieve_with_scope.

Verifies that WorkflowContext.get_scope_filter() output is forwarded
through RetrievalEngine.retrieve() down to the storage backend's
get_* methods, enabling workflow-scoped memory retrieval.
"""

from unittest.mock import MagicMock

import pytest

from alma.core import ALMA
from alma.testing.factories import create_test_heuristic, create_test_outcome
from alma.testing.mocks import MockStorage
from alma.types import MemorySlice, ScopeFilter
from alma.workflow import RetrievalScope, WorkflowContext

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_storage():
    """Create a mock storage backend with workflow support."""
    storage = MagicMock()
    storage.get_stats.return_value = {}
    storage.get_checkpoint.return_value = None
    storage.get_latest_checkpoint.return_value = None
    storage.cleanup_checkpoints.return_value = 0
    storage.save_checkpoint.return_value = None
    storage.save_workflow_outcome.return_value = None
    storage.save_artifact_link.return_value = None
    storage.get_artifact_links.return_value = []
    return storage


@pytest.fixture
def mock_retrieval():
    """Create a mock retrieval engine."""
    retrieval = MagicMock()
    retrieval.invalidate_cache.return_value = None
    empty_slice = MemorySlice(
        heuristics=[],
        outcomes=[],
        preferences=[],
        domain_knowledge=[],
        anti_patterns=[],
        query="test",
        agent="test-agent",
        retrieval_time_ms=5,
    )
    retrieval.retrieve.return_value = empty_slice
    return retrieval


@pytest.fixture
def mock_learning():
    """Create a mock learning protocol."""
    return MagicMock()


@pytest.fixture
def alma_instance(mock_storage, mock_retrieval, mock_learning):
    """Create an ALMA instance with mocks."""
    return ALMA(
        storage=mock_storage,
        retrieval_engine=mock_retrieval,
        learning_protocol=mock_learning,
        scopes={},
        project_id="test-project",
    )


# ---------------------------------------------------------------------------
# Tests: scope_filter passthrough via mocked retrieval engine
# ---------------------------------------------------------------------------


class TestScopeFilterPassthrough:
    """Verify scope_filter flows from core.retrieve_with_scope to the retrieval engine."""

    def test_scope_filter_passed_to_retrieval_engine(
        self, alma_instance, mock_retrieval
    ):
        """[UNIT] retrieve_with_scope -- should pass scope_filter to retrieval.retrieve()."""
        context = WorkflowContext(
            tenant_id="tenant-1",
            workflow_id="wf-001",
            run_id="run-123",
        )

        alma_instance.retrieve_with_scope(
            task="Find patterns",
            agent="test-agent",
            context=context,
            scope=RetrievalScope.WORKFLOW,
        )

        mock_retrieval.retrieve.assert_called_once()
        call_kwargs = mock_retrieval.retrieve.call_args.kwargs
        assert "scope_filter" in call_kwargs
        sf = call_kwargs["scope_filter"]
        assert isinstance(sf, ScopeFilter)
        assert sf.workflow_id == "wf-001"

    def test_scope_filter_agent_scope(self, alma_instance, mock_retrieval):
        """[UNIT] retrieve_with_scope -- should produce scope_filter with agent-level scope."""
        context = WorkflowContext(
            tenant_id="tenant-1",
            workflow_id="wf-001",
            run_id="run-123",
        )

        alma_instance.retrieve_with_scope(
            task="Find patterns",
            agent="test-agent",
            context=context,
            scope=RetrievalScope.AGENT,
        )

        call_kwargs = mock_retrieval.retrieve.call_args.kwargs
        sf = call_kwargs["scope_filter"]
        assert sf is not None

    def test_scope_filter_run_scope(self, alma_instance, mock_retrieval):
        """[UNIT] retrieve_with_scope -- should produce scope_filter scoped to run_id."""
        context = WorkflowContext(
            tenant_id="tenant-1",
            workflow_id="wf-001",
            run_id="run-456",
        )

        alma_instance.retrieve_with_scope(
            task="Find patterns",
            agent="test-agent",
            context=context,
            scope=RetrievalScope.RUN,
        )

        call_kwargs = mock_retrieval.retrieve.call_args.kwargs
        sf = call_kwargs["scope_filter"]
        assert sf is not None
        assert sf.run_id == "run-456"


# ---------------------------------------------------------------------------
# Tests: MockStorage scope_filter filtering
# ---------------------------------------------------------------------------


class TestMockStorageScopeFilter:
    """Verify MockStorage applies scope_filter in get_* methods."""

    def test_get_heuristics_filters_by_workflow_id(self):
        """[UNIT] MockStorage.get_heuristics -- should filter by workflow_id in scope_filter."""
        storage = MockStorage()

        h1 = create_test_heuristic(
            id="h-1",
            metadata={"workflow_id": "wf-001"},
        )
        h2 = create_test_heuristic(
            id="h-2",
            metadata={"workflow_id": "wf-002"},
        )
        h3 = create_test_heuristic(
            id="h-3",
            metadata={},
        )
        storage.save_heuristic(h1)
        storage.save_heuristic(h2)
        storage.save_heuristic(h3)

        sf = ScopeFilter(workflow_id="wf-001")
        results = storage.get_heuristics(
            project_id="test-project", scope_filter=sf
        )

        assert len(results) == 1
        assert results[0].id == "h-1"

    def test_get_heuristics_no_filter_returns_all(self):
        """[UNIT] MockStorage.get_heuristics -- should return all when no scope_filter."""
        storage = MockStorage()

        h1 = create_test_heuristic(id="h-1", metadata={"workflow_id": "wf-001"})
        h2 = create_test_heuristic(id="h-2", metadata={"workflow_id": "wf-002"})
        storage.save_heuristic(h1)
        storage.save_heuristic(h2)

        results = storage.get_heuristics(project_id="test-project")
        assert len(results) == 2

    def test_get_outcomes_filters_by_workflow_id(self):
        """[UNIT] MockStorage.get_outcomes -- should filter by workflow_id in scope_filter."""
        storage = MockStorage()

        o1 = create_test_outcome(
            id="o-1",
            metadata={"workflow_id": "wf-001"},
        )
        o2 = create_test_outcome(
            id="o-2",
            metadata={"workflow_id": "wf-002"},
        )
        storage.save_outcome(o1)
        storage.save_outcome(o2)

        sf = ScopeFilter(workflow_id="wf-001")
        results = storage.get_outcomes(
            project_id="test-project", scope_filter=sf
        )

        assert len(results) == 1
        assert results[0].id == "o-1"

    def test_get_heuristics_filters_by_run_id(self):
        """[UNIT] MockStorage.get_heuristics -- should filter by run_id in scope_filter."""
        storage = MockStorage()

        h1 = create_test_heuristic(
            id="h-1",
            metadata={"workflow_id": "wf-001", "run_id": "run-A"},
        )
        h2 = create_test_heuristic(
            id="h-2",
            metadata={"workflow_id": "wf-001", "run_id": "run-B"},
        )
        storage.save_heuristic(h1)
        storage.save_heuristic(h2)

        sf = ScopeFilter(workflow_id="wf-001", run_id="run-A")
        results = storage.get_heuristics(
            project_id="test-project", scope_filter=sf
        )

        assert len(results) == 1
        assert results[0].id == "h-1"

    def test_scope_filter_with_no_matching_items(self):
        """[UNIT] MockStorage.get_heuristics -- should return empty when nothing matches scope_filter."""
        storage = MockStorage()

        h1 = create_test_heuristic(
            id="h-1",
            metadata={"workflow_id": "wf-999"},
        )
        storage.save_heuristic(h1)

        sf = ScopeFilter(workflow_id="wf-001")
        results = storage.get_heuristics(
            project_id="test-project", scope_filter=sf
        )

        assert len(results) == 0
