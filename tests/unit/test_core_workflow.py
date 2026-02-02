"""
Unit tests for ALMA Core workflow integration methods.

Sprint 3 Tasks 3.1-3.5
Tests the workflow methods added to ALMA core in v0.6.0.
"""

from unittest.mock import MagicMock

import pytest

from alma.core import ALMA
from alma.workflow import (
    ArtifactRef,
    ArtifactType,
    Checkpoint,
    RetrievalScope,
    WorkflowContext,
    WorkflowOutcome,
    WorkflowResult,
)


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


class TestCheckpoint:
    """Tests for ALMA.checkpoint() method."""

    def test_checkpoint_creates_checkpoint(self, alma_instance, mock_storage):
        """Test that checkpoint() creates a checkpoint."""
        result = alma_instance.checkpoint(
            run_id="run-123",
            node_id="node-1",
            state={"key": "value"},
        )

        assert result is not None
        assert result.run_id == "run-123"
        assert result.node_id == "node-1"
        assert result.state == {"key": "value"}
        mock_storage.save_checkpoint.assert_called_once()

    def test_checkpoint_with_branch(self, alma_instance, mock_storage):
        """Test checkpoint with branch_id for parallel execution."""
        result = alma_instance.checkpoint(
            run_id="run-123",
            node_id="node-1",
            state={"data": 42},
            branch_id="branch-A",
        )

        assert result.branch_id == "branch-A"

    def test_checkpoint_with_parent(self, alma_instance, mock_storage):
        """Test checkpoint with parent reference."""
        result = alma_instance.checkpoint(
            run_id="run-123",
            node_id="node-2",
            state={"count": 1},
            parent_checkpoint_id="cp-001",
            skip_if_unchanged=False,
        )

        assert result.parent_checkpoint_id == "cp-001"

    def test_checkpoint_skip_if_unchanged(self, alma_instance, mock_storage):
        """Test that checkpoint skips if state unchanged."""
        # Set up existing checkpoint with same state
        existing_cp = Checkpoint(
            id="cp-001",
            run_id="run-123",
            node_id="node-1",
            state={"key": "value"},
            sequence_number=0,
        )
        mock_storage.get_checkpoint.return_value = existing_cp

        result = alma_instance.checkpoint(
            run_id="run-123",
            node_id="node-2",
            state={"key": "value"},
            parent_checkpoint_id="cp-001",
            skip_if_unchanged=True,
        )

        # Should return None since state unchanged
        assert result is None

    def test_checkpoint_validates_state_size(self, alma_instance):
        """Test that checkpoint validates state size."""
        # Create state larger than 1MB
        large_state = {"data": "x" * (1024 * 1024 + 1)}

        with pytest.raises(ValueError, match="exceeds maximum"):
            alma_instance.checkpoint(
                run_id="run-123",
                node_id="node-1",
                state=large_state,
            )


class TestGetResumePoint:
    """Tests for ALMA.get_resume_point() method."""

    def test_get_resume_point_returns_checkpoint(self, alma_instance, mock_storage):
        """Test get_resume_point returns latest checkpoint."""
        expected_cp = Checkpoint(
            id="cp-003",
            run_id="run-123",
            node_id="node-3",
            state={"progress": 75},
            sequence_number=3,
        )
        mock_storage.get_latest_checkpoint.return_value = expected_cp

        result = alma_instance.get_resume_point("run-123")

        assert result == expected_cp
        mock_storage.get_latest_checkpoint.assert_called_with("run-123", None)

    def test_get_resume_point_with_branch(self, alma_instance, mock_storage):
        """Test get_resume_point with branch filter."""
        alma_instance.get_resume_point("run-123", branch_id="branch-A")

        mock_storage.get_latest_checkpoint.assert_called_with("run-123", "branch-A")

    def test_get_resume_point_returns_none_if_no_checkpoints(
        self, alma_instance, mock_storage
    ):
        """Test get_resume_point returns None when no checkpoints."""
        mock_storage.get_latest_checkpoint.return_value = None

        result = alma_instance.get_resume_point("run-999")

        assert result is None


class TestMergeStates:
    """Tests for ALMA.merge_states() method."""

    def test_merge_states_default_reducer(self, alma_instance):
        """Test merge_states with default last_value reducer."""
        states = [
            {"count": 1, "name": "first"},
            {"count": 2, "name": "second"},
        ]

        result = alma_instance.merge_states(states)

        # Default reducer is last_value
        assert result["count"] == 2
        assert result["name"] == "second"

    def test_merge_states_with_sum_reducer(self, alma_instance):
        """Test merge_states with sum reducer for specific key."""
        states = [
            {"count": 5, "items": ["a"]},
            {"count": 3, "items": ["b", "c"]},
        ]

        result = alma_instance.merge_states(states, {"count": "sum", "items": "append"})

        assert result["count"] == 8
        assert result["items"] == ["a", "b", "c"]

    def test_merge_states_with_union_reducer(self, alma_instance):
        """Test merge_states with union reducer."""
        states = [
            {"tags": ["python", "ai"]},
            {"tags": ["ai", "ml"]},
        ]

        result = alma_instance.merge_states(states, {"tags": "union"})

        assert set(result["tags"]) == {"python", "ai", "ml"}

    def test_merge_states_empty_list(self, alma_instance):
        """Test merge_states with empty list."""
        result = alma_instance.merge_states([])

        assert result == {}

    def test_merge_states_single_state(self, alma_instance):
        """Test merge_states with single state."""
        states = [{"key": "value"}]

        result = alma_instance.merge_states(states)

        assert result == {"key": "value"}


class TestLearnFromWorkflow:
    """Tests for ALMA.learn_from_workflow() method."""

    def test_learn_from_workflow_success(
        self, alma_instance, mock_storage, mock_retrieval
    ):
        """Test recording a successful workflow outcome."""
        result = alma_instance.learn_from_workflow(
            agent="test-agent",
            workflow_id="wf-001",
            run_id="run-123",
            result="success",
            summary="Completed all tasks successfully",
            strategies_used=["parallel-processing", "caching"],
            successful_patterns=["batch-updates"],
            duration_seconds=45.5,
            node_count=5,
        )

        assert isinstance(result, WorkflowOutcome)
        assert result.workflow_id == "wf-001"
        assert result.run_id == "run-123"
        assert result.result == WorkflowResult.SUCCESS
        assert result.project_id == "test-project"
        mock_storage.save_workflow_outcome.assert_called_once()
        mock_retrieval.invalidate_cache.assert_called()

    def test_learn_from_workflow_failure(self, alma_instance, mock_storage):
        """Test recording a failed workflow outcome."""
        result = alma_instance.learn_from_workflow(
            agent="test-agent",
            workflow_id="wf-001",
            run_id="run-456",
            result="failure",
            summary="Failed due to network error",
            failed_patterns=["external-api-call"],
            error_message="Connection timeout",
            node_count=3,
        )

        assert result.result == WorkflowResult.FAILURE
        assert result.error_message == "Connection timeout"
        assert "external-api-call" in result.failed_patterns

    def test_learn_from_workflow_with_tenant(self, alma_instance, mock_storage):
        """Test workflow outcome with tenant isolation."""
        result = alma_instance.learn_from_workflow(
            agent="test-agent",
            workflow_id="wf-001",
            run_id="run-789",
            result="success",
            summary="Multi-tenant workflow completed",
            tenant_id="tenant-acme",
        )

        assert result.tenant_id == "tenant-acme"


class TestLinkArtifact:
    """Tests for ALMA.link_artifact() method."""

    def test_link_artifact_basic(self, alma_instance, mock_storage):
        """Test basic artifact linking."""
        result = alma_instance.link_artifact(
            memory_id="outcome-123",
            artifact_type="screenshot",
            storage_url="r2://alma-artifacts/test/screenshot.png",
        )

        assert isinstance(result, ArtifactRef)
        assert result.memory_id == "outcome-123"
        assert result.artifact_type == ArtifactType.SCREENSHOT
        assert result.storage_url == "r2://alma-artifacts/test/screenshot.png"
        mock_storage.save_artifact_link.assert_called_once()

    def test_link_artifact_with_metadata(self, alma_instance, mock_storage):
        """Test artifact linking with full metadata."""
        result = alma_instance.link_artifact(
            memory_id="outcome-456",
            artifact_type="log",
            storage_url="r2://alma-artifacts/test/build.log",
            filename="build.log",
            mime_type="text/plain",
            size_bytes=1024,
            checksum="abc123",
            metadata={"build_id": "b-001"},
        )

        assert result.filename == "build.log"
        assert result.mime_type == "text/plain"
        assert result.size_bytes == 1024
        assert result.checksum == "abc123"
        assert result.metadata["build_id"] == "b-001"

    def test_link_artifact_unknown_type(self, alma_instance, mock_storage):
        """Test artifact linking with unknown type defaults to OTHER."""
        result = alma_instance.link_artifact(
            memory_id="outcome-789",
            artifact_type="custom-type",
            storage_url="r2://test/file.bin",
        )

        assert result.artifact_type == ArtifactType.OTHER


class TestGetArtifacts:
    """Tests for ALMA.get_artifacts() method."""

    def test_get_artifacts_returns_list(self, alma_instance, mock_storage):
        """Test get_artifacts returns list of artifacts."""
        expected = [
            ArtifactRef(
                id="art-1",
                memory_id="outcome-123",
                artifact_type=ArtifactType.SCREENSHOT,
                storage_url="r2://test/1.png",
            ),
            ArtifactRef(
                id="art-2",
                memory_id="outcome-123",
                artifact_type=ArtifactType.LOG,
                storage_url="r2://test/log.txt",
            ),
        ]
        mock_storage.get_artifact_links.return_value = expected

        result = alma_instance.get_artifacts("outcome-123")

        assert len(result) == 2
        mock_storage.get_artifact_links.assert_called_with("outcome-123")

    def test_get_artifacts_empty(self, alma_instance, mock_storage):
        """Test get_artifacts returns empty list when no artifacts."""
        mock_storage.get_artifact_links.return_value = []

        result = alma_instance.get_artifacts("outcome-999")

        assert result == []


class TestCleanupCheckpoints:
    """Tests for ALMA.cleanup_checkpoints() method."""

    def test_cleanup_checkpoints(self, alma_instance, mock_storage):
        """Test cleanup removes old checkpoints."""
        mock_storage.cleanup_checkpoints.return_value = 5

        result = alma_instance.cleanup_checkpoints("run-123", keep_latest=2)

        assert result == 5
        mock_storage.cleanup_checkpoints.assert_called_with("run-123", 2)

    def test_cleanup_checkpoints_none_removed(self, alma_instance, mock_storage):
        """Test cleanup when no checkpoints to remove."""
        mock_storage.cleanup_checkpoints.return_value = 0

        result = alma_instance.cleanup_checkpoints("run-456")

        assert result == 0


class TestRetrieveWithScope:
    """Tests for ALMA.retrieve_with_scope() method."""

    def test_retrieve_with_scope_agent(self, alma_instance, mock_retrieval):
        """Test scoped retrieval at agent level."""
        from alma.types import MemorySlice

        mock_slice = MemorySlice(
            heuristics=[],
            outcomes=[],
            preferences=[],
            domain_knowledge=[],
            anti_patterns=[],
            query="test task",
            agent="test-agent",
            retrieval_time_ms=10,
        )
        mock_retrieval.retrieve.return_value = mock_slice

        context = WorkflowContext(
            tenant_id="tenant-1",
            workflow_id="wf-001",
            run_id="run-123",
        )

        result = alma_instance.retrieve_with_scope(
            task="Find relevant patterns",
            agent="test-agent",
            context=context,
            scope=RetrievalScope.AGENT,
        )

        assert result.metadata["scope"] == "agent"
        assert result.metadata["context"]["workflow_id"] == "wf-001"

    def test_retrieve_with_scope_workflow(self, alma_instance, mock_retrieval):
        """Test scoped retrieval at workflow level."""
        from alma.types import MemorySlice

        mock_slice = MemorySlice(
            heuristics=[],
            outcomes=[],
            preferences=[],
            domain_knowledge=[],
            anti_patterns=[],
            query="test",
            agent="test-agent",
            retrieval_time_ms=5,
        )
        mock_retrieval.retrieve.return_value = mock_slice

        context = WorkflowContext(
            workflow_id="wf-001",
            run_id="run-123",
        )

        result = alma_instance.retrieve_with_scope(
            task="Get workflow patterns",
            agent="test-agent",
            context=context,
            scope=RetrievalScope.WORKFLOW,
        )

        assert result.metadata["scope"] == "workflow"
        scope_filter = result.metadata["scope_filter"]
        assert scope_filter.get("workflow_id") == "wf-001"

    def test_retrieve_with_scope_run(self, alma_instance, mock_retrieval):
        """Test scoped retrieval at run level."""
        from alma.types import MemorySlice

        mock_slice = MemorySlice(
            heuristics=[],
            outcomes=[],
            preferences=[],
            domain_knowledge=[],
            anti_patterns=[],
            query="test",
            agent="test-agent",
            retrieval_time_ms=5,
        )
        mock_retrieval.retrieve.return_value = mock_slice

        context = WorkflowContext(
            workflow_id="wf-001",
            run_id="run-123",
            node_id="node-5",
        )

        result = alma_instance.retrieve_with_scope(
            task="Get run-specific context",
            agent="test-agent",
            context=context,
            scope=RetrievalScope.RUN,
        )

        scope_filter = result.metadata["scope_filter"]
        assert scope_filter.get("run_id") == "run-123"


class TestAsyncWorkflowMethods:
    """Tests for async workflow methods."""

    @pytest.mark.asyncio
    async def test_async_checkpoint(self, alma_instance, mock_storage):
        """Test async checkpoint creation."""
        result = await alma_instance.async_checkpoint(
            run_id="run-async",
            node_id="node-1",
            state={"async": True},
        )

        assert result is not None
        assert result.run_id == "run-async"

    @pytest.mark.asyncio
    async def test_async_get_resume_point(self, alma_instance, mock_storage):
        """Test async resume point retrieval."""
        expected_cp = Checkpoint(
            id="cp-async",
            run_id="run-async",
            node_id="node-3",
            state={"progress": 100},
        )
        mock_storage.get_latest_checkpoint.return_value = expected_cp

        result = await alma_instance.async_get_resume_point("run-async")

        assert result == expected_cp

    @pytest.mark.asyncio
    async def test_async_learn_from_workflow(self, alma_instance, mock_storage):
        """Test async workflow learning."""
        result = await alma_instance.async_learn_from_workflow(
            agent="async-agent",
            workflow_id="wf-async",
            run_id="run-async",
            result="success",
            summary="Async workflow completed",
        )

        assert isinstance(result, WorkflowOutcome)
        assert result.workflow_id == "wf-async"

    @pytest.mark.asyncio
    async def test_async_link_artifact(self, alma_instance, mock_storage):
        """Test async artifact linking."""
        result = await alma_instance.async_link_artifact(
            memory_id="async-outcome",
            artifact_type="screenshot",
            storage_url="r2://test/async.png",
        )

        assert isinstance(result, ArtifactRef)

    @pytest.mark.asyncio
    async def test_async_retrieve_with_scope(self, alma_instance, mock_retrieval):
        """Test async scoped retrieval."""
        from alma.types import MemorySlice

        mock_slice = MemorySlice(
            heuristics=[],
            outcomes=[],
            preferences=[],
            domain_knowledge=[],
            anti_patterns=[],
            query="async test",
            agent="async-agent",
            retrieval_time_ms=5,
        )
        mock_retrieval.retrieve.return_value = mock_slice

        context = WorkflowContext(workflow_id="wf-async")

        result = await alma_instance.async_retrieve_with_scope(
            task="Async retrieval",
            agent="async-agent",
            context=context,
        )

        assert result.metadata["scope"] == "agent"
