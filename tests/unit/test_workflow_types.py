"""
Unit tests for ALMA Workflow Types.

Tests for:
- RetrievalScope enum (Task 1.11)
- WorkflowContext dataclass (Task 1.11)
- Checkpoint dataclass (Task 1.11)
- ArtifactRef dataclass (Task 1.11)
- WorkflowOutcome dataclass (Task 1.11)

Sprint 1 Task 1.11
"""

from unittest.mock import MagicMock

import pytest

from alma.workflow.artifacts import ArtifactRef, ArtifactType, link_artifact
from alma.workflow.checkpoint import (
    DEFAULT_MAX_STATE_SIZE,
    Checkpoint,
    CheckpointManager,
)
from alma.workflow.context import RetrievalScope, WorkflowContext
from alma.workflow.outcomes import WorkflowOutcome, WorkflowResult

# =============================================================================
# RetrievalScope Tests
# =============================================================================


class TestRetrievalScope:
    """Tests for RetrievalScope enum."""

    def test_enum_values(self):
        """Test all scope values exist."""
        assert RetrievalScope.NODE.value == "node"
        assert RetrievalScope.RUN.value == "run"
        assert RetrievalScope.WORKFLOW.value == "workflow"
        assert RetrievalScope.AGENT.value == "agent"
        assert RetrievalScope.TENANT.value == "tenant"
        assert RetrievalScope.GLOBAL.value == "global"

    def test_from_string_valid(self):
        """Test creating scope from valid string."""
        assert RetrievalScope.from_string("node") == RetrievalScope.NODE
        assert RetrievalScope.from_string("NODE") == RetrievalScope.NODE
        assert RetrievalScope.from_string("Run") == RetrievalScope.RUN
        assert RetrievalScope.from_string("GLOBAL") == RetrievalScope.GLOBAL

    def test_from_string_invalid(self):
        """Test invalid string raises ValueError."""
        with pytest.raises(ValueError) as exc:
            RetrievalScope.from_string("invalid")
        assert "Invalid RetrievalScope" in str(exc.value)
        assert "invalid" in str(exc.value)

    def test_is_broader_than(self):
        """Test scope comparison."""
        assert RetrievalScope.GLOBAL.is_broader_than(RetrievalScope.TENANT)
        assert RetrievalScope.TENANT.is_broader_than(RetrievalScope.AGENT)
        assert RetrievalScope.AGENT.is_broader_than(RetrievalScope.WORKFLOW)
        assert RetrievalScope.WORKFLOW.is_broader_than(RetrievalScope.RUN)
        assert RetrievalScope.RUN.is_broader_than(RetrievalScope.NODE)

        # Same scope is not broader
        assert not RetrievalScope.NODE.is_broader_than(RetrievalScope.NODE)
        assert not RetrievalScope.GLOBAL.is_broader_than(RetrievalScope.GLOBAL)

        # Reverse should be False
        assert not RetrievalScope.NODE.is_broader_than(RetrievalScope.RUN)
        assert not RetrievalScope.TENANT.is_broader_than(RetrievalScope.GLOBAL)


# =============================================================================
# WorkflowContext Tests
# =============================================================================


class TestWorkflowContext:
    """Tests for WorkflowContext dataclass."""

    def test_default_values(self):
        """Test default context creation."""
        ctx = WorkflowContext()
        assert ctx.tenant_id is None
        assert ctx.workflow_id is None
        assert ctx.run_id is None
        assert ctx.node_id is None
        assert ctx.branch_id is None
        assert ctx.metadata == {}
        assert ctx.created_at is not None

    def test_full_context(self):
        """Test fully populated context."""
        ctx = WorkflowContext(
            tenant_id="tenant-1",
            workflow_id="wf-1",
            run_id="run-1",
            node_id="node-1",
            branch_id="branch-1",
            metadata={"key": "value"},
        )
        assert ctx.tenant_id == "tenant-1"
        assert ctx.workflow_id == "wf-1"
        assert ctx.run_id == "run-1"
        assert ctx.node_id == "node-1"
        assert ctx.branch_id == "branch-1"
        assert ctx.metadata == {"key": "value"}

    def test_validate_success(self):
        """Test successful validation."""
        # Minimal valid context
        ctx = WorkflowContext()
        ctx.validate()  # Should not raise

        # Full valid context
        ctx = WorkflowContext(
            tenant_id="t1",
            workflow_id="wf1",
            run_id="r1",
            node_id="n1",
        )
        ctx.validate()  # Should not raise

    def test_validate_require_tenant(self):
        """Test tenant validation."""
        ctx = WorkflowContext()
        with pytest.raises(ValueError) as exc:
            ctx.validate(require_tenant=True)
        assert "tenant_id is required" in str(exc.value)

        # With tenant_id should pass
        ctx = WorkflowContext(tenant_id="t1")
        ctx.validate(require_tenant=True)  # Should not raise

    def test_validate_node_requires_run(self):
        """Test node_id requires run_id."""
        ctx = WorkflowContext(node_id="n1")
        with pytest.raises(ValueError) as exc:
            ctx.validate()
        assert "node_id requires run_id" in str(exc.value)

    def test_validate_run_requires_workflow(self):
        """Test run_id requires workflow_id."""
        ctx = WorkflowContext(run_id="r1")
        with pytest.raises(ValueError) as exc:
            ctx.validate()
        assert "run_id requires workflow_id" in str(exc.value)

    def test_validate_branch_requires_run(self):
        """Test branch_id requires run_id."""
        ctx = WorkflowContext(branch_id="b1")
        with pytest.raises(ValueError) as exc:
            ctx.validate()
        assert "branch_id requires run_id" in str(exc.value)

    def test_get_scope_filter_global(self):
        """Test global scope filter returns empty dict."""
        ctx = WorkflowContext(tenant_id="t1", workflow_id="wf1")
        filters = ctx.get_scope_filter(RetrievalScope.GLOBAL)
        assert filters == {}

    def test_get_scope_filter_tenant(self):
        """Test tenant scope filter."""
        ctx = WorkflowContext(tenant_id="t1", workflow_id="wf1")
        filters = ctx.get_scope_filter(RetrievalScope.TENANT)
        assert filters == {"tenant_id": "t1"}

    def test_get_scope_filter_workflow(self):
        """Test workflow scope filter."""
        ctx = WorkflowContext(tenant_id="t1", workflow_id="wf1", run_id="r1")
        filters = ctx.get_scope_filter(RetrievalScope.WORKFLOW)
        assert filters == {"tenant_id": "t1", "workflow_id": "wf1"}

    def test_get_scope_filter_run(self):
        """Test run scope filter."""
        ctx = WorkflowContext(
            tenant_id="t1", workflow_id="wf1", run_id="r1", node_id="n1"
        )
        filters = ctx.get_scope_filter(RetrievalScope.RUN)
        assert filters == {"tenant_id": "t1", "workflow_id": "wf1", "run_id": "r1"}

    def test_get_scope_filter_node(self):
        """Test node scope filter."""
        ctx = WorkflowContext(
            tenant_id="t1", workflow_id="wf1", run_id="r1", node_id="n1"
        )
        filters = ctx.get_scope_filter(RetrievalScope.NODE)
        assert filters == {
            "tenant_id": "t1",
            "workflow_id": "wf1",
            "run_id": "r1",
            "node_id": "n1",
        }

    def test_with_node(self):
        """Test creating context with new node."""
        ctx = WorkflowContext(
            tenant_id="t1",
            workflow_id="wf1",
            run_id="r1",
            node_id="n1",
            metadata={"key": "value"},
        )
        new_ctx = ctx.with_node("n2")

        assert new_ctx.node_id == "n2"
        assert new_ctx.tenant_id == "t1"
        assert new_ctx.workflow_id == "wf1"
        assert new_ctx.run_id == "r1"
        assert new_ctx.metadata == {"key": "value"}
        # Original unchanged
        assert ctx.node_id == "n1"

    def test_with_branch(self):
        """Test creating context with new branch."""
        ctx = WorkflowContext(
            tenant_id="t1",
            workflow_id="wf1",
            run_id="r1",
        )
        new_ctx = ctx.with_branch("branch-1")

        assert new_ctx.branch_id == "branch-1"
        assert new_ctx.tenant_id == "t1"
        assert new_ctx.run_id == "r1"
        # Original unchanged
        assert ctx.branch_id is None

    def test_to_dict_and_from_dict(self):
        """Test serialization and deserialization."""
        ctx = WorkflowContext(
            tenant_id="t1",
            workflow_id="wf1",
            run_id="r1",
            node_id="n1",
            branch_id="b1",
            metadata={"key": "value"},
        )

        data = ctx.to_dict()
        assert data["tenant_id"] == "t1"
        assert data["workflow_id"] == "wf1"
        assert data["created_at"] is not None

        # Deserialize
        restored = WorkflowContext.from_dict(data)
        assert restored.tenant_id == ctx.tenant_id
        assert restored.workflow_id == ctx.workflow_id
        assert restored.run_id == ctx.run_id
        assert restored.node_id == ctx.node_id
        assert restored.branch_id == ctx.branch_id
        assert restored.metadata == ctx.metadata


# =============================================================================
# Checkpoint Tests
# =============================================================================


class TestCheckpoint:
    """Tests for Checkpoint dataclass."""

    def test_default_values(self):
        """Test default checkpoint creation."""
        cp = Checkpoint()
        assert cp.id is not None
        assert cp.run_id == ""
        assert cp.node_id == ""
        assert cp.state == {}
        assert cp.sequence_number == 0
        assert cp.branch_id is None
        assert cp.parent_checkpoint_id is None
        assert cp.state_hash == ""  # Empty state produces empty hash
        assert cp.metadata == {}
        assert cp.created_at is not None

    def test_state_hash_computed(self):
        """Test state hash is computed on creation."""
        cp = Checkpoint(
            run_id="run-1",
            node_id="node-1",
            state={"key": "value"},
        )
        assert cp.state_hash != ""
        assert len(cp.state_hash) == 64  # SHA256 hex length

    def test_state_hash_consistent(self):
        """Test same state produces same hash."""
        state = {"a": 1, "b": 2}
        cp1 = Checkpoint(state=state.copy())
        cp2 = Checkpoint(state=state.copy())
        assert cp1.state_hash == cp2.state_hash

    def test_state_hash_different_for_different_state(self):
        """Test different states produce different hashes."""
        cp1 = Checkpoint(state={"a": 1})
        cp2 = Checkpoint(state={"a": 2})
        assert cp1.state_hash != cp2.state_hash

    def test_has_changed(self):
        """Test change detection."""
        cp = Checkpoint(state={"a": 1, "b": 2})

        # Same state
        assert not cp.has_changed({"a": 1, "b": 2})

        # Different state
        assert cp.has_changed({"a": 1, "b": 3})
        assert cp.has_changed({"a": 1})
        assert cp.has_changed({})

    def test_get_state_size(self):
        """Test state size calculation."""
        cp = Checkpoint(state={"key": "value"})
        size = cp.get_state_size()
        assert size > 0
        assert isinstance(size, int)

    def test_validate_success(self):
        """Test successful validation."""
        cp = Checkpoint(
            run_id="run-1",
            node_id="node-1",
            state={"key": "value"},
        )
        cp.validate()  # Should not raise

    def test_validate_missing_run_id(self):
        """Test validation fails without run_id."""
        cp = Checkpoint(node_id="node-1")
        with pytest.raises(ValueError) as exc:
            cp.validate()
        assert "run_id is required" in str(exc.value)

    def test_validate_missing_node_id(self):
        """Test validation fails without node_id."""
        cp = Checkpoint(run_id="run-1")
        with pytest.raises(ValueError) as exc:
            cp.validate()
        assert "node_id is required" in str(exc.value)

    def test_validate_negative_sequence(self):
        """Test validation fails with negative sequence."""
        cp = Checkpoint(run_id="run-1", node_id="node-1", sequence_number=-1)
        with pytest.raises(ValueError) as exc:
            cp.validate()
        assert "sequence_number must be non-negative" in str(exc.value)

    def test_validate_state_size_exceeded(self):
        """Test validation fails when state too large."""
        large_state = {"data": "x" * (DEFAULT_MAX_STATE_SIZE + 1)}
        cp = Checkpoint(run_id="run-1", node_id="node-1", state=large_state)
        with pytest.raises(ValueError) as exc:
            cp.validate()
        assert "State size" in str(exc.value)
        assert "exceeds maximum" in str(exc.value)

    def test_validate_custom_max_size(self):
        """Test validation with custom max size."""
        cp = Checkpoint(run_id="run-1", node_id="node-1", state={"key": "value"})
        with pytest.raises(ValueError):
            cp.validate(max_state_size=1)  # 1 byte is too small

    def test_to_dict_and_from_dict(self):
        """Test serialization and deserialization."""
        cp = Checkpoint(
            run_id="run-1",
            node_id="node-1",
            state={"key": "value"},
            sequence_number=5,
            branch_id="branch-1",
            metadata={"meta": "data"},
        )

        data = cp.to_dict()
        assert data["run_id"] == "run-1"
        assert data["state"] == {"key": "value"}
        assert data["sequence_number"] == 5

        restored = Checkpoint.from_dict(data)
        assert restored.run_id == cp.run_id
        assert restored.node_id == cp.node_id
        assert restored.state == cp.state
        assert restored.sequence_number == cp.sequence_number
        assert restored.branch_id == cp.branch_id
        assert restored.state_hash == cp.state_hash


# =============================================================================
# ArtifactRef Tests
# =============================================================================


class TestArtifactType:
    """Tests for ArtifactType enum."""

    def test_enum_values(self):
        """Test artifact type values."""
        assert ArtifactType.FILE.value == "file"
        assert ArtifactType.SCREENSHOT.value == "screenshot"
        assert ArtifactType.LOG.value == "log"
        assert ArtifactType.REPORT.value == "report"
        assert ArtifactType.OTHER.value == "other"


class TestArtifactRef:
    """Tests for ArtifactRef dataclass."""

    def test_default_values(self):
        """Test default artifact ref creation."""
        ref = ArtifactRef()
        assert ref.id is not None
        assert ref.memory_id == ""
        assert ref.artifact_type == ArtifactType.OTHER
        assert ref.storage_url == ""
        assert ref.filename is None
        assert ref.mime_type is None
        assert ref.size_bytes is None
        assert ref.checksum is None
        assert ref.metadata == {}
        assert ref.created_at is not None

    def test_full_artifact_ref(self):
        """Test fully populated artifact ref."""
        ref = ArtifactRef(
            memory_id="mem-1",
            artifact_type=ArtifactType.SCREENSHOT,
            storage_url="r2://bucket/path/img.png",
            filename="img.png",
            mime_type="image/png",
            size_bytes=1024,
            checksum="abc123",
            metadata={"key": "value"},
        )
        assert ref.memory_id == "mem-1"
        assert ref.artifact_type == ArtifactType.SCREENSHOT
        assert ref.storage_url == "r2://bucket/path/img.png"
        assert ref.size_bytes == 1024

    def test_validate_success(self):
        """Test successful validation."""
        ref = ArtifactRef(
            memory_id="mem-1",
            storage_url="r2://bucket/path/file.txt",
        )
        ref.validate()  # Should not raise

    def test_validate_missing_memory_id(self):
        """Test validation fails without memory_id."""
        ref = ArtifactRef(storage_url="r2://bucket/file.txt")
        with pytest.raises(ValueError) as exc:
            ref.validate()
        assert "memory_id is required" in str(exc.value)

    def test_validate_missing_storage_url(self):
        """Test validation fails without storage_url."""
        ref = ArtifactRef(memory_id="mem-1")
        with pytest.raises(ValueError) as exc:
            ref.validate()
        assert "storage_url is required" in str(exc.value)

    def test_to_dict_and_from_dict(self):
        """Test serialization and deserialization."""
        ref = ArtifactRef(
            memory_id="mem-1",
            artifact_type=ArtifactType.LOG,
            storage_url="r2://bucket/log.txt",
            filename="log.txt",
            size_bytes=2048,
        )

        data = ref.to_dict()
        assert data["memory_id"] == "mem-1"
        assert data["artifact_type"] == "log"
        assert data["storage_url"] == "r2://bucket/log.txt"

        restored = ArtifactRef.from_dict(data)
        assert restored.memory_id == ref.memory_id
        assert restored.artifact_type == ArtifactType.LOG
        assert restored.storage_url == ref.storage_url
        assert restored.filename == ref.filename
        assert restored.size_bytes == ref.size_bytes


class TestLinkArtifact:
    """Tests for link_artifact helper function."""

    def test_link_artifact_basic(self):
        """Test basic artifact linking."""
        ref = link_artifact(
            memory_id="mem-1",
            artifact_type=ArtifactType.SCREENSHOT,
            storage_url="r2://bucket/img.png",
        )
        assert ref.memory_id == "mem-1"
        assert ref.artifact_type == ArtifactType.SCREENSHOT
        assert ref.storage_url == "r2://bucket/img.png"

    def test_link_artifact_full(self):
        """Test artifact linking with all fields."""
        ref = link_artifact(
            memory_id="mem-1",
            artifact_type=ArtifactType.REPORT,
            storage_url="r2://bucket/report.pdf",
            filename="report.pdf",
            mime_type="application/pdf",
            size_bytes=5000,
            checksum="sha256hash",
            metadata={"generated": "2024-01-01"},
        )
        assert ref.filename == "report.pdf"
        assert ref.mime_type == "application/pdf"
        assert ref.size_bytes == 5000
        assert ref.checksum == "sha256hash"

    def test_link_artifact_validates(self):
        """Test link_artifact validates the ref."""
        with pytest.raises(ValueError):
            link_artifact(
                memory_id="",  # Invalid
                artifact_type=ArtifactType.FILE,
                storage_url="r2://bucket/file.txt",
            )


# =============================================================================
# WorkflowOutcome Tests
# =============================================================================


class TestWorkflowResult:
    """Tests for WorkflowResult enum."""

    def test_enum_values(self):
        """Test workflow result values."""
        assert WorkflowResult.SUCCESS.value == "success"
        assert WorkflowResult.FAILURE.value == "failure"
        assert WorkflowResult.PARTIAL.value == "partial"
        assert WorkflowResult.CANCELLED.value == "cancelled"
        assert WorkflowResult.TIMEOUT.value == "timeout"


class TestWorkflowOutcome:
    """Tests for WorkflowOutcome dataclass."""

    def test_default_values(self):
        """Test default outcome creation."""
        outcome = WorkflowOutcome()
        assert outcome.id is not None
        assert outcome.tenant_id is None
        assert outcome.workflow_id == ""
        assert outcome.run_id == ""
        assert outcome.agent == ""
        assert outcome.project_id == ""
        assert outcome.result == WorkflowResult.SUCCESS
        assert outcome.summary == ""
        assert outcome.strategies_used == []
        assert outcome.successful_patterns == []
        assert outcome.failed_patterns == []
        assert outcome.extracted_heuristics == []
        assert outcome.extracted_anti_patterns == []
        assert outcome.duration_seconds is None
        assert outcome.node_count is None
        assert outcome.error_message is None
        assert outcome.embedding is None
        assert outcome.metadata == {}
        assert outcome.created_at is not None

    def test_full_outcome(self):
        """Test fully populated outcome."""
        outcome = WorkflowOutcome(
            tenant_id="t1",
            workflow_id="wf1",
            run_id="r1",
            agent="helena",
            project_id="proj1",
            result=WorkflowResult.SUCCESS,
            summary="Workflow completed successfully",
            strategies_used=["strategy1", "strategy2"],
            successful_patterns=["pattern1"],
            failed_patterns=[],
            duration_seconds=120.5,
            node_count=5,
        )
        assert outcome.tenant_id == "t1"
        assert outcome.workflow_id == "wf1"
        assert outcome.agent == "helena"
        assert outcome.duration_seconds == 120.5

    def test_validate_success(self):
        """Test successful validation."""
        outcome = WorkflowOutcome(
            workflow_id="wf1",
            run_id="r1",
            agent="helena",
            project_id="proj1",
        )
        outcome.validate()  # Should not raise

    def test_validate_require_tenant(self):
        """Test tenant validation."""
        outcome = WorkflowOutcome(
            workflow_id="wf1",
            run_id="r1",
            agent="helena",
            project_id="proj1",
        )
        with pytest.raises(ValueError) as exc:
            outcome.validate(require_tenant=True)
        assert "tenant_id is required" in str(exc.value)

    def test_validate_missing_workflow_id(self):
        """Test validation fails without workflow_id."""
        outcome = WorkflowOutcome(run_id="r1", agent="helena", project_id="proj1")
        with pytest.raises(ValueError) as exc:
            outcome.validate()
        assert "workflow_id is required" in str(exc.value)

    def test_validate_missing_run_id(self):
        """Test validation fails without run_id."""
        outcome = WorkflowOutcome(workflow_id="wf1", agent="helena", project_id="proj1")
        with pytest.raises(ValueError) as exc:
            outcome.validate()
        assert "run_id is required" in str(exc.value)

    def test_validate_missing_agent(self):
        """Test validation fails without agent."""
        outcome = WorkflowOutcome(workflow_id="wf1", run_id="r1", project_id="proj1")
        with pytest.raises(ValueError) as exc:
            outcome.validate()
        assert "agent is required" in str(exc.value)

    def test_validate_missing_project_id(self):
        """Test validation fails without project_id."""
        outcome = WorkflowOutcome(workflow_id="wf1", run_id="r1", agent="helena")
        with pytest.raises(ValueError) as exc:
            outcome.validate()
        assert "project_id is required" in str(exc.value)

    def test_is_success(self):
        """Test is_success property."""
        outcome = WorkflowOutcome(result=WorkflowResult.SUCCESS)
        assert outcome.is_success is True

        outcome = WorkflowOutcome(result=WorkflowResult.FAILURE)
        assert outcome.is_success is False

    def test_is_failure(self):
        """Test is_failure property."""
        outcome = WorkflowOutcome(result=WorkflowResult.FAILURE)
        assert outcome.is_failure is True

        outcome = WorkflowOutcome(result=WorkflowResult.TIMEOUT)
        assert outcome.is_failure is True

        outcome = WorkflowOutcome(result=WorkflowResult.SUCCESS)
        assert outcome.is_failure is False

        outcome = WorkflowOutcome(result=WorkflowResult.PARTIAL)
        assert outcome.is_failure is False

    def test_get_searchable_text(self):
        """Test searchable text generation."""
        outcome = WorkflowOutcome(
            summary="Login workflow test",
            strategies_used=["form validation", "error handling"],
            successful_patterns=["wait for element"],
            failed_patterns=["fixed sleep"],
            error_message="Timeout on submit",
        )
        text = outcome.get_searchable_text()
        assert "Login workflow test" in text
        assert "form validation" in text
        assert "error handling" in text
        assert "wait for element" in text
        assert "fixed sleep" in text
        assert "Timeout on submit" in text

    def test_get_searchable_text_minimal(self):
        """Test searchable text with minimal data."""
        outcome = WorkflowOutcome(summary="Simple test")
        text = outcome.get_searchable_text()
        assert text == "Simple test"

    def test_to_dict_and_from_dict(self):
        """Test serialization and deserialization."""
        outcome = WorkflowOutcome(
            tenant_id="t1",
            workflow_id="wf1",
            run_id="r1",
            agent="helena",
            project_id="proj1",
            result=WorkflowResult.PARTIAL,
            summary="Partial success",
            strategies_used=["s1", "s2"],
            successful_patterns=["p1"],
            failed_patterns=["p2"],
            duration_seconds=60.0,
            node_count=3,
            error_message="Some nodes failed",
            metadata={"key": "value"},
        )

        data = outcome.to_dict()
        assert data["tenant_id"] == "t1"
        assert data["result"] == "partial"
        assert data["strategies_used"] == ["s1", "s2"]

        restored = WorkflowOutcome.from_dict(data)
        assert restored.tenant_id == outcome.tenant_id
        assert restored.workflow_id == outcome.workflow_id
        assert restored.result == WorkflowResult.PARTIAL
        assert restored.strategies_used == outcome.strategies_used
        assert restored.duration_seconds == outcome.duration_seconds


# =============================================================================
# CheckpointManager Tests (Basic - full tests require storage mock)
# =============================================================================


class TestCheckpointManager:
    """Basic tests for CheckpointManager."""

    @pytest.fixture
    def mock_storage(self):
        """Create a mock storage backend."""
        storage = MagicMock()
        storage.get_checkpoint.return_value = None
        storage.get_latest_checkpoint.return_value = None
        storage.cleanup_checkpoints.return_value = 0
        return storage

    @pytest.fixture
    def manager(self, mock_storage):
        """Create a checkpoint manager with mock storage."""
        return CheckpointManager(storage=mock_storage)

    def test_create_checkpoint(self, manager, mock_storage):
        """Test creating a checkpoint."""
        cp = manager.create_checkpoint(
            run_id="run-1",
            node_id="node-1",
            state={"key": "value"},
        )

        assert cp is not None
        assert cp.run_id == "run-1"
        assert cp.node_id == "node-1"
        assert cp.state == {"key": "value"}
        assert cp.sequence_number == 0
        mock_storage.save_checkpoint.assert_called_once()

    def test_create_checkpoint_increments_sequence(self, manager, mock_storage):
        """Test sequence number increments."""
        # First checkpoint
        manager.create_checkpoint(run_id="run-1", node_id="node-1", state={"a": 1})

        # Second checkpoint - sequence should increment
        cp2 = manager.create_checkpoint(
            run_id="run-1", node_id="node-2", state={"b": 2}
        )
        assert cp2.sequence_number == 1

    def test_create_checkpoint_skip_unchanged(self, manager, mock_storage):
        """Test skipping checkpoint when state unchanged."""
        parent_cp = Checkpoint(
            id="parent-1",
            run_id="run-1",
            node_id="node-1",
            state={"key": "value"},
        )
        mock_storage.get_checkpoint.return_value = parent_cp

        # Try to create with same state
        result = manager.create_checkpoint(
            run_id="run-1",
            node_id="node-2",
            state={"key": "value"},
            parent_checkpoint_id="parent-1",
            skip_if_unchanged=True,
        )

        assert result is None  # Should be skipped

    def test_create_checkpoint_state_too_large(self, manager):
        """Test validation rejects oversized state."""
        large_state = {"data": "x" * (DEFAULT_MAX_STATE_SIZE + 1)}
        with pytest.raises(ValueError) as exc:
            manager.create_checkpoint(
                run_id="run-1",
                node_id="node-1",
                state=large_state,
            )
        assert "exceeds maximum" in str(exc.value)

    def test_get_resume_point(self, manager, mock_storage):
        """Test get_resume_point delegates to get_latest_checkpoint."""
        expected = Checkpoint(run_id="run-1", node_id="node-1")
        mock_storage.get_latest_checkpoint.return_value = expected

        result = manager.get_resume_point("run-1")
        assert result == expected
        mock_storage.get_latest_checkpoint.assert_called_with("run-1", None)

    def test_get_branch_checkpoints(self, manager, mock_storage):
        """Test getting checkpoints for multiple branches."""
        cp1 = Checkpoint(run_id="run-1", node_id="n1", branch_id="b1")
        cp2 = Checkpoint(run_id="run-1", node_id="n2", branch_id="b2")

        def side_effect(run_id, branch_id):
            if branch_id == "b1":
                return cp1
            elif branch_id == "b2":
                return cp2
            return None

        mock_storage.get_latest_checkpoint.side_effect = side_effect

        result = manager.get_branch_checkpoints("run-1", ["b1", "b2", "b3"])
        assert len(result) == 2
        assert result["b1"] == cp1
        assert result["b2"] == cp2
        assert "b3" not in result

    def test_cleanup_checkpoints(self, manager, mock_storage):
        """Test checkpoint cleanup."""
        mock_storage.cleanup_checkpoints.return_value = 5

        count = manager.cleanup_checkpoints("run-1", keep_latest=2)
        assert count == 5
        mock_storage.cleanup_checkpoints.assert_called_with("run-1", 2)
