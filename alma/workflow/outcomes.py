"""
ALMA Workflow Outcomes.

Provides the WorkflowOutcome dataclass for capturing learnings
from completed workflow executions.

Sprint 1 Task 1.5
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4


class WorkflowResult(Enum):
    """Result status of a workflow execution."""

    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"  # Partially succeeded
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


@dataclass
class WorkflowOutcome:
    """
    Captures learnings from a completed workflow execution.

    WorkflowOutcome records what was learned from running a workflow,
    including the strategies used, what worked, what didn't, and any
    extracted heuristics or anti-patterns.

    Attributes:
        id: Unique outcome identifier
        tenant_id: Multi-tenant isolation identifier
        workflow_id: The workflow definition that was executed
        run_id: The specific run this outcome is from
        agent: The agent that executed the workflow
        project_id: Project scope identifier
        result: Overall result status
        summary: Human-readable summary of what happened
        strategies_used: List of strategies/approaches attempted
        successful_patterns: Patterns that worked well
        failed_patterns: Patterns that didn't work
        extracted_heuristics: IDs of heuristics created from this run
        extracted_anti_patterns: IDs of anti-patterns created from this run
        duration_seconds: How long the workflow took
        node_count: Number of nodes executed
        error_message: Error details if failed
        embedding: Vector embedding for semantic search
        metadata: Additional outcome metadata
        created_at: When this outcome was recorded
    """

    id: str = field(default_factory=lambda: str(uuid4()))
    tenant_id: Optional[str] = None
    workflow_id: str = ""
    run_id: str = ""
    agent: str = ""
    project_id: str = ""
    result: WorkflowResult = WorkflowResult.SUCCESS
    summary: str = ""
    strategies_used: List[str] = field(default_factory=list)
    successful_patterns: List[str] = field(default_factory=list)
    failed_patterns: List[str] = field(default_factory=list)
    extracted_heuristics: List[str] = field(default_factory=list)
    extracted_anti_patterns: List[str] = field(default_factory=list)
    duration_seconds: Optional[float] = None
    node_count: Optional[int] = None
    error_message: Optional[str] = None
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def validate(self, require_tenant: bool = False) -> None:
        """
        Validate the workflow outcome.

        Args:
            require_tenant: If True, tenant_id must be provided.

        Raises:
            ValueError: If validation fails.
        """
        if require_tenant and not self.tenant_id:
            raise ValueError("tenant_id is required for multi-tenant deployments")
        if not self.workflow_id:
            raise ValueError("workflow_id is required")
        if not self.run_id:
            raise ValueError("run_id is required")
        if not self.agent:
            raise ValueError("agent is required")
        if not self.project_id:
            raise ValueError("project_id is required")

    @property
    def is_success(self) -> bool:
        """Check if the workflow succeeded."""
        return self.result == WorkflowResult.SUCCESS

    @property
    def is_failure(self) -> bool:
        """Check if the workflow failed."""
        return self.result in (WorkflowResult.FAILURE, WorkflowResult.TIMEOUT)

    def get_searchable_text(self) -> str:
        """
        Get text suitable for embedding generation.

        Combines summary, strategies, and patterns into a single
        searchable string.
        """
        parts = [self.summary]

        if self.strategies_used:
            parts.append("Strategies: " + ", ".join(self.strategies_used))

        if self.successful_patterns:
            parts.append("Successful: " + ", ".join(self.successful_patterns))

        if self.failed_patterns:
            parts.append("Failed: " + ", ".join(self.failed_patterns))

        if self.error_message:
            parts.append("Error: " + self.error_message)

        return " | ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "tenant_id": self.tenant_id,
            "workflow_id": self.workflow_id,
            "run_id": self.run_id,
            "agent": self.agent,
            "project_id": self.project_id,
            "result": self.result.value,
            "summary": self.summary,
            "strategies_used": self.strategies_used,
            "successful_patterns": self.successful_patterns,
            "failed_patterns": self.failed_patterns,
            "extracted_heuristics": self.extracted_heuristics,
            "extracted_anti_patterns": self.extracted_anti_patterns,
            "duration_seconds": self.duration_seconds,
            "node_count": self.node_count,
            "error_message": self.error_message,
            "embedding": self.embedding,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkflowOutcome":
        """Create from dictionary."""
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        elif created_at is None:
            created_at = datetime.now(timezone.utc)

        result = data.get("result", "success")
        if isinstance(result, str):
            result = WorkflowResult(result)

        return cls(
            id=data.get("id", str(uuid4())),
            tenant_id=data.get("tenant_id"),
            workflow_id=data.get("workflow_id", ""),
            run_id=data.get("run_id", ""),
            agent=data.get("agent", ""),
            project_id=data.get("project_id", ""),
            result=result,
            summary=data.get("summary", ""),
            strategies_used=data.get("strategies_used", []),
            successful_patterns=data.get("successful_patterns", []),
            failed_patterns=data.get("failed_patterns", []),
            extracted_heuristics=data.get("extracted_heuristics", []),
            extracted_anti_patterns=data.get("extracted_anti_patterns", []),
            duration_seconds=data.get("duration_seconds"),
            node_count=data.get("node_count"),
            error_message=data.get("error_message"),
            embedding=data.get("embedding"),
            metadata=data.get("metadata", {}),
            created_at=created_at,
        )
