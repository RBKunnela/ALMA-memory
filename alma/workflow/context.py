"""
ALMA Workflow Context.

Defines the WorkflowContext dataclass and RetrievalScope enum for
scoped memory retrieval in workflow orchestration systems.

Sprint 1 Tasks 1.1, 1.2
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional


class RetrievalScope(Enum):
    """
    Defines the scope for memory retrieval operations.

    The hierarchy from most specific to most general:
    NODE -> RUN -> WORKFLOW -> AGENT -> TENANT -> GLOBAL

    Note: Named RetrievalScope (not MemoryScope) to avoid collision
    with the existing MemoryScope dataclass in alma/types.py which
    defines what an agent is *allowed* to learn, not *where* to search.
    """

    # Most specific - only memories from this specific node execution
    NODE = "node"

    # Memories from this specific workflow run
    RUN = "run"

    # Memories from all runs of this workflow definition
    WORKFLOW = "workflow"

    # Memories from all workflows for this agent (default)
    AGENT = "agent"

    # Memories from all agents within this tenant
    TENANT = "tenant"

    # All memories across all tenants (admin only)
    GLOBAL = "global"

    @classmethod
    def from_string(cls, value: str) -> "RetrievalScope":
        """Convert string to RetrievalScope enum."""
        try:
            return cls(value.lower())
        except ValueError:
            raise ValueError(
                f"Invalid RetrievalScope: '{value}'. "
                f"Valid options: {[s.value for s in cls]}"
            )

    def is_broader_than(self, other: "RetrievalScope") -> bool:
        """Check if this scope is broader than another scope."""
        hierarchy = [
            RetrievalScope.NODE,
            RetrievalScope.RUN,
            RetrievalScope.WORKFLOW,
            RetrievalScope.AGENT,
            RetrievalScope.TENANT,
            RetrievalScope.GLOBAL,
        ]
        return hierarchy.index(self) > hierarchy.index(other)


@dataclass
class WorkflowContext:
    """
    Context for workflow-scoped memory operations.

    Provides hierarchical scoping for AGtestari and similar workflow
    orchestration systems. All fields are optional except when
    require_tenant=True is passed to validate().

    Attributes:
        tenant_id: Multi-tenant isolation identifier
        workflow_id: Workflow definition identifier
        run_id: Specific workflow execution identifier
        node_id: Current node within the workflow
        branch_id: Parallel branch identifier (for fan-out patterns)
        metadata: Additional context data
        created_at: When this context was created
    """

    tenant_id: Optional[str] = None
    workflow_id: Optional[str] = None
    run_id: Optional[str] = None
    node_id: Optional[str] = None
    branch_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def validate(self, require_tenant: bool = False) -> None:
        """
        Validate the workflow context.

        Args:
            require_tenant: If True, tenant_id must be provided.
                           Use for multi-tenant deployments.

        Raises:
            ValueError: If validation fails.
        """
        if require_tenant and not self.tenant_id:
            raise ValueError(
                "tenant_id is required for multi-tenant deployments. "
                "Set require_tenant=False for single-tenant mode."
            )

        # If node_id is provided, run_id should also be provided
        if self.node_id and not self.run_id:
            raise ValueError("node_id requires run_id to be set")

        # If run_id is provided, workflow_id should also be provided
        if self.run_id and not self.workflow_id:
            raise ValueError("run_id requires workflow_id to be set")

        # If branch_id is provided, run_id should also be provided
        if self.branch_id and not self.run_id:
            raise ValueError("branch_id requires run_id to be set")

    def get_scope_filter(self, scope: RetrievalScope) -> Dict[str, Any]:
        """
        Build a filter dict for the given retrieval scope.

        Returns a dictionary that can be used to filter memories
        based on the workflow context and requested scope.

        Args:
            scope: The retrieval scope to filter by.

        Returns:
            Dictionary with filter criteria.
        """
        filters: Dict[str, Any] = {}

        if scope == RetrievalScope.GLOBAL:
            # No filters - return everything
            pass
        elif scope == RetrievalScope.TENANT:
            if self.tenant_id:
                filters["tenant_id"] = self.tenant_id
        elif scope == RetrievalScope.AGENT:
            if self.tenant_id:
                filters["tenant_id"] = self.tenant_id
            # Agent filtering is done separately via the agent parameter
        elif scope == RetrievalScope.WORKFLOW:
            if self.tenant_id:
                filters["tenant_id"] = self.tenant_id
            if self.workflow_id:
                filters["workflow_id"] = self.workflow_id
        elif scope == RetrievalScope.RUN:
            if self.tenant_id:
                filters["tenant_id"] = self.tenant_id
            if self.workflow_id:
                filters["workflow_id"] = self.workflow_id
            if self.run_id:
                filters["run_id"] = self.run_id
        elif scope == RetrievalScope.NODE:
            if self.tenant_id:
                filters["tenant_id"] = self.tenant_id
            if self.workflow_id:
                filters["workflow_id"] = self.workflow_id
            if self.run_id:
                filters["run_id"] = self.run_id
            if self.node_id:
                filters["node_id"] = self.node_id

        return filters

    def with_node(self, node_id: str) -> "WorkflowContext":
        """Create a new context with a different node_id."""
        return WorkflowContext(
            tenant_id=self.tenant_id,
            workflow_id=self.workflow_id,
            run_id=self.run_id,
            node_id=node_id,
            branch_id=self.branch_id,
            metadata=self.metadata.copy(),
            created_at=self.created_at,
        )

    def with_branch(self, branch_id: str) -> "WorkflowContext":
        """Create a new context for a parallel branch."""
        return WorkflowContext(
            tenant_id=self.tenant_id,
            workflow_id=self.workflow_id,
            run_id=self.run_id,
            node_id=self.node_id,
            branch_id=branch_id,
            metadata=self.metadata.copy(),
            created_at=self.created_at,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "tenant_id": self.tenant_id,
            "workflow_id": self.workflow_id,
            "run_id": self.run_id,
            "node_id": self.node_id,
            "branch_id": self.branch_id,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkflowContext":
        """Create from dictionary."""
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        elif created_at is None:
            created_at = datetime.now(timezone.utc)

        return cls(
            tenant_id=data.get("tenant_id"),
            workflow_id=data.get("workflow_id"),
            run_id=data.get("run_id"),
            node_id=data.get("node_id"),
            branch_id=data.get("branch_id"),
            metadata=data.get("metadata", {}),
            created_at=created_at,
        )
