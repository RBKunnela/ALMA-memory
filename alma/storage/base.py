"""
ALMA Storage Backend Interface.

Abstract base class that all storage backends must implement.

v0.6.0 adds workflow context support:
- Checkpoint CRUD operations
- WorkflowOutcome storage and retrieval
- ArtifactRef linking
- scope_filter parameter for workflow-scoped queries
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from alma.types import (
    AntiPattern,
    DomainKnowledge,
    Heuristic,
    Outcome,
    UserPreference,
)

if TYPE_CHECKING:
    from alma.session import SessionHandoff
    from alma.workflow import ArtifactRef, Checkpoint, WorkflowOutcome


class StorageBackend(ABC):
    """
    Abstract base class for ALMA storage backends.

    Implementations:
    - FileBasedStorage: JSON files (testing/fallback)
    - SQLiteStorage: Local SQLite + FAISS vectors
    - AzureCosmosStorage: Production Azure Cosmos DB
    """

    # ==================== WRITE OPERATIONS ====================

    @abstractmethod
    def save_heuristic(self, heuristic: Heuristic) -> str:
        """Save a heuristic, return its ID."""
        pass

    @abstractmethod
    def save_outcome(self, outcome: Outcome) -> str:
        """Save an outcome, return its ID."""
        pass

    @abstractmethod
    def save_user_preference(self, preference: UserPreference) -> str:
        """Save a user preference, return its ID."""
        pass

    @abstractmethod
    def save_domain_knowledge(self, knowledge: DomainKnowledge) -> str:
        """Save domain knowledge, return its ID."""
        pass

    @abstractmethod
    def save_anti_pattern(self, anti_pattern: AntiPattern) -> str:
        """Save an anti-pattern, return its ID."""
        pass

    # ==================== BATCH WRITE OPERATIONS ====================

    def save_heuristics(self, heuristics: List[Heuristic]) -> List[str]:
        """Save multiple heuristics in a batch. Default implementation calls save_heuristic in a loop."""
        return [self.save_heuristic(h) for h in heuristics]

    def save_outcomes(self, outcomes: List[Outcome]) -> List[str]:
        """Save multiple outcomes in a batch. Default implementation calls save_outcome in a loop."""
        return [self.save_outcome(o) for o in outcomes]

    def save_domain_knowledge_batch(
        self, knowledge_items: List[DomainKnowledge]
    ) -> List[str]:
        """Save multiple domain knowledge items in a batch. Default implementation calls save_domain_knowledge in a loop."""
        return [self.save_domain_knowledge(k) for k in knowledge_items]

    # ==================== READ OPERATIONS ====================

    @abstractmethod
    def get_heuristics(
        self,
        project_id: str,
        agent: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        top_k: int = 5,
        min_confidence: float = 0.0,
        scope_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Heuristic]:
        """
        Get heuristics, optionally filtered by agent and similarity.

        Args:
            project_id: Project to query
            agent: Filter by agent name
            embedding: Query embedding for semantic search
            top_k: Max results to return
            min_confidence: Minimum confidence threshold
            scope_filter: Optional workflow scope filter (v0.6.0+)
                         Keys: tenant_id, workflow_id, run_id, node_id

        Returns:
            List of matching heuristics
        """
        pass

    @abstractmethod
    def get_outcomes(
        self,
        project_id: str,
        agent: Optional[str] = None,
        task_type: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        top_k: int = 5,
        success_only: bool = False,
        scope_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Outcome]:
        """
        Get outcomes, optionally filtered.

        Args:
            project_id: Project to query
            agent: Filter by agent name
            task_type: Filter by task type
            embedding: Query embedding for semantic search
            top_k: Max results
            success_only: Only return successful outcomes
            scope_filter: Optional workflow scope filter (v0.6.0+)
                         Keys: tenant_id, workflow_id, run_id, node_id

        Returns:
            List of matching outcomes
        """
        pass

    @abstractmethod
    def get_user_preferences(
        self,
        user_id: str,
        category: Optional[str] = None,
    ) -> List[UserPreference]:
        """
        Get user preferences.

        Args:
            user_id: User to query
            category: Optional category filter

        Returns:
            List of user preferences
        """
        pass

    @abstractmethod
    def get_domain_knowledge(
        self,
        project_id: str,
        agent: Optional[str] = None,
        domain: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        top_k: int = 5,
        scope_filter: Optional[Dict[str, Any]] = None,
    ) -> List[DomainKnowledge]:
        """
        Get domain knowledge.

        Args:
            project_id: Project to query
            agent: Filter by agent
            domain: Filter by domain
            embedding: Query embedding for semantic search
            top_k: Max results
            scope_filter: Optional workflow scope filter (v0.6.0+)
                         Keys: tenant_id, workflow_id, run_id, node_id

        Returns:
            List of domain knowledge items
        """
        pass

    @abstractmethod
    def get_anti_patterns(
        self,
        project_id: str,
        agent: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        top_k: int = 5,
        scope_filter: Optional[Dict[str, Any]] = None,
    ) -> List[AntiPattern]:
        """
        Get anti-patterns.

        Args:
            project_id: Project to query
            agent: Filter by agent
            embedding: Query embedding for semantic search
            top_k: Max results
            scope_filter: Optional workflow scope filter (v0.6.0+)
                         Keys: tenant_id, workflow_id, run_id, node_id

        Returns:
            List of anti-patterns
        """
        pass

    # ==================== UPDATE OPERATIONS ====================

    @abstractmethod
    def update_heuristic(
        self,
        heuristic_id: str,
        updates: Dict[str, Any],
    ) -> bool:
        """
        Update a heuristic's fields.

        Args:
            heuristic_id: ID of heuristic to update
            updates: Dict of field->value updates

        Returns:
            True if updated, False if not found
        """
        pass

    @abstractmethod
    def increment_heuristic_occurrence(
        self,
        heuristic_id: str,
        success: bool,
    ) -> bool:
        """
        Increment heuristic occurrence count.

        Args:
            heuristic_id: ID of heuristic
            success: Whether this occurrence was successful

        Returns:
            True if updated
        """
        pass

    @abstractmethod
    def update_heuristic_confidence(
        self,
        heuristic_id: str,
        new_confidence: float,
    ) -> bool:
        """
        Update a heuristic's confidence value.

        Args:
            heuristic_id: ID of heuristic to update
            new_confidence: New confidence value (0.0 - 1.0)

        Returns:
            True if updated, False if not found
        """
        pass

    @abstractmethod
    def update_knowledge_confidence(
        self,
        knowledge_id: str,
        new_confidence: float,
    ) -> bool:
        """
        Update domain knowledge confidence value.

        Args:
            knowledge_id: ID of knowledge to update
            new_confidence: New confidence value (0.0 - 1.0)

        Returns:
            True if updated, False if not found
        """
        pass

    # ==================== DELETE OPERATIONS ====================

    @abstractmethod
    def delete_heuristic(self, heuristic_id: str) -> bool:
        """
        Delete a heuristic by ID.

        Args:
            heuristic_id: ID of heuristic to delete

        Returns:
            True if deleted, False if not found
        """
        pass

    @abstractmethod
    def delete_outcome(self, outcome_id: str) -> bool:
        """
        Delete an outcome by ID.

        Args:
            outcome_id: ID of outcome to delete

        Returns:
            True if deleted, False if not found
        """
        pass

    @abstractmethod
    def delete_domain_knowledge(self, knowledge_id: str) -> bool:
        """
        Delete domain knowledge by ID.

        Args:
            knowledge_id: ID of knowledge to delete

        Returns:
            True if deleted, False if not found
        """
        pass

    @abstractmethod
    def delete_anti_pattern(self, anti_pattern_id: str) -> bool:
        """
        Delete an anti-pattern by ID.

        Args:
            anti_pattern_id: ID of anti-pattern to delete

        Returns:
            True if deleted, False if not found
        """
        pass

    @abstractmethod
    def delete_outcomes_older_than(
        self,
        project_id: str,
        older_than: datetime,
        agent: Optional[str] = None,
    ) -> int:
        """
        Delete old outcomes.

        Args:
            project_id: Project to prune
            older_than: Delete outcomes older than this
            agent: Optional agent filter

        Returns:
            Number of items deleted
        """
        pass

    @abstractmethod
    def delete_low_confidence_heuristics(
        self,
        project_id: str,
        below_confidence: float,
        agent: Optional[str] = None,
    ) -> int:
        """
        Delete low-confidence heuristics.

        Args:
            project_id: Project to prune
            below_confidence: Delete below this threshold
            agent: Optional agent filter

        Returns:
            Number of items deleted
        """
        pass

    # ==================== MULTI-AGENT MEMORY SHARING ====================

    def get_heuristics_for_agents(
        self,
        project_id: str,
        agents: List[str],
        embedding: Optional[List[float]] = None,
        top_k: int = 5,
        min_confidence: float = 0.0,
    ) -> List[Heuristic]:
        """
        Get heuristics from multiple agents in one call.

        This enables multi-agent memory sharing where an agent can
        read memories from agents it inherits from.

        Args:
            project_id: Project to query
            agents: List of agent names to query
            embedding: Query embedding for semantic search
            top_k: Max results to return per agent
            min_confidence: Minimum confidence threshold

        Returns:
            List of matching heuristics from all specified agents
        """
        # Default implementation: query each agent individually
        results = []
        for agent in agents:
            agent_heuristics = self.get_heuristics(
                project_id=project_id,
                agent=agent,
                embedding=embedding,
                top_k=top_k,
                min_confidence=min_confidence,
            )
            results.extend(agent_heuristics)
        return results

    def get_outcomes_for_agents(
        self,
        project_id: str,
        agents: List[str],
        task_type: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        top_k: int = 5,
        success_only: bool = False,
    ) -> List[Outcome]:
        """
        Get outcomes from multiple agents in one call.

        Args:
            project_id: Project to query
            agents: List of agent names to query
            task_type: Filter by task type
            embedding: Query embedding for semantic search
            top_k: Max results to return per agent
            success_only: Only return successful outcomes

        Returns:
            List of matching outcomes from all specified agents
        """
        results = []
        for agent in agents:
            agent_outcomes = self.get_outcomes(
                project_id=project_id,
                agent=agent,
                task_type=task_type,
                embedding=embedding,
                top_k=top_k,
                success_only=success_only,
            )
            results.extend(agent_outcomes)
        return results

    def get_domain_knowledge_for_agents(
        self,
        project_id: str,
        agents: List[str],
        domain: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        top_k: int = 5,
    ) -> List[DomainKnowledge]:
        """
        Get domain knowledge from multiple agents in one call.

        Args:
            project_id: Project to query
            agents: List of agent names to query
            domain: Filter by domain
            embedding: Query embedding for semantic search
            top_k: Max results to return per agent

        Returns:
            List of matching domain knowledge from all specified agents
        """
        results = []
        for agent in agents:
            agent_knowledge = self.get_domain_knowledge(
                project_id=project_id,
                agent=agent,
                domain=domain,
                embedding=embedding,
                top_k=top_k,
            )
            results.extend(agent_knowledge)
        return results

    def get_anti_patterns_for_agents(
        self,
        project_id: str,
        agents: List[str],
        embedding: Optional[List[float]] = None,
        top_k: int = 5,
    ) -> List[AntiPattern]:
        """
        Get anti-patterns from multiple agents in one call.

        Args:
            project_id: Project to query
            agents: List of agent names to query
            embedding: Query embedding for semantic search
            top_k: Max results to return per agent

        Returns:
            List of matching anti-patterns from all specified agents
        """
        results = []
        for agent in agents:
            agent_patterns = self.get_anti_patterns(
                project_id=project_id,
                agent=agent,
                embedding=embedding,
                top_k=top_k,
            )
            results.extend(agent_patterns)
        return results

    # ==================== STATS ====================

    @abstractmethod
    def get_stats(
        self,
        project_id: str,
        agent: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get memory statistics.

        Returns:
            Dict with counts per memory type, total size, etc.
        """
        pass

    # ==================== MIGRATION SUPPORT ====================

    def get_schema_version(self) -> Optional[str]:
        """
        Get the current schema version.

        Returns:
            Current schema version string, or None if not tracked
        """
        # Default implementation returns None (no version tracking)
        return None

    def get_migration_status(self) -> Dict[str, Any]:
        """
        Get migration status information.

        Returns:
            Dict with current version, pending migrations, etc.
        """
        return {
            "current_version": self.get_schema_version(),
            "target_version": None,
            "pending_count": 0,
            "pending_versions": [],
            "needs_migration": False,
            "migration_supported": False,
        }

    def migrate(
        self, target_version: Optional[str] = None, dry_run: bool = False
    ) -> List[str]:
        """
        Apply pending schema migrations.

        Args:
            target_version: Optional target version (applies all if not specified)
            dry_run: If True, show what would be done without making changes

        Returns:
            List of applied migration versions
        """
        # Default implementation does nothing
        return []

    def rollback(self, target_version: str, dry_run: bool = False) -> List[str]:
        """
        Roll back schema to a previous version.

        Args:
            target_version: Version to roll back to
            dry_run: If True, show what would be done without making changes

        Returns:
            List of rolled back migration versions
        """
        # Default implementation does nothing
        return []

    # ==================== CHECKPOINT OPERATIONS (v0.6.0+) ====================

    def save_checkpoint(self, checkpoint: "Checkpoint") -> str:
        """
        Save a workflow checkpoint.

        Args:
            checkpoint: Checkpoint to save

        Returns:
            The checkpoint ID

        Note: Default implementation raises NotImplementedError.
              Backends should override for workflow support.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support checkpoints. "
            "Use SQLiteStorage or PostgreSQLStorage for workflow features."
        )

    def get_checkpoint(self, checkpoint_id: str) -> Optional["Checkpoint"]:
        """
        Get a checkpoint by ID.

        Args:
            checkpoint_id: The checkpoint ID

        Returns:
            The checkpoint, or None if not found
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support checkpoints."
        )

    def get_latest_checkpoint(
        self,
        run_id: str,
        branch_id: Optional[str] = None,
    ) -> Optional["Checkpoint"]:
        """
        Get the most recent checkpoint for a workflow run.

        Args:
            run_id: The workflow run identifier
            branch_id: Optional branch to filter by

        Returns:
            The latest checkpoint, or None if no checkpoints exist
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support checkpoints."
        )

    def get_checkpoints_for_run(
        self,
        run_id: str,
        branch_id: Optional[str] = None,
        limit: int = 100,
    ) -> List["Checkpoint"]:
        """
        Get all checkpoints for a workflow run.

        Args:
            run_id: The workflow run identifier
            branch_id: Optional branch filter
            limit: Maximum checkpoints to return

        Returns:
            List of checkpoints ordered by sequence number
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support checkpoints."
        )

    def cleanup_checkpoints(
        self,
        run_id: str,
        keep_latest: int = 1,
    ) -> int:
        """
        Clean up old checkpoints for a completed run.

        Args:
            run_id: The workflow run identifier
            keep_latest: Number of latest checkpoints to keep

        Returns:
            Number of checkpoints deleted
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support checkpoints."
        )

    # ==================== WORKFLOW OUTCOME OPERATIONS (v0.6.0+) ====================

    def save_workflow_outcome(self, outcome: "WorkflowOutcome") -> str:
        """
        Save a workflow outcome.

        Args:
            outcome: WorkflowOutcome to save

        Returns:
            The outcome ID
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support workflow outcomes."
        )

    def get_workflow_outcome(self, outcome_id: str) -> Optional["WorkflowOutcome"]:
        """
        Get a workflow outcome by ID.

        Args:
            outcome_id: The outcome ID

        Returns:
            The workflow outcome, or None if not found
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support workflow outcomes."
        )

    def get_workflow_outcomes(
        self,
        project_id: str,
        agent: Optional[str] = None,
        workflow_id: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        top_k: int = 10,
        scope_filter: Optional[Dict[str, Any]] = None,
    ) -> List["WorkflowOutcome"]:
        """
        Get workflow outcomes with optional filtering.

        Args:
            project_id: Project to query
            agent: Filter by agent
            workflow_id: Filter by workflow definition
            embedding: Query embedding for semantic search
            top_k: Max results
            scope_filter: Optional workflow scope filter

        Returns:
            List of matching workflow outcomes
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support workflow outcomes."
        )

    # ==================== ARTIFACT LINK OPERATIONS (v0.6.0+) ====================

    def save_artifact_link(self, artifact_ref: "ArtifactRef") -> str:
        """
        Save an artifact reference linked to a memory.

        Args:
            artifact_ref: ArtifactRef to save

        Returns:
            The artifact reference ID
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support artifact links."
        )

    def get_artifact_links(
        self,
        memory_id: str,
    ) -> List["ArtifactRef"]:
        """
        Get all artifact references linked to a memory.

        Args:
            memory_id: The memory ID to get artifacts for

        Returns:
            List of artifact references
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support artifact links."
        )

    def delete_artifact_link(self, artifact_id: str) -> bool:
        """
        Delete an artifact reference.

        Args:
            artifact_id: The artifact reference ID

        Returns:
            True if deleted, False if not found
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support artifact links."
        )

    # ==================== SESSION HANDOFFS ====================

    def save_session_handoff(self, handoff: "SessionHandoff") -> str:
        """
        Save a session handoff for persistence across restarts.

        Args:
            handoff: SessionHandoff to save

        Returns:
            The handoff ID

        Note: Default implementation raises NotImplementedError.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support session handoffs."
        )

    def get_session_handoffs(
        self,
        project_id: str,
        agent: str,
        limit: int = 50,
    ) -> List["SessionHandoff"]:
        """
        Get session handoffs for an agent, most recent first.

        Args:
            project_id: Project identifier
            agent: Agent identifier
            limit: Maximum number of handoffs to return

        Returns:
            List of SessionHandoff, most recent first
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support session handoffs."
        )

    def get_latest_session_handoff(
        self,
        project_id: str,
        agent: str,
    ) -> Optional["SessionHandoff"]:
        """
        Get the most recent session handoff for an agent.

        Args:
            project_id: Project identifier
            agent: Agent identifier

        Returns:
            Most recent SessionHandoff or None
        """
        handoffs = self.get_session_handoffs(project_id, agent, limit=1)
        return handoffs[0] if handoffs else None

    def delete_session_handoffs(
        self,
        project_id: str,
        agent: Optional[str] = None,
    ) -> int:
        """
        Delete session handoffs.

        Args:
            project_id: Project identifier
            agent: If provided, only delete for this agent

        Returns:
            Number of handoffs deleted
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support session handoffs."
        )

    # ==================== MEMORY STRENGTH OPERATIONS (v0.7.0+) ====================

    def save_memory_strength(self, strength: Any) -> str:
        """
        Save or update a memory strength record.

        Args:
            strength: MemoryStrength instance to save

        Returns:
            The memory ID

        Note: Default implementation raises NotImplementedError.
              Backends should override for decay-based forgetting support.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support memory strength tracking. "
            "Use SQLiteStorage for decay-based forgetting features."
        )

    def get_memory_strength(self, memory_id: str) -> Optional[Any]:
        """
        Get a memory strength record by memory ID.

        Args:
            memory_id: The memory ID to look up

        Returns:
            MemoryStrength instance, or None if not found
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support memory strength tracking."
        )

    def get_all_memory_strengths(
        self,
        project_id: str,
        agent: Optional[str] = None,
    ) -> List[Any]:
        """
        Get all memory strength records for a project/agent.

        Args:
            project_id: Project to query
            agent: Optional agent filter

        Returns:
            List of MemoryStrength instances
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support memory strength tracking."
        )

    def delete_memory_strength(self, memory_id: str) -> bool:
        """
        Delete a memory strength record.

        Args:
            memory_id: The memory ID

        Returns:
            True if deleted, False if not found
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support memory strength tracking."
        )

    # ==================== ARCHIVE OPERATIONS (v0.7.0+) ====================

    def archive_memory(
        self,
        memory_id: str,
        memory_type: str,
        reason: str,
        final_strength: float,
    ) -> Any:
        """
        Archive a memory before deletion.

        Captures full memory data including content, embedding, and metadata
        for potential future recovery or compliance auditing.

        Args:
            memory_id: ID of the memory to archive
            memory_type: Type of memory (heuristic, outcome, etc.)
            reason: Why being archived (decay, manual, consolidation, etc.)
            final_strength: Memory strength at time of archival

        Returns:
            ArchivedMemory instance

        Note: Default implementation raises NotImplementedError.
              Backends should override for archive support.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support memory archiving. "
            "Use SQLiteStorage for archive features."
        )

    def get_archive(self, archive_id: str) -> Optional[Any]:
        """
        Get an archived memory by its archive ID.

        Args:
            archive_id: The archive ID

        Returns:
            ArchivedMemory instance, or None if not found
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support memory archiving."
        )

    def list_archives(
        self,
        project_id: str,
        agent: Optional[str] = None,
        reason: Optional[str] = None,
        memory_type: Optional[str] = None,
        older_than: Optional[datetime] = None,
        younger_than: Optional[datetime] = None,
        include_restored: bool = False,
        limit: int = 100,
    ) -> List[Any]:
        """
        List archived memories with filtering.

        Args:
            project_id: Project to query
            agent: Optional agent filter
            reason: Optional archive reason filter
            memory_type: Optional memory type filter
            older_than: Optional filter for archives older than this time
            younger_than: Optional filter for archives younger than this time
            include_restored: Whether to include archives that have been restored
            limit: Maximum number of archives to return

        Returns:
            List of ArchivedMemory instances
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support memory archiving."
        )

    def restore_from_archive(self, archive_id: str) -> str:
        """
        Restore an archived memory, creating a new memory from archive data.

        The original archive is marked as restored but retained for audit purposes.

        Args:
            archive_id: The archive ID to restore

        Returns:
            New memory ID of the restored memory

        Raises:
            ValueError: If archive not found or already restored
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support memory archiving."
        )

    def purge_archives(
        self,
        older_than: datetime,
        project_id: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> int:
        """
        Permanently delete archived memories.

        This is a destructive operation - archives cannot be recovered after purging.

        Args:
            older_than: Delete archives older than this datetime
            project_id: Optional project filter
            reason: Optional reason filter

        Returns:
            Number of archives permanently deleted
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support memory archiving."
        )

    def get_archive_stats(
        self,
        project_id: str,
        agent: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get statistics about archived memories.

        Args:
            project_id: Project to query
            agent: Optional agent filter

        Returns:
            Dict with archive statistics (counts, by reason, by type, etc.)
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support memory archiving."
        )

    # ==================== UTILITY ====================

    @classmethod
    @abstractmethod
    def from_config(cls, config: Dict[str, Any]) -> "StorageBackend":
        """
        Create instance from configuration dict.

        Args:
            config: Configuration dictionary

        Returns:
            Configured storage backend instance
        """
        pass
