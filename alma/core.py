"""
ALMA Core - Main interface for the Agent Learning Memory Architecture.

API Return Type Conventions:
- Create operations: Return created object or raise exception
- Update operations: Return updated object or raise exception
- Delete operations: Return bool (success) or int (count), raise on failure
- Query operations: Return list (empty if none) or object

All scope violations raise ScopeViolationError for consistent error handling.

Async API:
ALMA provides both synchronous and asynchronous APIs. The async variants
(async_retrieve, async_learn, etc.) use asyncio.to_thread() to run
blocking storage operations in a thread pool, enabling better concurrency
in async applications without blocking the event loop.

Workflow Integration (v0.6.0):
ALMA supports AGtestari workflow integration with:
- Checkpoints: Crash recovery and state persistence
- Workflow Outcomes: Learning from completed workflows
- Artifact Links: Connecting external files to memories
- Scoped Retrieval: Filtering by workflow context
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

from alma.config.loader import ConfigLoader
from alma.exceptions import ScopeViolationError
from alma.learning.protocols import LearningProtocol
from alma.observability.logging import get_logger
from alma.observability.metrics import get_metrics
from alma.observability.tracing import SpanKind, get_tracer, trace_method
from alma.retrieval.engine import RetrievalEngine
from alma.storage.base import StorageBackend
from alma.types import (
    DomainKnowledge,
    MemoryScope,
    MemorySlice,
    Outcome,
    UserPreference,
)
from alma.workflow import (
    ArtifactRef,
    ArtifactType,
    Checkpoint,
    CheckpointManager,
    ReducerConfig,
    RetrievalScope,
    StateMerger,
    WorkflowContext,
    WorkflowOutcome,
    WorkflowResult,
)

logger = logging.getLogger(__name__)
structured_logger = get_logger(__name__)
tracer = get_tracer(__name__)


class ALMA:
    """
    Agent Learning Memory Architecture - Main Interface.

    Provides methods for:
    - Retrieving relevant memories for a task
    - Learning from task outcomes
    - Managing agent memory scopes
    """

    def __init__(
        self,
        storage: StorageBackend,
        retrieval_engine: RetrievalEngine,
        learning_protocol: LearningProtocol,
        scopes: Dict[str, MemoryScope],
        project_id: str,
    ):
        self.storage = storage
        self.retrieval = retrieval_engine
        self.learning = learning_protocol
        self.scopes = scopes
        self.project_id = project_id

    @classmethod
    def from_config(cls, config_path: str) -> "ALMA":
        """
        Initialize ALMA from a configuration file.

        Args:
            config_path: Path to .alma/config.yaml

        Returns:
            Configured ALMA instance
        """
        config = ConfigLoader.load(config_path)

        # Initialize storage backend based on config
        storage = cls._create_storage(config)

        # Initialize retrieval engine
        retrieval = RetrievalEngine(
            storage=storage,
            embedding_provider=config.get("embedding_provider", "local"),
        )

        # Initialize learning protocol
        learning = LearningProtocol(
            storage=storage,
            scopes={
                name: MemoryScope(
                    agent_name=name,
                    can_learn=scope.get("can_learn", []),
                    cannot_learn=scope.get("cannot_learn", []),
                    min_occurrences_for_heuristic=scope.get(
                        "min_occurrences_for_heuristic", 3
                    ),
                )
                for name, scope in config.get("agents", {}).items()
            },
        )

        # Build scopes dict
        scopes = {
            name: MemoryScope(
                agent_name=name,
                can_learn=scope.get("can_learn", []),
                cannot_learn=scope.get("cannot_learn", []),
                min_occurrences_for_heuristic=scope.get(
                    "min_occurrences_for_heuristic", 3
                ),
            )
            for name, scope in config.get("agents", {}).items()
        }

        return cls(
            storage=storage,
            retrieval_engine=retrieval,
            learning_protocol=learning,
            scopes=scopes,
            project_id=config.get("project_id", "default"),
        )

    @staticmethod
    def _create_storage(config: Dict[str, Any]) -> StorageBackend:
        """Create appropriate storage backend based on config."""
        storage_type = config.get("storage", "file")

        if storage_type == "azure":
            from alma.storage.azure_cosmos import AzureCosmosStorage

            return AzureCosmosStorage.from_config(config)
        elif storage_type == "postgres":
            from alma.storage.postgresql import PostgreSQLStorage

            return PostgreSQLStorage.from_config(config)
        elif storage_type == "sqlite":
            from alma.storage.sqlite_local import SQLiteStorage

            return SQLiteStorage.from_config(config)
        else:
            from alma.storage.file_based import FileBasedStorage

            return FileBasedStorage.from_config(config)

    @trace_method(name="ALMA.retrieve", kind=SpanKind.INTERNAL)
    def retrieve(
        self,
        task: str,
        agent: str,
        user_id: Optional[str] = None,
        top_k: int = 5,
    ) -> MemorySlice:
        """
        Retrieve relevant memories for a task.

        Args:
            task: Description of the task to perform
            agent: Name of the agent requesting memories
            user_id: Optional user ID for preference retrieval
            top_k: Maximum items per memory type

        Returns:
            MemorySlice with relevant memories for context injection
        """
        start_time = time.time()
        metrics = get_metrics()

        # Validate agent has a defined scope
        if agent not in self.scopes:
            structured_logger.warning(
                f"Agent '{agent}' has no defined scope, using defaults",
                agent=agent,
                project_id=self.project_id,
            )

        result = self.retrieval.retrieve(
            query=task,
            agent=agent,
            project_id=self.project_id,
            user_id=user_id,
            top_k=top_k,
            scope=self.scopes.get(agent),
        )

        # Record metrics
        duration_ms = (time.time() - start_time) * 1000
        cache_hit = result.retrieval_time_ms < 10  # Approximate cache hit detection
        metrics.record_retrieve_latency(
            duration_ms=duration_ms,
            agent=agent,
            project_id=self.project_id,
            cache_hit=cache_hit,
            items_returned=result.total_items,
        )

        structured_logger.info(
            "Memory retrieval completed",
            agent=agent,
            project_id=self.project_id,
            task_preview=task[:50] if task else "",
            items_returned=result.total_items,
            duration_ms=duration_ms,
            cache_hit=cache_hit,
        )

        return result

    @trace_method(name="ALMA.learn", kind=SpanKind.INTERNAL)
    def learn(
        self,
        agent: str,
        task: str,
        outcome: str,  # "success" or "failure"
        strategy_used: str,
        task_type: Optional[str] = None,
        duration_ms: Optional[int] = None,
        error_message: Optional[str] = None,
        feedback: Optional[str] = None,
    ) -> Outcome:
        """
        Learn from a task outcome.

        Validates that learning is within agent's scope before committing.
        Invalidates cache after learning to ensure fresh retrieval results.

        Args:
            agent: Name of the agent that executed the task
            task: Description of the task
            outcome: "success" or "failure"
            strategy_used: What approach was taken
            task_type: Category of task (for grouping)
            duration_ms: How long the task took
            error_message: Error details if failed
            feedback: User feedback if provided

        Returns:
            The created Outcome record

        Raises:
            ScopeViolationError: If learning is outside agent's scope
        """
        start_time = time.time()
        metrics = get_metrics()

        outcome_record = self.learning.learn(
            agent=agent,
            project_id=self.project_id,
            task=task,
            outcome=outcome == "success",
            strategy_used=strategy_used,
            task_type=task_type,
            duration_ms=duration_ms,
            error_message=error_message,
            feedback=feedback,
        )

        # Invalidate cache for this agent/project after learning
        self.retrieval.invalidate_cache(agent=agent, project_id=self.project_id)

        # Record metrics
        learn_duration_ms = (time.time() - start_time) * 1000
        metrics.record_learn_operation(
            duration_ms=learn_duration_ms,
            agent=agent,
            project_id=self.project_id,
            memory_type="outcome",
            success=True,
        )

        structured_logger.info(
            "Learning operation completed",
            agent=agent,
            project_id=self.project_id,
            task_type=task_type,
            outcome=outcome,
            duration_ms=learn_duration_ms,
        )

        return outcome_record

    def add_user_preference(
        self,
        user_id: str,
        category: str,
        preference: str,
        source: str = "explicit_instruction",
    ) -> UserPreference:
        """
        Add a user preference to memory.

        Args:
            user_id: User identifier
            category: Category (communication, code_style, workflow)
            preference: The preference text
            source: How this was learned

        Returns:
            The created UserPreference
        """
        result = self.learning.add_preference(
            user_id=user_id,
            category=category,
            preference=preference,
            source=source,
        )

        # Invalidate cache for project (user preferences affect all agents)
        self.retrieval.invalidate_cache(project_id=self.project_id)

        return result

    def add_domain_knowledge(
        self,
        agent: str,
        domain: str,
        fact: str,
        source: str = "user_stated",
    ) -> DomainKnowledge:
        """
        Add domain knowledge within agent's scope.

        Args:
            agent: Agent this knowledge belongs to
            domain: Knowledge domain
            fact: The fact to remember
            source: How this was learned

        Returns:
            The created DomainKnowledge

        Raises:
            ScopeViolationError: If agent is not allowed to learn in this domain
        """
        # Check scope
        scope = self.scopes.get(agent)
        if scope and not scope.is_allowed(domain):
            raise ScopeViolationError(
                f"Agent '{agent}' is not allowed to learn in domain '{domain}'"
            )

        result = self.learning.add_domain_knowledge(
            agent=agent,
            project_id=self.project_id,
            domain=domain,
            fact=fact,
            source=source,
        )

        # Invalidate cache for this agent/project after adding knowledge
        self.retrieval.invalidate_cache(agent=agent, project_id=self.project_id)

        return result

    @trace_method(name="ALMA.forget", kind=SpanKind.INTERNAL)
    def forget(
        self,
        agent: Optional[str] = None,
        older_than_days: int = 90,
        below_confidence: float = 0.3,
    ) -> int:
        """
        Prune stale or low-confidence memories.

        This is a delete operation that invalidates cache after pruning
        to ensure fresh retrieval results.

        Args:
            agent: Specific agent to prune, or None for all
            older_than_days: Remove outcomes older than this
            below_confidence: Remove heuristics below this confidence

        Returns:
            Number of items pruned (0 if nothing was pruned)

        Raises:
            StorageError: If the delete operation fails
        """
        start_time = time.time()
        metrics = get_metrics()

        count = self.learning.forget(
            project_id=self.project_id,
            agent=agent,
            older_than_days=older_than_days,
            below_confidence=below_confidence,
        )

        # Invalidate cache after forgetting (memories were removed)
        if count > 0:
            self.retrieval.invalidate_cache(agent=agent, project_id=self.project_id)

        # Record metrics
        duration_ms = (time.time() - start_time) * 1000
        metrics.record_forget_operation(
            duration_ms=duration_ms,
            agent=agent,
            project_id=self.project_id,
            items_removed=count,
        )

        structured_logger.info(
            "Forget operation completed",
            agent=agent or "all",
            project_id=self.project_id,
            items_removed=count,
            older_than_days=older_than_days,
            below_confidence=below_confidence,
            duration_ms=duration_ms,
        )

        return count

    def get_stats(self, agent: Optional[str] = None) -> Dict[str, Any]:
        """
        Get memory statistics.

        This is a query operation that returns statistics about stored memories.

        Args:
            agent: Specific agent or None for all

        Returns:
            Dict with counts and metadata (always returns a dict, may be empty)

        Raises:
            StorageError: If the query operation fails
        """
        return self.storage.get_stats(
            project_id=self.project_id,
            agent=agent,
        )

    # ==================== WORKFLOW INTEGRATION (v0.6.0) ====================
    #
    # Methods for AGtestari workflow integration: checkpointing, scoped
    # retrieval, learning from workflows, and artifact linking.

    def _get_checkpoint_manager(self) -> CheckpointManager:
        """Get or create the checkpoint manager."""
        if not hasattr(self, "_checkpoint_manager"):
            self._checkpoint_manager = CheckpointManager(storage=self.storage)
        return self._checkpoint_manager

    @trace_method(name="ALMA.checkpoint", kind=SpanKind.INTERNAL)
    def checkpoint(
        self,
        run_id: str,
        node_id: str,
        state: Dict[str, Any],
        branch_id: Optional[str] = None,
        parent_checkpoint_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        skip_if_unchanged: bool = True,
    ) -> Optional[Checkpoint]:
        """
        Create a checkpoint for crash recovery.

        Checkpoints persist workflow state at key points during execution,
        enabling recovery after crashes or failures.

        Args:
            run_id: The workflow run identifier.
            node_id: The node creating this checkpoint.
            state: The state to persist.
            branch_id: Optional branch identifier for parallel execution.
            parent_checkpoint_id: Previous checkpoint in the chain.
            metadata: Additional checkpoint metadata.
            skip_if_unchanged: If True, skip creating checkpoint if state
                              hasn't changed from the last checkpoint.

        Returns:
            The created Checkpoint, or None if skipped due to no changes.

        Raises:
            ValueError: If state exceeds max_state_size (1MB by default).
        """
        manager = self._get_checkpoint_manager()
        checkpoint = manager.create_checkpoint(
            run_id=run_id,
            node_id=node_id,
            state=state,
            branch_id=branch_id,
            parent_checkpoint_id=parent_checkpoint_id,
            metadata=metadata,
            skip_if_unchanged=skip_if_unchanged,
        )

        if checkpoint:
            structured_logger.info(
                "Checkpoint created",
                run_id=run_id,
                node_id=node_id,
                checkpoint_id=checkpoint.id,
                sequence_number=checkpoint.sequence_number,
            )

        return checkpoint

    def get_resume_point(
        self,
        run_id: str,
        branch_id: Optional[str] = None,
    ) -> Optional[Checkpoint]:
        """
        Get the checkpoint to resume from after a crash.

        Args:
            run_id: The workflow run identifier.
            branch_id: Optional branch to filter by.

        Returns:
            The checkpoint to resume from, or None if no checkpoints.
        """
        manager = self._get_checkpoint_manager()
        return manager.get_latest_checkpoint(run_id, branch_id)

    def merge_states(
        self,
        states: List[Dict[str, Any]],
        reducer_config: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Merge multiple branch states after parallel execution.

        Uses configurable reducers to handle each key in the state.
        Default reducer is 'last_value' which takes the value from
        the last state.

        Args:
            states: List of state dicts from parallel branches.
            reducer_config: Optional mapping of key -> reducer name.
                           Available reducers: append, merge_dict,
                           last_value, first_value, sum, max, min, union.

        Returns:
            Merged state dictionary.

        Example:
            >>> states = [
            ...     {"count": 5, "items": ["a"]},
            ...     {"count": 3, "items": ["b", "c"]},
            ... ]
            >>> alma.merge_states(states, {"count": "sum", "items": "append"})
            {"count": 8, "items": ["a", "b", "c"]}
        """
        config = ReducerConfig(field_reducers=reducer_config or {})
        merger = StateMerger(config)
        return merger.merge(states)

    @trace_method(name="ALMA.learn_from_workflow", kind=SpanKind.INTERNAL)
    def learn_from_workflow(
        self,
        agent: str,
        workflow_id: str,
        run_id: str,
        result: str,
        summary: str,
        strategies_used: Optional[List[str]] = None,
        successful_patterns: Optional[List[str]] = None,
        failed_patterns: Optional[List[str]] = None,
        duration_seconds: Optional[float] = None,
        node_count: Optional[int] = None,
        error_message: Optional[str] = None,
        tenant_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> WorkflowOutcome:
        """
        Record learnings from a completed workflow execution.

        Captures what was learned from running a workflow, including
        strategies used, what worked, what didn't, and error details.

        Args:
            agent: The agent that executed the workflow.
            workflow_id: The workflow definition that was executed.
            run_id: The specific run this outcome is from.
            result: Result status ("success", "failure", "partial",
                   "cancelled", "timeout").
            summary: Human-readable summary of what happened.
            strategies_used: List of strategies/approaches attempted.
            successful_patterns: Patterns that worked well.
            failed_patterns: Patterns that didn't work.
            duration_seconds: How long the workflow took.
            node_count: Number of nodes executed.
            error_message: Error details if failed.
            tenant_id: Multi-tenant isolation identifier.
            metadata: Additional outcome metadata.

        Returns:
            The created WorkflowOutcome.
        """
        start_time = time.time()
        metrics = get_metrics()

        # Create the outcome
        outcome = WorkflowOutcome(
            tenant_id=tenant_id,
            workflow_id=workflow_id,
            run_id=run_id,
            agent=agent,
            project_id=self.project_id,
            result=WorkflowResult(result),
            summary=summary,
            strategies_used=strategies_used or [],
            successful_patterns=successful_patterns or [],
            failed_patterns=failed_patterns or [],
            duration_seconds=duration_seconds,
            node_count=node_count,
            error_message=error_message,
            metadata=metadata or {},
        )

        # Validate
        outcome.validate()

        # Save to storage
        self.storage.save_workflow_outcome(outcome)

        # Invalidate cache
        self.retrieval.invalidate_cache(agent=agent, project_id=self.project_id)

        # Record metrics
        learn_duration_ms = (time.time() - start_time) * 1000
        metrics.record_learn_operation(
            duration_ms=learn_duration_ms,
            agent=agent,
            project_id=self.project_id,
            memory_type="workflow_outcome",
            success=True,
        )

        structured_logger.info(
            "Workflow outcome recorded",
            agent=agent,
            workflow_id=workflow_id,
            run_id=run_id,
            result=result,
            duration_ms=learn_duration_ms,
        )

        return outcome

    def link_artifact(
        self,
        memory_id: str,
        artifact_type: str,
        storage_url: str,
        filename: Optional[str] = None,
        mime_type: Optional[str] = None,
        size_bytes: Optional[int] = None,
        checksum: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ArtifactRef:
        """
        Link an external artifact to a memory.

        Artifacts are stored externally (e.g., Cloudflare R2, S3) and
        referenced by URL. This allows memories to reference large files
        without bloating the memory database.

        Args:
            memory_id: The memory to link the artifact to.
            artifact_type: Type of artifact ("screenshot", "log", "report",
                          "file", "document", "image", etc.).
            storage_url: URL or path to the artifact in storage.
            filename: Original filename.
            mime_type: MIME type.
            size_bytes: Size in bytes.
            checksum: SHA256 checksum for integrity verification.
            metadata: Additional artifact metadata.

        Returns:
            The created ArtifactRef.
        """
        # Convert string to enum
        try:
            artifact_type_enum = ArtifactType(artifact_type)
        except ValueError:
            artifact_type_enum = ArtifactType.OTHER

        artifact = ArtifactRef(
            memory_id=memory_id,
            artifact_type=artifact_type_enum,
            storage_url=storage_url,
            filename=filename,
            mime_type=mime_type,
            size_bytes=size_bytes,
            checksum=checksum,
            metadata=metadata or {},
        )

        # Validate
        artifact.validate()

        # Save to storage
        self.storage.save_artifact_link(artifact)

        structured_logger.info(
            "Artifact linked",
            memory_id=memory_id,
            artifact_id=artifact.id,
            artifact_type=artifact_type,
            storage_url=storage_url[:50] if storage_url else None,
        )

        return artifact

    def get_artifacts(self, memory_id: str) -> List[ArtifactRef]:
        """
        Get all artifacts linked to a memory.

        Args:
            memory_id: The memory to get artifacts for.

        Returns:
            List of ArtifactRef objects.
        """
        return self.storage.get_artifact_links(memory_id)

    def cleanup_checkpoints(
        self,
        run_id: str,
        keep_latest: int = 1,
    ) -> int:
        """
        Clean up old checkpoints for a completed run.

        Call this after a workflow completes to free up storage.

        Args:
            run_id: The workflow run identifier.
            keep_latest: Number of latest checkpoints to keep.

        Returns:
            Number of checkpoints deleted.
        """
        manager = self._get_checkpoint_manager()
        count = manager.cleanup_checkpoints(run_id, keep_latest)

        if count > 0:
            structured_logger.info(
                "Checkpoints cleaned up",
                run_id=run_id,
                deleted_count=count,
                kept=keep_latest,
            )

        return count

    def retrieve_with_scope(
        self,
        task: str,
        agent: str,
        context: WorkflowContext,
        scope: RetrievalScope = RetrievalScope.AGENT,
        user_id: Optional[str] = None,
        top_k: int = 5,
    ) -> MemorySlice:
        """
        Retrieve memories with workflow scope filtering.

        This is an enhanced version of retrieve() that supports
        filtering by workflow context and scope level.

        Args:
            task: Description of the task to perform.
            agent: Name of the agent requesting memories.
            context: Workflow context for scoping.
            scope: How broadly to search for memories.
            user_id: Optional user ID for preference retrieval.
            top_k: Maximum items per memory type.

        Returns:
            MemorySlice with relevant memories for context injection.
        """
        start_time = time.time()
        metrics = get_metrics()

        # Build scope filter from context
        scope_filter = context.get_scope_filter(scope)

        # For now, scope_filter is passed to the retrieval as metadata
        # Future: pass to storage.get_* methods for proper filtering
        result = self.retrieval.retrieve(
            query=task,
            agent=agent,
            project_id=self.project_id,
            user_id=user_id,
            top_k=top_k,
            scope=self.scopes.get(agent),
        )

        # Add scope context to result metadata
        result.metadata = {
            "scope": scope.value,
            "scope_filter": scope_filter,
            "context": context.to_dict(),
        }

        # Record metrics
        duration_ms = (time.time() - start_time) * 1000
        cache_hit = result.retrieval_time_ms < 10
        metrics.record_retrieve_latency(
            duration_ms=duration_ms,
            agent=agent,
            project_id=self.project_id,
            cache_hit=cache_hit,
            items_returned=result.total_items,
        )

        structured_logger.info(
            "Scoped memory retrieval completed",
            agent=agent,
            project_id=self.project_id,
            scope=scope.value,
            workflow_id=context.workflow_id,
            run_id=context.run_id,
            items_returned=result.total_items,
            duration_ms=duration_ms,
        )

        return result

    # ==================== ASYNC API ====================
    #
    # Async variants of core methods for better concurrency support.
    # These use asyncio.to_thread() to run blocking operations in a
    # thread pool, preventing event loop blocking in async applications.

    async def async_retrieve(
        self,
        task: str,
        agent: str,
        user_id: Optional[str] = None,
        top_k: int = 5,
    ) -> MemorySlice:
        """
        Async version of retrieve(). Retrieve relevant memories for a task.

        Runs the blocking storage operations in a thread pool to avoid
        blocking the event loop.

        Args:
            task: Description of the task to perform
            agent: Name of the agent requesting memories
            user_id: Optional user ID for preference retrieval
            top_k: Maximum items per memory type

        Returns:
            MemorySlice with relevant memories for context injection
        """
        return await asyncio.to_thread(
            self.retrieve,
            task=task,
            agent=agent,
            user_id=user_id,
            top_k=top_k,
        )

    async def async_learn(
        self,
        agent: str,
        task: str,
        outcome: str,
        strategy_used: str,
        task_type: Optional[str] = None,
        duration_ms: Optional[int] = None,
        error_message: Optional[str] = None,
        feedback: Optional[str] = None,
    ) -> Outcome:
        """
        Async version of learn(). Learn from a task outcome.

        Validates that learning is within agent's scope before committing.
        Invalidates cache after learning to ensure fresh retrieval results.

        Args:
            agent: Name of the agent that executed the task
            task: Description of the task
            outcome: "success" or "failure"
            strategy_used: What approach was taken
            task_type: Category of task (for grouping)
            duration_ms: How long the task took
            error_message: Error details if failed
            feedback: User feedback if provided

        Returns:
            The created Outcome record

        Raises:
            ScopeViolationError: If learning is outside agent's scope
        """
        return await asyncio.to_thread(
            self.learn,
            agent=agent,
            task=task,
            outcome=outcome,
            strategy_used=strategy_used,
            task_type=task_type,
            duration_ms=duration_ms,
            error_message=error_message,
            feedback=feedback,
        )

    async def async_add_user_preference(
        self,
        user_id: str,
        category: str,
        preference: str,
        source: str = "explicit_instruction",
    ) -> UserPreference:
        """
        Async version of add_user_preference(). Add a user preference to memory.

        Args:
            user_id: User identifier
            category: Category (communication, code_style, workflow)
            preference: The preference text
            source: How this was learned

        Returns:
            The created UserPreference
        """
        return await asyncio.to_thread(
            self.add_user_preference,
            user_id=user_id,
            category=category,
            preference=preference,
            source=source,
        )

    async def async_add_domain_knowledge(
        self,
        agent: str,
        domain: str,
        fact: str,
        source: str = "user_stated",
    ) -> DomainKnowledge:
        """
        Async version of add_domain_knowledge(). Add domain knowledge within agent's scope.

        Args:
            agent: Agent this knowledge belongs to
            domain: Knowledge domain
            fact: The fact to remember
            source: How this was learned

        Returns:
            The created DomainKnowledge

        Raises:
            ScopeViolationError: If agent is not allowed to learn in this domain
        """
        return await asyncio.to_thread(
            self.add_domain_knowledge,
            agent=agent,
            domain=domain,
            fact=fact,
            source=source,
        )

    async def async_forget(
        self,
        agent: Optional[str] = None,
        older_than_days: int = 90,
        below_confidence: float = 0.3,
    ) -> int:
        """
        Async version of forget(). Prune stale or low-confidence memories.

        This is a delete operation that invalidates cache after pruning
        to ensure fresh retrieval results.

        Args:
            agent: Specific agent to prune, or None for all
            older_than_days: Remove outcomes older than this
            below_confidence: Remove heuristics below this confidence

        Returns:
            Number of items pruned (0 if nothing was pruned)

        Raises:
            StorageError: If the delete operation fails
        """
        return await asyncio.to_thread(
            self.forget,
            agent=agent,
            older_than_days=older_than_days,
            below_confidence=below_confidence,
        )

    async def async_get_stats(self, agent: Optional[str] = None) -> Dict[str, Any]:
        """
        Async version of get_stats(). Get memory statistics.

        Args:
            agent: Specific agent or None for all

        Returns:
            Dict with counts and metadata (always returns a dict, may be empty)

        Raises:
            StorageError: If the query operation fails
        """
        return await asyncio.to_thread(
            self.get_stats,
            agent=agent,
        )

    # ==================== ASYNC WORKFLOW API ====================

    async def async_checkpoint(
        self,
        run_id: str,
        node_id: str,
        state: Dict[str, Any],
        branch_id: Optional[str] = None,
        parent_checkpoint_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        skip_if_unchanged: bool = True,
    ) -> Optional[Checkpoint]:
        """Async version of checkpoint()."""
        return await asyncio.to_thread(
            self.checkpoint,
            run_id=run_id,
            node_id=node_id,
            state=state,
            branch_id=branch_id,
            parent_checkpoint_id=parent_checkpoint_id,
            metadata=metadata,
            skip_if_unchanged=skip_if_unchanged,
        )

    async def async_get_resume_point(
        self,
        run_id: str,
        branch_id: Optional[str] = None,
    ) -> Optional[Checkpoint]:
        """Async version of get_resume_point()."""
        return await asyncio.to_thread(
            self.get_resume_point,
            run_id=run_id,
            branch_id=branch_id,
        )

    async def async_learn_from_workflow(
        self,
        agent: str,
        workflow_id: str,
        run_id: str,
        result: str,
        summary: str,
        strategies_used: Optional[List[str]] = None,
        successful_patterns: Optional[List[str]] = None,
        failed_patterns: Optional[List[str]] = None,
        duration_seconds: Optional[float] = None,
        node_count: Optional[int] = None,
        error_message: Optional[str] = None,
        tenant_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> WorkflowOutcome:
        """Async version of learn_from_workflow()."""
        return await asyncio.to_thread(
            self.learn_from_workflow,
            agent=agent,
            workflow_id=workflow_id,
            run_id=run_id,
            result=result,
            summary=summary,
            strategies_used=strategies_used,
            successful_patterns=successful_patterns,
            failed_patterns=failed_patterns,
            duration_seconds=duration_seconds,
            node_count=node_count,
            error_message=error_message,
            tenant_id=tenant_id,
            metadata=metadata,
        )

    async def async_link_artifact(
        self,
        memory_id: str,
        artifact_type: str,
        storage_url: str,
        filename: Optional[str] = None,
        mime_type: Optional[str] = None,
        size_bytes: Optional[int] = None,
        checksum: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ArtifactRef:
        """Async version of link_artifact()."""
        return await asyncio.to_thread(
            self.link_artifact,
            memory_id=memory_id,
            artifact_type=artifact_type,
            storage_url=storage_url,
            filename=filename,
            mime_type=mime_type,
            size_bytes=size_bytes,
            checksum=checksum,
            metadata=metadata,
        )

    async def async_retrieve_with_scope(
        self,
        task: str,
        agent: str,
        context: WorkflowContext,
        scope: RetrievalScope = RetrievalScope.AGENT,
        user_id: Optional[str] = None,
        top_k: int = 5,
    ) -> MemorySlice:
        """Async version of retrieve_with_scope()."""
        return await asyncio.to_thread(
            self.retrieve_with_scope,
            task=task,
            agent=agent,
            context=context,
            scope=scope,
            user_id=user_id,
            top_k=top_k,
        )
