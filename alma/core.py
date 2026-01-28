"""
ALMA Core - Main interface for the Agent Learning Memory Architecture.

API Return Type Conventions:
- Create operations: Return created object or raise exception
- Update operations: Return updated object or raise exception
- Delete operations: Return bool (success) or int (count), raise on failure
- Query operations: Return list (empty if none) or object

All scope violations raise ScopeViolationError for consistent error handling.
"""

import logging
import time
from typing import Any, Dict, Optional

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
            logger.warning(f"Agent '{agent}' has no defined scope, using defaults")
            structured_logger.warning(
                "Agent has no defined scope, using defaults",
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
