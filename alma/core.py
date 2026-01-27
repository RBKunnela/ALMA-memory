"""
ALMA Core - Main interface for the Agent Learning Memory Architecture.
"""

import logging
from typing import Any, Dict, Optional

from alma.config.loader import ConfigLoader
from alma.learning.protocols import LearningProtocol
from alma.retrieval.engine import RetrievalEngine
from alma.storage.base import StorageBackend
from alma.types import (
    DomainKnowledge,
    MemoryScope,
    MemorySlice,
    UserPreference,
)

logger = logging.getLogger(__name__)


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
        # Validate agent has a defined scope
        if agent not in self.scopes:
            logger.warning(f"Agent '{agent}' has no defined scope, using defaults")

        return self.retrieval.retrieve(
            query=task,
            agent=agent,
            project_id=self.project_id,
            user_id=user_id,
            top_k=top_k,
            scope=self.scopes.get(agent),
        )

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
    ) -> bool:
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
            True if learning was accepted, False if rejected (scope violation)
        """
        result = self.learning.learn(
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
        if result:
            self.retrieval.invalidate_cache(agent=agent, project_id=self.project_id)

        return result

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
    ) -> Optional[DomainKnowledge]:
        """
        Add domain knowledge within agent's scope.

        Args:
            agent: Agent this knowledge belongs to
            domain: Knowledge domain
            fact: The fact to remember
            source: How this was learned

        Returns:
            The created DomainKnowledge or None if scope violation
        """
        # Check scope
        scope = self.scopes.get(agent)
        if scope and not scope.is_allowed(domain):
            logger.warning(f"Agent '{agent}' not allowed to learn in domain '{domain}'")
            return None

        result = self.learning.add_domain_knowledge(
            agent=agent,
            project_id=self.project_id,
            domain=domain,
            fact=fact,
            source=source,
        )

        # Invalidate cache for this agent/project after adding knowledge
        if result:
            self.retrieval.invalidate_cache(agent=agent, project_id=self.project_id)

        return result

    def forget(
        self,
        agent: Optional[str] = None,
        older_than_days: int = 90,
        below_confidence: float = 0.3,
    ) -> int:
        """
        Prune stale or low-confidence memories.

        Invalidates cache after pruning to ensure fresh retrieval results.

        Args:
            agent: Specific agent to prune, or None for all
            older_than_days: Remove outcomes older than this
            below_confidence: Remove heuristics below this confidence

        Returns:
            Number of items pruned
        """
        count = self.learning.forget(
            project_id=self.project_id,
            agent=agent,
            older_than_days=older_than_days,
            below_confidence=below_confidence,
        )

        # Invalidate cache after forgetting (memories were removed)
        if count > 0:
            self.retrieval.invalidate_cache(agent=agent, project_id=self.project_id)

        return count

    def get_stats(self, agent: Optional[str] = None) -> Dict[str, Any]:
        """
        Get memory statistics.

        Args:
            agent: Specific agent or None for all

        Returns:
            Dict with counts and metadata
        """
        return self.storage.get_stats(
            project_id=self.project_id,
            agent=agent,
        )
