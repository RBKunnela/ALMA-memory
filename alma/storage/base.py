"""
ALMA Storage Backend Interface.

Abstract base class that all storage backends must implement.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional

from alma.types import (
    AntiPattern,
    DomainKnowledge,
    Heuristic,
    Outcome,
    UserPreference,
)


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
    ) -> List[Heuristic]:
        """
        Get heuristics, optionally filtered by agent and similarity.

        Args:
            project_id: Project to query
            agent: Filter by agent name
            embedding: Query embedding for semantic search
            top_k: Max results to return
            min_confidence: Minimum confidence threshold

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
    ) -> List[DomainKnowledge]:
        """
        Get domain knowledge.

        Args:
            project_id: Project to query
            agent: Filter by agent
            domain: Filter by domain
            embedding: Query embedding for semantic search
            top_k: Max results

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
    ) -> List[AntiPattern]:
        """
        Get anti-patterns.

        Args:
            project_id: Project to query
            agent: Filter by agent
            embedding: Query embedding for semantic search
            top_k: Max results

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

    def migrate(self, target_version: Optional[str] = None, dry_run: bool = False) -> List[str]:
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
