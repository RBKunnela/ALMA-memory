"""
ALMA Storage Backend Interface.

Abstract base class that all storage backends must implement.
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
from datetime import datetime

from alma.types import (
    Heuristic,
    Outcome,
    UserPreference,
    DomainKnowledge,
    AntiPattern,
    MemoryType,
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

    # ==================== DELETE OPERATIONS ====================

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
