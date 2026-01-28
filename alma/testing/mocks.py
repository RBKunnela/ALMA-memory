"""
ALMA Testing Mocks.

Provides mock implementations of ALMA interfaces for testing.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from alma.retrieval.embeddings import MockEmbedder
from alma.storage.base import StorageBackend
from alma.types import (
    AntiPattern,
    DomainKnowledge,
    Heuristic,
    Outcome,
    UserPreference,
)

# Re-export MockEmbedder for convenience
__all__ = ["MockStorage", "MockEmbedder"]


class MockStorage(StorageBackend):
    """
    In-memory mock storage backend for testing.

    Stores all memory types in dictionaries for fast, isolated testing.
    Does not persist data between test runs.

    Example:
        >>> from alma.testing import MockStorage, create_test_heuristic
        >>> storage = MockStorage()
        >>> heuristic = create_test_heuristic(agent="test-agent")
        >>> storage.save_heuristic(heuristic)
        >>> found = storage.get_heuristics("test-project", agent="test-agent")
        >>> assert len(found) == 1
    """

    def __init__(self):
        """Initialize empty in-memory storage."""
        self._heuristics: Dict[str, Heuristic] = {}
        self._outcomes: Dict[str, Outcome] = {}
        self._preferences: Dict[str, UserPreference] = {}
        self._domain_knowledge: Dict[str, DomainKnowledge] = {}
        self._anti_patterns: Dict[str, AntiPattern] = {}

    # ==================== WRITE OPERATIONS ====================

    def save_heuristic(self, heuristic: Heuristic) -> str:
        """Save a heuristic, return its ID."""
        self._heuristics[heuristic.id] = heuristic
        return heuristic.id

    def save_outcome(self, outcome: Outcome) -> str:
        """Save an outcome, return its ID."""
        self._outcomes[outcome.id] = outcome
        return outcome.id

    def save_user_preference(self, preference: UserPreference) -> str:
        """Save a user preference, return its ID."""
        self._preferences[preference.id] = preference
        return preference.id

    def save_domain_knowledge(self, knowledge: DomainKnowledge) -> str:
        """Save domain knowledge, return its ID."""
        self._domain_knowledge[knowledge.id] = knowledge
        return knowledge.id

    def save_anti_pattern(self, anti_pattern: AntiPattern) -> str:
        """Save an anti-pattern, return its ID."""
        self._anti_patterns[anti_pattern.id] = anti_pattern
        return anti_pattern.id

    # ==================== READ OPERATIONS ====================

    def get_heuristics(
        self,
        project_id: str,
        agent: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        top_k: int = 5,
        min_confidence: float = 0.0,
    ) -> List[Heuristic]:
        """Get heuristics, optionally filtered by agent."""
        results = []
        for h in self._heuristics.values():
            if h.project_id != project_id:
                continue
            if agent and h.agent != agent:
                continue
            if h.confidence < min_confidence:
                continue
            results.append(h)

        # Sort by confidence descending
        results.sort(key=lambda x: x.confidence, reverse=True)
        return results[:top_k]

    def get_outcomes(
        self,
        project_id: str,
        agent: Optional[str] = None,
        task_type: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        top_k: int = 5,
        success_only: bool = False,
    ) -> List[Outcome]:
        """Get outcomes, optionally filtered."""
        results = []
        for o in self._outcomes.values():
            if o.project_id != project_id:
                continue
            if agent and o.agent != agent:
                continue
            if task_type and o.task_type != task_type:
                continue
            if success_only and not o.success:
                continue
            results.append(o)

        # Sort by timestamp descending (most recent first)
        results.sort(key=lambda x: x.timestamp, reverse=True)
        return results[:top_k]

    def get_user_preferences(
        self,
        user_id: str,
        category: Optional[str] = None,
    ) -> List[UserPreference]:
        """Get user preferences."""
        results = []
        for p in self._preferences.values():
            if p.user_id != user_id:
                continue
            if category and p.category != category:
                continue
            results.append(p)
        return results

    def get_domain_knowledge(
        self,
        project_id: str,
        agent: Optional[str] = None,
        domain: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        top_k: int = 5,
    ) -> List[DomainKnowledge]:
        """Get domain knowledge."""
        results = []
        for dk in self._domain_knowledge.values():
            if dk.project_id != project_id:
                continue
            if agent and dk.agent != agent:
                continue
            if domain and dk.domain != domain:
                continue
            results.append(dk)

        # Sort by confidence descending
        results.sort(key=lambda x: x.confidence, reverse=True)
        return results[:top_k]

    def get_anti_patterns(
        self,
        project_id: str,
        agent: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        top_k: int = 5,
    ) -> List[AntiPattern]:
        """Get anti-patterns."""
        results = []
        for ap in self._anti_patterns.values():
            if ap.project_id != project_id:
                continue
            if agent and ap.agent != agent:
                continue
            results.append(ap)

        # Sort by occurrence_count descending
        results.sort(key=lambda x: x.occurrence_count, reverse=True)
        return results[:top_k]

    # ==================== UPDATE OPERATIONS ====================

    def update_heuristic(
        self,
        heuristic_id: str,
        updates: Dict[str, Any],
    ) -> bool:
        """Update a heuristic's fields."""
        if heuristic_id not in self._heuristics:
            return False

        h = self._heuristics[heuristic_id]
        for key, value in updates.items():
            if hasattr(h, key):
                setattr(h, key, value)
        return True

    def increment_heuristic_occurrence(
        self,
        heuristic_id: str,
        success: bool,
    ) -> bool:
        """Increment heuristic occurrence count."""
        if heuristic_id not in self._heuristics:
            return False

        h = self._heuristics[heuristic_id]
        h.occurrence_count += 1
        if success:
            h.success_count += 1
        h.last_validated = datetime.now(timezone.utc)
        return True

    def update_heuristic_confidence(
        self,
        heuristic_id: str,
        new_confidence: float,
    ) -> bool:
        """Update a heuristic's confidence value."""
        if heuristic_id not in self._heuristics:
            return False

        self._heuristics[heuristic_id].confidence = new_confidence
        return True

    def update_knowledge_confidence(
        self,
        knowledge_id: str,
        new_confidence: float,
    ) -> bool:
        """Update domain knowledge confidence value."""
        if knowledge_id not in self._domain_knowledge:
            return False

        self._domain_knowledge[knowledge_id].confidence = new_confidence
        return True

    # ==================== DELETE OPERATIONS ====================

    def delete_heuristic(self, heuristic_id: str) -> bool:
        """Delete a heuristic by ID."""
        if heuristic_id not in self._heuristics:
            return False
        del self._heuristics[heuristic_id]
        return True

    def delete_outcome(self, outcome_id: str) -> bool:
        """Delete an outcome by ID."""
        if outcome_id not in self._outcomes:
            return False
        del self._outcomes[outcome_id]
        return True

    def delete_domain_knowledge(self, knowledge_id: str) -> bool:
        """Delete domain knowledge by ID."""
        if knowledge_id not in self._domain_knowledge:
            return False
        del self._domain_knowledge[knowledge_id]
        return True

    def delete_anti_pattern(self, anti_pattern_id: str) -> bool:
        """Delete an anti-pattern by ID."""
        if anti_pattern_id not in self._anti_patterns:
            return False
        del self._anti_patterns[anti_pattern_id]
        return True

    def delete_outcomes_older_than(
        self,
        project_id: str,
        older_than: datetime,
        agent: Optional[str] = None,
    ) -> int:
        """Delete old outcomes."""
        to_delete = []
        for oid, o in self._outcomes.items():
            if o.project_id != project_id:
                continue
            if agent and o.agent != agent:
                continue
            if o.timestamp < older_than:
                to_delete.append(oid)

        for oid in to_delete:
            del self._outcomes[oid]
        return len(to_delete)

    def delete_low_confidence_heuristics(
        self,
        project_id: str,
        below_confidence: float,
        agent: Optional[str] = None,
    ) -> int:
        """Delete low-confidence heuristics."""
        to_delete = []
        for hid, h in self._heuristics.items():
            if h.project_id != project_id:
                continue
            if agent and h.agent != agent:
                continue
            if h.confidence < below_confidence:
                to_delete.append(hid)

        for hid in to_delete:
            del self._heuristics[hid]
        return len(to_delete)

    # ==================== STATS ====================

    def get_stats(
        self,
        project_id: str,
        agent: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get memory statistics."""
        stats = {
            "heuristics": 0,
            "outcomes": 0,
            "preferences": 0,
            "domain_knowledge": 0,
            "anti_patterns": 0,
        }

        for h in self._heuristics.values():
            if h.project_id == project_id and (not agent or h.agent == agent):
                stats["heuristics"] += 1

        for o in self._outcomes.values():
            if o.project_id == project_id and (not agent or o.agent == agent):
                stats["outcomes"] += 1

        for dk in self._domain_knowledge.values():
            if dk.project_id == project_id and (not agent or dk.agent == agent):
                stats["domain_knowledge"] += 1

        for ap in self._anti_patterns.values():
            if ap.project_id == project_id and (not agent or ap.agent == agent):
                stats["anti_patterns"] += 1

        # Note: preferences are user-scoped, not project-scoped
        stats["preferences"] = len(self._preferences)

        stats["total_count"] = sum(stats.values())
        return stats

    # ==================== UTILITY ====================

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "MockStorage":
        """Create instance from configuration dict (ignores config for mock)."""
        return cls()

    # ==================== MOCK-SPECIFIC METHODS ====================

    def clear(self) -> None:
        """Clear all stored data. Useful for test cleanup."""
        self._heuristics.clear()
        self._outcomes.clear()
        self._preferences.clear()
        self._domain_knowledge.clear()
        self._anti_patterns.clear()

    @property
    def heuristic_count(self) -> int:
        """Get total number of stored heuristics."""
        return len(self._heuristics)

    @property
    def outcome_count(self) -> int:
        """Get total number of stored outcomes."""
        return len(self._outcomes)

    @property
    def preference_count(self) -> int:
        """Get total number of stored preferences."""
        return len(self._preferences)

    @property
    def knowledge_count(self) -> int:
        """Get total number of stored domain knowledge items."""
        return len(self._domain_knowledge)

    @property
    def anti_pattern_count(self) -> int:
        """Get total number of stored anti-patterns."""
        return len(self._anti_patterns)
