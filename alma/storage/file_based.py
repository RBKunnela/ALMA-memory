"""
ALMA File-Based Storage Backend.

Simple JSON file storage for testing and fallback scenarios.
No vector search - uses basic text matching for retrieval.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from alma.storage.base import StorageBackend
from alma.types import (
    AntiPattern,
    DomainKnowledge,
    Heuristic,
    Outcome,
    UserPreference,
)

logger = logging.getLogger(__name__)


class FileBasedStorage(StorageBackend):
    """
    File-based storage using JSON files.

    Structure:
        .alma/
        ├── heuristics.json
        ├── outcomes.json
        ├── preferences.json
        ├── domain_knowledge.json
        └── anti_patterns.json

    Note: This backend does NOT support vector search.
    Use SQLiteStorage or AzureCosmosStorage for semantic retrieval.
    """

    def __init__(self, storage_dir: Path):
        """
        Initialize file-based storage.

        Args:
            storage_dir: Directory to store JSON files
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # File paths
        self._files = {
            "heuristics": self.storage_dir / "heuristics.json",
            "outcomes": self.storage_dir / "outcomes.json",
            "preferences": self.storage_dir / "preferences.json",
            "domain_knowledge": self.storage_dir / "domain_knowledge.json",
            "anti_patterns": self.storage_dir / "anti_patterns.json",
        }

        # Initialize empty files if they don't exist
        for file_path in self._files.values():
            if not file_path.exists():
                self._write_json(file_path, [])

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "FileBasedStorage":
        """Create instance from configuration."""
        storage_dir = config.get("storage_dir", ".alma")
        return cls(storage_dir=Path(storage_dir))

    # ==================== WRITE OPERATIONS ====================

    def save_heuristic(self, heuristic: Heuristic) -> str:
        """Save a heuristic (UPSERT - update if exists, insert if new)."""
        data = self._read_json(self._files["heuristics"])
        record = self._to_dict(heuristic)
        # Find and replace existing, or append new
        found = False
        for i, existing in enumerate(data):
            if existing.get("id") == record["id"]:
                data[i] = record
                found = True
                break
        if not found:
            data.append(record)
        self._write_json(self._files["heuristics"], data)
        logger.debug(f"Saved heuristic: {heuristic.id}")
        return heuristic.id

    def save_outcome(self, outcome: Outcome) -> str:
        """Save an outcome (UPSERT - update if exists, insert if new)."""
        data = self._read_json(self._files["outcomes"])
        record = self._to_dict(outcome)
        # Find and replace existing, or append new
        found = False
        for i, existing in enumerate(data):
            if existing.get("id") == record["id"]:
                data[i] = record
                found = True
                break
        if not found:
            data.append(record)
        self._write_json(self._files["outcomes"], data)
        logger.debug(f"Saved outcome: {outcome.id}")
        return outcome.id

    def save_user_preference(self, preference: UserPreference) -> str:
        """Save a user preference (UPSERT - update if exists, insert if new)."""
        data = self._read_json(self._files["preferences"])
        record = self._to_dict(preference)
        # Find and replace existing, or append new
        found = False
        for i, existing in enumerate(data):
            if existing.get("id") == record["id"]:
                data[i] = record
                found = True
                break
        if not found:
            data.append(record)
        self._write_json(self._files["preferences"], data)
        logger.debug(f"Saved preference: {preference.id}")
        return preference.id

    def save_domain_knowledge(self, knowledge: DomainKnowledge) -> str:
        """Save domain knowledge (UPSERT - update if exists, insert if new)."""
        data = self._read_json(self._files["domain_knowledge"])
        record = self._to_dict(knowledge)
        # Find and replace existing, or append new
        found = False
        for i, existing in enumerate(data):
            if existing.get("id") == record["id"]:
                data[i] = record
                found = True
                break
        if not found:
            data.append(record)
        self._write_json(self._files["domain_knowledge"], data)
        logger.debug(f"Saved domain knowledge: {knowledge.id}")
        return knowledge.id

    def save_anti_pattern(self, anti_pattern: AntiPattern) -> str:
        """Save an anti-pattern (UPSERT - update if exists, insert if new)."""
        data = self._read_json(self._files["anti_patterns"])
        record = self._to_dict(anti_pattern)
        # Find and replace existing, or append new
        found = False
        for i, existing in enumerate(data):
            if existing.get("id") == record["id"]:
                data[i] = record
                found = True
                break
        if not found:
            data.append(record)
        self._write_json(self._files["anti_patterns"], data)
        logger.debug(f"Saved anti-pattern: {anti_pattern.id}")
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
        """Get heuristics (no vector search - returns all matching filters)."""
        data = self._read_json(self._files["heuristics"])

        # Filter
        results = []
        for record in data:
            if record.get("project_id") != project_id:
                continue
            if agent and record.get("agent") != agent:
                continue
            if record.get("confidence", 0) < min_confidence:
                continue
            results.append(self._to_heuristic(record))

        # Sort by confidence and return top_k
        results.sort(key=lambda x: -x.confidence)
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
        """Get outcomes (no vector search)."""
        data = self._read_json(self._files["outcomes"])

        results = []
        for record in data:
            if record.get("project_id") != project_id:
                continue
            if agent and record.get("agent") != agent:
                continue
            if task_type and record.get("task_type") != task_type:
                continue
            if success_only and not record.get("success"):
                continue
            results.append(self._to_outcome(record))

        # Sort by timestamp (most recent first) and return top_k
        results.sort(key=lambda x: x.timestamp, reverse=True)
        return results[:top_k]

    def get_user_preferences(
        self,
        user_id: str,
        category: Optional[str] = None,
    ) -> List[UserPreference]:
        """Get user preferences."""
        data = self._read_json(self._files["preferences"])

        results = []
        for record in data:
            if record.get("user_id") != user_id:
                continue
            if category and record.get("category") != category:
                continue
            results.append(self._to_user_preference(record))

        return results

    def get_domain_knowledge(
        self,
        project_id: str,
        agent: Optional[str] = None,
        domain: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        top_k: int = 5,
    ) -> List[DomainKnowledge]:
        """Get domain knowledge (no vector search)."""
        data = self._read_json(self._files["domain_knowledge"])

        results = []
        for record in data:
            if record.get("project_id") != project_id:
                continue
            if agent and record.get("agent") != agent:
                continue
            if domain and record.get("domain") != domain:
                continue
            results.append(self._to_domain_knowledge(record))

        # Sort by confidence and return top_k
        results.sort(key=lambda x: -x.confidence)
        return results[:top_k]

    def get_anti_patterns(
        self,
        project_id: str,
        agent: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        top_k: int = 5,
    ) -> List[AntiPattern]:
        """Get anti-patterns (no vector search)."""
        data = self._read_json(self._files["anti_patterns"])

        results = []
        for record in data:
            if record.get("project_id") != project_id:
                continue
            if agent and record.get("agent") != agent:
                continue
            results.append(self._to_anti_pattern(record))

        # Sort by occurrence count and return top_k
        results.sort(key=lambda x: -x.occurrence_count)
        return results[:top_k]

    # ==================== UPDATE OPERATIONS ====================

    def update_heuristic(
        self,
        heuristic_id: str,
        updates: Dict[str, Any],
    ) -> bool:
        """Update a heuristic's fields."""
        data = self._read_json(self._files["heuristics"])

        for i, record in enumerate(data):
            if record.get("id") == heuristic_id:
                data[i].update(updates)
                self._write_json(self._files["heuristics"], data)
                return True

        return False

    def increment_heuristic_occurrence(
        self,
        heuristic_id: str,
        success: bool,
    ) -> bool:
        """Increment heuristic occurrence count."""
        data = self._read_json(self._files["heuristics"])

        for i, record in enumerate(data):
            if record.get("id") == heuristic_id:
                data[i]["occurrence_count"] = record.get("occurrence_count", 0) + 1
                if success:
                    data[i]["success_count"] = record.get("success_count", 0) + 1
                data[i]["last_validated"] = datetime.now(timezone.utc).isoformat()
                self._write_json(self._files["heuristics"], data)
                return True

        return False

    # ==================== UPDATE CONFIDENCE OPERATIONS ====================

    def update_heuristic_confidence(
        self,
        heuristic_id: str,
        new_confidence: float,
    ) -> bool:
        """Update a heuristic's confidence score."""
        data = self._read_json(self._files["heuristics"])

        for i, record in enumerate(data):
            if record.get("id") == heuristic_id:
                data[i]["confidence"] = new_confidence
                data[i]["last_validated"] = datetime.now(timezone.utc).isoformat()
                self._write_json(self._files["heuristics"], data)
                return True

        return False

    def update_knowledge_confidence(
        self,
        knowledge_id: str,
        new_confidence: float,
    ) -> bool:
        """Update domain knowledge confidence score."""
        data = self._read_json(self._files["domain_knowledge"])

        for i, record in enumerate(data):
            if record.get("id") == knowledge_id:
                data[i]["confidence"] = new_confidence
                data[i]["last_verified"] = datetime.now(timezone.utc).isoformat()
                self._write_json(self._files["domain_knowledge"], data)
                return True

        return False

    # ==================== DELETE OPERATIONS ====================

    def delete_heuristic(self, heuristic_id: str) -> bool:
        """Delete a single heuristic by ID."""
        data = self._read_json(self._files["heuristics"])
        original_count = len(data)

        filtered = [r for r in data if r.get("id") != heuristic_id]
        self._write_json(self._files["heuristics"], filtered)

        deleted = original_count != len(filtered)
        if deleted:
            logger.debug(f"Deleted heuristic: {heuristic_id}")
        return deleted

    def delete_outcome(self, outcome_id: str) -> bool:
        """Delete a single outcome by ID."""
        data = self._read_json(self._files["outcomes"])
        original_count = len(data)

        filtered = [r for r in data if r.get("id") != outcome_id]
        self._write_json(self._files["outcomes"], filtered)

        deleted = original_count != len(filtered)
        if deleted:
            logger.debug(f"Deleted outcome: {outcome_id}")
        return deleted

    def delete_domain_knowledge(self, knowledge_id: str) -> bool:
        """Delete a single domain knowledge entry by ID."""
        data = self._read_json(self._files["domain_knowledge"])
        original_count = len(data)

        filtered = [r for r in data if r.get("id") != knowledge_id]
        self._write_json(self._files["domain_knowledge"], filtered)

        deleted = original_count != len(filtered)
        if deleted:
            logger.debug(f"Deleted domain knowledge: {knowledge_id}")
        return deleted

    def delete_anti_pattern(self, anti_pattern_id: str) -> bool:
        """Delete a single anti-pattern by ID."""
        data = self._read_json(self._files["anti_patterns"])
        original_count = len(data)

        filtered = [r for r in data if r.get("id") != anti_pattern_id]
        self._write_json(self._files["anti_patterns"], filtered)

        deleted = original_count != len(filtered)
        if deleted:
            logger.debug(f"Deleted anti-pattern: {anti_pattern_id}")
        return deleted

    def delete_outcomes_older_than(
        self,
        project_id: str,
        older_than: datetime,
        agent: Optional[str] = None,
    ) -> int:
        """Delete old outcomes."""
        data = self._read_json(self._files["outcomes"])
        original_count = len(data)

        filtered = []
        for record in data:
            if record.get("project_id") != project_id:
                filtered.append(record)
                continue
            if agent and record.get("agent") != agent:
                filtered.append(record)
                continue

            timestamp = self._parse_datetime(record.get("timestamp"))
            if timestamp and timestamp >= older_than:
                filtered.append(record)

        self._write_json(self._files["outcomes"], filtered)
        deleted = original_count - len(filtered)
        logger.info(f"Deleted {deleted} old outcomes")
        return deleted

    def delete_low_confidence_heuristics(
        self,
        project_id: str,
        below_confidence: float,
        agent: Optional[str] = None,
    ) -> int:
        """Delete low-confidence heuristics."""
        data = self._read_json(self._files["heuristics"])
        original_count = len(data)

        filtered = []
        for record in data:
            if record.get("project_id") != project_id:
                filtered.append(record)
                continue
            if agent and record.get("agent") != agent:
                filtered.append(record)
                continue

            if record.get("confidence", 0) >= below_confidence:
                filtered.append(record)

        self._write_json(self._files["heuristics"], filtered)
        deleted = original_count - len(filtered)
        logger.info(f"Deleted {deleted} low-confidence heuristics")
        return deleted

    # ==================== STATS ====================

    def get_stats(
        self,
        project_id: str,
        agent: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get memory statistics."""
        stats = {
            "project_id": project_id,
            "agent": agent,
            "heuristics_count": 0,
            "outcomes_count": 0,
            "preferences_count": 0,
            "domain_knowledge_count": 0,
            "anti_patterns_count": 0,
        }

        for name, file_path in self._files.items():
            data = self._read_json(file_path)
            count = 0
            for record in data:
                if name == "preferences":
                    # Preferences don't have project_id
                    count += 1
                elif record.get("project_id") == project_id:
                    if agent is None or record.get("agent") == agent:
                        count += 1
            stats[f"{name}_count"] = count

        stats["total_count"] = sum(
            stats[k] for k in stats if k.endswith("_count")
        )

        return stats

    # ==================== HELPERS ====================

    def _read_json(self, file_path: Path) -> List[Dict]:
        """Read JSON file."""
        try:
            with open(file_path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return []

    def _write_json(self, file_path: Path, data: List[Dict]):
        """Write JSON file."""
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def _to_dict(self, obj: Any) -> Dict:
        """Convert dataclass to dict with datetime handling."""
        if hasattr(obj, "__dataclass_fields__"):
            result = {}
            for field_name in obj.__dataclass_fields__:
                value = getattr(obj, field_name)
                if isinstance(value, datetime):
                    result[field_name] = value.isoformat()
                elif value is not None:
                    result[field_name] = value
            return result
        return dict(obj)

    def _parse_datetime(self, value: Any) -> Optional[datetime]:
        """Parse datetime from string or return as-is."""
        if value is None:
            return None
        if isinstance(value, datetime):
            return value
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            return None

    def _to_heuristic(self, record: Dict) -> Heuristic:
        """Convert dict to Heuristic."""
        return Heuristic(
            id=record["id"],
            agent=record["agent"],
            project_id=record["project_id"],
            condition=record["condition"],
            strategy=record["strategy"],
            confidence=record.get("confidence", 0.0),
            occurrence_count=record.get("occurrence_count", 0),
            success_count=record.get("success_count", 0),
            last_validated=self._parse_datetime(record.get("last_validated"))
            or datetime.now(timezone.utc),
            created_at=self._parse_datetime(record.get("created_at"))
            or datetime.now(timezone.utc),
            embedding=record.get("embedding"),
            metadata=record.get("metadata", {}),
        )

    def _to_outcome(self, record: Dict) -> Outcome:
        """Convert dict to Outcome."""
        return Outcome(
            id=record["id"],
            agent=record["agent"],
            project_id=record["project_id"],
            task_type=record.get("task_type", "general"),
            task_description=record["task_description"],
            success=record.get("success", False),
            strategy_used=record.get("strategy_used", ""),
            duration_ms=record.get("duration_ms"),
            error_message=record.get("error_message"),
            user_feedback=record.get("user_feedback"),
            timestamp=self._parse_datetime(record.get("timestamp"))
            or datetime.now(timezone.utc),
            embedding=record.get("embedding"),
            metadata=record.get("metadata", {}),
        )

    def _to_user_preference(self, record: Dict) -> UserPreference:
        """Convert dict to UserPreference."""
        return UserPreference(
            id=record["id"],
            user_id=record["user_id"],
            category=record.get("category", "general"),
            preference=record["preference"],
            source=record.get("source", "unknown"),
            confidence=record.get("confidence", 1.0),
            timestamp=self._parse_datetime(record.get("timestamp"))
            or datetime.now(timezone.utc),
            metadata=record.get("metadata", {}),
        )

    def _to_domain_knowledge(self, record: Dict) -> DomainKnowledge:
        """Convert dict to DomainKnowledge."""
        return DomainKnowledge(
            id=record["id"],
            agent=record["agent"],
            project_id=record["project_id"],
            domain=record.get("domain", "general"),
            fact=record["fact"],
            source=record.get("source", "unknown"),
            confidence=record.get("confidence", 1.0),
            last_verified=self._parse_datetime(record.get("last_verified"))
            or datetime.now(timezone.utc),
            embedding=record.get("embedding"),
            metadata=record.get("metadata", {}),
        )

    def _to_anti_pattern(self, record: Dict) -> AntiPattern:
        """Convert dict to AntiPattern."""
        return AntiPattern(
            id=record["id"],
            agent=record["agent"],
            project_id=record["project_id"],
            pattern=record["pattern"],
            why_bad=record.get("why_bad", ""),
            better_alternative=record.get("better_alternative", ""),
            occurrence_count=record.get("occurrence_count", 1),
            last_seen=self._parse_datetime(record.get("last_seen"))
            or datetime.now(timezone.utc),
            created_at=self._parse_datetime(record.get("created_at"))
            or datetime.now(timezone.utc),
            embedding=record.get("embedding"),
            metadata=record.get("metadata", {}),
        )
