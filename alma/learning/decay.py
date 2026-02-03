"""
ALMA Decay-Based Forgetting.

Implements memory strength tracking with natural decay over time,
reinforced by access patterns - mimicking human memory behavior.

Key Principles from Memory Wall:
- "Forgetting is a technology" - lossy compression with importance weighting
- Memories weaken over time if not accessed
- Accessing a memory strengthens it
- Users can explicitly mark memories as important
- Weak memories can be rescued before deletion

Strength Formula:
    Strength = (Base Decay + Access Bonus + Reinforcement Bonus) × Importance Factor

Where:
    - Base Decay = e^(-0.693 × days_since_access / half_life)
    - Access Bonus = min(0.4, log(1 + access_count) × 0.1)
    - Reinforcement Bonus = min(0.3, recent_reinforcements × 0.1)
    - Importance Factor = 0.5 + (explicit_importance × 0.5)
"""

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class StrengthState(Enum):
    """Memory strength states based on current strength value."""

    STRONG = "strong"  # > 0.7 - Normal retrieval, highly likely to be returned
    NORMAL = "normal"  # 0.3 - 0.7 - Normal retrieval
    WEAK = "weak"  # 0.1 - 0.3 - Recoverable, shown in warnings
    FORGETTABLE = "forgettable"  # < 0.1 - Ready for archive/deletion


@dataclass
class MemoryStrength:
    """
    Tracks decay and reinforcement of a memory.

    Implements human-like memory behavior where memories naturally
    decay over time but are strengthened through access and
    explicit reinforcement.

    Attributes:
        memory_id: Unique identifier for the memory
        memory_type: Type of memory (heuristic, outcome, knowledge, etc.)
        initial_strength: Starting strength when created (default 1.0)
        decay_half_life_days: Days until strength halves without access
        created_at: When the memory was created
        last_accessed: When the memory was last accessed
        access_count: Number of times memory has been accessed
        reinforcement_events: Timestamps of explicit reinforcement
        explicit_importance: User-set importance (0-1), affects decay rate
    """

    memory_id: str
    memory_type: str = "unknown"
    initial_strength: float = 1.0
    decay_half_life_days: int = 30
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_accessed: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    access_count: int = 0
    reinforcement_events: List[datetime] = field(default_factory=list)
    explicit_importance: float = 0.5  # Default middle importance

    def current_strength(self) -> float:
        """
        Calculate current memory strength with decay and reinforcement.

        The formula combines:
        1. Base decay using half-life formula
        2. Access bonus with diminishing returns
        3. Recent reinforcement bonus
        4. Importance factor multiplier

        Returns:
            Current strength value between 0.0 and 1.0
        """
        now = datetime.now(timezone.utc)

        # Handle naive datetimes by assuming UTC
        last_accessed = self.last_accessed
        if last_accessed.tzinfo is None:
            last_accessed = last_accessed.replace(tzinfo=timezone.utc)

        days_since_access = (now - last_accessed).total_seconds() / 86400

        # Base decay using half-life formula
        # After half_life days, strength = 0.5 of original
        if self.decay_half_life_days > 0:
            base_decay = math.exp(
                -0.693 * days_since_access / self.decay_half_life_days
            )
        else:
            base_decay = 1.0  # No decay if half-life is 0

        # Access reinforcement (diminishing returns via log)
        # More accesses = stronger memory, but capped at 0.4 bonus
        access_bonus = min(0.4, math.log1p(self.access_count) * 0.1)

        # Recency of reinforcements (within last 7 days)
        # Each recent reinforcement adds 0.1, capped at 0.3
        recent_reinforcements = sum(
            1 for r in self.reinforcement_events
            if self._days_ago(r) < 7
        )
        reinforcement_bonus = min(0.3, recent_reinforcements * 0.1)

        # Explicit importance factor (0.5 to 1.0 range)
        # importance=0 means 0.5x strength, importance=1 means 1x strength
        importance_factor = 0.5 + (self.explicit_importance * 0.5)

        # Combine factors
        strength = (base_decay + access_bonus + reinforcement_bonus) * importance_factor
        return min(1.0, max(0.0, strength))

    def _days_ago(self, dt: datetime) -> float:
        """Calculate days since a datetime."""
        now = datetime.now(timezone.utc)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return (now - dt).total_seconds() / 86400

    def access(self) -> float:
        """
        Record an access to this memory.

        Returns:
            New strength after access
        """
        self.last_accessed = datetime.now(timezone.utc)
        self.access_count += 1
        return self.current_strength()

    def reinforce(self) -> float:
        """
        Explicitly reinforce this memory.

        Reinforcement is stronger than a simple access and records
        a reinforcement event for the bonus calculation.

        Returns:
            New strength after reinforcement
        """
        now = datetime.now(timezone.utc)
        self.reinforcement_events.append(now)
        self.last_accessed = now
        # Keep only last 10 reinforcement events to prevent unbounded growth
        self.reinforcement_events = self.reinforcement_events[-10:]
        return self.current_strength()

    def set_importance(self, importance: float) -> float:
        """
        Set explicit importance level.

        Args:
            importance: Value between 0.0 and 1.0

        Returns:
            New strength after importance change
        """
        self.explicit_importance = max(0.0, min(1.0, importance))
        return self.current_strength()

    def get_state(self) -> StrengthState:
        """
        Get the current state based on strength.

        Returns:
            StrengthState enum value
        """
        strength = self.current_strength()
        if strength > 0.7:
            return StrengthState.STRONG
        elif strength >= 0.3:
            return StrengthState.NORMAL
        elif strength >= 0.1:
            return StrengthState.WEAK
        else:
            return StrengthState.FORGETTABLE

    def should_forget(self, threshold: float = 0.1) -> bool:
        """
        Determine if memory should be forgotten.

        Args:
            threshold: Strength threshold below which to forget

        Returns:
            True if strength is below threshold
        """
        return self.current_strength() < threshold

    def is_recoverable(self) -> bool:
        """
        Check if memory is weak but can be recovered.

        Recoverable memories are weak enough to warrant attention
        but not so weak they should be deleted.

        Returns:
            True if in recoverable range (0.1 <= strength < 0.3)
        """
        strength = self.current_strength()
        return 0.1 <= strength < 0.3

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "memory_id": self.memory_id,
            "memory_type": self.memory_type,
            "initial_strength": self.initial_strength,
            "decay_half_life_days": self.decay_half_life_days,
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "access_count": self.access_count,
            "reinforcement_events": [r.isoformat() for r in self.reinforcement_events],
            "explicit_importance": self.explicit_importance,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryStrength":
        """Create from dictionary."""
        reinforcement_events = []
        for r in data.get("reinforcement_events", []):
            if isinstance(r, str):
                # Parse ISO format, handle both Z and +00:00 suffixes
                dt_str = r.replace("Z", "+00:00")
                reinforcement_events.append(datetime.fromisoformat(dt_str))
            elif isinstance(r, datetime):
                reinforcement_events.append(r)

        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        elif created_at is None:
            created_at = datetime.now(timezone.utc)

        last_accessed = data.get("last_accessed")
        if isinstance(last_accessed, str):
            last_accessed = datetime.fromisoformat(last_accessed.replace("Z", "+00:00"))
        elif last_accessed is None:
            last_accessed = datetime.now(timezone.utc)

        return cls(
            memory_id=data["memory_id"],
            memory_type=data.get("memory_type", "unknown"),
            initial_strength=data.get("initial_strength", 1.0),
            decay_half_life_days=data.get("decay_half_life_days", 30),
            created_at=created_at,
            last_accessed=last_accessed,
            access_count=data.get("access_count", 0),
            reinforcement_events=reinforcement_events,
            explicit_importance=data.get("explicit_importance", 0.5),
        )


@dataclass
class DecayConfig:
    """
    Configuration for decay-based forgetting.

    Allows customization of half-lives per memory type and
    global forget thresholds.
    """

    enabled: bool = True
    default_half_life_days: int = 30
    forget_threshold: float = 0.1
    weak_threshold: float = 0.3
    strong_threshold: float = 0.7

    # Half-life by memory type (days until half strength)
    half_life_by_type: Dict[str, int] = field(default_factory=lambda: {
        "heuristic": 60,      # Heuristics are valuable, decay slowly
        "outcome": 30,        # Outcomes decay at normal rate
        "preference": 365,    # User preferences are very stable
        "knowledge": 90,      # Domain knowledge decays slowly
        "anti_pattern": 45,   # Anti-patterns decay moderately
    })

    def get_half_life(self, memory_type: str) -> int:
        """Get half-life for a memory type."""
        return self.half_life_by_type.get(memory_type, self.default_half_life_days)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DecayConfig":
        """Create from configuration dictionary."""
        return cls(
            enabled=data.get("enabled", True),
            default_half_life_days=data.get("default_half_life_days", 30),
            forget_threshold=data.get("forget_threshold", 0.1),
            weak_threshold=data.get("weak_threshold", 0.3),
            strong_threshold=data.get("strong_threshold", 0.7),
            half_life_by_type=data.get("half_life_by_type", {
                "heuristic": 60,
                "outcome": 30,
                "preference": 365,
                "knowledge": 90,
                "anti_pattern": 45,
            }),
        )


class DecayManager:
    """
    Manages memory decay across the system.

    Provides methods to:
    - Track and update memory strength
    - Record accesses and reinforcements
    - Find memories ready to forget
    - Find weak but recoverable memories
    """

    def __init__(
        self,
        storage: Any,  # StorageBackend with strength methods
        config: Optional[DecayConfig] = None,
    ):
        """
        Initialize decay manager.

        Args:
            storage: Storage backend with memory strength methods
            config: Decay configuration (uses defaults if not provided)
        """
        self.storage = storage
        self.config = config or DecayConfig()
        self._strength_cache: Dict[str, MemoryStrength] = {}

    def get_strength(self, memory_id: str, memory_type: str = "unknown") -> MemoryStrength:
        """
        Get or create strength record for a memory.

        Args:
            memory_id: Memory identifier
            memory_type: Type of memory for half-life lookup

        Returns:
            MemoryStrength record
        """
        # Check cache first
        if memory_id in self._strength_cache:
            return self._strength_cache[memory_id]

        # Try to load from storage
        strength = self.storage.get_memory_strength(memory_id)

        if strength is None:
            # Create new strength record with appropriate half-life
            half_life = self.config.get_half_life(memory_type)
            strength = MemoryStrength(
                memory_id=memory_id,
                memory_type=memory_type,
                decay_half_life_days=half_life,
            )
            self.storage.save_memory_strength(strength)

        self._strength_cache[memory_id] = strength
        return strength

    def record_access(self, memory_id: str, memory_type: str = "unknown") -> float:
        """
        Record memory access, return new strength.

        Args:
            memory_id: Memory that was accessed
            memory_type: Type of memory

        Returns:
            New strength value after access
        """
        strength = self.get_strength(memory_id, memory_type)
        new_strength = strength.access()
        self.storage.save_memory_strength(strength)
        return new_strength

    def reinforce_memory(self, memory_id: str, memory_type: str = "unknown") -> float:
        """
        Explicitly reinforce a memory.

        Args:
            memory_id: Memory to reinforce
            memory_type: Type of memory

        Returns:
            New strength value after reinforcement
        """
        strength = self.get_strength(memory_id, memory_type)
        new_strength = strength.reinforce()
        self.storage.save_memory_strength(strength)
        logger.info(f"Reinforced memory {memory_id}, new strength: {new_strength:.3f}")
        return new_strength

    def set_importance(
        self,
        memory_id: str,
        importance: float,
        memory_type: str = "unknown",
    ) -> float:
        """
        Set explicit importance for a memory.

        Args:
            memory_id: Memory to update
            importance: Importance value (0.0 to 1.0)
            memory_type: Type of memory

        Returns:
            New strength value after update
        """
        strength = self.get_strength(memory_id, memory_type)
        new_strength = strength.set_importance(importance)
        self.storage.save_memory_strength(strength)
        logger.info(
            f"Set importance for {memory_id} to {importance:.2f}, "
            f"new strength: {new_strength:.3f}"
        )
        return new_strength

    def get_forgettable_memories(
        self,
        project_id: str,
        agent: Optional[str] = None,
        threshold: Optional[float] = None,
    ) -> List[Tuple[str, str, float]]:
        """
        Get memories that should be forgotten.

        Args:
            project_id: Project to scan
            agent: Specific agent or None for all
            threshold: Custom threshold or use config default

        Returns:
            List of (memory_id, memory_type, strength) tuples
        """
        threshold = threshold or self.config.forget_threshold
        all_strengths = self.storage.get_all_memory_strengths(project_id, agent)

        forgettable = []
        for strength in all_strengths:
            current = strength.current_strength()
            if current < threshold:
                forgettable.append((
                    strength.memory_id,
                    strength.memory_type,
                    current,
                ))

        # Sort by strength (weakest first)
        return sorted(forgettable, key=lambda x: x[2])

    def get_weak_memories(
        self,
        project_id: str,
        agent: Optional[str] = None,
    ) -> List[Tuple[str, str, float]]:
        """
        Get recoverable memories sorted by strength.

        These are memories that are weak but can still be saved
        through reinforcement.

        Args:
            project_id: Project to scan
            agent: Specific agent or None for all

        Returns:
            List of (memory_id, memory_type, strength) tuples
        """
        all_strengths = self.storage.get_all_memory_strengths(project_id, agent)

        weak = []
        for strength in all_strengths:
            if strength.is_recoverable():
                weak.append((
                    strength.memory_id,
                    strength.memory_type,
                    strength.current_strength(),
                ))

        # Sort by strength (weakest first for prioritization)
        return sorted(weak, key=lambda x: x[2])

    def get_strong_memories(
        self,
        project_id: str,
        agent: Optional[str] = None,
    ) -> List[Tuple[str, str, float]]:
        """
        Get strong memories.

        Args:
            project_id: Project to scan
            agent: Specific agent or None for all

        Returns:
            List of (memory_id, memory_type, strength) tuples
        """
        all_strengths = self.storage.get_all_memory_strengths(project_id, agent)

        strong = []
        for strength in all_strengths:
            current = strength.current_strength()
            if current > self.config.strong_threshold:
                strong.append((
                    strength.memory_id,
                    strength.memory_type,
                    current,
                ))

        # Sort by strength (strongest first)
        return sorted(strong, key=lambda x: x[2], reverse=True)

    def get_memory_stats(
        self,
        project_id: str,
        agent: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get statistics about memory strength distribution.

        Args:
            project_id: Project to analyze
            agent: Specific agent or None for all

        Returns:
            Dictionary with strength statistics
        """
        all_strengths = self.storage.get_all_memory_strengths(project_id, agent)

        if not all_strengths:
            return {
                "total": 0,
                "strong": 0,
                "normal": 0,
                "weak": 0,
                "forgettable": 0,
                "average_strength": 0.0,
                "by_type": {},
            }

        stats = {
            "total": len(all_strengths),
            "strong": 0,
            "normal": 0,
            "weak": 0,
            "forgettable": 0,
            "by_type": {},
        }

        strength_sum = 0.0
        for s in all_strengths:
            current = s.current_strength()
            strength_sum += current

            state = s.get_state()
            if state == StrengthState.STRONG:
                stats["strong"] += 1
            elif state == StrengthState.NORMAL:
                stats["normal"] += 1
            elif state == StrengthState.WEAK:
                stats["weak"] += 1
            else:
                stats["forgettable"] += 1

            # Track by type
            if s.memory_type not in stats["by_type"]:
                stats["by_type"][s.memory_type] = {
                    "count": 0,
                    "avg_strength": 0.0,
                    "strength_sum": 0.0,
                }
            stats["by_type"][s.memory_type]["count"] += 1
            stats["by_type"][s.memory_type]["strength_sum"] += current

        stats["average_strength"] = strength_sum / len(all_strengths)

        # Calculate per-type averages
        for type_stats in stats["by_type"].values():
            if type_stats["count"] > 0:
                type_stats["avg_strength"] = (
                    type_stats["strength_sum"] / type_stats["count"]
                )
            del type_stats["strength_sum"]

        return stats

    def cleanup_forgettable(
        self,
        project_id: str,
        agent: Optional[str] = None,
        dry_run: bool = True,
    ) -> Dict[str, Any]:
        """
        Clean up memories below forget threshold.

        Args:
            project_id: Project to clean
            agent: Specific agent or None for all
            dry_run: If True, only report what would be deleted

        Returns:
            Cleanup results
        """
        forgettable = self.get_forgettable_memories(project_id, agent)

        result = {
            "dry_run": dry_run,
            "count": len(forgettable),
            "memories": forgettable,
            "deleted": 0,
        }

        if not dry_run:
            for memory_id, memory_type, _ in forgettable:
                try:
                    # Delete from main storage based on type
                    deleted = self._delete_memory(memory_id, memory_type)
                    if deleted:
                        # Also delete strength record
                        self.storage.delete_memory_strength(memory_id)
                        # Clear from cache
                        self._strength_cache.pop(memory_id, None)
                        result["deleted"] += 1
                except Exception as e:
                    logger.warning(f"Failed to delete memory {memory_id}: {e}")

        return result

    def _delete_memory(self, memory_id: str, memory_type: str) -> bool:
        """Delete a memory from main storage."""
        try:
            if memory_type == "heuristic":
                return self.storage.delete_heuristic(memory_id)
            elif memory_type == "outcome":
                return self.storage.delete_outcome(memory_id)
            elif memory_type == "knowledge":
                return self.storage.delete_domain_knowledge(memory_id)
            elif memory_type == "anti_pattern":
                return self.storage.delete_anti_pattern(memory_id)
            else:
                logger.warning(f"Unknown memory type for deletion: {memory_type}")
                return False
        except Exception as e:
            logger.error(f"Error deleting memory {memory_id}: {e}")
            return False

    def smart_forget(
        self,
        project_id: str,
        agent: Optional[str] = None,
        threshold: Optional[float] = None,
        archive: bool = True,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """
        Forget weak memories with optional archiving.

        This is the recommended method for memory cleanup as it:
        1. Identifies memories below the forget threshold
        2. Archives them before deletion (if enabled)
        3. Deletes the memory and its strength record
        4. Returns a detailed report

        Args:
            project_id: Project to scan and clean
            agent: Specific agent or None for all
            threshold: Custom forget threshold or use config default
            archive: If True, archive memories before deletion
            dry_run: If True, only report what would be done

        Returns:
            Dictionary with:
                - dry_run: Whether this was a dry run
                - threshold: Threshold used
                - archive_enabled: Whether archiving was enabled
                - total_found: Number of forgettable memories found
                - archived: List of archived memory IDs
                - deleted: List of deleted memory IDs
                - errors: List of any errors encountered
        """
        threshold = threshold or self.config.forget_threshold
        forgettable = self.get_forgettable_memories(project_id, agent, threshold)

        result = {
            "dry_run": dry_run,
            "threshold": threshold,
            "archive_enabled": archive,
            "total_found": len(forgettable),
            "archived": [],
            "deleted": [],
            "errors": [],
        }

        if dry_run:
            result["would_archive"] = [
                {"id": mid, "type": mtype, "strength": s}
                for mid, mtype, s in forgettable
            ] if archive else []
            result["would_delete"] = [
                {"id": mid, "type": mtype, "strength": s}
                for mid, mtype, s in forgettable
            ]
            return result

        for memory_id, memory_type, current_strength in forgettable:
            try:
                # Archive before deletion if enabled
                if archive:
                    try:
                        archived = self.storage.archive_memory(
                            memory_id=memory_id,
                            memory_type=memory_type,
                            reason="decay",
                            final_strength=current_strength,
                        )
                        result["archived"].append({
                            "memory_id": memory_id,
                            "archive_id": archived.id,
                            "memory_type": memory_type,
                            "final_strength": current_strength,
                        })
                        logger.info(
                            f"Archived memory {memory_id} as {archived.id} "
                            f"(strength: {current_strength:.3f})"
                        )
                    except Exception as e:
                        # Log but continue - still try to delete
                        logger.warning(
                            f"Failed to archive memory {memory_id}: {e}"
                        )
                        result["errors"].append({
                            "memory_id": memory_id,
                            "operation": "archive",
                            "error": str(e),
                        })

                # Delete the memory
                deleted = self._delete_memory(memory_id, memory_type)
                if deleted:
                    # Also delete strength record
                    self.storage.delete_memory_strength(memory_id)
                    # Clear from cache
                    self._strength_cache.pop(memory_id, None)
                    result["deleted"].append({
                        "memory_id": memory_id,
                        "memory_type": memory_type,
                    })
                    logger.info(
                        f"Deleted memory {memory_id} (type: {memory_type}, "
                        f"strength: {current_strength:.3f})"
                    )
                else:
                    result["errors"].append({
                        "memory_id": memory_id,
                        "operation": "delete",
                        "error": "Delete returned False",
                    })

            except Exception as e:
                logger.error(f"Error processing memory {memory_id}: {e}")
                result["errors"].append({
                    "memory_id": memory_id,
                    "operation": "process",
                    "error": str(e),
                })

        return result

    def invalidate_cache(self, memory_id: Optional[str] = None) -> None:
        """
        Invalidate strength cache.

        Args:
            memory_id: Specific memory to invalidate, or None for all
        """
        if memory_id:
            self._strength_cache.pop(memory_id, None)
        else:
            self._strength_cache.clear()


def calculate_projected_strength(
    strength: MemoryStrength,
    days_ahead: int,
) -> float:
    """
    Project what strength will be in the future.

    Useful for predicting when a memory will become weak/forgettable.

    Args:
        strength: Current MemoryStrength record
        days_ahead: Days to project forward

    Returns:
        Projected strength value
    """
    if days_ahead <= 0:
        return strength.current_strength()

    # Calculate decay factor for future date
    if strength.decay_half_life_days > 0:
        decay_factor = math.exp(
            -0.693 * days_ahead / strength.decay_half_life_days
        )
    else:
        decay_factor = 1.0

    # Current strength minus future decay
    current = strength.current_strength()
    # Simplified projection: apply decay to current strength
    return max(0.0, current * decay_factor)


def days_until_threshold(
    strength: MemoryStrength,
    threshold: float = 0.1,
) -> Optional[int]:
    """
    Calculate days until memory reaches threshold.

    Args:
        strength: MemoryStrength record
        threshold: Target threshold

    Returns:
        Days until threshold, or None if already below or won't reach
    """
    current = strength.current_strength()

    if current <= threshold:
        return 0

    if strength.decay_half_life_days <= 0:
        return None  # No decay, won't reach threshold

    # Solve: threshold = current * e^(-0.693 * days / half_life)
    # days = -half_life * ln(threshold / current) / 0.693
    ratio = threshold / current
    if ratio >= 1:
        return 0

    days = -strength.decay_half_life_days * math.log(ratio) / 0.693
    return int(math.ceil(days))
