"""
ALMA Event Types.

Defines event types and the MemoryEvent dataclass for the event system.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional


class MemoryEventType(Enum):
    """Types of events that can be emitted by the memory system."""

    # Core memory operations
    CREATED = "memory.created"
    UPDATED = "memory.updated"
    DELETED = "memory.deleted"
    CONSOLIDATED = "memory.consolidated"

    # Learning-specific events
    HEURISTIC_FORMED = "heuristic.formed"
    ANTIPATTERN_DETECTED = "antipattern.detected"
    PREFERENCE_ADDED = "preference.added"
    KNOWLEDGE_ADDED = "knowledge.added"

    # Confidence events
    CONFIDENCE_UPDATED = "confidence.updated"
    CONFIDENCE_DECAYED = "confidence.decayed"


@dataclass
class MemoryEvent:
    """
    Represents an event in the memory system.

    Events are emitted when memory operations occur, allowing external
    systems to react to changes through callbacks or webhooks.

    Attributes:
        event_type: The type of event that occurred
        agent: Name of the agent associated with the event
        project_id: Project identifier
        memory_type: Type of memory (heuristics, outcomes, etc.)
        memory_id: Unique identifier of the affected memory
        timestamp: When the event occurred
        payload: Event-specific data (e.g., the created memory)
        metadata: Optional additional context
    """

    event_type: MemoryEventType
    agent: str
    project_id: str
    memory_type: (
        str  # heuristics, outcomes, preferences, domain_knowledge, anti_patterns
    )
    memory_id: str
    timestamp: datetime
    payload: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization."""
        return {
            "event_type": self.event_type.value,
            "agent": self.agent,
            "project_id": self.project_id,
            "memory_type": self.memory_type,
            "memory_id": self.memory_id,
            "timestamp": self.timestamp.isoformat(),
            "payload": self.payload,
            "metadata": self.metadata or {},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryEvent":
        """Create event from dictionary."""
        return cls(
            event_type=MemoryEventType(data["event_type"]),
            agent=data["agent"],
            project_id=data["project_id"],
            memory_type=data["memory_type"],
            memory_id=data["memory_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            payload=data["payload"],
            metadata=data.get("metadata"),
        )


def create_memory_event(
    event_type: MemoryEventType,
    agent: str,
    project_id: str,
    memory_type: str,
    memory_id: str,
    payload: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None,
    *,
    default_metadata: bool = True,
) -> MemoryEvent:
    """
    Factory function to create a MemoryEvent with current timestamp.

    Args:
        event_type: Type of event
        agent: Agent name
        project_id: Project identifier
        memory_type: Type of memory
        memory_id: Memory identifier
        payload: Event-specific data
        metadata: Optional additional context

    Returns:
        A new MemoryEvent instance
    """
    return MemoryEvent(
        event_type=event_type,
        agent=agent,
        project_id=project_id,
        memory_type=memory_type,
        memory_id=memory_id,
        timestamp=datetime.now(timezone.utc),
        payload=payload,
        metadata=metadata if metadata is not None else {},
    )
