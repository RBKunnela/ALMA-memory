"""
ALMA Event-Aware Storage Mixin.

Provides event emission capabilities for storage backends.
This is a mixin class that can be used by any storage implementation
to emit events when memory operations occur.
"""

import logging
from typing import Dict, Any, Optional
from dataclasses import asdict

from alma.events.types import MemoryEvent, MemoryEventType, create_memory_event
from alma.events.emitter import get_emitter, EventEmitter

logger = logging.getLogger(__name__)


class EventAwareStorageMixin:
    """
    Mixin that adds event emission to storage backends.

    This mixin provides helper methods to emit events for various
    storage operations. Events are only emitted if there are subscribers,
    making the overhead negligible when events are not used.

    Usage:
        class MyStorage(StorageBackend, EventAwareStorageMixin):
            def save_heuristic(self, heuristic):
                result_id = super().save_heuristic(heuristic)
                self._emit_memory_event(
                    event_type=MemoryEventType.CREATED,
                    agent=heuristic.agent,
                    project_id=heuristic.project_id,
                    memory_type="heuristics",
                    memory_id=result_id,
                    payload=self._heuristic_to_dict(heuristic),
                )
                return result_id
    """

    _events_enabled: bool = True
    _custom_emitter: Optional[EventEmitter] = None

    def enable_events(self) -> None:
        """Enable event emission for this storage instance."""
        self._events_enabled = True

    def disable_events(self) -> None:
        """Disable event emission for this storage instance."""
        self._events_enabled = False

    def set_emitter(self, emitter: EventEmitter) -> None:
        """
        Set a custom event emitter for this storage instance.

        Args:
            emitter: The event emitter to use
        """
        self._custom_emitter = emitter

    def _get_emitter(self) -> EventEmitter:
        """Get the event emitter to use."""
        if self._custom_emitter is not None:
            return self._custom_emitter
        return get_emitter()

    def _should_emit(self, event_type: MemoryEventType) -> bool:
        """
        Check if events should be emitted.

        Returns False if:
        - Events are disabled for this storage instance
        - There are no subscribers for this event type

        This optimization ensures event creation overhead is avoided
        when no one is listening.

        Args:
            event_type: The type of event being considered

        Returns:
            True if an event should be emitted
        """
        if not self._events_enabled:
            return False

        emitter = self._get_emitter()
        return emitter.has_subscribers(event_type)

    def _emit_memory_event(
        self,
        event_type: MemoryEventType,
        agent: str,
        project_id: str,
        memory_type: str,
        memory_id: str,
        payload: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Emit a memory event if there are subscribers.

        This is a convenience method that checks for subscribers before
        creating and emitting the event.

        Args:
            event_type: Type of event
            agent: Agent name
            project_id: Project identifier
            memory_type: Type of memory
            memory_id: Memory identifier
            payload: Event-specific data
            metadata: Optional additional context
        """
        if not self._should_emit(event_type):
            return

        event = create_memory_event(
            event_type=event_type,
            agent=agent,
            project_id=project_id,
            memory_type=memory_type,
            memory_id=memory_id,
            payload=payload,
            metadata=metadata,
        )

        emitter = self._get_emitter()
        emitter.emit(event)

        logger.debug(
            f"Emitted {event_type.value} for {memory_type}/{memory_id}"
        )

    def _emit_created_event(
        self,
        agent: str,
        project_id: str,
        memory_type: str,
        memory_id: str,
        payload: Dict[str, Any],
    ) -> None:
        """Emit a CREATED event."""
        self._emit_memory_event(
            event_type=MemoryEventType.CREATED,
            agent=agent,
            project_id=project_id,
            memory_type=memory_type,
            memory_id=memory_id,
            payload=payload,
        )

    def _emit_updated_event(
        self,
        agent: str,
        project_id: str,
        memory_type: str,
        memory_id: str,
        payload: Dict[str, Any],
    ) -> None:
        """Emit an UPDATED event."""
        self._emit_memory_event(
            event_type=MemoryEventType.UPDATED,
            agent=agent,
            project_id=project_id,
            memory_type=memory_type,
            memory_id=memory_id,
            payload=payload,
        )

    def _emit_deleted_event(
        self,
        agent: str,
        project_id: str,
        memory_type: str,
        memory_id: str,
    ) -> None:
        """Emit a DELETED event."""
        self._emit_memory_event(
            event_type=MemoryEventType.DELETED,
            agent=agent,
            project_id=project_id,
            memory_type=memory_type,
            memory_id=memory_id,
            payload={"deleted_id": memory_id},
        )


def emit_on_save(memory_type: str, event_type: MemoryEventType = MemoryEventType.CREATED):
    """
    Decorator to emit events when save methods are called.

    This decorator can be used on storage methods that save memories.
    It will emit an event after the save completes successfully.

    Args:
        memory_type: The type of memory being saved (e.g., "heuristics")
        event_type: The event type to emit (default: CREATED)

    Example:
        @emit_on_save("heuristics")
        def save_heuristic(self, heuristic: Heuristic) -> str:
            # Original implementation
            ...
    """
    def decorator(func):
        def wrapper(self, memory_item):
            # Call the original method
            result_id = func(self, memory_item)

            # Emit event if storage supports events
            if hasattr(self, "_emit_memory_event") and hasattr(self, "_should_emit"):
                if self._should_emit(event_type):
                    # Extract common fields
                    agent = getattr(memory_item, "agent", "unknown")
                    project_id = getattr(memory_item, "project_id", "unknown")

                    # Convert to dict, handling dataclasses
                    try:
                        if hasattr(memory_item, "__dataclass_fields__"):
                            payload = {
                                k: v for k, v in asdict(memory_item).items()
                                if k != "embedding"  # Exclude large embedding vectors
                            }
                        else:
                            payload = {"id": result_id}
                    except Exception:
                        payload = {"id": result_id}

                    self._emit_memory_event(
                        event_type=event_type,
                        agent=agent,
                        project_id=project_id,
                        memory_type=memory_type,
                        memory_id=result_id,
                        payload=payload,
                    )

            return result_id

        return wrapper
    return decorator
