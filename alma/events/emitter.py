"""
ALMA Event Emitter.

Provides a pub/sub mechanism for memory events, allowing components
and external systems to subscribe to and receive notifications about
memory changes.
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Awaitable, Callable, Dict, List, Optional, Union

from alma.events.types import MemoryEvent, MemoryEventType

logger = logging.getLogger(__name__)

# Type aliases for callbacks
SyncCallback = Callable[[MemoryEvent], None]
AsyncCallback = Callable[[MemoryEvent], Awaitable[None]]
EventCallback = Union[SyncCallback, AsyncCallback]


class EventEmitter:
    """
    Event emitter for memory system events.

    Supports both synchronous and asynchronous callbacks, with options
    to subscribe to specific event types or all events.

    The emitter is designed to be non-blocking - callbacks are executed
    in a way that doesn't slow down the main storage operations.

    Example:
        ```python
        emitter = EventEmitter()

        def on_created(event: MemoryEvent):
            print(f"Memory created: {event.memory_id}")

        emitter.subscribe(MemoryEventType.CREATED, on_created)
        emitter.emit(event)
        ```
    """

    def __init__(self, max_workers: int = 4):
        """
        Initialize the event emitter.

        Args:
            max_workers: Maximum number of worker threads for async callback execution
        """
        self._subscribers: Dict[MemoryEventType, List[EventCallback]] = {}
        self._global_subscribers: List[EventCallback] = []
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._enabled = True

    def subscribe(
        self,
        event_type: MemoryEventType,
        callback: EventCallback,
    ) -> None:
        """
        Subscribe to a specific event type.

        Args:
            event_type: The type of event to subscribe to
            callback: Function to call when event occurs (sync or async)
        """
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []

        if callback not in self._subscribers[event_type]:
            self._subscribers[event_type].append(callback)
            callback_name = getattr(callback, "__name__", repr(callback))
            logger.debug(f"Subscribed to {event_type.value}: {callback_name}")

    def subscribe_all(self, callback: EventCallback) -> None:
        """
        Subscribe to all events.

        Args:
            callback: Function to call for any event
        """
        if callback not in self._global_subscribers:
            self._global_subscribers.append(callback)
            callback_name = getattr(callback, "__name__", repr(callback))
            logger.debug(f"Subscribed to all events: {callback_name}")

    def unsubscribe(
        self,
        event_type: MemoryEventType,
        callback: EventCallback,
    ) -> bool:
        """
        Unsubscribe from a specific event type.

        Args:
            event_type: The event type to unsubscribe from
            callback: The callback to remove

        Returns:
            True if callback was removed, False if not found
        """
        if event_type in self._subscribers:
            try:
                self._subscribers[event_type].remove(callback)
                callback_name = getattr(callback, "__name__", repr(callback))
                logger.debug(f"Unsubscribed from {event_type.value}: {callback_name}")
                return True
            except ValueError:
                pass
        return False

    def unsubscribe_all(self, callback: EventCallback) -> bool:
        """
        Unsubscribe a callback from all events.

        Args:
            callback: The callback to remove

        Returns:
            True if callback was removed, False if not found
        """
        try:
            self._global_subscribers.remove(callback)
            callback_name = getattr(callback, "__name__", repr(callback))
            logger.debug(f"Unsubscribed from all events: {callback_name}")
            return True
        except ValueError:
            return False

    def has_subscribers(self, event_type: Optional[MemoryEventType] = None) -> bool:
        """
        Check if there are any subscribers.

        Args:
            event_type: Optional specific event type to check

        Returns:
            True if there are subscribers
        """
        if event_type is None:
            return bool(self._global_subscribers) or any(
                bool(subs) for subs in self._subscribers.values()
            )
        return bool(self._subscribers.get(event_type)) or bool(self._global_subscribers)

    def emit(self, event: MemoryEvent) -> None:
        """
        Emit an event to all matching subscribers (non-blocking).

        Callbacks are executed in a thread pool to avoid blocking
        the main thread. Any exceptions in callbacks are logged
        but do not propagate.

        Args:
            event: The event to emit
        """
        if not self._enabled:
            return

        callbacks = self._get_callbacks(event.event_type)
        if not callbacks:
            return

        # Execute callbacks in thread pool (non-blocking)
        for callback in callbacks:
            self._executor.submit(self._safe_call, callback, event)

    async def emit_async(self, event: MemoryEvent) -> None:
        """
        Emit an event to all matching subscribers asynchronously.

        For async callbacks, awaits them directly. For sync callbacks,
        runs them in the executor.

        Args:
            event: The event to emit
        """
        if not self._enabled:
            return

        callbacks = self._get_callbacks(event.event_type)
        if not callbacks:
            return

        tasks = []
        for callback in callbacks:
            if asyncio.iscoroutinefunction(callback):
                tasks.append(self._safe_call_async(callback, event))
            else:
                # Run sync callbacks in executor
                loop = asyncio.get_event_loop()
                tasks.append(
                    loop.run_in_executor(
                        self._executor,
                        self._safe_call,
                        callback,
                        event,
                    )
                )

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def _get_callbacks(self, event_type: MemoryEventType) -> List[EventCallback]:
        """Get all callbacks for an event type."""
        callbacks = list(self._global_subscribers)
        callbacks.extend(self._subscribers.get(event_type, []))
        return callbacks

    def _safe_call(self, callback: SyncCallback, event: MemoryEvent) -> None:
        """Safely call a sync callback, catching exceptions."""
        try:
            callback(event)
        except Exception as e:
            callback_name = getattr(callback, "__name__", repr(callback))
            logger.error(
                f"Error in event callback {callback_name}: {e}",
                exc_info=True,
            )

    async def _safe_call_async(
        self,
        callback: AsyncCallback,
        event: MemoryEvent,
    ) -> None:
        """Safely call an async callback, catching exceptions."""
        try:
            await callback(event)
        except Exception as e:
            callback_name = getattr(callback, "__name__", repr(callback))
            logger.error(
                f"Error in async event callback {callback_name}: {e}",
                exc_info=True,
            )

    def enable(self) -> None:
        """Enable event emission."""
        self._enabled = True

    def disable(self) -> None:
        """Disable event emission (events will be silently dropped)."""
        self._enabled = False

    def clear(self) -> None:
        """Remove all subscribers."""
        self._subscribers.clear()
        self._global_subscribers.clear()

    def shutdown(self) -> None:
        """Shutdown the executor and clear subscribers."""
        self.clear()
        self._executor.shutdown(wait=False)


# Global emitter instance (singleton pattern)
_emitter: Optional[EventEmitter] = None


def get_emitter() -> EventEmitter:
    """
    Get the global event emitter instance.

    Returns:
        The singleton EventEmitter instance
    """
    global _emitter
    if _emitter is None:
        _emitter = EventEmitter()
    return _emitter


def reset_emitter() -> None:
    """
    Reset the global emitter (mainly for testing).

    Creates a fresh emitter instance, clearing all subscriptions.
    """
    global _emitter
    if _emitter is not None:
        _emitter.shutdown()
    _emitter = EventEmitter()
