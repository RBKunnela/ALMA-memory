"""
ALMA Event System.

Provides event emission and webhook delivery for memory operations.

The event system allows external systems to react to memory changes through:
1. In-process callbacks (subscribe to event types)
2. Webhooks (HTTP delivery with signatures)

Example - In-process subscription:
    ```python
    from alma.events import get_emitter, MemoryEventType

    def on_memory_created(event):
        print(f"Memory created: {event.memory_id}")

    emitter = get_emitter()
    emitter.subscribe(MemoryEventType.CREATED, on_memory_created)
    ```

Example - Webhook delivery:
    ```python
    from alma.events import WebhookConfig, WebhookManager, get_emitter

    manager = WebhookManager()
    manager.add_webhook(WebhookConfig(
        url="https://example.com/webhook",
        events=[MemoryEventType.CREATED, MemoryEventType.UPDATED],
        secret="my-webhook-secret"
    ))
    manager.start(get_emitter())
    ```
"""

from alma.events.emitter import (
    EventEmitter,
    get_emitter,
    reset_emitter,
)
from alma.events.storage_mixin import (
    EventAwareStorageMixin,
    emit_on_save,
)
from alma.events.types import (
    MemoryEvent,
    MemoryEventType,
    create_memory_event,
)
from alma.events.webhook import (
    WebhookConfig,
    WebhookDelivery,
    WebhookDeliveryResult,
    WebhookDeliveryStatus,
    WebhookManager,
)

__all__ = [
    # Types
    "MemoryEvent",
    "MemoryEventType",
    "create_memory_event",
    # Emitter
    "EventEmitter",
    "get_emitter",
    "reset_emitter",
    # Webhook
    "WebhookConfig",
    "WebhookDelivery",
    "WebhookDeliveryResult",
    "WebhookDeliveryStatus",
    "WebhookManager",
    # Storage Mixin
    "EventAwareStorageMixin",
    "emit_on_save",
]
