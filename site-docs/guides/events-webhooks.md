# Event System & Webhooks Guide

ALMA's event system allows external systems to react to memory changes through in-process callbacks or HTTP webhooks.

## Overview

The event system supports two delivery mechanisms:

1. **In-Process Callbacks** - For same-process reactions
2. **Webhooks** - For external service notifications

## Event Types

| Event | Trigger | Payload |
|-------|---------|---------|
| `CREATED` | New memory saved | Memory details |
| `UPDATED` | Memory modified | Old + new values |
| `DELETED` | Memory removed | Memory ID |
| `ACCESSED` | Memory retrieved | Query details |
| `CONSOLIDATED` | Memories merged | Merge details |

## In-Process Callbacks

### Basic Usage

```python
from alma.events import get_emitter, MemoryEventType

# Get the global event emitter
emitter = get_emitter()

# Define callback
def on_memory_created(event):
    print(f"Memory created: {event.memory_id}")
    print(f"Agent: {event.agent}")
    print(f"Type: {event.memory_type}")

# Subscribe to events
emitter.subscribe(MemoryEventType.CREATED, on_memory_created)
```

### Event Object

```python
from alma.events import MemoryEvent, MemoryEventType

def handler(event: MemoryEvent):
    event.event_type      # MemoryEventType enum
    event.memory_id       # UUID of the memory
    event.memory_type     # 'heuristic', 'outcome', etc.
    event.agent           # Agent name
    event.project_id      # Project ID
    event.timestamp       # When event occurred
    event.data            # Additional event data
```

### Multiple Subscriptions

```python
def on_created(event):
    log.info(f"Created: {event.memory_id}")

def on_updated(event):
    log.info(f"Updated: {event.memory_id}")

def on_any(event):
    # Runs for all events
    metrics.increment(f"alma.events.{event.event_type.value}")

emitter.subscribe(MemoryEventType.CREATED, on_created)
emitter.subscribe(MemoryEventType.UPDATED, on_updated)

# Subscribe to multiple types
for event_type in MemoryEventType:
    emitter.subscribe(event_type, on_any)
```

### Unsubscribing

```python
# Returns subscription ID
sub_id = emitter.subscribe(MemoryEventType.CREATED, handler)

# Later, unsubscribe
emitter.unsubscribe(sub_id)
```

### Async Handlers

```python
import asyncio

async def async_handler(event):
    await asyncio.sleep(1)  # Some async operation
    await notify_external_service(event)

# Async handlers are supported
emitter.subscribe(MemoryEventType.CREATED, async_handler)
```

## Webhooks

### Basic Configuration

```python
from alma.events import WebhookConfig, WebhookManager, get_emitter

manager = WebhookManager()

# Add webhook endpoint
manager.add_webhook(WebhookConfig(
    url="https://your-app.com/alma-webhook",
    events=[MemoryEventType.CREATED, MemoryEventType.UPDATED],
    secret="your-secret-key"  # For signature verification
))

# Start webhook delivery
manager.start(get_emitter())
```

### WebhookConfig Options

```python
WebhookConfig(
    # Required
    url="https://your-app.com/webhook",
    events=[MemoryEventType.CREATED],

    # Authentication
    secret="hmac-secret",        # For X-ALMA-Signature header
    headers={                    # Custom headers
        "Authorization": "Bearer token"
    },

    # Retry configuration
    retry_count=3,               # Max retries (default: 3)
    retry_delay=5.0,             # Initial delay in seconds
    retry_backoff=2.0,           # Backoff multiplier

    # Filtering
    agent_filter=["agent1"],     # Only these agents
    project_filter=["proj1"],    # Only these projects

    # Delivery
    timeout=30.0,                # Request timeout in seconds
    batch_size=10,               # Max events per request
    batch_delay=1.0              # Wait time to batch events
)
```

### Webhook Payload

```json
{
  "event_type": "CREATED",
  "memory_id": "550e8400-e29b-41d4-a716-446655440000",
  "memory_type": "heuristic",
  "agent": "helena",
  "project_id": "my-project",
  "timestamp": "2026-01-28T12:00:00Z",
  "data": {
    "strategy": "Use incremental validation for large forms",
    "category": "testing_strategies",
    "confidence": 0.85
  }
}
```

### Signature Verification

ALMA signs webhook payloads using HMAC-SHA256:

```
X-ALMA-Signature: sha256=<hex-encoded-signature>
```

Verify in your webhook handler:

```python
import hmac
import hashlib

def verify_signature(payload: bytes, signature: str, secret: str) -> bool:
    expected = 'sha256=' + hmac.new(
        secret.encode(),
        payload,
        hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(signature, expected)

# In your webhook endpoint
@app.post("/alma-webhook")
async def webhook(request: Request):
    payload = await request.body()
    signature = request.headers.get("X-ALMA-Signature", "")

    if not verify_signature(payload, signature, WEBHOOK_SECRET):
        return Response(status_code=401)

    event = json.loads(payload)
    # Process event...
```

### Multiple Webhooks

```python
manager = WebhookManager()

# Different endpoints for different events
manager.add_webhook(WebhookConfig(
    url="https://analytics.example.com/events",
    events=[MemoryEventType.CREATED, MemoryEventType.UPDATED],
    secret="analytics-secret"
))

manager.add_webhook(WebhookConfig(
    url="https://alerts.example.com/alma",
    events=[MemoryEventType.DELETED],  # Alert on deletions
    secret="alerts-secret"
))

manager.add_webhook(WebhookConfig(
    url="https://backup.example.com/sync",
    events=list(MemoryEventType),  # All events
    secret="backup-secret"
))

manager.start(get_emitter())
```

### Webhook Delivery Status

```python
from alma.events import WebhookDeliveryStatus

# After processing
for delivery in manager.get_recent_deliveries():
    print(f"URL: {delivery.url}")
    print(f"Status: {delivery.status}")  # SUCCESS, FAILED, PENDING
    print(f"Attempts: {delivery.attempts}")
    print(f"Last error: {delivery.last_error}")
```

### Retry Logic

Failed deliveries are retried with exponential backoff:

```
Attempt 1: Immediate
Attempt 2: retry_delay seconds
Attempt 3: retry_delay * retry_backoff seconds
Attempt 4: retry_delay * retry_backoff^2 seconds
...
```

Default: 3 retries with 5s initial delay and 2x backoff.

## Event-Aware Storage

Automatically emit events when using storage backends:

```python
from alma.events import EventAwareStorageMixin, get_emitter
from alma.storage import SQLiteStorage

# Storage that emits events
class EventAwareSQLite(EventAwareStorageMixin, SQLiteStorage):
    pass

storage = EventAwareSQLite(config)
storage.set_emitter(get_emitter())

# Now all save operations emit events
storage.save_heuristic(heuristic)  # Emits CREATED event
```

### Using the Decorator

```python
from alma.events import emit_on_save

class MyStorage(StorageBackend):
    @emit_on_save(MemoryEventType.CREATED, memory_type='heuristic')
    def save_heuristic(self, heuristic):
        # Your implementation
        return heuristic.id
```

## Common Patterns

### Analytics Tracking

```python
def track_metrics(event):
    metrics.increment('alma.memory.created', tags={
        'agent': event.agent,
        'type': event.memory_type
    })

emitter.subscribe(MemoryEventType.CREATED, track_metrics)
```

### Cache Invalidation

```python
def invalidate_cache(event):
    cache_key = f"memories:{event.agent}:{event.project_id}"
    cache.delete(cache_key)

emitter.subscribe(MemoryEventType.CREATED, invalidate_cache)
emitter.subscribe(MemoryEventType.UPDATED, invalidate_cache)
emitter.subscribe(MemoryEventType.DELETED, invalidate_cache)
```

### Audit Logging

```python
def audit_log(event):
    audit_db.insert({
        'event_type': event.event_type.value,
        'memory_id': event.memory_id,
        'agent': event.agent,
        'timestamp': event.timestamp,
        'data': event.data
    })

for event_type in MemoryEventType:
    emitter.subscribe(event_type, audit_log)
```

### Real-time Dashboard

```python
import socketio

sio = socketio.AsyncServer()

async def broadcast_event(event):
    await sio.emit('alma_event', {
        'type': event.event_type.value,
        'agent': event.agent,
        'memory': event.data
    })

emitter.subscribe(MemoryEventType.CREATED, broadcast_event)
```

## Testing

### Resetting the Emitter

```python
from alma.events import reset_emitter

def test_my_handler():
    # Fresh emitter for each test
    reset_emitter()
    emitter = get_emitter()

    received = []
    emitter.subscribe(MemoryEventType.CREATED, lambda e: received.append(e))

    # Trigger your code...

    assert len(received) == 1
```

### Mocking Webhooks

```python
from unittest.mock import patch

def test_webhook_delivery():
    manager = WebhookManager()
    manager.add_webhook(WebhookConfig(
        url="https://test.example.com",
        events=[MemoryEventType.CREATED]
    ))

    with patch('aiohttp.ClientSession.post') as mock_post:
        mock_post.return_value.__aenter__.return_value.status = 200

        # Trigger event...

        assert mock_post.called
```

## Best Practices

1. **Idempotent handlers** - Events may be delivered more than once
2. **Fast callbacks** - Keep in-process handlers quick; offload heavy work
3. **Verify signatures** - Always verify webhook signatures in production
4. **Monitor delivery** - Track webhook success/failure rates
5. **Set reasonable timeouts** - Prevent hanging requests
6. **Use filtering** - Only subscribe to events you need
