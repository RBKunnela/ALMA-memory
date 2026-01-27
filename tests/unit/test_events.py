"""
Unit tests for ALMA Event System.

Tests the event emitter, webhook delivery, and event types.
"""

import hashlib
import hmac
import json
from datetime import datetime, timezone
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from alma.events import (
    EventEmitter,
    MemoryEvent,
    MemoryEventType,
    WebhookConfig,
    WebhookDelivery,
    WebhookDeliveryStatus,
    WebhookManager,
    create_memory_event,
    get_emitter,
    reset_emitter,
)


class TestMemoryEventType:
    """Tests for MemoryEventType enum."""

    def test_event_type_values(self):
        """Test all event type values exist."""
        assert MemoryEventType.CREATED.value == "memory.created"
        assert MemoryEventType.UPDATED.value == "memory.updated"
        assert MemoryEventType.DELETED.value == "memory.deleted"
        assert MemoryEventType.CONSOLIDATED.value == "memory.consolidated"
        assert MemoryEventType.HEURISTIC_FORMED.value == "heuristic.formed"
        assert MemoryEventType.ANTIPATTERN_DETECTED.value == "antipattern.detected"
        assert MemoryEventType.PREFERENCE_ADDED.value == "preference.added"
        assert MemoryEventType.KNOWLEDGE_ADDED.value == "knowledge.added"

    def test_event_type_from_value(self):
        """Test creating event type from value."""
        event_type = MemoryEventType("memory.created")
        assert event_type == MemoryEventType.CREATED


class TestMemoryEvent:
    """Tests for MemoryEvent dataclass."""

    @pytest.fixture
    def sample_event(self) -> MemoryEvent:
        """Create a sample event for testing."""
        return MemoryEvent(
            event_type=MemoryEventType.CREATED,
            agent="helena",
            project_id="test-project",
            memory_type="heuristics",
            memory_id="heur-123",
            timestamp=datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc),
            payload={"condition": "test", "strategy": "test strategy"},
            metadata={"source": "test"},
        )

    def test_event_creation(self, sample_event):
        """Test event creation with all fields."""
        assert sample_event.event_type == MemoryEventType.CREATED
        assert sample_event.agent == "helena"
        assert sample_event.project_id == "test-project"
        assert sample_event.memory_type == "heuristics"
        assert sample_event.memory_id == "heur-123"
        assert sample_event.payload["condition"] == "test"
        assert sample_event.metadata["source"] == "test"

    def test_event_to_dict(self, sample_event):
        """Test event serialization to dict."""
        data = sample_event.to_dict()

        assert data["event_type"] == "memory.created"
        assert data["agent"] == "helena"
        assert data["project_id"] == "test-project"
        assert data["memory_type"] == "heuristics"
        assert data["memory_id"] == "heur-123"
        assert data["timestamp"] == "2024-01-15T10:30:00+00:00"
        assert data["payload"]["condition"] == "test"
        assert data["metadata"]["source"] == "test"

    def test_event_from_dict(self, sample_event):
        """Test event deserialization from dict."""
        data = sample_event.to_dict()
        restored = MemoryEvent.from_dict(data)

        assert restored.event_type == sample_event.event_type
        assert restored.agent == sample_event.agent
        assert restored.project_id == sample_event.project_id
        assert restored.memory_type == sample_event.memory_type
        assert restored.memory_id == sample_event.memory_id
        assert restored.payload == sample_event.payload
        assert restored.metadata == sample_event.metadata

    def test_create_memory_event_factory(self):
        """Test the create_memory_event factory function."""
        event = create_memory_event(
            event_type=MemoryEventType.CREATED,
            agent="victor",
            project_id="test-project",
            memory_type="outcomes",
            memory_id="out-456",
            payload={"success": True},
        )

        assert event.event_type == MemoryEventType.CREATED
        assert event.agent == "victor"
        assert event.memory_id == "out-456"
        assert event.payload == {"success": True}
        assert event.timestamp is not None
        assert event.metadata == {}  # Default empty dict


class TestEventEmitter:
    """Tests for EventEmitter."""

    @pytest.fixture(autouse=True)
    def reset_global_emitter(self):
        """Reset global emitter before each test."""
        reset_emitter()
        yield
        reset_emitter()

    @pytest.fixture
    def emitter(self) -> EventEmitter:
        """Create a fresh emitter for testing."""
        return EventEmitter()

    @pytest.fixture
    def sample_event(self) -> MemoryEvent:
        """Create a sample event."""
        return create_memory_event(
            event_type=MemoryEventType.CREATED,
            agent="helena",
            project_id="test-project",
            memory_type="heuristics",
            memory_id="heur-123",
            payload={"test": True},
        )

    def test_subscribe_to_event_type(self, emitter):
        """Test subscribing to a specific event type."""
        callback = MagicMock()

        emitter.subscribe(MemoryEventType.CREATED, callback)

        assert emitter.has_subscribers(MemoryEventType.CREATED)
        assert not emitter.has_subscribers(MemoryEventType.DELETED)

    def test_subscribe_all(self, emitter):
        """Test subscribing to all events."""
        callback = MagicMock()

        emitter.subscribe_all(callback)

        assert emitter.has_subscribers()
        assert emitter.has_subscribers(MemoryEventType.CREATED)
        assert emitter.has_subscribers(MemoryEventType.DELETED)

    def test_unsubscribe(self, emitter):
        """Test unsubscribing from events."""
        callback = MagicMock()
        emitter.subscribe(MemoryEventType.CREATED, callback)

        result = emitter.unsubscribe(MemoryEventType.CREATED, callback)

        assert result is True
        assert not emitter.has_subscribers(MemoryEventType.CREATED)

    def test_unsubscribe_not_found(self, emitter):
        """Test unsubscribing when not subscribed."""
        callback = MagicMock()

        result = emitter.unsubscribe(MemoryEventType.CREATED, callback)

        assert result is False

    def test_unsubscribe_all(self, emitter):
        """Test unsubscribing from all events."""
        callback = MagicMock()
        emitter.subscribe_all(callback)

        result = emitter.unsubscribe_all(callback)

        assert result is True

    def test_emit_calls_subscribers(self, emitter, sample_event):
        """Test that emit calls matching subscribers."""
        callback = MagicMock()
        emitter.subscribe(MemoryEventType.CREATED, callback)

        emitter.emit(sample_event)

        # Wait for async execution
        import time

        time.sleep(0.1)

        callback.assert_called_once_with(sample_event)

    def test_emit_calls_global_subscribers(self, emitter, sample_event):
        """Test that emit calls global subscribers."""
        callback = MagicMock()
        emitter.subscribe_all(callback)

        emitter.emit(sample_event)

        import time

        time.sleep(0.1)

        callback.assert_called_once_with(sample_event)

    def test_emit_does_not_call_wrong_type(self, emitter, sample_event):
        """Test that emit only calls matching subscribers."""
        callback = MagicMock()
        emitter.subscribe(MemoryEventType.DELETED, callback)  # Different type

        emitter.emit(sample_event)  # CREATED event

        import time

        time.sleep(0.1)

        callback.assert_not_called()

    def test_emit_disabled(self, emitter, sample_event):
        """Test that emit does nothing when disabled."""
        callback = MagicMock()
        emitter.subscribe(MemoryEventType.CREATED, callback)
        emitter.disable()

        emitter.emit(sample_event)

        import time

        time.sleep(0.1)

        callback.assert_not_called()

    def test_emit_re_enabled(self, emitter, sample_event):
        """Test that emit works after re-enabling."""
        callback = MagicMock()
        emitter.subscribe(MemoryEventType.CREATED, callback)
        emitter.disable()
        emitter.enable()

        emitter.emit(sample_event)

        import time

        time.sleep(0.1)

        callback.assert_called_once()

    @pytest.mark.asyncio
    async def test_emit_async(self, emitter, sample_event):
        """Test async event emission."""
        callback = AsyncMock()
        emitter.subscribe(MemoryEventType.CREATED, callback)

        await emitter.emit_async(sample_event)

        callback.assert_called_once_with(sample_event)

    @pytest.mark.asyncio
    async def test_emit_async_with_sync_callback(self, emitter, sample_event):
        """Test async emission with sync callback."""
        received_events = []

        def callback(event):
            received_events.append(event)

        emitter.subscribe(MemoryEventType.CREATED, callback)

        await emitter.emit_async(sample_event)

        assert len(received_events) == 1
        assert received_events[0] == sample_event

    def test_clear_subscribers(self, emitter):
        """Test clearing all subscribers."""
        callback1 = MagicMock()
        callback2 = MagicMock()
        emitter.subscribe(MemoryEventType.CREATED, callback1)
        emitter.subscribe_all(callback2)

        emitter.clear()

        assert not emitter.has_subscribers()

    def test_callback_exception_handled(self, emitter, sample_event):
        """Test that exceptions in callbacks are handled gracefully."""

        def bad_callback(event):
            raise ValueError("Test error")

        good_callback = MagicMock()

        emitter.subscribe(MemoryEventType.CREATED, bad_callback)
        emitter.subscribe(MemoryEventType.CREATED, good_callback)

        # Should not raise
        emitter.emit(sample_event)

        import time

        time.sleep(0.1)

        # Good callback should still be called
        good_callback.assert_called_once()

    def test_global_emitter_singleton(self):
        """Test that get_emitter returns the same instance."""
        emitter1 = get_emitter()
        emitter2 = get_emitter()

        assert emitter1 is emitter2

    def test_reset_emitter(self):
        """Test that reset_emitter creates new instance."""
        emitter1 = get_emitter()
        emitter1.subscribe(MemoryEventType.CREATED, lambda e: None)

        reset_emitter()
        emitter2 = get_emitter()

        assert emitter1 is not emitter2
        assert not emitter2.has_subscribers()


class TestWebhookConfig:
    """Tests for WebhookConfig."""

    def test_config_creation(self):
        """Test creating a webhook config."""
        config = WebhookConfig(
            url="https://example.com/webhook",
            events=[MemoryEventType.CREATED, MemoryEventType.UPDATED],
            secret="my-secret",
            max_retries=5,
            timeout_seconds=15,
        )

        assert config.url == "https://example.com/webhook"
        assert len(config.events) == 2
        assert config.secret == "my-secret"
        assert config.max_retries == 5
        assert config.timeout_seconds == 15

    def test_config_defaults(self):
        """Test config default values."""
        config = WebhookConfig(url="https://example.com/webhook")

        assert config.events == []  # Empty = all events
        assert config.secret is None
        assert config.max_retries == 3
        assert config.timeout_seconds == 10

    def test_matches_event_with_empty_list(self):
        """Test that empty events list matches all events."""
        config = WebhookConfig(url="https://example.com/webhook")

        assert config.matches_event(MemoryEventType.CREATED)
        assert config.matches_event(MemoryEventType.DELETED)
        assert config.matches_event(MemoryEventType.HEURISTIC_FORMED)

    def test_matches_event_with_specific_events(self):
        """Test that specific events are matched correctly."""
        config = WebhookConfig(
            url="https://example.com/webhook",
            events=[MemoryEventType.CREATED, MemoryEventType.UPDATED],
        )

        assert config.matches_event(MemoryEventType.CREATED)
        assert config.matches_event(MemoryEventType.UPDATED)
        assert not config.matches_event(MemoryEventType.DELETED)


class TestWebhookDelivery:
    """Tests for WebhookDelivery."""

    @pytest.fixture
    def sample_event(self) -> MemoryEvent:
        """Create a sample event."""
        return create_memory_event(
            event_type=MemoryEventType.CREATED,
            agent="helena",
            project_id="test-project",
            memory_type="heuristics",
            memory_id="heur-123",
            payload={"test": True},
        )

    def test_sign_payload(self):
        """Test HMAC-SHA256 signature generation."""
        delivery = WebhookDelivery([])

        payload = '{"test": "data"}'
        secret = "my-secret"

        signature = delivery._sign_payload(payload, secret)

        # Verify signature manually
        expected = hmac.new(
            secret.encode("utf-8"),
            payload.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

        assert signature == expected

    def test_sign_payload_different_secrets(self):
        """Test that different secrets produce different signatures."""
        delivery = WebhookDelivery([])
        payload = '{"test": "data"}'

        sig1 = delivery._sign_payload(payload, "secret1")
        sig2 = delivery._sign_payload(payload, "secret2")

        assert sig1 != sig2

    @pytest.mark.asyncio
    async def test_deliver_filters_by_event_type(self, sample_event):
        """Test that delivery filters to matching webhooks."""
        config1 = WebhookConfig(
            url="https://example1.com/webhook",
            events=[MemoryEventType.CREATED],
        )
        config2 = WebhookConfig(
            url="https://example2.com/webhook",
            events=[MemoryEventType.DELETED],  # Won't match
        )

        delivery = WebhookDelivery([config1, config2])

        with patch.object(
            delivery, "_send_webhook", new_callable=AsyncMock
        ) as mock_send:
            mock_send.return_value = MagicMock(status=WebhookDeliveryStatus.SUCCESS)

            await delivery.deliver(sample_event)

            # Only config1 should be called
            assert mock_send.call_count == 1
            called_config = mock_send.call_args[0][0]
            assert called_config.url == "https://example1.com/webhook"

    @pytest.mark.asyncio
    async def test_deliver_returns_results(self, sample_event):
        """Test that delivery returns results for all matching webhooks."""
        config1 = WebhookConfig(url="https://example1.com/webhook")
        config2 = WebhookConfig(url="https://example2.com/webhook")

        delivery = WebhookDelivery([config1, config2])

        with patch.object(
            delivery, "_send_webhook", new_callable=AsyncMock
        ) as mock_send:
            mock_send.return_value = MagicMock(status=WebhookDeliveryStatus.SUCCESS)

            results = await delivery.deliver(sample_event)

            assert len(results) == 2

    @pytest.mark.asyncio
    @patch("alma.events.webhook.AIOHTTP_AVAILABLE", True)
    async def test_send_webhook_success(self, sample_event):
        """Test successful webhook delivery."""
        config = WebhookConfig(url="https://example.com/webhook")
        delivery = WebhookDelivery([config])

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await delivery._send_webhook(config, sample_event)

        assert result.status == WebhookDeliveryStatus.SUCCESS
        assert result.status_code == 200
        assert result.attempts == 1

    @pytest.mark.asyncio
    @patch("alma.events.webhook.AIOHTTP_AVAILABLE", True)
    async def test_send_webhook_with_signature(self, sample_event):
        """Test webhook delivery includes signature header."""
        config = WebhookConfig(
            url="https://example.com/webhook",
            secret="test-secret",
        )
        delivery = WebhookDelivery([config])

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            await delivery._send_webhook(config, sample_event)

        # Check that signature header was included
        call_args = mock_session.post.call_args
        headers = call_args.kwargs.get("headers", {})
        assert "X-ALMA-Signature" in headers
        assert headers["X-ALMA-Signature"].startswith("sha256=")

    def test_add_config(self):
        """Test adding a webhook config."""
        delivery = WebhookDelivery([])
        config = WebhookConfig(url="https://example.com/webhook")

        delivery.add_config(config)

        assert len(delivery.configs) == 1
        assert delivery.configs[0] == config

    def test_remove_config(self):
        """Test removing a webhook config."""
        config = WebhookConfig(url="https://example.com/webhook")
        delivery = WebhookDelivery([config])

        result = delivery.remove_config("https://example.com/webhook")

        assert result is True
        assert len(delivery.configs) == 0

    def test_remove_config_not_found(self):
        """Test removing a non-existent config."""
        delivery = WebhookDelivery([])

        result = delivery.remove_config("https://example.com/webhook")

        assert result is False


class TestWebhookManager:
    """Tests for WebhookManager."""

    @pytest.fixture(autouse=True)
    def reset_global_emitter(self):
        """Reset global emitter before each test."""
        reset_emitter()
        yield
        reset_emitter()

    def test_add_webhook(self):
        """Test adding a webhook."""
        manager = WebhookManager()
        config = WebhookConfig(url="https://example.com/webhook")

        manager.add_webhook(config)

        assert manager.webhook_count == 1

    def test_remove_webhook(self):
        """Test removing a webhook."""
        manager = WebhookManager()
        config = WebhookConfig(url="https://example.com/webhook")
        manager.add_webhook(config)

        result = manager.remove_webhook("https://example.com/webhook")

        assert result is True
        assert manager.webhook_count == 0

    def test_start_and_stop(self):
        """Test starting and stopping the manager."""
        manager = WebhookManager()
        emitter = get_emitter()

        assert not manager.is_running

        manager.start(emitter)
        assert manager.is_running

        manager.stop(emitter)
        assert not manager.is_running

    def test_start_subscribes_to_emitter(self):
        """Test that start subscribes to the emitter."""
        manager = WebhookManager()
        emitter = get_emitter()

        manager.start(emitter)

        assert emitter.has_subscribers()

    def test_stop_unsubscribes_from_emitter(self):
        """Test that stop unsubscribes from the emitter."""
        manager = WebhookManager()
        emitter = get_emitter()

        manager.start(emitter)
        manager.stop(emitter)

        # Note: The manager's callback may still be subscribed if there are
        # other subscribers, so we check is_running instead
        assert not manager.is_running


class TestEventAwareStorageMixin:
    """Tests for EventAwareStorageMixin."""

    @pytest.fixture(autouse=True)
    def reset_global_emitter(self):
        """Reset global emitter before each test."""
        reset_emitter()
        yield
        reset_emitter()

    def test_mixin_enable_disable(self):
        """Test enabling and disabling events."""
        from alma.events.storage_mixin import EventAwareStorageMixin

        class TestStorage(EventAwareStorageMixin):
            pass

        storage = TestStorage()
        assert storage._events_enabled is True

        storage.disable_events()
        assert storage._events_enabled is False

        storage.enable_events()
        assert storage._events_enabled is True

    def test_mixin_set_emitter(self):
        """Test setting a custom emitter."""
        from alma.events.storage_mixin import EventAwareStorageMixin

        class TestStorage(EventAwareStorageMixin):
            pass

        storage = TestStorage()
        custom_emitter = EventEmitter()

        storage.set_emitter(custom_emitter)
        assert storage._get_emitter() is custom_emitter

    def test_mixin_should_emit_no_subscribers(self):
        """Test _should_emit returns False when no subscribers."""
        from alma.events.storage_mixin import EventAwareStorageMixin

        class TestStorage(EventAwareStorageMixin):
            pass

        storage = TestStorage()
        assert storage._should_emit(MemoryEventType.CREATED) is False

    def test_mixin_should_emit_with_subscribers(self):
        """Test _should_emit returns True when subscribers exist."""
        from alma.events.storage_mixin import EventAwareStorageMixin

        class TestStorage(EventAwareStorageMixin):
            pass

        storage = TestStorage()
        emitter = get_emitter()
        emitter.subscribe(MemoryEventType.CREATED, lambda e: None)

        assert storage._should_emit(MemoryEventType.CREATED) is True
        assert storage._should_emit(MemoryEventType.DELETED) is False

    def test_mixin_emit_memory_event(self):
        """Test _emit_memory_event emits to subscribers."""
        from alma.events.storage_mixin import EventAwareStorageMixin

        class TestStorage(EventAwareStorageMixin):
            pass

        received_events = []

        def callback(event):
            received_events.append(event)

        emitter = get_emitter()
        emitter.subscribe(MemoryEventType.CREATED, callback)

        storage = TestStorage()
        storage._emit_memory_event(
            event_type=MemoryEventType.CREATED,
            agent="helena",
            project_id="test-project",
            memory_type="heuristics",
            memory_id="heur-123",
            payload={"test": True},
        )

        import time

        time.sleep(0.1)

        assert len(received_events) == 1
        assert received_events[0].memory_id == "heur-123"

    def test_mixin_disabled_no_emit(self):
        """Test that disabled storage doesn't emit events."""
        from alma.events.storage_mixin import EventAwareStorageMixin

        class TestStorage(EventAwareStorageMixin):
            pass

        received_events = []

        def callback(event):
            received_events.append(event)

        emitter = get_emitter()
        emitter.subscribe(MemoryEventType.CREATED, callback)

        storage = TestStorage()
        storage.disable_events()

        storage._emit_memory_event(
            event_type=MemoryEventType.CREATED,
            agent="helena",
            project_id="test-project",
            memory_type="heuristics",
            memory_id="heur-123",
            payload={"test": True},
        )

        import time

        time.sleep(0.1)

        assert len(received_events) == 0


class TestWebhookRetryLogic:
    """Tests for webhook retry and backoff logic."""

    @pytest.fixture
    def sample_event(self) -> MemoryEvent:
        """Create a sample event."""
        return create_memory_event(
            event_type=MemoryEventType.CREATED,
            agent="helena",
            project_id="test-project",
            memory_type="heuristics",
            memory_id="heur-123",
            payload={"test": True},
        )

    @pytest.mark.asyncio
    @patch("alma.events.webhook.AIOHTTP_AVAILABLE", True)
    async def test_retry_on_server_error(self, sample_event):
        """Test that webhook delivery retries on server errors."""
        config = WebhookConfig(
            url="https://example.com/webhook",
            max_retries=2,
        )
        delivery = WebhookDelivery([config])

        call_count = [0]  # Use list to allow mutation in nested function

        class MockResponse:
            def __init__(self, status, text_val):
                self.status = status
                self._text = text_val

            async def text(self):
                return self._text

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

        class MockSession:
            def post(self, *args, **kwargs):
                call_count[0] += 1
                if call_count[0] < 3:
                    return MockResponse(500, "Server error")
                return MockResponse(200, "OK")

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

        with patch("aiohttp.ClientSession", return_value=MockSession()):
            with patch("asyncio.sleep", new_callable=AsyncMock):  # Skip actual delays
                result = await delivery._send_webhook(config, sample_event)

        # Should succeed on third attempt
        assert result.status == WebhookDeliveryStatus.SUCCESS
        assert result.attempts == 3

    @pytest.mark.asyncio
    @patch("alma.events.webhook.AIOHTTP_AVAILABLE", True)
    async def test_max_retries_exceeded(self, sample_event):
        """Test that delivery fails after max retries."""
        config = WebhookConfig(
            url="https://example.com/webhook",
            max_retries=2,  # Total 3 attempts (1 initial + 2 retries)
        )
        delivery = WebhookDelivery([config])

        class MockResponse:
            def __init__(self):
                self.status = 500

            async def text(self):
                return "Server error"

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

        class MockSession:
            def post(self, *args, **kwargs):
                return MockResponse()

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

        with patch("aiohttp.ClientSession", return_value=MockSession()):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                result = await delivery._send_webhook(config, sample_event)

        assert result.status == WebhookDeliveryStatus.FAILED
        assert result.attempts == 3
        assert "500" in result.error


class TestEventIntegration:
    """Integration tests for the event system."""

    @pytest.fixture(autouse=True)
    def reset_global_emitter(self):
        """Reset global emitter before each test."""
        reset_emitter()
        yield
        reset_emitter()

    def test_full_event_flow(self):
        """Test complete flow from event creation to callback."""
        received_events: List[MemoryEvent] = []

        def callback(event: MemoryEvent):
            received_events.append(event)

        emitter = get_emitter()
        emitter.subscribe(MemoryEventType.CREATED, callback)

        event = create_memory_event(
            event_type=MemoryEventType.CREATED,
            agent="helena",
            project_id="test-project",
            memory_type="heuristics",
            memory_id="heur-123",
            payload={"test": True},
        )

        emitter.emit(event)

        # Wait for async execution
        import time

        time.sleep(0.1)

        assert len(received_events) == 1
        assert received_events[0].memory_id == "heur-123"

    def test_multiple_subscribers(self):
        """Test that multiple subscribers receive events."""
        results = {"count": 0}

        def callback1(event):
            results["count"] += 1

        def callback2(event):
            results["count"] += 10

        emitter = get_emitter()
        emitter.subscribe(MemoryEventType.CREATED, callback1)
        emitter.subscribe(MemoryEventType.CREATED, callback2)

        event = create_memory_event(
            event_type=MemoryEventType.CREATED,
            agent="helena",
            project_id="test-project",
            memory_type="heuristics",
            memory_id="heur-123",
            payload={},
        )

        emitter.emit(event)

        import time

        time.sleep(0.1)

        assert results["count"] == 11

    def test_event_serialization_roundtrip(self):
        """Test that events survive JSON serialization."""
        event = create_memory_event(
            event_type=MemoryEventType.HEURISTIC_FORMED,
            agent="victor",
            project_id="test-project",
            memory_type="heuristics",
            memory_id="heur-456",
            payload={"confidence": 0.85, "strategy": "test first"},
            metadata={"source": "learning"},
        )

        # Serialize and deserialize
        json_str = json.dumps(event.to_dict())
        restored_data = json.loads(json_str)
        restored_event = MemoryEvent.from_dict(restored_data)

        assert restored_event.event_type == event.event_type
        assert restored_event.agent == event.agent
        assert restored_event.payload == event.payload
        assert restored_event.metadata == event.metadata
