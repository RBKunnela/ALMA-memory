"""
ALMA Webhook Delivery.

Provides webhook delivery capabilities for external system integration.
Webhooks are delivered asynchronously with retry logic and signature
verification support.
"""

import asyncio
import hashlib
import hmac
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum

# TYPE_CHECKING import for forward references
from typing import TYPE_CHECKING, Dict, List, Optional

from alma.events.types import MemoryEvent, MemoryEventType

if TYPE_CHECKING:
    from alma.events.emitter import EventEmitter

logger = logging.getLogger(__name__)

# Try to import aiohttp, provide fallback warning
try:
    import aiohttp

    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    logger.warning("aiohttp not installed - webhook delivery will be unavailable")


class WebhookDeliveryStatus(Enum):
    """Status of webhook delivery attempt."""

    SUCCESS = "success"
    FAILED = "failed"
    RETRYING = "retrying"
    PENDING = "pending"


@dataclass
class WebhookConfig:
    """
    Configuration for a webhook endpoint.

    Attributes:
        url: The URL to send webhook payloads to
        events: List of event types to deliver (empty = all)
        secret: Optional secret for HMAC signature generation
        max_retries: Maximum number of retry attempts
        timeout_seconds: Request timeout in seconds
        headers: Optional additional headers to include
    """

    url: str
    events: List[MemoryEventType] = field(default_factory=list)
    secret: Optional[str] = None
    max_retries: int = 3
    timeout_seconds: int = 10
    headers: Dict[str, str] = field(default_factory=dict)

    def matches_event(self, event_type: MemoryEventType) -> bool:
        """Check if this webhook should receive the given event type."""
        if not self.events:
            return True  # Empty list means all events
        return event_type in self.events


@dataclass
class WebhookDeliveryResult:
    """Result of a webhook delivery attempt."""

    config: WebhookConfig
    event: MemoryEvent
    status: WebhookDeliveryStatus
    status_code: Optional[int] = None
    attempts: int = 0
    error: Optional[str] = None
    response_body: Optional[str] = None


class WebhookDelivery:
    """
    Handles webhook delivery with retry logic and signature generation.

    Features:
    - HMAC-SHA256 signature for payload verification
    - Exponential backoff retry logic
    - Async delivery to avoid blocking
    - Configurable timeouts

    Example:
        ```python
        configs = [
            WebhookConfig(
                url="https://example.com/webhook",
                events=[MemoryEventType.CREATED],
                secret="my-secret"
            )
        ]
        delivery = WebhookDelivery(configs)
        results = await delivery.deliver(event)
        ```
    """

    def __init__(self, configs: List[WebhookConfig]):
        """
        Initialize webhook delivery.

        Args:
            configs: List of webhook configurations
        """
        self.configs = configs

    async def deliver(self, event: MemoryEvent) -> List[WebhookDeliveryResult]:
        """
        Deliver event to all matching webhooks.

        Args:
            event: The event to deliver

        Returns:
            List of delivery results for each webhook
        """
        if not AIOHTTP_AVAILABLE:
            logger.error("Cannot deliver webhooks: aiohttp not installed")
            return []

        # Filter to matching webhooks
        matching_configs = [
            config for config in self.configs if config.matches_event(event.event_type)
        ]

        if not matching_configs:
            return []

        # Deliver to all matching webhooks concurrently
        tasks = [self._send_webhook(config, event) for config in matching_configs]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle any exceptions that occurred
        delivery_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                delivery_results.append(
                    WebhookDeliveryResult(
                        config=matching_configs[i],
                        event=event,
                        status=WebhookDeliveryStatus.FAILED,
                        error=str(result),
                    )
                )
            else:
                delivery_results.append(result)

        return delivery_results

    def _sign_payload(self, payload: str, secret: str) -> str:
        """
        Sign payload with HMAC-SHA256.

        Args:
            payload: JSON payload string
            secret: Secret key for signing

        Returns:
            Hexadecimal signature string
        """
        return hmac.new(
            secret.encode("utf-8"),
            payload.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    def _build_headers(
        self,
        config: WebhookConfig,
        payload: str,
        timestamp: int,
    ) -> Dict[str, str]:
        """Build headers for webhook request."""
        headers = {
            "Content-Type": "application/json",
            "X-ALMA-Event-Type": payload,  # Will be replaced with actual event type
            "X-ALMA-Timestamp": str(timestamp),
            **config.headers,
        }

        if config.secret:
            # Create signature from timestamp + payload
            signature_payload = f"{timestamp}.{payload}"
            signature = self._sign_payload(signature_payload, config.secret)
            headers["X-ALMA-Signature"] = f"sha256={signature}"

        return headers

    async def _send_webhook(
        self,
        config: WebhookConfig,
        event: MemoryEvent,
    ) -> WebhookDeliveryResult:
        """
        Send a single webhook with retry logic.

        Uses exponential backoff for retries:
        - Attempt 1: immediate
        - Attempt 2: 1 second delay
        - Attempt 3: 2 second delay
        - Attempt 4: 4 second delay

        Args:
            config: Webhook configuration
            event: Event to deliver

        Returns:
            Delivery result
        """
        payload = json.dumps(event.to_dict(), default=str)
        timestamp = int(time.time())

        headers = {
            "Content-Type": "application/json",
            "X-ALMA-Event-Type": event.event_type.value,
            "X-ALMA-Timestamp": str(timestamp),
            **config.headers,
        }

        if config.secret:
            signature_payload = f"{timestamp}.{payload}"
            signature = self._sign_payload(signature_payload, config.secret)
            headers["X-ALMA-Signature"] = f"sha256={signature}"

        attempts = 0
        last_error = None
        last_status_code = None

        async with aiohttp.ClientSession() as session:
            for attempt in range(config.max_retries + 1):
                attempts = attempt + 1

                try:
                    async with session.post(
                        config.url,
                        data=payload,
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=config.timeout_seconds),
                    ) as response:
                        last_status_code = response.status

                        if 200 <= response.status < 300:
                            logger.info(
                                f"Webhook delivered successfully to {config.url} "
                                f"(attempt {attempts})"
                            )
                            return WebhookDeliveryResult(
                                config=config,
                                event=event,
                                status=WebhookDeliveryStatus.SUCCESS,
                                status_code=response.status,
                                attempts=attempts,
                            )

                        # Non-2xx response
                        response_body = await response.text()
                        last_error = f"HTTP {response.status}: {response_body[:200]}"
                        logger.warning(
                            f"Webhook delivery failed to {config.url}: {last_error}"
                        )

                except asyncio.TimeoutError:
                    last_error = f"Timeout after {config.timeout_seconds} seconds"
                    logger.warning(
                        f"Webhook delivery timeout to {config.url} (attempt {attempts})"
                    )

                except aiohttp.ClientError as e:
                    last_error = str(e)
                    logger.warning(
                        f"Webhook delivery error to {config.url}: {e} (attempt {attempts})"
                    )

                # Calculate backoff for next attempt
                if attempt < config.max_retries:
                    backoff = 2**attempt  # 1, 2, 4 seconds
                    await asyncio.sleep(backoff)

        # All retries exhausted
        logger.error(
            f"Webhook delivery failed after {attempts} attempts to {config.url}"
        )
        return WebhookDeliveryResult(
            config=config,
            event=event,
            status=WebhookDeliveryStatus.FAILED,
            status_code=last_status_code,
            attempts=attempts,
            error=last_error,
        )

    def add_config(self, config: WebhookConfig) -> None:
        """Add a webhook configuration."""
        self.configs.append(config)

    def remove_config(self, url: str) -> bool:
        """
        Remove a webhook configuration by URL.

        Args:
            url: URL of the webhook to remove

        Returns:
            True if removed, False if not found
        """
        for i, config in enumerate(self.configs):
            if config.url == url:
                self.configs.pop(i)
                return True
        return False


class WebhookManager:
    """
    High-level manager for webhook delivery integrated with the event emitter.

    Automatically subscribes to the event emitter and delivers webhooks
    for configured events.

    Example:
        ```python
        from alma.events import get_emitter
        from alma.events.webhook import WebhookManager, WebhookConfig

        manager = WebhookManager()
        manager.add_webhook(WebhookConfig(
            url="https://example.com/webhook",
            secret="my-secret"
        ))
        manager.start(get_emitter())
        ```
    """

    def __init__(self):
        """Initialize the webhook manager."""
        self._configs: List[WebhookConfig] = []
        self._delivery: Optional[WebhookDelivery] = None
        self._running = False

    def add_webhook(self, config: WebhookConfig) -> None:
        """Add a webhook configuration."""
        self._configs.append(config)
        if self._delivery:
            self._delivery.add_config(config)

    def remove_webhook(self, url: str) -> bool:
        """Remove a webhook by URL."""
        for i, config in enumerate(self._configs):
            if config.url == url:
                self._configs.pop(i)
                if self._delivery:
                    self._delivery.remove_config(url)
                return True
        return False

    def start(self, emitter: "EventEmitter") -> None:
        """
        Start the webhook manager.

        Subscribes to all events from the emitter.

        Args:
            emitter: The event emitter to subscribe to
        """

        if self._running:
            return

        self._delivery = WebhookDelivery(self._configs)
        emitter.subscribe_all(self._on_event)
        self._running = True
        logger.info(f"Webhook manager started with {len(self._configs)} webhooks")

    def stop(self, emitter: "EventEmitter") -> None:
        """
        Stop the webhook manager.

        Args:
            emitter: The event emitter to unsubscribe from
        """

        if not self._running:
            return

        emitter.unsubscribe_all(self._on_event)
        self._running = False
        logger.info("Webhook manager stopped")

    def _on_event(self, event: MemoryEvent) -> None:
        """Handle incoming events by delivering webhooks."""
        if not self._delivery or not AIOHTTP_AVAILABLE:
            return

        # Run async delivery in a new event loop if needed
        try:
            asyncio.get_running_loop()
            asyncio.create_task(self._delivery.deliver(event))
        except RuntimeError:
            # No running event loop, create one
            asyncio.run(self._delivery.deliver(event))

    @property
    def webhook_count(self) -> int:
        """Get the number of configured webhooks."""
        return len(self._configs)

    @property
    def is_running(self) -> bool:
        """Check if the manager is running."""
        return self._running
