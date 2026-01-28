"""
ALMA Metrics Collection.

Provides metrics collection using OpenTelemetry with fallback
to in-memory collection when OTel is not available.

Metrics tracked:
- Memory operation latency (retrieve, learn, forget)
- Embedding generation time
- Cache hit/miss rates
- Storage backend query times
- Memory counts by type
"""

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Try to import OpenTelemetry
_otel_available = False
try:
    from opentelemetry import metrics

    _otel_available = True
except ImportError:
    pass

# Global metrics instance
_metrics_instance: Optional["ALMAMetrics"] = None
_metrics_lock = threading.Lock()


@dataclass
class MetricValue:
    """Container for metric values with metadata."""

    name: str
    value: float
    timestamp: float
    labels: Dict[str, str] = field(default_factory=dict)


class InMemoryMetricsCollector:
    """
    In-memory metrics collection for when OpenTelemetry is not available.

    Stores metric values in memory for later retrieval.
    """

    def __init__(self, max_samples: int = 10000):
        """Initialize in-memory collector."""
        self._counters: Dict[str, float] = {}
        self._histograms: Dict[str, List[float]] = {}
        self._gauges: Dict[str, float] = {}
        self._lock = threading.RLock()
        self._max_samples = max_samples

    def increment_counter(
        self,
        name: str,
        value: float = 1.0,
        labels: Optional[Dict[str, str]] = None,
    ):
        """Increment a counter metric."""
        key = self._make_key(name, labels)
        with self._lock:
            self._counters[key] = self._counters.get(key, 0.0) + value

    def record_histogram(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ):
        """Record a histogram value."""
        key = self._make_key(name, labels)
        with self._lock:
            if key not in self._histograms:
                self._histograms[key] = []
            self._histograms[key].append(value)
            # Trim if needed
            if len(self._histograms[key]) > self._max_samples:
                self._histograms[key] = self._histograms[key][-self._max_samples :]

    def set_gauge(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ):
        """Set a gauge value."""
        key = self._make_key(name, labels)
        with self._lock:
            self._gauges[key] = value

    def increment_gauge(
        self,
        name: str,
        value: float = 1.0,
        labels: Optional[Dict[str, str]] = None,
    ):
        """Increment a gauge (up-down counter)."""
        key = self._make_key(name, labels)
        with self._lock:
            self._gauges[key] = self._gauges.get(key, 0.0) + value

    def _make_key(
        self,
        name: str,
        labels: Optional[Dict[str, str]] = None,
    ) -> str:
        """Create a unique key for the metric with labels."""
        if not labels:
            return name
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"

    def get_counter(self, name: str, labels: Optional[Dict[str, str]] = None) -> float:
        """Get counter value."""
        key = self._make_key(name, labels)
        with self._lock:
            return self._counters.get(key, 0.0)

    def get_histogram_stats(
        self,
        name: str,
        labels: Optional[Dict[str, str]] = None,
    ) -> Dict[str, float]:
        """Get histogram statistics."""
        key = self._make_key(name, labels)
        with self._lock:
            values = self._histograms.get(key, [])
            if not values:
                return {
                    "count": 0,
                    "sum": 0,
                    "min": 0,
                    "max": 0,
                    "avg": 0,
                    "p50": 0,
                    "p95": 0,
                    "p99": 0,
                }

            sorted_values = sorted(values)
            count = len(sorted_values)
            return {
                "count": count,
                "sum": sum(sorted_values),
                "min": sorted_values[0],
                "max": sorted_values[-1],
                "avg": sum(sorted_values) / count,
                "p50": sorted_values[int(count * 0.5)],
                "p95": sorted_values[min(int(count * 0.95), count - 1)],
                "p99": sorted_values[min(int(count * 0.99), count - 1)],
            }

    def get_gauge(self, name: str, labels: Optional[Dict[str, str]] = None) -> float:
        """Get gauge value."""
        key = self._make_key(name, labels)
        with self._lock:
            return self._gauges.get(key, 0.0)

    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all metrics as a dictionary."""
        with self._lock:
            return {
                "counters": dict(self._counters),
                "histograms": {
                    k: self.get_histogram_stats(k.split("{")[0])
                    for k in self._histograms
                },
                "gauges": dict(self._gauges),
            }

    def reset(self):
        """Reset all metrics."""
        with self._lock:
            self._counters.clear()
            self._histograms.clear()
            self._gauges.clear()


class MetricsCollector:
    """
    Unified metrics collector that uses OpenTelemetry when available,
    falling back to in-memory collection otherwise.
    """

    def __init__(
        self,
        service_name: str = "alma-memory",
        use_otel: bool = True,
    ):
        """
        Initialize metrics collector.

        Args:
            service_name: Service name for metrics
            use_otel: Whether to use OpenTelemetry (if available)
        """
        self.service_name = service_name
        self._use_otel = use_otel and _otel_available
        self._fallback = InMemoryMetricsCollector()

        # OpenTelemetry instruments
        self._otel_counters: Dict[str, Any] = {}
        self._otel_histograms: Dict[str, Any] = {}
        self._otel_gauges: Dict[str, Any] = {}

        if self._use_otel:
            self._meter = metrics.get_meter(service_name)

    def counter(
        self,
        name: str,
        value: float = 1.0,
        labels: Optional[Dict[str, str]] = None,
    ):
        """Increment a counter metric."""
        if self._use_otel:
            if name not in self._otel_counters:
                self._otel_counters[name] = self._meter.create_counter(
                    name=f"alma.{name}",
                    description=f"ALMA counter: {name}",
                )
            self._otel_counters[name].add(value, labels or {})
        else:
            self._fallback.increment_counter(name, value, labels)

    def histogram(
        self,
        name: str,
        value: float,
        unit: str = "ms",
        labels: Optional[Dict[str, str]] = None,
    ):
        """Record a histogram value (typically latency)."""
        if self._use_otel:
            if name not in self._otel_histograms:
                self._otel_histograms[name] = self._meter.create_histogram(
                    name=f"alma.{name}",
                    unit=unit,
                    description=f"ALMA histogram: {name}",
                )
            self._otel_histograms[name].record(value, labels or {})
        else:
            self._fallback.record_histogram(name, value, labels)

    def gauge(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ):
        """Set a gauge value."""
        if self._use_otel:
            # OTel gauges require callbacks, so we use up-down counter
            if name not in self._otel_gauges:
                self._otel_gauges[name] = self._meter.create_up_down_counter(
                    name=f"alma.{name}",
                    description=f"ALMA gauge: {name}",
                )
            # Note: OTel up-down counters don't support setting absolute values
            # We track the last value and adjust
            pass
        # Always use fallback for gauges to support get operations
        self._fallback.set_gauge(name, value, labels)

    def gauge_increment(
        self,
        name: str,
        value: float = 1.0,
        labels: Optional[Dict[str, str]] = None,
    ):
        """Increment (or decrement if negative) a gauge."""
        if self._use_otel:
            if name not in self._otel_gauges:
                self._otel_gauges[name] = self._meter.create_up_down_counter(
                    name=f"alma.{name}",
                    description=f"ALMA gauge: {name}",
                )
            self._otel_gauges[name].add(value, labels or {})
        self._fallback.increment_gauge(name, value, labels)

    def get_stats(self) -> Dict[str, Any]:
        """Get all metrics as a dictionary (from fallback collector)."""
        return self._fallback.get_all_metrics()

    def timer(self, name: str, labels: Optional[Dict[str, str]] = None) -> "Timer":
        """Create a timer context manager for measuring duration."""
        return Timer(self, name, labels)


class Timer:
    """Context manager for timing operations."""

    def __init__(
        self,
        collector: MetricsCollector,
        name: str,
        labels: Optional[Dict[str, str]] = None,
    ):
        self._collector = collector
        self._name = name
        self._labels = labels
        self._start_time: Optional[float] = None
        self.duration_ms: float = 0

    def __enter__(self) -> "Timer":
        self._start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._start_time is not None:
            self.duration_ms = (time.time() - self._start_time) * 1000
            labels = dict(self._labels or {})
            labels["success"] = "false" if exc_type else "true"
            self._collector.histogram(self._name, self.duration_ms, "ms", labels)
        return False


class ALMAMetrics:
    """
    High-level metrics interface for ALMA operations.

    Provides semantic methods for tracking ALMA-specific metrics.
    """

    def __init__(self, collector: Optional[MetricsCollector] = None):
        """Initialize ALMA metrics."""
        self._collector = collector or MetricsCollector()

    @property
    def collector(self) -> MetricsCollector:
        """Get underlying metrics collector."""
        return self._collector

    # ==================== Memory Operations ====================

    def record_retrieve_latency(
        self,
        duration_ms: float,
        agent: str,
        project_id: str,
        cache_hit: bool,
        items_returned: int,
    ):
        """Record memory retrieval latency."""
        self._collector.histogram(
            "memory.retrieve.latency",
            duration_ms,
            "ms",
            {
                "agent": agent,
                "project_id": project_id,
                "cache_hit": str(cache_hit).lower(),
            },
        )
        self._collector.counter(
            "memory.retrieve.count",
            1,
            {"agent": agent, "project_id": project_id},
        )
        self._collector.counter(
            "memory.retrieve.items",
            items_returned,
            {"agent": agent, "project_id": project_id},
        )

    def record_learn_operation(
        self,
        duration_ms: float,
        agent: str,
        project_id: str,
        memory_type: str,
        success: bool,
    ):
        """Record a learning operation."""
        self._collector.histogram(
            "memory.learn.latency",
            duration_ms,
            "ms",
            {
                "agent": agent,
                "project_id": project_id,
                "memory_type": memory_type,
                "success": str(success).lower(),
            },
        )
        self._collector.counter(
            "memory.learn.count",
            1,
            {
                "agent": agent,
                "memory_type": memory_type,
                "success": str(success).lower(),
            },
        )

    def record_forget_operation(
        self,
        duration_ms: float,
        agent: Optional[str],
        project_id: str,
        items_removed: int,
    ):
        """Record a forget (pruning) operation."""
        self._collector.histogram(
            "memory.forget.latency",
            duration_ms,
            "ms",
            {"project_id": project_id},
        )
        self._collector.counter(
            "memory.forget.items",
            items_removed,
            {"project_id": project_id, "agent": agent or "all"},
        )

    # ==================== Embedding Operations ====================

    def record_embedding_latency(
        self,
        duration_ms: float,
        provider: str,
        batch_size: int = 1,
    ):
        """Record embedding generation latency."""
        self._collector.histogram(
            "embedding.latency",
            duration_ms,
            "ms",
            {"provider": provider, "batch_size": str(batch_size)},
        )
        self._collector.counter(
            "embedding.count",
            batch_size,
            {"provider": provider},
        )

    # ==================== Cache Operations ====================

    def record_cache_hit(self, cache_type: str = "retrieval"):
        """Record a cache hit."""
        self._collector.counter("cache.hit", 1, {"cache_type": cache_type})

    def record_cache_miss(self, cache_type: str = "retrieval"):
        """Record a cache miss."""
        self._collector.counter("cache.miss", 1, {"cache_type": cache_type})

    def record_cache_eviction(self, cache_type: str = "retrieval", count: int = 1):
        """Record cache evictions."""
        self._collector.counter("cache.eviction", count, {"cache_type": cache_type})

    def set_cache_size(self, size: int, cache_type: str = "retrieval"):
        """Set current cache size."""
        self._collector.gauge("cache.size", size, {"cache_type": cache_type})

    # ==================== Storage Operations ====================

    def record_storage_query_latency(
        self,
        duration_ms: float,
        operation: str,
        backend: str,
        success: bool = True,
    ):
        """Record storage query latency."""
        self._collector.histogram(
            "storage.query.latency",
            duration_ms,
            "ms",
            {
                "operation": operation,
                "backend": backend,
                "success": str(success).lower(),
            },
        )
        self._collector.counter(
            "storage.query.count",
            1,
            {"operation": operation, "backend": backend},
        )

    def record_storage_error(self, backend: str, operation: str, error_type: str):
        """Record a storage error."""
        self._collector.counter(
            "storage.error.count",
            1,
            {"backend": backend, "operation": operation, "error_type": error_type},
        )

    # ==================== Memory Counts ====================

    def set_memory_count(
        self,
        count: int,
        memory_type: str,
        agent: Optional[str] = None,
        project_id: Optional[str] = None,
    ):
        """Set memory item count gauge."""
        labels = {"memory_type": memory_type}
        if agent:
            labels["agent"] = agent
        if project_id:
            labels["project_id"] = project_id
        self._collector.gauge("memory.count", count, labels)

    # ==================== Session Operations ====================

    def record_session_start(self, agent: str, project_id: str):
        """Record a session start."""
        self._collector.counter(
            "session.start",
            1,
            {"agent": agent, "project_id": project_id},
        )

    def record_session_end(
        self,
        agent: str,
        project_id: str,
        duration_ms: float,
        outcome: str,
    ):
        """Record a session end."""
        self._collector.histogram(
            "session.duration",
            duration_ms,
            "ms",
            {"agent": agent, "project_id": project_id, "outcome": outcome},
        )
        self._collector.counter(
            "session.end",
            1,
            {"agent": agent, "project_id": project_id, "outcome": outcome},
        )

    # ==================== Utility ====================

    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all collected metrics."""
        return self._collector.get_stats()

    def timer(self, name: str, labels: Optional[Dict[str, str]] = None) -> Timer:
        """Create a timer for measuring operation duration."""
        return self._collector.timer(name, labels)


def get_meter(name: str = "alma"):
    """
    Get an OpenTelemetry meter.

    Falls back to a no-op meter if OTel is not available.
    """
    if _otel_available:
        return metrics.get_meter(name)
    return None


def get_metrics() -> ALMAMetrics:
    """
    Get the global ALMAMetrics instance.

    Creates one if it doesn't exist.
    """
    global _metrics_instance

    with _metrics_lock:
        if _metrics_instance is None:
            _metrics_instance = ALMAMetrics()
        return _metrics_instance


def set_metrics(metrics_instance: ALMAMetrics):
    """Set the global ALMAMetrics instance."""
    global _metrics_instance

    with _metrics_lock:
        _metrics_instance = metrics_instance
