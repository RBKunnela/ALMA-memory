"""
ALMA Observability Module.

Provides comprehensive observability features including:
- OpenTelemetry integration for distributed tracing
- Structured JSON logging
- Metrics collection (counters, histograms, gauges)
- Performance monitoring

This module follows the OpenTelemetry specification and supports
integration with common observability backends (Jaeger, Prometheus,
DataDog, etc.).

Usage:
    from alma.observability import (
        get_tracer,
        get_meter,
        get_logger,
        configure_observability,
        ALMAMetrics,
    )

    # Initialize observability (typically at app startup)
    configure_observability(
        service_name="alma-memory",
        enable_tracing=True,
        enable_metrics=True,
        log_format="json",
    )

    # Use in code
    tracer = get_tracer(__name__)
    with tracer.start_as_current_span("my_operation"):
        # ... your code
        pass
"""

from alma.observability.config import (
    ObservabilityConfig,
    configure_observability,
    shutdown_observability,
)
from alma.observability.guidelines import (
    OPERATION_LOG_LEVELS,
    get_recommended_level,
)
from alma.observability.logging import (
    JSONFormatter,
    StructuredLogger,
    get_logger,
    setup_logging,
)
from alma.observability.metrics import (
    ALMAMetrics,
    MetricsCollector,
    get_meter,
    get_metrics,
)
from alma.observability.tracing import (
    SpanKind,
    TracingContext,
    get_tracer,
    trace_async,
    trace_method,
)

__all__ = [
    # Configuration
    "ObservabilityConfig",
    "configure_observability",
    "shutdown_observability",
    # Logging
    "JSONFormatter",
    "StructuredLogger",
    "get_logger",
    "setup_logging",
    # Metrics
    "ALMAMetrics",
    "MetricsCollector",
    "get_meter",
    "get_metrics",
    # Tracing
    "SpanKind",
    "TracingContext",
    "get_tracer",
    "trace_method",
    "trace_async",
    # Guidelines
    "OPERATION_LOG_LEVELS",
    "get_recommended_level",
]
