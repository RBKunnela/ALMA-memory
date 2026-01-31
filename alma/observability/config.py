"""
ALMA Observability Configuration.

Centralized configuration for observability features including
tracing, metrics, and logging setup.
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

# Global state for observability configuration
_observability_initialized = False
_tracer_provider = None
_meter_provider = None


@dataclass
class ObservabilityConfig:
    """
    Configuration for ALMA observability features.

    Attributes:
        service_name: Name of the service for tracing/metrics
        service_version: Version of the service
        environment: Deployment environment (dev, staging, prod)
        enable_tracing: Whether to enable distributed tracing
        enable_metrics: Whether to enable metrics collection
        enable_logging: Whether to enable structured logging
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_format: Log format ("json" or "text")
        otlp_endpoint: OpenTelemetry collector endpoint
        otlp_headers: Headers for OTLP exporter
        trace_sample_rate: Sampling rate for traces (0.0-1.0)
        metric_export_interval_ms: How often to export metrics
        resource_attributes: Additional resource attributes
    """

    service_name: str = "alma-memory"
    service_version: str = "0.5.1"
    environment: str = field(
        default_factory=lambda: os.environ.get("ALMA_ENVIRONMENT", "development")
    )
    enable_tracing: bool = True
    enable_metrics: bool = True
    enable_logging: bool = True
    log_level: str = field(
        default_factory=lambda: os.environ.get("ALMA_LOG_LEVEL", "INFO")
    )
    log_format: str = field(
        default_factory=lambda: os.environ.get("ALMA_LOG_FORMAT", "json")
    )
    otlp_endpoint: Optional[str] = field(
        default_factory=lambda: os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    )
    otlp_headers: Dict[str, str] = field(default_factory=dict)
    trace_sample_rate: float = 1.0
    metric_export_interval_ms: int = 60000
    resource_attributes: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "service_name": self.service_name,
            "service_version": self.service_version,
            "environment": self.environment,
            "enable_tracing": self.enable_tracing,
            "enable_metrics": self.enable_metrics,
            "enable_logging": self.enable_logging,
            "log_level": self.log_level,
            "log_format": self.log_format,
            "otlp_endpoint": self.otlp_endpoint,
            "trace_sample_rate": self.trace_sample_rate,
            "metric_export_interval_ms": self.metric_export_interval_ms,
        }


def configure_observability(
    service_name: str = "alma-memory",
    service_version: str = "0.5.1",
    environment: Optional[str] = None,
    enable_tracing: bool = True,
    enable_metrics: bool = True,
    enable_logging: bool = True,
    log_level: str = "INFO",
    log_format: str = "json",
    otlp_endpoint: Optional[str] = None,
    trace_sample_rate: float = 1.0,
    resource_attributes: Optional[Dict[str, str]] = None,
) -> ObservabilityConfig:
    """
    Configure ALMA observability features.

    This function should be called once at application startup to initialize
    tracing, metrics, and logging.

    Args:
        service_name: Name of the service
        service_version: Version of the service
        environment: Deployment environment
        enable_tracing: Enable distributed tracing
        enable_metrics: Enable metrics collection
        enable_logging: Enable structured logging
        log_level: Logging level
        log_format: Log format ("json" or "text")
        otlp_endpoint: OpenTelemetry collector endpoint
        trace_sample_rate: Sampling rate for traces
        resource_attributes: Additional resource attributes

    Returns:
        ObservabilityConfig with applied settings
    """
    global _observability_initialized, _tracer_provider, _meter_provider

    config = ObservabilityConfig(
        service_name=service_name,
        service_version=service_version,
        environment=environment or os.environ.get("ALMA_ENVIRONMENT", "development"),
        enable_tracing=enable_tracing,
        enable_metrics=enable_metrics,
        enable_logging=enable_logging,
        log_level=log_level,
        log_format=log_format,
        otlp_endpoint=otlp_endpoint,
        trace_sample_rate=trace_sample_rate,
        resource_attributes=resource_attributes or {},
    )

    # Setup logging first
    if config.enable_logging:
        from alma.observability.logging import setup_logging

        setup_logging(
            level=config.log_level,
            format_type=config.log_format,
            service_name=config.service_name,
        )

    # Setup tracing
    if config.enable_tracing:
        _tracer_provider = _setup_tracing(config)

    # Setup metrics
    if config.enable_metrics:
        _meter_provider = _setup_metrics(config)

    _observability_initialized = True

    logger = logging.getLogger(__name__)
    logger.info(
        "ALMA observability configured",
        extra={
            "service_name": config.service_name,
            "environment": config.environment,
            "tracing_enabled": config.enable_tracing,
            "metrics_enabled": config.enable_metrics,
        },
    )

    return config


def _setup_tracing(config: ObservabilityConfig):
    """Setup OpenTelemetry tracing."""
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.sampling import TraceIdRatioBased

        # Build resource attributes
        resource_attrs = {
            "service.name": config.service_name,
            "service.version": config.service_version,
            "deployment.environment": config.environment,
        }
        resource_attrs.update(config.resource_attributes)

        resource = Resource.create(resource_attrs)

        # Create sampler
        sampler = TraceIdRatioBased(config.trace_sample_rate)

        # Create and set tracer provider
        provider = TracerProvider(resource=resource, sampler=sampler)

        # Add OTLP exporter if endpoint is configured
        if config.otlp_endpoint:
            try:
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                    OTLPSpanExporter,
                )
                from opentelemetry.sdk.trace.export import BatchSpanProcessor

                otlp_exporter = OTLPSpanExporter(
                    endpoint=config.otlp_endpoint,
                    headers=config.otlp_headers or {},
                )
                provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
            except ImportError:
                logging.getLogger(__name__).warning(
                    "OTLP exporter not available. Install with: "
                    "pip install opentelemetry-exporter-otlp-proto-grpc"
                )

        trace.set_tracer_provider(provider)
        return provider

    except ImportError:
        logging.getLogger(__name__).warning(
            "OpenTelemetry SDK not available. Tracing disabled. "
            "Install with: pip install opentelemetry-sdk"
        )
        return None


def _setup_metrics(config: ObservabilityConfig):
    """Setup OpenTelemetry metrics."""
    try:
        from opentelemetry import metrics
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.resources import Resource

        # Build resource attributes
        resource_attrs = {
            "service.name": config.service_name,
            "service.version": config.service_version,
            "deployment.environment": config.environment,
        }
        resource_attrs.update(config.resource_attributes)

        resource = Resource.create(resource_attrs)

        # Create meter provider
        provider = MeterProvider(resource=resource)

        # Add OTLP exporter if endpoint is configured
        if config.otlp_endpoint:
            try:
                from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
                    OTLPMetricExporter,
                )
                from opentelemetry.sdk.metrics.export import (
                    PeriodicExportingMetricReader,
                )

                otlp_exporter = OTLPMetricExporter(
                    endpoint=config.otlp_endpoint,
                    headers=config.otlp_headers or {},
                )
                reader = PeriodicExportingMetricReader(
                    otlp_exporter,
                    export_interval_millis=config.metric_export_interval_ms,
                )
                provider = MeterProvider(resource=resource, metric_readers=[reader])
            except ImportError:
                logging.getLogger(__name__).warning(
                    "OTLP metric exporter not available. Install with: "
                    "pip install opentelemetry-exporter-otlp-proto-grpc"
                )

        metrics.set_meter_provider(provider)
        return provider

    except ImportError:
        logging.getLogger(__name__).warning(
            "OpenTelemetry SDK not available. Metrics disabled. "
            "Install with: pip install opentelemetry-sdk"
        )
        return None


def shutdown_observability():
    """
    Shutdown observability providers.

    Should be called at application shutdown to ensure all telemetry
    data is exported.
    """
    global _observability_initialized, _tracer_provider, _meter_provider

    if _tracer_provider is not None:
        try:
            _tracer_provider.shutdown()
        except Exception as e:
            logging.getLogger(__name__).error(f"Error shutting down tracer: {e}")

    if _meter_provider is not None:
        try:
            _meter_provider.shutdown()
        except Exception as e:
            logging.getLogger(__name__).error(f"Error shutting down meter: {e}")

    _observability_initialized = False
    _tracer_provider = None
    _meter_provider = None


def is_observability_initialized() -> bool:
    """Check if observability has been initialized."""
    return _observability_initialized
