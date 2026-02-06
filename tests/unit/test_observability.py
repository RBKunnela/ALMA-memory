"""
Tests for ALMA Observability Module.

Tests structured logging, metrics collection, and tracing functionality.
"""

import json
import logging
import time
from unittest.mock import patch

import pytest

from alma.observability import (
    ALMAMetrics,
    JSONFormatter,
    MetricsCollector,
    ObservabilityConfig,
    SpanKind,
    StructuredLogger,
    TracingContext,
    configure_observability,
    get_logger,
    get_metrics,
    get_tracer,
    setup_logging,
    shutdown_observability,
    trace_async,
    trace_method,
)
from alma.observability.guidelines import (
    OPERATION_LOG_LEVELS,
    get_recommended_level,
)
from alma.observability.metrics import InMemoryMetricsCollector

# =============================================================================
# Logging Tests
# =============================================================================


class TestJSONFormatter:
    """Tests for JSON log formatter."""

    def test_format_basic_message(self):
        """Test formatting a basic log message."""
        formatter = JSONFormatter(service_name="test-service")
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)
        data = json.loads(result)

        assert data["level"] == "INFO"
        assert data["logger"] == "test.logger"
        assert data["message"] == "Test message"
        assert data["service"] == "test-service"
        assert "timestamp" in data
        assert "source" in data
        assert data["source"]["line"] == 42

    def test_format_with_extra_fields(self):
        """Test formatting with extra fields."""
        formatter = JSONFormatter(
            service_name="test-service",
            extra_fields={"environment": "test"},
        )
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        record.custom_field = "custom_value"

        result = formatter.format(record)
        data = json.loads(result)

        assert data["environment"] == "test"
        assert data["extra"]["custom_field"] == "custom_value"

    def test_format_with_exception(self):
        """Test formatting with exception info."""
        formatter = JSONFormatter(service_name="test-service")

        try:
            raise ValueError("Test error")
        except ValueError:
            import sys

            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="test.logger",
            level=logging.ERROR,
            pathname="test.py",
            lineno=42,
            msg="Error occurred",
            args=(),
            exc_info=exc_info,
        )

        result = formatter.format(record)
        data = json.loads(result)

        assert "exception" in data
        assert data["exception"]["type"] == "ValueError"
        assert data["exception"]["message"] == "Test error"
        assert "traceback" in data["exception"]


class TestStructuredLogger:
    """Tests for structured logger."""

    def test_logger_creation(self):
        """Test creating a structured logger."""
        logger = StructuredLogger("test.module")
        assert logger._logger.name == "test.module"

    def test_set_context(self):
        """Test setting context on logger."""
        logger = StructuredLogger("test.module")
        logger.set_context(request_id="123", user_id="456")

        assert logger._context["request_id"] == "123"
        assert logger._context["user_id"] == "456"

    def test_clear_context(self):
        """Test clearing context."""
        logger = StructuredLogger("test.module")
        logger.set_context(request_id="123")
        logger.clear_context()

        assert len(logger._context) == 0

    def test_with_context_manager(self):
        """Test context manager for temporary context."""
        logger = StructuredLogger("test.module")
        logger.set_context(persistent="value")

        with logger.with_context(temporary="temp_value") as ctx_logger:
            assert ctx_logger._context["persistent"] == "value"
            assert ctx_logger._context["temporary"] == "temp_value"

        # Temporary context should be removed
        assert "temporary" not in logger._context
        assert logger._context["persistent"] == "value"

    def test_metric_logging(self):
        """Test metric method."""
        logger = StructuredLogger("test.module")
        with patch.object(logger, "_log") as mock_log:
            logger.metric("test_metric", 42.5, "ms", {"tag": "value"})
            mock_log.assert_called_once()
            args, kwargs = mock_log.call_args
            assert "METRIC test_metric=42.5ms" in args[1]

    def test_timing_logging(self):
        """Test timing method."""
        logger = StructuredLogger("test.module")
        with patch.object(logger, "_log") as mock_log:
            logger.timing("operation", 150.5, success=True, tags={"type": "test"})
            mock_log.assert_called_once()
            args, kwargs = mock_log.call_args
            assert "TIMING operation completed" in args[1]


class TestSetupLogging:
    """Tests for logging setup."""

    def test_setup_json_logging(self):
        """Test setting up JSON logging."""
        setup_logging(level="DEBUG", format_type="json", service_name="test")

        logger = logging.getLogger("alma.test")
        assert logger.level <= logging.DEBUG

    def test_setup_text_logging(self):
        """Test setting up text logging."""
        setup_logging(level="INFO", format_type="text", service_name="test")

        logger = logging.getLogger("alma.test")
        assert logger.level <= logging.INFO

    def test_get_logger(self):
        """Test getting a structured logger."""
        logger1 = get_logger("test.module")
        logger2 = get_logger("test.module")

        # Should return the same instance
        assert logger1 is logger2


# =============================================================================
# Metrics Tests
# =============================================================================


class TestInMemoryMetricsCollector:
    """Tests for in-memory metrics collector."""

    def test_counter_increment(self):
        """Test counter increment."""
        collector = InMemoryMetricsCollector()

        collector.increment_counter("requests", 1)
        collector.increment_counter("requests", 2)

        assert collector.get_counter("requests") == 3

    def test_counter_with_labels(self):
        """Test counter with labels."""
        collector = InMemoryMetricsCollector()

        collector.increment_counter("requests", 1, {"method": "GET"})
        collector.increment_counter("requests", 1, {"method": "POST"})
        collector.increment_counter("requests", 1, {"method": "GET"})

        assert collector.get_counter("requests", {"method": "GET"}) == 2
        assert collector.get_counter("requests", {"method": "POST"}) == 1

    def test_histogram_recording(self):
        """Test histogram recording."""
        collector = InMemoryMetricsCollector()

        for i in range(100):
            collector.record_histogram("latency", i)

        stats = collector.get_histogram_stats("latency")
        assert stats["count"] == 100
        assert stats["min"] == 0
        assert stats["max"] == 99
        assert stats["avg"] == 49.5

    def test_histogram_percentiles(self):
        """Test histogram percentile calculation."""
        collector = InMemoryMetricsCollector()

        for i in range(100):
            collector.record_histogram("latency", i)

        stats = collector.get_histogram_stats("latency")
        assert stats["p50"] == 50
        assert stats["p95"] >= 95

    def test_gauge_set_and_get(self):
        """Test gauge set and get."""
        collector = InMemoryMetricsCollector()

        collector.set_gauge("active_connections", 10)
        assert collector.get_gauge("active_connections") == 10

        collector.set_gauge("active_connections", 5)
        assert collector.get_gauge("active_connections") == 5

    def test_gauge_increment(self):
        """Test gauge increment."""
        collector = InMemoryMetricsCollector()

        collector.increment_gauge("active_connections", 5)
        collector.increment_gauge("active_connections", 3)
        collector.increment_gauge("active_connections", -2)

        assert collector.get_gauge("active_connections") == 6

    def test_get_all_metrics(self):
        """Test getting all metrics."""
        collector = InMemoryMetricsCollector()

        collector.increment_counter("requests", 10)
        collector.record_histogram("latency", 50)
        collector.set_gauge("connections", 5)

        all_metrics = collector.get_all_metrics()

        assert "counters" in all_metrics
        assert "histograms" in all_metrics
        assert "gauges" in all_metrics

    def test_reset(self):
        """Test resetting all metrics."""
        collector = InMemoryMetricsCollector()

        collector.increment_counter("requests", 10)
        collector.record_histogram("latency", 50)
        collector.set_gauge("connections", 5)

        collector.reset()

        assert collector.get_counter("requests") == 0
        assert collector.get_gauge("connections") == 0


class TestMetricsCollector:
    """Tests for unified metrics collector."""

    def test_counter_fallback(self):
        """Test counter with fallback (no OTel)."""
        collector = MetricsCollector(use_otel=False)

        collector.counter("test_counter", 5, {"label": "value"})

        stats = collector.get_stats()
        assert "counters" in stats

    def test_histogram_fallback(self):
        """Test histogram with fallback (no OTel)."""
        collector = MetricsCollector(use_otel=False)

        collector.histogram("test_histogram", 100, "ms", {"label": "value"})

        stats = collector.get_stats()
        assert "histograms" in stats

    def test_timer_context_manager(self):
        """Test timer context manager."""
        collector = MetricsCollector(use_otel=False)

        with collector.timer("operation") as timer:
            time.sleep(0.01)  # 10ms

        assert timer.duration_ms >= 10

    def test_timer_records_success(self):
        """Test timer records success."""
        collector = MetricsCollector(use_otel=False)

        with collector.timer("operation"):
            pass  # Successful operation

        stats = collector.get_stats()
        assert "histograms" in stats

    def test_timer_records_failure(self):
        """Test timer records failure on exception."""
        collector = MetricsCollector(use_otel=False)

        try:
            with collector.timer("operation"):
                raise ValueError("Test error")
        except ValueError:
            pass

        stats = collector.get_stats()
        assert "histograms" in stats


class TestALMAMetrics:
    """Tests for ALMA-specific metrics."""

    def test_record_retrieve_latency(self):
        """Test recording retrieval latency."""
        metrics = ALMAMetrics()

        metrics.record_retrieve_latency(
            duration_ms=50.5,
            agent="helena",
            project_id="test-project",
            cache_hit=True,
            items_returned=5,
        )

        stats = metrics.get_all_metrics()
        assert "histograms" in stats

    def test_record_learn_operation(self):
        """Test recording learn operation."""
        metrics = ALMAMetrics()

        metrics.record_learn_operation(
            duration_ms=25.0,
            agent="helena",
            project_id="test-project",
            memory_type="outcome",
            success=True,
        )

        stats = metrics.get_all_metrics()
        assert "counters" in stats

    def test_record_forget_operation(self):
        """Test recording forget operation."""
        metrics = ALMAMetrics()

        metrics.record_forget_operation(
            duration_ms=100.0,
            agent="helena",
            project_id="test-project",
            items_removed=10,
        )

        stats = metrics.get_all_metrics()
        assert "counters" in stats

    def test_record_embedding_latency(self):
        """Test recording embedding latency."""
        metrics = ALMAMetrics()

        metrics.record_embedding_latency(
            duration_ms=30.0,
            provider="local",
            batch_size=5,
        )

        stats = metrics.get_all_metrics()
        assert "histograms" in stats

    def test_cache_metrics(self):
        """Test cache hit/miss metrics."""
        metrics = ALMAMetrics()

        metrics.record_cache_hit()
        metrics.record_cache_hit()
        metrics.record_cache_miss()
        metrics.record_cache_eviction(count=5)
        metrics.set_cache_size(100)

        stats = metrics.get_all_metrics()
        assert "counters" in stats
        assert "gauges" in stats

    def test_storage_metrics(self):
        """Test storage metrics."""
        metrics = ALMAMetrics()

        metrics.record_storage_query_latency(
            duration_ms=15.0,
            operation="get_heuristics",
            backend="sqlite",
            success=True,
        )
        metrics.record_storage_error("sqlite", "save", "timeout")

        stats = metrics.get_all_metrics()
        assert "histograms" in stats
        assert "counters" in stats

    def test_session_metrics(self):
        """Test session metrics."""
        metrics = ALMAMetrics()

        metrics.record_session_start("helena", "test-project")
        metrics.record_session_end(
            "helena", "test-project", duration_ms=5000.0, outcome="success"
        )

        stats = metrics.get_all_metrics()
        assert "counters" in stats
        assert "histograms" in stats


class TestGetMetrics:
    """Tests for global metrics access."""

    def test_get_metrics_singleton(self):
        """Test that get_metrics returns same instance."""
        metrics1 = get_metrics()
        metrics2 = get_metrics()

        assert metrics1 is metrics2


# =============================================================================
# Tracing Tests
# =============================================================================


class TestTracingContext:
    """Tests for tracing context."""

    def test_create_span(self):
        """Test creating a span."""
        ctx = TracingContext("test-tracer")

        with ctx.span("test-operation") as span:
            span.set_attribute("key", "value")

    def test_span_with_kind(self):
        """Test creating span with specific kind."""
        ctx = TracingContext("test-tracer")

        with ctx.span("test-operation", kind=SpanKind.CLIENT) as _span:
            pass

    def test_span_with_attributes(self):
        """Test creating span with initial attributes."""
        ctx = TracingContext("test-tracer")

        with ctx.span(
            "test-operation",
            attributes={"initial": "value"},
        ) as span:
            span.set_attribute("additional", "value")


class TestTraceDecorators:
    """Tests for tracing decorators."""

    def test_trace_method_decorator(self):
        """Test trace_method decorator."""

        @trace_method(name="test_function")
        def test_function(arg1, arg2):
            return arg1 + arg2

        result = test_function(1, 2)
        assert result == 3

    def test_trace_method_with_args_recording(self):
        """Test trace_method with argument recording."""

        @trace_method(name="test_function", record_args=True)
        def test_function(x, y):
            return x * y

        result = test_function(3, 4)
        assert result == 12

    def test_trace_method_with_result_recording(self):
        """Test trace_method with result recording."""

        @trace_method(name="test_function", record_result=True)
        def test_function():
            return "result"

        result = test_function()
        assert result == "result"

    def test_trace_method_on_class_method(self):
        """Test trace_method on class method."""

        class TestClass:
            @trace_method(name="TestClass.method")
            def method(self, value):
                return value * 2

        obj = TestClass()
        result = obj.method(5)
        assert result == 10

    def test_trace_method_handles_exception(self):
        """Test trace_method handles exceptions properly."""

        @trace_method(name="failing_function")
        def failing_function():
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            failing_function()

    @pytest.mark.asyncio
    async def test_trace_async_decorator(self):
        """Test trace_async decorator."""

        @trace_async(name="async_function")
        async def async_function(value):
            return value * 2

        result = await async_function(5)
        assert result == 10

    @pytest.mark.asyncio
    async def test_trace_async_handles_exception(self):
        """Test trace_async handles exceptions."""

        @trace_async(name="failing_async")
        async def failing_async():
            raise ValueError("Async error")

        with pytest.raises(ValueError):
            await failing_async()


class TestGetTracer:
    """Tests for tracer access."""

    def test_get_tracer_returns_tracer(self):
        """Test get_tracer returns a usable tracer."""
        tracer = get_tracer("test.module")
        assert tracer is not None

    def test_tracer_can_create_spans(self):
        """Test tracer can create spans."""
        tracer = get_tracer("test.module")

        with tracer.start_as_current_span("test-span") as _span:
            pass


# =============================================================================
# Configuration Tests
# =============================================================================


class TestObservabilityConfig:
    """Tests for observability configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ObservabilityConfig()

        assert config.service_name == "alma-memory"
        assert config.enable_tracing is True
        assert config.enable_metrics is True
        assert config.enable_logging is True

    def test_config_from_env(self):
        """Test configuration from environment variables."""
        with patch.dict(
            "os.environ",
            {
                "ALMA_ENVIRONMENT": "production",
                "ALMA_LOG_LEVEL": "ERROR",
                "ALMA_LOG_FORMAT": "text",
            },
        ):
            config = ObservabilityConfig()
            assert config.environment == "production"
            assert config.log_level == "ERROR"
            assert config.log_format == "text"

    def test_config_to_dict(self):
        """Test config serialization."""
        config = ObservabilityConfig(
            service_name="test-service",
            environment="test",
        )

        data = config.to_dict()

        assert data["service_name"] == "test-service"
        assert data["environment"] == "test"


class TestConfigureObservability:
    """Tests for observability configuration function."""

    def test_configure_with_defaults(self):
        """Test configuration with default values."""
        config = configure_observability()

        assert config.service_name == "alma-memory"
        assert config.enable_logging is True

        # Cleanup
        shutdown_observability()

    def test_configure_with_custom_values(self):
        """Test configuration with custom values."""
        config = configure_observability(
            service_name="custom-service",
            environment="test",
            enable_tracing=False,
            enable_metrics=False,
            log_level="DEBUG",
            log_format="text",
        )

        assert config.service_name == "custom-service"
        assert config.environment == "test"
        assert config.enable_tracing is False
        assert config.enable_metrics is False
        assert config.log_level == "DEBUG"
        assert config.log_format == "text"

        # Cleanup
        shutdown_observability()


# =============================================================================
# Integration Tests
# =============================================================================


class TestObservabilityIntegration:
    """Integration tests for observability features."""

    def test_full_observability_flow(self):
        """Test complete observability flow."""
        # Configure
        _config = configure_observability(
            service_name="integration-test",
            enable_tracing=True,
            enable_metrics=True,
            enable_logging=True,
            log_format="json",
        )

        # Get components
        logger = get_logger("integration.test")
        metrics = get_metrics()
        tracer = get_tracer("integration.test")

        # Use components
        logger.info("Test message", extra_field="value")

        with tracer.start_as_current_span("test-operation") as span:
            span.set_attribute("test", "value")
            metrics.record_retrieve_latency(
                duration_ms=50.0,
                agent="test",
                project_id="test-project",
                cache_hit=False,
                items_returned=5,
            )

        # Verify metrics recorded
        stats = metrics.get_all_metrics()
        assert "histograms" in stats
        assert "counters" in stats

        # Cleanup
        shutdown_observability()

    def test_observability_disabled(self):
        """Test with observability features disabled."""
        _config = configure_observability(
            service_name="disabled-test",
            enable_tracing=False,
            enable_metrics=False,
            enable_logging=False,
        )

        # Components should still work (no-op)
        logger = get_logger("disabled.test")
        _metrics = get_metrics()
        tracer = get_tracer("disabled.test")

        logger.info("This should not cause errors")

        with tracer.start_as_current_span("test"):
            pass

        # Cleanup
        shutdown_observability()


# =============================================================================
# Logging Guidelines Tests
# =============================================================================


class TestLoggingGuidelines:
    """Tests for logging level guidelines."""

    def test_operation_log_levels_defined(self):
        """Test that all expected operation types have defined levels."""
        expected_operations = [
            "save_success",
            "save_failure",
            "cache_hit",
            "cache_miss",
            "retrieval_complete",
            "service_start",
            "service_stop",
            "missing_scope",
            "operation_failure",
        ]
        for op in expected_operations:
            assert op in OPERATION_LOG_LEVELS, f"Missing operation: {op}"

    def test_get_recommended_level_returns_correct_levels(self):
        """Test that get_recommended_level returns appropriate levels."""
        import logging

        # DEBUG level operations
        assert get_recommended_level("save_success") == logging.DEBUG
        assert get_recommended_level("cache_hit") == logging.DEBUG
        assert get_recommended_level("cache_miss") == logging.DEBUG

        # INFO level operations
        assert get_recommended_level("service_start") == logging.INFO
        assert get_recommended_level("retrieval_complete") == logging.INFO

        # WARNING level operations
        assert get_recommended_level("missing_scope") == logging.WARNING
        assert get_recommended_level("retry_attempt") == logging.WARNING

        # ERROR level operations
        assert get_recommended_level("operation_failure") == logging.ERROR
        assert get_recommended_level("save_failure") == logging.ERROR

    def test_get_recommended_level_unknown_operation(self):
        """Test that unknown operations default to INFO."""
        import logging

        assert get_recommended_level("unknown_operation") == logging.INFO
        assert get_recommended_level("") == logging.INFO

    def test_log_levels_consistency(self):
        """Test that similar operations have consistent log levels."""
        import logging

        # All save success operations should be DEBUG
        assert OPERATION_LOG_LEVELS["save_success"] == logging.DEBUG
        assert OPERATION_LOG_LEVELS["batch_save_success"] == logging.DEBUG
        assert OPERATION_LOG_LEVELS["delete_success"] == logging.DEBUG

        # All failure operations should be ERROR
        assert OPERATION_LOG_LEVELS["save_failure"] == logging.ERROR
        assert OPERATION_LOG_LEVELS["delete_failure"] == logging.ERROR
        assert OPERATION_LOG_LEVELS["operation_failure"] == logging.ERROR

        # All cache operations should be DEBUG
        assert OPERATION_LOG_LEVELS["cache_hit"] == logging.DEBUG
        assert OPERATION_LOG_LEVELS["cache_miss"] == logging.DEBUG
        assert OPERATION_LOG_LEVELS["cache_invalidate"] == logging.DEBUG

        # All lifecycle events should be INFO
        assert OPERATION_LOG_LEVELS["service_start"] == logging.INFO
        assert OPERATION_LOG_LEVELS["service_stop"] == logging.INFO
        assert OPERATION_LOG_LEVELS["config_loaded"] == logging.INFO
