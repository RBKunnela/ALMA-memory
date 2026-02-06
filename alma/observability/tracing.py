"""
ALMA Distributed Tracing.

Provides distributed tracing using OpenTelemetry with fallback
to logging when OTel is not available.
"""

import functools
import logging
from contextlib import contextmanager
from enum import Enum
from typing import Any, Callable, Dict, Optional, TypeVar, Union

# Try to import OpenTelemetry
_otel_available = False
_NoOpSpan = None
_NoOpTracer = None

try:
    from opentelemetry import trace
    from opentelemetry.trace import SpanKind as OTelSpanKind
    from opentelemetry.trace import Status, StatusCode

    _otel_available = True
except ImportError:
    pass


logger = logging.getLogger(__name__)

# Type variable for decorated functions
F = TypeVar("F", bound=Callable[..., Any])


class SpanKind(Enum):
    """Span kind enum (mirrors OpenTelemetry SpanKind)."""

    INTERNAL = "internal"
    SERVER = "server"
    CLIENT = "client"
    PRODUCER = "producer"
    CONSUMER = "consumer"


class NoOpSpan:
    """No-op span implementation when OpenTelemetry is not available."""

    def __init__(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        self.name = name
        self.attributes = attributes or {}
        self._logger = logging.getLogger(f"alma.trace.{name}")

    def set_attribute(self, key: str, value: Any):
        """Set a span attribute."""
        self.attributes[key] = value

    def set_attributes(self, attributes: Dict[str, Any]):
        """Set multiple span attributes."""
        self.attributes.update(attributes)

    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        """Add an event to the span."""
        self._logger.debug(f"Event: {name}", extra={"event_attributes": attributes})

    def set_status(self, status: Any, description: Optional[str] = None):
        """Set span status."""
        pass

    def record_exception(
        self, exception: BaseException, attributes: Optional[Dict[str, Any]] = None
    ):
        """Record an exception."""
        self._logger.error(f"Exception in span {self.name}: {exception}", exc_info=True)

    def end(self, end_time: Optional[int] = None):
        """End the span."""
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.record_exception(exc_val)
        return False


class NoOpTracer:
    """No-op tracer implementation when OpenTelemetry is not available."""

    def __init__(self, name: str):
        self.name = name

    def start_span(
        self,
        name: str,
        context: Optional[Any] = None,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Optional[Dict[str, Any]] = None,
        start_time: Optional[int] = None,
    ) -> NoOpSpan:
        """Start a new span."""
        return NoOpSpan(name, attributes)

    @contextmanager
    def start_as_current_span(
        self,
        name: str,
        context: Optional[Any] = None,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Optional[Dict[str, Any]] = None,
        start_time: Optional[int] = None,
    ):
        """Start a span as the current span."""
        span = NoOpSpan(name, attributes)
        try:
            yield span
        except Exception as e:
            span.record_exception(e)
            raise
        finally:
            span.end()


class TracingContext:
    """
    Context for managing trace propagation and span creation.

    Provides a unified interface for tracing regardless of
    whether OpenTelemetry is available.
    """

    def __init__(self, tracer_name: str = "alma"):
        """
        Initialize tracing context.

        Args:
            tracer_name: Name for the tracer
        """
        self.tracer_name = tracer_name
        self._tracer = None

    @property
    def tracer(self):
        """Get the tracer (lazy initialization)."""
        if self._tracer is None:
            self._tracer = get_tracer(self.tracer_name)
        return self._tracer

    @contextmanager
    def span(
        self,
        name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Optional[Dict[str, Any]] = None,
    ):
        """
        Create a span context manager.

        Args:
            name: Span name
            kind: Span kind
            attributes: Initial span attributes

        Yields:
            The created span
        """
        if _otel_available:
            otel_kind = _map_span_kind(kind)
            with self.tracer.start_as_current_span(
                name,
                kind=otel_kind,
                attributes=attributes,
            ) as span:
                yield span
        else:
            with self.tracer.start_as_current_span(
                name,
                kind=kind,
                attributes=attributes,
            ) as span:
                yield span

    def create_span(
        self,
        name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Optional[Dict[str, Any]] = None,
    ):
        """
        Create a span (not automatically set as current).

        Args:
            name: Span name
            kind: Span kind
            attributes: Initial span attributes

        Returns:
            The created span
        """
        if _otel_available:
            otel_kind = _map_span_kind(kind)
            return self.tracer.start_span(
                name,
                kind=otel_kind,
                attributes=attributes,
            )
        else:
            return self.tracer.start_span(
                name,
                kind=kind,
                attributes=attributes,
            )


def _map_span_kind(kind: SpanKind):
    """Map our SpanKind to OpenTelemetry SpanKind."""
    if not _otel_available:
        return kind

    mapping = {
        SpanKind.INTERNAL: OTelSpanKind.INTERNAL,
        SpanKind.SERVER: OTelSpanKind.SERVER,
        SpanKind.CLIENT: OTelSpanKind.CLIENT,
        SpanKind.PRODUCER: OTelSpanKind.PRODUCER,
        SpanKind.CONSUMER: OTelSpanKind.CONSUMER,
    }
    return mapping.get(kind, OTelSpanKind.INTERNAL)


def get_tracer(name: str = "alma") -> Union["NoOpTracer", Any]:
    """
    Get a tracer for the given name.

    Uses OpenTelemetry tracer if available, otherwise returns
    a no-op tracer that logs span information.

    Args:
        name: Tracer name (typically module name)

    Returns:
        Tracer instance
    """
    if _otel_available:
        return trace.get_tracer(name)
    return NoOpTracer(name)


def get_current_span():
    """
    Get the current span.

    Returns:
        Current span or NoOpSpan if no span is active
    """
    if _otel_available:
        return trace.get_current_span()
    return NoOpSpan("current")


def trace_method(
    name: Optional[str] = None,
    kind: SpanKind = SpanKind.INTERNAL,
    record_args: bool = True,
    record_result: bool = False,
) -> Callable[[F], F]:
    """
    Decorator to trace a synchronous method.

    Args:
        name: Span name (defaults to function name)
        kind: Span kind
        record_args: Whether to record function arguments as attributes
        record_result: Whether to record the return value

    Usage:
        @trace_method(name="my_operation")
        def my_function(arg1, arg2):
            return result
    """

    def decorator(func: F) -> F:
        span_name = name or func.__qualname__

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            tracer = get_tracer(func.__module__)

            attributes: Dict[str, Any] = {
                "code.function": func.__name__,
                "code.namespace": func.__module__,
            }

            if record_args:
                # Record positional args (skip 'self' for methods)
                arg_names = func.__code__.co_varnames[: func.__code__.co_argcount]
                start_idx = 1 if arg_names and arg_names[0] in ("self", "cls") else 0
                for i, arg in enumerate(args[start_idx:], start=start_idx):
                    if i < len(arg_names):
                        arg_val = _safe_attribute_value(arg)
                        if arg_val is not None:
                            attributes[f"arg.{arg_names[i]}"] = arg_val

                # Record keyword args
                for key, value in kwargs.items():
                    arg_val = _safe_attribute_value(value)
                    if arg_val is not None:
                        attributes[f"arg.{key}"] = arg_val

            if _otel_available:
                otel_kind = _map_span_kind(kind)
                with tracer.start_as_current_span(
                    span_name,
                    kind=otel_kind,
                    attributes=attributes,
                ) as span:
                    try:
                        result = func(*args, **kwargs)
                        if record_result:
                            result_val = _safe_attribute_value(result)
                            if result_val is not None:
                                span.set_attribute("result", result_val)
                        return result
                    except Exception as e:
                        span.record_exception(e)
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        raise
            else:
                with tracer.start_as_current_span(
                    span_name,
                    kind=kind,
                    attributes=attributes,
                ) as span:
                    return func(*args, **kwargs)

        return wrapper  # type: ignore

    return decorator


def trace_async(
    name: Optional[str] = None,
    kind: SpanKind = SpanKind.INTERNAL,
    record_args: bool = True,
    record_result: bool = False,
) -> Callable[[F], F]:
    """
    Decorator to trace an async method.

    Args:
        name: Span name (defaults to function name)
        kind: Span kind
        record_args: Whether to record function arguments as attributes
        record_result: Whether to record the return value

    Usage:
        @trace_async(name="my_async_operation")
        async def my_async_function(arg1, arg2):
            return result
    """

    def decorator(func: F) -> F:
        span_name = name or func.__qualname__

        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            tracer = get_tracer(func.__module__)

            attributes: Dict[str, Any] = {
                "code.function": func.__name__,
                "code.namespace": func.__module__,
            }

            if record_args:
                # Record positional args (skip 'self' for methods)
                arg_names = func.__code__.co_varnames[: func.__code__.co_argcount]
                start_idx = 1 if arg_names and arg_names[0] in ("self", "cls") else 0
                for i, arg in enumerate(args[start_idx:], start=start_idx):
                    if i < len(arg_names):
                        arg_val = _safe_attribute_value(arg)
                        if arg_val is not None:
                            attributes[f"arg.{arg_names[i]}"] = arg_val

                # Record keyword args
                for key, value in kwargs.items():
                    arg_val = _safe_attribute_value(value)
                    if arg_val is not None:
                        attributes[f"arg.{key}"] = arg_val

            if _otel_available:
                otel_kind = _map_span_kind(kind)
                with tracer.start_as_current_span(
                    span_name,
                    kind=otel_kind,
                    attributes=attributes,
                ) as span:
                    try:
                        result = await func(*args, **kwargs)
                        if record_result:
                            result_val = _safe_attribute_value(result)
                            if result_val is not None:
                                span.set_attribute("result", result_val)
                        return result
                    except Exception as e:
                        span.record_exception(e)
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        raise
            else:
                with tracer.start_as_current_span(
                    span_name,
                    kind=kind,
                    attributes=attributes,
                ) as span:
                    return await func(*args, **kwargs)

        return wrapper  # type: ignore

    return decorator


def _safe_attribute_value(value: Any) -> Optional[Union[str, int, float, bool]]:
    """
    Convert a value to a safe attribute value for tracing.

    OpenTelemetry only supports certain types for attributes.
    """
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        if len(value) <= 10:  # Limit list size
            return str(value)
        return f"[{len(value)} items]"
    if isinstance(value, dict):
        if len(value) <= 5:  # Limit dict size
            return str(value)
        return f"{{{len(value)} items}}"
    # For complex objects, return type and id
    return f"<{type(value).__name__}>"
