"""
ALMA Structured Logging.

Provides structured JSON logging for log aggregation systems
(ELK, Splunk, DataDog, etc.) with contextual information.
"""

import json
import logging
import sys
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, Optional

# Global logger registry
_loggers: Dict[str, "StructuredLogger"] = {}
_logging_configured = False
_default_service_name = "alma-memory"


class JSONFormatter(logging.Formatter):
    """
    JSON formatter for structured logging.

    Outputs log records as JSON objects with consistent field names
    for easy parsing by log aggregation systems.
    """

    def __init__(
        self,
        service_name: str = "alma-memory",
        include_traceback: bool = True,
        extra_fields: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize JSON formatter.

        Args:
            service_name: Name of the service for logs
            include_traceback: Include full traceback for exceptions
            extra_fields: Additional fields to include in every log
        """
        super().__init__()
        self.service_name = service_name
        self.include_traceback = include_traceback
        self.extra_fields = extra_fields or {}

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        # Build base log entry
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "service": self.service_name,
        }

        # Add source location
        log_entry["source"] = {
            "file": record.filename,
            "line": record.lineno,
            "function": record.funcName,
        }

        # Add thread/process info
        log_entry["thread"] = {
            "id": record.thread,
            "name": record.threadName,
        }
        log_entry["process"] = {
            "id": record.process,
            "name": record.processName,
        }

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
            }
            if self.include_traceback:
                log_entry["exception"]["traceback"] = traceback.format_exception(
                    *record.exc_info
                )

        # Add extra fields from record
        # Skip standard LogRecord attributes
        skip_attrs = {
            "name",
            "msg",
            "args",
            "created",
            "filename",
            "funcName",
            "levelname",
            "levelno",
            "lineno",
            "module",
            "msecs",
            "pathname",
            "process",
            "processName",
            "relativeCreated",
            "stack_info",
            "exc_info",
            "exc_text",
            "thread",
            "threadName",
            "message",
            "taskName",
        }

        extra = {}
        for key, value in record.__dict__.items():
            if key not in skip_attrs and not key.startswith("_"):
                # Try to serialize, skip if not possible
                try:
                    json.dumps(value)
                    extra[key] = value
                except (TypeError, ValueError):
                    extra[key] = str(value)

        if extra:
            log_entry["extra"] = extra

        # Add configured extra fields
        if self.extra_fields:
            log_entry.update(self.extra_fields)

        return json.dumps(log_entry, default=str)


class TextFormatter(logging.Formatter):
    """
    Enhanced text formatter with structured context.

    Provides readable text output with optional context fields
    for development and debugging.
    """

    def __init__(
        self,
        include_context: bool = True,
        include_location: bool = True,
    ):
        """
        Initialize text formatter.

        Args:
            include_context: Include extra context fields
            include_location: Include source file and line
        """
        super().__init__()
        self.include_context = include_context
        self.include_location = include_location

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as readable text."""
        # Base format: timestamp level [logger] message
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        parts = [
            f"{timestamp} {record.levelname:8s} [{record.name}] {record.getMessage()}"
        ]

        # Add location if enabled
        if self.include_location:
            parts.append(f"  at {record.filename}:{record.lineno} in {record.funcName}")

        # Add exception info if present
        if record.exc_info:
            parts.append(self.formatException(record.exc_info))

        # Add extra context if enabled
        if self.include_context:
            skip_attrs = {
                "name",
                "msg",
                "args",
                "created",
                "filename",
                "funcName",
                "levelname",
                "levelno",
                "lineno",
                "module",
                "msecs",
                "pathname",
                "process",
                "processName",
                "relativeCreated",
                "stack_info",
                "exc_info",
                "exc_text",
                "thread",
                "threadName",
                "message",
                "taskName",
            }

            extra_items = []
            for key, value in record.__dict__.items():
                if key not in skip_attrs and not key.startswith("_"):
                    extra_items.append(f"{key}={value}")

            if extra_items:
                parts.append(f"  context: {', '.join(extra_items)}")

        return "\n".join(parts)


class StructuredLogger:
    """
    Structured logger wrapper with context management.

    Provides a convenient interface for logging with structured
    context that is automatically included in all log messages.
    """

    def __init__(
        self,
        name: str,
        level: int = logging.INFO,
    ):
        """
        Initialize structured logger.

        Args:
            name: Logger name (typically __name__)
            level: Logging level
        """
        self._logger = logging.getLogger(name)
        self._logger.setLevel(level)
        self._context: Dict[str, Any] = {}

    def set_context(self, **kwargs):
        """Set persistent context fields for all subsequent logs."""
        self._context.update(kwargs)

    def clear_context(self):
        """Clear all context fields."""
        self._context.clear()

    def with_context(self, **kwargs) -> "LogContext":
        """
        Create a context manager for temporary context.

        Usage:
            with logger.with_context(request_id="123"):
                logger.info("Processing request")
        """
        return LogContext(self, kwargs)

    def _log(
        self,
        level: int,
        msg: str,
        *args,
        exc_info: bool = False,
        **kwargs,
    ):
        """Internal log method with context injection."""
        # Merge context with kwargs
        extra = {**self._context, **kwargs}
        self._logger.log(level, msg, *args, exc_info=exc_info, extra=extra)

    def debug(self, msg: str, *args, **kwargs):
        """Log debug message."""
        self._log(logging.DEBUG, msg, *args, **kwargs)

    def info(self, msg: str, *args, **kwargs):
        """Log info message."""
        self._log(logging.INFO, msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs):
        """Log warning message."""
        self._log(logging.WARNING, msg, *args, **kwargs)

    def error(self, msg: str, *args, exc_info: bool = False, **kwargs):
        """Log error message."""
        self._log(logging.ERROR, msg, *args, exc_info=exc_info, **kwargs)

    def exception(self, msg: str, *args, **kwargs):
        """Log exception with traceback."""
        self._log(logging.ERROR, msg, *args, exc_info=True, **kwargs)

    def critical(self, msg: str, *args, **kwargs):
        """Log critical message."""
        self._log(logging.CRITICAL, msg, *args, **kwargs)

    # Metrics-related log methods

    def metric(
        self,
        name: str,
        value: float,
        unit: str = "",
        tags: Optional[Dict[str, str]] = None,
    ):
        """
        Log a metric value.

        This is useful for logging metrics when OpenTelemetry
        metrics are not available.
        """
        self.info(
            f"METRIC {name}={value}{unit}",
            metric_name=name,
            metric_value=value,
            metric_unit=unit,
            metric_tags=tags or {},
        )

    def timing(
        self,
        operation: str,
        duration_ms: float,
        success: bool = True,
        tags: Optional[Dict[str, str]] = None,
    ):
        """Log an operation timing."""
        self.info(
            f"TIMING {operation} completed in {duration_ms:.2f}ms",
            operation=operation,
            duration_ms=duration_ms,
            success=success,
            **(tags or {}),
        )


class LogContext:
    """Context manager for temporary logging context."""

    def __init__(self, logger: StructuredLogger, context: Dict[str, Any]):
        self._logger = logger
        self._context = context
        self._original_context: Dict[str, Any] = {}

    def __enter__(self):
        # Save original values for keys we're overwriting
        for key in self._context:
            if key in self._logger._context:
                self._original_context[key] = self._logger._context[key]
        # Set new context
        self._logger.set_context(**self._context)
        return self._logger

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Remove added context
        for key in self._context:
            if key in self._original_context:
                self._logger._context[key] = self._original_context[key]
            else:
                self._logger._context.pop(key, None)
        return False


def setup_logging(
    level: str = "INFO",
    format_type: str = "json",
    service_name: str = "alma-memory",
    output: str = "stderr",
    extra_fields: Optional[Dict[str, Any]] = None,
):
    """
    Setup logging with the specified configuration.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_type: "json" or "text"
        service_name: Service name for logs
        output: "stderr", "stdout", or a file path
        extra_fields: Additional fields to include in JSON logs
    """
    global _logging_configured, _default_service_name

    _default_service_name = service_name

    # Get root logger for alma
    root_logger = logging.getLogger("alma")
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Remove existing handlers
    root_logger.handlers = []

    # Create handler based on output
    if output == "stderr":
        handler = logging.StreamHandler(sys.stderr)
    elif output == "stdout":
        handler = logging.StreamHandler(sys.stdout)
    else:
        handler = logging.FileHandler(output)

    # Create formatter based on format type
    if format_type.lower() == "json":
        formatter = JSONFormatter(
            service_name=service_name,
            extra_fields=extra_fields,
        )
    else:
        formatter = TextFormatter()

    handler.setFormatter(formatter)
    root_logger.addHandler(handler)

    _logging_configured = True


def get_logger(name: str) -> StructuredLogger:
    """
    Get a structured logger for the given name.

    Args:
        name: Logger name (typically __name__)

    Returns:
        StructuredLogger instance
    """
    global _loggers

    if name not in _loggers:
        _loggers[name] = StructuredLogger(name)

    return _loggers[name]
