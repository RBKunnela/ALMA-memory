"""
ALMA Logging Level Guidelines.

This module documents the standardized logging levels for ALMA.
All modules should follow these guidelines for consistent logging.

Logging Level Standards
=======================

DEBUG
-----
Use for detailed diagnostic information useful during development
and debugging. This level should NOT be enabled in production under
normal circumstances.

Examples:
- Cache hits/misses for individual queries
- Internal state changes (e.g., "Updated work item: {item_id}")
- Function entry/exit with parameters
- Detailed timing breakdowns
- Query/embedding details

INFO
----
Use for high-level operation completion and significant lifecycle events.
These logs confirm that things are working as expected.

Examples:
- Service startup/shutdown
- Memory retrieval/learning operations completed (with summary metrics)
- Scheduled job execution completed
- Configuration loaded successfully
- Database/storage connections established
- Batch operations completed with counts

WARNING
-------
Use for recoverable issues that may indicate problems but don't prevent
operation. The system continues to function but something unexpected happened.

Examples:
- Agent has no defined scope, using defaults
- Optional feature unavailable (e.g., "aiohttp not installed")
- Failed to retrieve optional data (e.g., git commands timeout)
- Near-quota conditions
- Deprecated feature usage
- Retry attempts
- Missing optional configuration

ERROR
-----
Use for failures that need attention. These indicate that an operation
failed and likely requires investigation or intervention.

Examples:
- Storage operation failures
- Failed to process required data
- Authentication/authorization failures
- Configuration errors preventing operation
- Unrecoverable API errors
- Data corruption detected
- Resource exhaustion

CRITICAL
--------
Use for severe failures that may cause application shutdown or
complete loss of functionality. These require immediate attention.

Examples:
- Database connection permanently lost
- Critical configuration missing
- Unrecoverable system state
- Security breach detected

Implementation Notes
====================

1. Logger Initialization:
   - Use standard logging: logger = logging.getLogger(__name__)
   - For structured logging features, also use: structured_logger = get_logger(__name__)

2. Message Format:
   - Start with action or subject (e.g., "Memory retrieval completed", "Failed to save heuristic")
   - Include relevant context as structured fields
   - Keep messages concise but informative

3. Structured Logging:
   - Use structured_logger for operations that benefit from structured fields
   - Pass context as keyword arguments for JSON serialization
   - Example: structured_logger.info("Retrieved memories", agent=agent, count=count)

4. Exception Logging:
   - Use logger.error(..., exc_info=True) or logger.exception() for exceptions
   - Include context about what operation was being attempted

5. Performance Logging:
   - Log duration for operations > 100ms at INFO level
   - Log internal operation timing at DEBUG level

Examples
========

# DEBUG - Internal diagnostics
logger.debug(f"Cache hit for query: {query[:50]}...")
logger.debug(f"Updated heuristic: {heuristic_id}")
logger.debug(f"Saved outcome: {outcome.id}")

# INFO - Operation completions
logger.info(f"Memory retrieval completed: {count} items in {duration_ms}ms")
logger.info(f"Cleanup scheduler started (interval: {interval}s)")
logger.info("ChromaDB storage closed")

# WARNING - Recoverable issues
logger.warning(f"Agent '{agent}' has no defined scope, using defaults")
logger.warning("aiohttp not installed - webhook delivery unavailable")
logger.warning(f"Failed to retrieve memories: {e}")

# ERROR - Failures requiring attention
logger.error(f"Failed to save heuristic {heuristic_id}: {e}")
logger.error(f"Redis connection error: {e}")
logger.error("Config file not found, cannot proceed")
"""

# Log level constants for programmatic use
import logging

# Map of operation types to recommended log levels
OPERATION_LOG_LEVELS = {
    # Storage operations
    "save_success": logging.DEBUG,
    "save_failure": logging.ERROR,
    "batch_save_success": logging.DEBUG,
    "delete_success": logging.DEBUG,
    "delete_failure": logging.ERROR,

    # Retrieval operations
    "cache_hit": logging.DEBUG,
    "cache_miss": logging.DEBUG,
    "cache_invalidate": logging.DEBUG,
    "retrieval_complete": logging.INFO,

    # Learning operations
    "learn_complete": logging.INFO,
    "heuristic_updated": logging.DEBUG,

    # Lifecycle events
    "service_start": logging.INFO,
    "service_stop": logging.INFO,
    "connection_established": logging.INFO,
    "config_loaded": logging.INFO,

    # Warnings
    "missing_scope": logging.WARNING,
    "optional_feature_unavailable": logging.WARNING,
    "retry_attempt": logging.WARNING,
    "deprecation": logging.WARNING,

    # Errors
    "operation_failure": logging.ERROR,
    "connection_failure": logging.ERROR,
    "validation_failure": logging.ERROR,
}


def get_recommended_level(operation: str) -> int:
    """
    Get the recommended log level for a given operation type.

    Args:
        operation: The operation type (e.g., "save_success", "cache_hit")

    Returns:
        Logging level constant (e.g., logging.DEBUG, logging.INFO)
    """
    return OPERATION_LOG_LEVELS.get(operation, logging.INFO)
