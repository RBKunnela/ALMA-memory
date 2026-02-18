"""
ALMA Consolidation Exception Hierarchy.

Wraps all consolidation-specific errors, abstracting away implementation details
(e.g., LLM provider, storage backend, cache systems).

This fixes the boundary violation where callers had to understand OpenAI,
Redis, or other implementation-specific exceptions.
"""


class ConsolidationError(Exception):
    """Base exception for all consolidation errors."""

    pass


class LLMError(ConsolidationError):
    """Raised when LLM API call fails (timeout, rate limit, auth, connection)."""

    pass


class InvalidLLMResponse(ConsolidationError):
    """Raised when LLM response has invalid structure (not JSON, missing fields)."""

    pass


class CacheError(ConsolidationError):
    """Raised when cache operation fails."""

    pass


class ValidationError(ConsolidationError):
    """Raised when validation of consolidation inputs or outputs fails."""

    pass


class StorageError(ConsolidationError):
    """Raised when storage backend operation fails."""

    pass
