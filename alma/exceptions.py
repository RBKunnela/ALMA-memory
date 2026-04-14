"""
ALMA Exception Hierarchy

Custom exceptions for the ALMA memory system, providing clear error
categorization for configuration, storage, embedding, retrieval,
and extraction operations.
"""


class ALMAError(Exception):
    """Base exception for all ALMA errors."""

    pass


class ConfigurationError(ALMAError):
    """Raised when configuration is invalid or missing."""

    pass


class ScopeViolationError(ALMAError):
    """Raised when an agent attempts to learn outside its scope."""

    pass


class StorageError(ALMAError):
    """Raised when storage operations fail."""

    pass


class EmbeddingError(ALMAError):
    """Raised when embedding generation fails."""

    pass


class RetrievalError(ALMAError):
    """Raised when memory retrieval fails."""

    pass


class ExtractionError(ALMAError):
    """Raised when fact extraction fails."""

    pass
