"""
Storage Backend Factory - Decouples backend selection from implementation.

Applies Factory Pattern to reduce coupling between storage backends and
their consumers. Previously backends were directly imported/instantiated
throughout the codebase. Now centralized through this factory.

IMPROVEMENTS:
- Reduces coupling score by separating backend instantiation
- Enables easier backend swapping
- Improves testability (can mock factory)
- Centralizes backend configuration
"""

from typing import Dict, Optional, Any, Type
import logging

from alma.storage.base import StorageBackend

logger = logging.getLogger(__name__)


class StorageFactory:
    """Factory for creating storage backend instances."""

    _backends: Dict[str, Type[StorageBackend]] = {}
    _default_backend: Optional[str] = None

    @classmethod
    def register(cls, name: str, backend_class: Type[StorageBackend]) -> None:
        """Register a storage backend type."""
        cls._backends[name] = backend_class
        logger.info(f"Registered storage backend: {name}")

    @classmethod
    def create(
        cls,
        backend_type: str,
        **kwargs: Any,
    ) -> StorageBackend:
        """
        Create a storage backend instance.

        Args:
            backend_type: Type of backend (sqlite, postgresql, azure_cosmos, etc.)
            **kwargs: Backend-specific configuration

        Returns:
            StorageBackend instance

        Raises:
            ValueError: If backend type not registered
        """
        if backend_type not in cls._backends:
            available = list(cls._backends.keys())
            raise ValueError(
                f"Unknown backend type: {backend_type}. "
                f"Available: {available}"
            )

        backend_class = cls._backends[backend_type]
        return backend_class(**kwargs)

    @classmethod
    def get_available_backends(cls) -> list[str]:
        """Get list of registered backend types."""
        return list(cls._backends.keys())

    @classmethod
    def set_default(cls, backend_type: str) -> None:
        """Set default backend type."""
        if backend_type not in cls._backends:
            raise ValueError(f"Backend {backend_type} not registered")
        cls._default_backend = backend_type

    @classmethod
    def create_default(cls, **kwargs: Any) -> StorageBackend:
        """Create instance of default backend type."""
        if cls._default_backend is None:
            raise ValueError("No default backend configured")
        return cls.create(cls._default_backend, **kwargs)


# Register all backends
def _register_backends() -> None:
    """Register all available storage backends."""
    try:
        from alma.storage.sqlite_local import SQLiteStorage
        StorageFactory.register("sqlite", SQLiteStorage)
    except ImportError:
        pass

    try:
        from alma.storage.postgresql import PostgreSQLStorage
        StorageFactory.register("postgresql", PostgreSQLStorage)
    except ImportError:
        pass

    try:
        from alma.storage.azure_cosmos import AzureCosmosStorage
        StorageFactory.register("azure_cosmos", AzureCosmosStorage)
    except ImportError:
        pass

    try:
        from alma.storage.chroma import ChromaStorage
        StorageFactory.register("chroma", ChromaStorage)
    except ImportError:
        pass

    try:
        from alma.storage.qdrant import QdrantStorage
        StorageFactory.register("qdrant", QdrantStorage)
    except ImportError:
        pass

    try:
        from alma.storage.pinecone import PineconeStorage
        StorageFactory.register("pinecone", PineconeStorage)
    except ImportError:
        pass

    try:
        from alma.storage.file_based import FileBasedStorage
        StorageFactory.register("file", FileBasedStorage)
    except ImportError:
        pass


# Auto-register on import
_register_backends()
