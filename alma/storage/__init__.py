"""ALMA Storage Backends."""

from alma.storage.archive import (
    ArchiveConfig,
    ArchivedMemory,
    ArchiveReason,
    ArchiveStats,
)
from alma.storage.base import StorageBackend
from alma.storage.constants import (
    AZURE_COSMOS_CONTAINER_NAMES,
    POSTGRESQL_TABLE_NAMES,
    SQLITE_TABLE_NAMES,
    MemoryType,
    get_table_name,
    get_table_names,
)
from alma.storage.file_based import FileBasedStorage
from alma.storage.migrations import (
    Migration,
    MigrationError,
    MigrationRegistry,
    MigrationRunner,
    SchemaVersion,
)
from alma.storage.sqlite_local import SQLiteStorage

# Azure Cosmos DB is optional - requires azure-cosmos package
try:
    from alma.storage.azure_cosmos import AzureCosmosStorage

    _HAS_AZURE = True
except ImportError:
    AzureCosmosStorage = None  # type: ignore
    _HAS_AZURE = False

# PostgreSQL is optional - requires psycopg package
try:
    from alma.storage.postgresql import PostgreSQLStorage

    _HAS_POSTGRES = True
except ImportError:
    PostgreSQLStorage = None  # type: ignore
    _HAS_POSTGRES = False

# Qdrant is optional - requires qdrant-client package
try:
    from alma.storage.qdrant import QdrantStorage

    _HAS_QDRANT = True
except ImportError:
    QdrantStorage = None  # type: ignore
    _HAS_QDRANT = False

# ChromaDB is optional - requires chromadb package
try:
    from alma.storage.chroma import ChromaStorage

    _HAS_CHROMA = True
except ImportError:
    ChromaStorage = None  # type: ignore
    _HAS_CHROMA = False

# Pinecone is optional - requires pinecone-client package
try:
    from alma.storage.pinecone import PineconeStorage

    _HAS_PINECONE = True
except ImportError:
    PineconeStorage = None  # type: ignore
    _HAS_PINECONE = False

# Atlas / Chefe 1756: anti-pattern write guard on every storage save_* door
# (not only LearningProtocol.learn). install is idempotent.
from alma.storage.write_guard_hooks import (  # noqa: E402
    enforce_storage_write_guard,
    install_storage_write_guards,
)

install_storage_write_guards(FileBasedStorage)
install_storage_write_guards(SQLiteStorage)
if _HAS_AZURE and AzureCosmosStorage is not None:
    install_storage_write_guards(AzureCosmosStorage)
if _HAS_POSTGRES and PostgreSQLStorage is not None:
    install_storage_write_guards(PostgreSQLStorage)
if _HAS_QDRANT and QdrantStorage is not None:
    install_storage_write_guards(QdrantStorage)
if _HAS_CHROMA and ChromaStorage is not None:
    install_storage_write_guards(ChromaStorage)
if _HAS_PINECONE and PineconeStorage is not None:
    install_storage_write_guards(PineconeStorage)

__all__ = [
    # Storage backends
    "StorageBackend",
    "FileBasedStorage",
    "SQLiteStorage",
    "AzureCosmosStorage",
    "PostgreSQLStorage",
    "QdrantStorage",
    "ChromaStorage",
    "PineconeStorage",
    # Migration framework
    "Migration",
    "MigrationError",
    "MigrationRegistry",
    "MigrationRunner",
    "SchemaVersion",
    # Archive system (v0.7.0+)
    "ArchivedMemory",
    "ArchiveConfig",
    "ArchiveReason",
    "ArchiveStats",
    # Constants for consistent naming
    "MemoryType",
    "get_table_name",
    "get_table_names",
    "POSTGRESQL_TABLE_NAMES",
    "SQLITE_TABLE_NAMES",
    "AZURE_COSMOS_CONTAINER_NAMES",
    # Write guard (storage layer)
    "install_storage_write_guards",
    "enforce_storage_write_guard",
]
