"""ALMA Storage Backends."""

from alma.storage.base import StorageBackend
from alma.storage.file_based import FileBasedStorage
from alma.storage.sqlite_local import SQLiteStorage

# Azure Cosmos DB is optional - requires azure-cosmos package
try:
    from alma.storage.azure_cosmos import AzureCosmosStorage
    _HAS_AZURE = True
except ImportError:
    AzureCosmosStorage = None  # type: ignore
    _HAS_AZURE = False

__all__ = [
    "StorageBackend",
    "FileBasedStorage",
    "SQLiteStorage",
    "AzureCosmosStorage",
]
