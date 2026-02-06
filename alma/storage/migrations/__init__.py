"""
ALMA Schema Migration Framework.

Provides version tracking and migration capabilities for storage backends.
"""

from alma.storage.migrations.base import (
    Migration,
    MigrationError,
    MigrationRegistry,
    SchemaVersion,
)
from alma.storage.migrations.runner import MigrationRunner

__all__ = [
    "Migration",
    "MigrationError",
    "MigrationRegistry",
    "MigrationRunner",
    "SchemaVersion",
]
