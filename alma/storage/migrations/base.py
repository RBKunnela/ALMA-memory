"""
ALMA Migration Framework - Base Classes.

Provides abstract migration classes and version tracking utilities.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Type

logger = logging.getLogger(__name__)


class MigrationError(Exception):
    """Exception raised when a migration fails."""

    def __init__(
        self,
        message: str,
        version: Optional[str] = None,
        cause: Optional[Exception] = None,
    ):
        self.version = version
        self.cause = cause
        super().__init__(message)


@dataclass
class SchemaVersion:
    """
    Represents a schema version record.

    Attributes:
        version: Semantic version string (e.g., "1.0.0")
        applied_at: When the migration was applied
        description: Human-readable description of changes
        checksum: Optional hash for integrity verification
    """

    version: str
    applied_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    description: str = ""
    checksum: Optional[str] = None

    def __lt__(self, other: "SchemaVersion") -> bool:
        """Compare versions for sorting."""
        return self._parse_version(self.version) < self._parse_version(other.version)

    @staticmethod
    def _parse_version(version: str) -> tuple:
        """Parse version string into comparable tuple."""
        parts = version.split(".")
        result = []
        for part in parts:
            try:
                result.append(int(part))
            except ValueError:
                result.append(part)
        return tuple(result)


class Migration(ABC):
    """
    Abstract base class for schema migrations.

    Subclasses must implement upgrade() and optionally downgrade().

    Example:
        class AddTagsColumn(Migration):
            version = "1.1.0"
            description = "Add tags column to heuristics table"

            def upgrade(self, connection):
                connection.execute(
                    "ALTER TABLE heuristics ADD COLUMN tags TEXT"
                )

            def downgrade(self, connection):
                connection.execute(
                    "ALTER TABLE heuristics DROP COLUMN tags"
                )
    """

    # These must be set by subclasses
    version: str = ""
    description: str = ""
    # Optional: previous version this migration depends on
    depends_on: Optional[str] = None

    @abstractmethod
    def upgrade(self, connection: Any) -> None:
        """
        Apply the migration.

        Args:
            connection: Database connection or storage instance
        """
        pass

    def downgrade(self, connection: Any) -> None:
        """
        Revert the migration (optional).

        Args:
            connection: Database connection or storage instance

        Raises:
            NotImplementedError: If downgrade is not supported
        """
        raise NotImplementedError(
            f"Downgrade not implemented for migration {self.version}"
        )

    def pre_check(self, connection: Any) -> bool:
        """
        Optional pre-migration check.

        Override to verify prerequisites before migration.

        Args:
            connection: Database connection

        Returns:
            True if migration can proceed, False otherwise
        """
        return True

    def post_check(self, connection: Any) -> bool:
        """
        Optional post-migration verification.

        Override to verify migration was successful.

        Args:
            connection: Database connection

        Returns:
            True if migration was successful
        """
        return True


class MigrationRegistry:
    """
    Registry for available migrations.

    Manages migration discovery, ordering, and execution planning.
    """

    def __init__(self) -> None:
        self._migrations: Dict[str, Type[Migration]] = {}
        self._backend_migrations: Dict[str, Dict[str, Type[Migration]]] = {}

    def register(
        self, migration_class: Type[Migration], backend: Optional[str] = None
    ) -> Type[Migration]:
        """
        Register a migration class.

        Args:
            migration_class: The migration class to register
            backend: Optional backend name (e.g., "sqlite", "postgresql")

        Returns:
            The migration class (for use as decorator)
        """
        version = migration_class.version
        if not version:
            raise ValueError(
                f"Migration {migration_class.__name__} must have a version"
            )

        if backend:
            if backend not in self._backend_migrations:
                self._backend_migrations[backend] = {}
            self._backend_migrations[backend][version] = migration_class
            logger.debug(f"Registered migration {version} for backend {backend}")
        else:
            self._migrations[version] = migration_class
            logger.debug(f"Registered global migration {version}")

        return migration_class

    def get_migration(
        self, version: str, backend: Optional[str] = None
    ) -> Optional[Type[Migration]]:
        """
        Get a migration class by version.

        Args:
            version: Version string to look up
            backend: Optional backend name

        Returns:
            Migration class or None if not found
        """
        if backend and backend in self._backend_migrations:
            migration = self._backend_migrations[backend].get(version)
            if migration:
                return migration
        return self._migrations.get(version)

    def get_all_migrations(
        self, backend: Optional[str] = None
    ) -> List[Type[Migration]]:
        """
        Get all migrations in version order.

        Args:
            backend: Optional backend name to filter migrations

        Returns:
            List of migration classes sorted by version
        """
        migrations = dict(self._migrations)
        if backend and backend in self._backend_migrations:
            migrations.update(self._backend_migrations[backend])

        return [
            cls
            for _, cls in sorted(
                migrations.items(),
                key=lambda x: SchemaVersion._parse_version(x[0]),
            )
        ]

    def get_pending_migrations(
        self,
        current_version: Optional[str],
        backend: Optional[str] = None,
    ) -> List[Type[Migration]]:
        """
        Get migrations that need to be applied.

        Args:
            current_version: Current schema version (None if fresh install)
            backend: Optional backend name

        Returns:
            List of migration classes that need to be applied
        """
        all_migrations = self.get_all_migrations(backend)

        if current_version is None:
            return all_migrations

        current = SchemaVersion._parse_version(current_version)
        return [
            m
            for m in all_migrations
            if SchemaVersion._parse_version(m.version) > current
        ]

    def get_rollback_migrations(
        self,
        current_version: str,
        target_version: str,
        backend: Optional[str] = None,
    ) -> List[Type[Migration]]:
        """
        Get migrations that need to be rolled back.

        Args:
            current_version: Current schema version
            target_version: Target version to roll back to
            backend: Optional backend name

        Returns:
            List of migration classes to roll back (in reverse order)
        """
        all_migrations = self.get_all_migrations(backend)

        current = SchemaVersion._parse_version(current_version)
        target = SchemaVersion._parse_version(target_version)

        rollback = [
            m
            for m in all_migrations
            if target < SchemaVersion._parse_version(m.version) <= current
        ]

        # Return in reverse order for rollback
        return list(reversed(rollback))


# Global registry instance
_global_registry = MigrationRegistry()


def register_migration(
    backend: Optional[str] = None,
) -> Callable[[Type[Migration]], Type[Migration]]:
    """
    Decorator to register a migration class.

    Args:
        backend: Optional backend name

    Example:
        @register_migration()
        class MyMigration(Migration):
            version = "1.0.0"
            ...

        @register_migration(backend="postgresql")
        class PostgresSpecificMigration(Migration):
            version = "1.0.1"
            ...
    """

    def decorator(cls: Type[Migration]) -> Type[Migration]:
        return _global_registry.register(cls, backend)

    return decorator


def get_registry() -> MigrationRegistry:
    """Get the global migration registry."""
    return _global_registry
