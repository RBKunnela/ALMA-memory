"""
ALMA Migration Framework - Migration Runner.

Executes migrations and manages schema version tracking.
"""

import hashlib
import logging
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Protocol, Type

from alma.storage.migrations.base import (
    Migration,
    MigrationError,
    MigrationRegistry,
    SchemaVersion,
    get_registry,
)

logger = logging.getLogger(__name__)


class VersionStore(Protocol):
    """Protocol for schema version storage."""

    def get_current_version(self) -> Optional[str]:
        """Get the current schema version."""
        ...

    def get_version_history(self) -> List[SchemaVersion]:
        """Get all applied versions in order."""
        ...

    def record_version(self, version: SchemaVersion) -> None:
        """Record a new version."""
        ...

    def remove_version(self, version: str) -> None:
        """Remove a version record (for rollback)."""
        ...


class MigrationRunner:
    """
    Executes schema migrations and tracks versions.

    Handles:
    - Forward migrations (upgrades)
    - Backward migrations (rollbacks)
    - Version tracking
    - Pre/post migration checks
    - Dry run mode
    """

    # Current schema version for fresh installations
    CURRENT_SCHEMA_VERSION = "1.0.0"

    def __init__(
        self,
        version_store: VersionStore,
        registry: Optional[MigrationRegistry] = None,
        backend: Optional[str] = None,
    ):
        """
        Initialize migration runner.

        Args:
            version_store: Storage for version tracking
            registry: Migration registry (uses global if not provided)
            backend: Backend name for filtering migrations
        """
        self.version_store = version_store
        self.registry = registry or get_registry()
        self.backend = backend
        self._hooks: Dict[str, List[Callable]] = {
            "pre_migrate": [],
            "post_migrate": [],
            "pre_rollback": [],
            "post_rollback": [],
        }

    def add_hook(self, event: str, callback: Callable) -> None:
        """
        Add a hook callback for migration events.

        Args:
            event: Event name (pre_migrate, post_migrate, etc.)
            callback: Function to call
        """
        if event in self._hooks:
            self._hooks[event].append(callback)

    def _run_hooks(self, event: str, *args: Any, **kwargs: Any) -> None:
        """Run all hooks for an event."""
        for callback in self._hooks.get(event, []):
            try:
                callback(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Hook {callback.__name__} failed: {e}")

    def get_current_version(self) -> Optional[str]:
        """Get the current schema version."""
        return self.version_store.get_current_version()

    def get_pending_migrations(self) -> List[Type[Migration]]:
        """Get list of migrations that need to be applied."""
        current = self.get_current_version()
        return self.registry.get_pending_migrations(current, self.backend)

    def needs_migration(self) -> bool:
        """Check if there are pending migrations."""
        return len(self.get_pending_migrations()) > 0

    def migrate(
        self,
        connection: Any,
        target_version: Optional[str] = None,
        dry_run: bool = False,
    ) -> List[str]:
        """
        Apply pending migrations.

        Args:
            connection: Database connection or storage instance
            target_version: Optional target version (applies all if not specified)
            dry_run: If True, show what would be done without making changes

        Returns:
            List of applied migration versions

        Raises:
            MigrationError: If a migration fails
        """
        current = self.get_current_version()
        pending = self.registry.get_pending_migrations(current, self.backend)

        if target_version:
            target_tuple = SchemaVersion._parse_version(target_version)
            pending = [
                m
                for m in pending
                if SchemaVersion._parse_version(m.version) <= target_tuple
            ]

        if not pending:
            logger.info("No pending migrations")
            return []

        applied = []
        logger.info(f"Found {len(pending)} pending migrations")

        for migration_class in pending:
            version = migration_class.version
            migration = migration_class()

            if dry_run:
                logger.info(
                    f"[DRY RUN] Would apply migration {version}: {migration.description}"
                )
                applied.append(version)
                continue

            logger.info(f"Applying migration {version}: {migration.description}")

            try:
                # Run pre-check
                if not migration.pre_check(connection):
                    raise MigrationError(
                        f"Pre-check failed for migration {version}",
                        version=version,
                    )

                # Run pre-migrate hooks
                self._run_hooks("pre_migrate", migration, connection)

                # Apply migration
                migration.upgrade(connection)

                # Run post-check
                if not migration.post_check(connection):
                    raise MigrationError(
                        f"Post-check failed for migration {version}",
                        version=version,
                    )

                # Record version
                schema_version = SchemaVersion(
                    version=version,
                    applied_at=datetime.now(timezone.utc),
                    description=migration.description,
                    checksum=self._compute_checksum(migration_class),
                )
                self.version_store.record_version(schema_version)

                # Run post-migrate hooks
                self._run_hooks("post_migrate", migration, connection)

                applied.append(version)
                logger.info(f"Successfully applied migration {version}")

            except MigrationError:
                raise
            except Exception as e:
                raise MigrationError(
                    f"Migration {version} failed: {str(e)}",
                    version=version,
                    cause=e,
                ) from e

        return applied

    def rollback(
        self,
        connection: Any,
        target_version: str,
        dry_run: bool = False,
    ) -> List[str]:
        """
        Roll back to a target version.

        Args:
            connection: Database connection or storage instance
            target_version: Version to roll back to
            dry_run: If True, show what would be done without making changes

        Returns:
            List of rolled back migration versions

        Raises:
            MigrationError: If a rollback fails
        """
        current = self.get_current_version()
        if current is None:
            logger.info("No migrations to roll back")
            return []

        rollback_migrations = self.registry.get_rollback_migrations(
            current, target_version, self.backend
        )

        if not rollback_migrations:
            logger.info("No migrations to roll back")
            return []

        rolled_back = []
        logger.info(f"Rolling back {len(rollback_migrations)} migrations")

        for migration_class in rollback_migrations:
            version = migration_class.version
            migration = migration_class()

            if dry_run:
                logger.info(f"[DRY RUN] Would roll back migration {version}")
                rolled_back.append(version)
                continue

            logger.info(f"Rolling back migration {version}")

            try:
                # Run pre-rollback hooks
                self._run_hooks("pre_rollback", migration, connection)

                # Apply downgrade
                migration.downgrade(connection)

                # Remove version record
                self.version_store.remove_version(version)

                # Run post-rollback hooks
                self._run_hooks("post_rollback", migration, connection)

                rolled_back.append(version)
                logger.info(f"Successfully rolled back migration {version}")

            except NotImplementedError as e:
                raise MigrationError(
                    f"Migration {version} does not support rollback",
                    version=version,
                ) from e
            except Exception as e:
                raise MigrationError(
                    f"Rollback of {version} failed: {str(e)}",
                    version=version,
                    cause=e,
                ) from e

        return rolled_back

    def get_status(self) -> Dict[str, Any]:
        """
        Get migration status information.

        Returns:
            Dict with current version, pending migrations, and history
        """
        current = self.get_current_version()
        pending = self.get_pending_migrations()
        history = self.version_store.get_version_history()

        return {
            "current_version": current,
            "target_version": self.CURRENT_SCHEMA_VERSION,
            "pending_count": len(pending),
            "pending_versions": [m.version for m in pending],
            "applied_count": len(history),
            "history": [
                {
                    "version": v.version,
                    "applied_at": v.applied_at.isoformat(),
                    "description": v.description,
                }
                for v in history
            ],
            "needs_migration": len(pending) > 0,
        }

    @staticmethod
    def _compute_checksum(migration_class: Type[Migration]) -> str:
        """Compute checksum of migration for integrity tracking."""
        import inspect

        source = inspect.getsource(migration_class)
        return hashlib.sha256(source.encode()).hexdigest()[:16]
