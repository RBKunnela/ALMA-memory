"""
Unit tests for ALMA schema migration framework.
"""

import shutil
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from alma.storage.migrations.base import (
    Migration,
    MigrationRegistry,
    SchemaVersion,
)
from alma.storage.migrations.runner import MigrationRunner
from alma.storage.migrations.version_stores import (
    FileBasedVersionStore,
    SQLiteVersionStore,
)


class TestSchemaVersion:
    """Tests for SchemaVersion."""

    def test_version_ordering(self):
        """Test that versions are ordered correctly."""
        v1 = SchemaVersion(version="1.0.0")
        v2 = SchemaVersion(version="1.0.1")
        v3 = SchemaVersion(version="1.1.0")
        v4 = SchemaVersion(version="2.0.0")

        assert v1 < v2
        assert v2 < v3
        assert v3 < v4
        assert not v4 < v1

    def test_version_parsing(self):
        """Test version string parsing."""
        versions = ["1.0.0", "1.2.3", "2.0.0-beta", "10.20.30"]
        for v in versions:
            sv = SchemaVersion(version=v)
            assert sv.version == v

    def test_version_with_metadata(self):
        """Test version with all metadata."""
        now = datetime.now(timezone.utc)
        sv = SchemaVersion(
            version="1.0.0",
            applied_at=now,
            description="Test migration",
            checksum="abc123",
        )

        assert sv.version == "1.0.0"
        assert sv.applied_at == now
        assert sv.description == "Test migration"
        assert sv.checksum == "abc123"


class TestMigrationRegistry:
    """Tests for MigrationRegistry."""

    def test_register_migration(self):
        """Test registering a migration."""
        registry = MigrationRegistry()

        class TestMigration(Migration):
            version = "1.0.0"
            description = "Test"

            def upgrade(self, connection: Any) -> None:
                pass

        registry.register(TestMigration)
        assert registry.get_migration("1.0.0") is TestMigration

    def test_register_backend_specific_migration(self):
        """Test registering a backend-specific migration."""
        registry = MigrationRegistry()

        class SQLiteMigration(Migration):
            version = "1.0.0"
            description = "SQLite specific"

            def upgrade(self, connection: Any) -> None:
                pass

        class PostgresMigration(Migration):
            version = "1.0.0"
            description = "PostgreSQL specific"

            def upgrade(self, connection: Any) -> None:
                pass

        registry.register(SQLiteMigration, backend="sqlite")
        registry.register(PostgresMigration, backend="postgresql")

        assert registry.get_migration("1.0.0", backend="sqlite") is SQLiteMigration
        assert (
            registry.get_migration("1.0.0", backend="postgresql") is PostgresMigration
        )

    def test_get_all_migrations_ordered(self):
        """Test that migrations are returned in version order."""
        registry = MigrationRegistry()

        versions = ["1.2.0", "1.0.0", "2.0.0", "1.1.0"]
        for v in versions:

            class M(Migration):
                version = v
                description = f"Migration {v}"

                def upgrade(self, connection: Any) -> None:
                    pass

            # Create unique class for each version
            M.version = v
            registry.register(type(f"Migration_{v.replace('.', '_')}", (M,), {}))

        all_migrations = registry.get_all_migrations()
        returned_versions = [m.version for m in all_migrations]

        # Should be sorted
        assert returned_versions == sorted(
            returned_versions, key=SchemaVersion._parse_version
        )

    def test_get_pending_migrations(self):
        """Test getting pending migrations."""
        registry = MigrationRegistry()

        for v in ["1.0.0", "1.1.0", "1.2.0"]:

            class M(Migration):
                version = v
                description = f"Migration {v}"

                def upgrade(self, connection: Any) -> None:
                    pass

            M.version = v
            registry.register(type(f"Migration_{v.replace('.', '_')}", (M,), {}))

        # From fresh install, all are pending
        pending = registry.get_pending_migrations(None)
        assert len(pending) == 3

        # From 1.0.0, two are pending
        pending = registry.get_pending_migrations("1.0.0")
        assert len(pending) == 2
        assert all(
            SchemaVersion._parse_version(m.version)
            > SchemaVersion._parse_version("1.0.0")
            for m in pending
        )

        # From 1.2.0, none are pending
        pending = registry.get_pending_migrations("1.2.0")
        assert len(pending) == 0


class TestSQLiteVersionStore:
    """Tests for SQLiteVersionStore."""

    @pytest.fixture
    def version_store(self):
        """Create temporary SQLite version store."""
        temp_dir = tempfile.mkdtemp()
        db_path = Path(temp_dir) / "test.db"
        store = SQLiteVersionStore(db_path)
        yield store
        shutil.rmtree(temp_dir)

    def test_initial_version_is_none(self, version_store):
        """Test that initial version is None."""
        assert version_store.get_current_version() is None

    def test_record_and_get_version(self, version_store):
        """Test recording and getting a version."""
        version = SchemaVersion(
            version="1.0.0",
            description="Initial schema",
        )
        version_store.record_version(version)

        assert version_store.get_current_version() == "1.0.0"

    def test_version_history(self, version_store):
        """Test getting version history."""
        for v in ["1.0.0", "1.1.0", "1.2.0"]:
            version = SchemaVersion(version=v, description=f"Version {v}")
            version_store.record_version(version)

        history = version_store.get_version_history()
        assert len(history) == 3
        assert [h.version for h in history] == ["1.0.0", "1.1.0", "1.2.0"]

    def test_remove_version(self, version_store):
        """Test removing a version."""
        for v in ["1.0.0", "1.1.0"]:
            version = SchemaVersion(version=v)
            version_store.record_version(version)

        version_store.remove_version("1.1.0")

        assert version_store.get_current_version() == "1.0.0"
        history = version_store.get_version_history()
        assert len(history) == 1


class TestFileBasedVersionStore:
    """Tests for FileBasedVersionStore."""

    @pytest.fixture
    def version_store(self):
        """Create temporary file-based version store."""
        temp_dir = tempfile.mkdtemp()
        store = FileBasedVersionStore(Path(temp_dir))
        yield store
        shutil.rmtree(temp_dir)

    def test_initial_version_is_none(self, version_store):
        """Test that initial version is None."""
        assert version_store.get_current_version() is None

    def test_record_and_get_version(self, version_store):
        """Test recording and getting a version."""
        version = SchemaVersion(
            version="1.0.0",
            description="Initial schema",
        )
        version_store.record_version(version)

        assert version_store.get_current_version() == "1.0.0"

    def test_version_history(self, version_store):
        """Test getting version history."""
        for v in ["1.0.0", "1.1.0"]:
            version = SchemaVersion(version=v)
            version_store.record_version(version)

        history = version_store.get_version_history()
        assert len(history) == 2


class TestMigrationRunner:
    """Tests for MigrationRunner."""

    @pytest.fixture
    def temp_db(self):
        """Create temporary SQLite database."""
        temp_dir = tempfile.mkdtemp()
        db_path = Path(temp_dir) / "test.db"
        yield db_path
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def version_store(self, temp_db):
        """Create version store."""
        return SQLiteVersionStore(temp_db)

    @pytest.fixture
    def test_registry(self):
        """Create test registry with sample migrations."""
        registry = MigrationRegistry()

        class Migration100(Migration):
            version = "1.0.0"
            description = "Create test table"

            def upgrade(self, connection: Any) -> None:
                cursor = connection.cursor()
                cursor.execute(
                    "CREATE TABLE IF NOT EXISTS test_table (id INTEGER PRIMARY KEY)"
                )
                connection.commit()

            def downgrade(self, connection: Any) -> None:
                cursor = connection.cursor()
                cursor.execute("DROP TABLE IF EXISTS test_table")
                connection.commit()

        class Migration110(Migration):
            version = "1.1.0"
            description = "Add name column"

            def upgrade(self, connection: Any) -> None:
                cursor = connection.cursor()
                cursor.execute("ALTER TABLE test_table ADD COLUMN name TEXT")
                connection.commit()

            def downgrade(self, connection: Any) -> None:
                # SQLite doesn't support DROP COLUMN easily
                pass

        registry.register(Migration100)
        registry.register(Migration110)

        return registry

    def test_needs_migration_fresh_db(self, version_store, test_registry):
        """Test detecting need for migration on fresh database."""
        runner = MigrationRunner(version_store, test_registry)
        assert runner.needs_migration()

    def test_migrate_fresh_db(self, temp_db, version_store, test_registry):
        """Test migrating a fresh database."""
        runner = MigrationRunner(version_store, test_registry)

        conn = sqlite3.connect(temp_db)
        try:
            applied = runner.migrate(conn)
            assert applied == ["1.0.0", "1.1.0"]
            assert version_store.get_current_version() == "1.1.0"

            # Verify table was created
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='test_table'"
            )
            assert cursor.fetchone() is not None
        finally:
            conn.close()

    def test_migrate_dry_run(self, temp_db, version_store, test_registry):
        """Test dry run migration."""
        runner = MigrationRunner(version_store, test_registry)

        conn = sqlite3.connect(temp_db)
        try:
            applied = runner.migrate(conn, dry_run=True)
            assert applied == ["1.0.0", "1.1.0"]

            # Version should not be recorded
            assert version_store.get_current_version() is None

            # Table should not be created
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='test_table'"
            )
            assert cursor.fetchone() is None
        finally:
            conn.close()

    def test_migrate_to_specific_version(self, temp_db, version_store, test_registry):
        """Test migrating to a specific version."""
        runner = MigrationRunner(version_store, test_registry)

        conn = sqlite3.connect(temp_db)
        try:
            applied = runner.migrate(conn, target_version="1.0.0")
            assert applied == ["1.0.0"]
            assert version_store.get_current_version() == "1.0.0"

            # Only first migration applied
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(test_table)")
            columns = [row[1] for row in cursor.fetchall()]
            assert "name" not in columns
        finally:
            conn.close()

    def test_no_pending_migrations(self, temp_db, version_store, test_registry):
        """Test when all migrations are already applied."""
        runner = MigrationRunner(version_store, test_registry)

        conn = sqlite3.connect(temp_db)
        try:
            # Apply all migrations
            runner.migrate(conn)

            # Try to migrate again
            applied = runner.migrate(conn)
            assert applied == []
            assert not runner.needs_migration()
        finally:
            conn.close()

    def test_get_status(self, temp_db, version_store, test_registry):
        """Test getting migration status."""
        runner = MigrationRunner(version_store, test_registry)

        # Before migration
        status = runner.get_status()
        assert status["current_version"] is None
        assert status["pending_count"] == 2
        assert status["needs_migration"] is True

        conn = sqlite3.connect(temp_db)
        try:
            runner.migrate(conn)
        finally:
            conn.close()

        # After migration
        status = runner.get_status()
        assert status["current_version"] == "1.1.0"
        assert status["pending_count"] == 0
        assert status["needs_migration"] is False
        assert len(status["history"]) == 2


class TestMigrationDecorator:
    """Tests for migration registration decorator."""

    def test_register_migration_decorator(self):
        """Test using the decorator to register migrations."""
        # Use a separate registry to avoid polluting global
        test_registry = MigrationRegistry()

        class DecoratedMigration(Migration):
            version = "99.0.0"
            description = "Decorated migration"

            def upgrade(self, connection: Any) -> None:
                pass

        test_registry.register(DecoratedMigration)

        # Should be registered in test registry
        assert test_registry.get_migration("99.0.0") is DecoratedMigration


class TestSQLiteStorageMigration:
    """Integration tests for SQLite storage migration."""

    @pytest.fixture
    def storage(self):
        """Create temporary SQLite storage."""
        temp_dir = tempfile.mkdtemp()
        db_path = Path(temp_dir) / "test.db"

        from alma.storage.sqlite_local import SQLiteStorage

        storage = SQLiteStorage(db_path=db_path, auto_migrate=True)
        yield storage
        shutil.rmtree(temp_dir)

    def test_storage_has_migration_methods(self, storage):
        """Test that storage has migration methods."""
        assert hasattr(storage, "get_schema_version")
        assert hasattr(storage, "get_migration_status")
        assert hasattr(storage, "migrate")
        assert hasattr(storage, "rollback")

    def test_storage_migration_status(self, storage):
        """Test getting migration status from storage."""
        status = storage.get_migration_status()
        assert "current_version" in status
        assert "pending_count" in status
        assert "migration_supported" in status
        assert status["migration_supported"] is True

    def test_storage_schema_version(self, storage):
        """Test getting schema version from storage."""
        # After auto-migrate, should have a version
        version = storage.get_schema_version()
        # May be None, "1.0.0", or "1.1.0" depending on migrations applied
        assert version is None or version in ("1.0.0", "1.1.0")
