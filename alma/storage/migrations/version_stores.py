"""
ALMA Migration Framework - Version Stores.

Implementations of version tracking for different storage backends.
"""

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

from alma.storage.migrations.base import SchemaVersion

logger = logging.getLogger(__name__)


class SQLiteVersionStore:
    """
    Version store using SQLite for tracking schema versions.

    Creates a `_schema_versions` table to track applied migrations.
    """

    def __init__(self, db_path: Path):
        """
        Initialize SQLite version store.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self._init_version_table()

    @contextmanager
    def _get_connection(self):
        """Get database connection with context manager."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_version_table(self) -> None:
        """Create schema versions table if it doesn't exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS _schema_versions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    version TEXT NOT NULL UNIQUE,
                    applied_at TEXT NOT NULL,
                    description TEXT,
                    checksum TEXT
                )
            """)
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_schema_version "
                "ON _schema_versions(version)"
            )

    def get_current_version(self) -> Optional[str]:
        """Get the current (latest) schema version."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT version FROM _schema_versions
                ORDER BY id DESC LIMIT 1
            """)
            row = cursor.fetchone()
            return row["version"] if row else None

    def get_version_history(self) -> List[SchemaVersion]:
        """Get all applied versions in chronological order."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT version, applied_at, description, checksum
                FROM _schema_versions
                ORDER BY id ASC
            """)
            rows = cursor.fetchall()

        return [
            SchemaVersion(
                version=row["version"],
                applied_at=datetime.fromisoformat(row["applied_at"]),
                description=row["description"] or "",
                checksum=row["checksum"],
            )
            for row in rows
        ]

    def record_version(self, version: SchemaVersion) -> None:
        """Record a new schema version."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO _schema_versions (version, applied_at, description, checksum)
                VALUES (?, ?, ?, ?)
                """,
                (
                    version.version,
                    version.applied_at.isoformat(),
                    version.description,
                    version.checksum,
                ),
            )
        logger.debug(f"Recorded schema version {version.version}")

    def remove_version(self, version: str) -> None:
        """Remove a version record (for rollback)."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM _schema_versions WHERE version = ?",
                (version,),
            )
        logger.debug(f"Removed schema version {version}")

    def has_version_table(self) -> bool:
        """Check if the version table exists."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name='_schema_versions'
            """)
            return cursor.fetchone() is not None


class PostgreSQLVersionStore:
    """
    Version store using PostgreSQL for tracking schema versions.

    Creates an `_schema_versions` table in the configured schema.
    """

    def __init__(self, pool: Any, schema: str = "public"):
        """
        Initialize PostgreSQL version store.

        Args:
            pool: psycopg connection pool
            schema: Database schema name
        """
        self._pool = pool
        self.schema = schema
        self._init_version_table()

    @contextmanager
    def _get_connection(self):
        """Get database connection from pool."""
        with self._pool.connection() as conn:
            yield conn

    def _init_version_table(self) -> None:
        """Create schema versions table if it doesn't exist."""
        with self._get_connection() as conn:
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.schema}._schema_versions (
                    id SERIAL PRIMARY KEY,
                    version TEXT NOT NULL UNIQUE,
                    applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    description TEXT,
                    checksum TEXT
                )
            """)
            conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_schema_version
                ON {self.schema}._schema_versions(version)
            """)
            conn.commit()

    def get_current_version(self) -> Optional[str]:
        """Get the current (latest) schema version."""
        with self._get_connection() as conn:
            cursor = conn.execute(f"""
                SELECT version FROM {self.schema}._schema_versions
                ORDER BY id DESC LIMIT 1
            """)
            row = cursor.fetchone()
            return row["version"] if row else None

    def get_version_history(self) -> List[SchemaVersion]:
        """Get all applied versions in chronological order."""
        with self._get_connection() as conn:
            cursor = conn.execute(f"""
                SELECT version, applied_at, description, checksum
                FROM {self.schema}._schema_versions
                ORDER BY id ASC
            """)
            rows = cursor.fetchall()

        return [
            SchemaVersion(
                version=row["version"],
                applied_at=row["applied_at"]
                if isinstance(row["applied_at"], datetime)
                else datetime.fromisoformat(str(row["applied_at"])),
                description=row["description"] or "",
                checksum=row["checksum"],
            )
            for row in rows
        ]

    def record_version(self, version: SchemaVersion) -> None:
        """Record a new schema version."""
        with self._get_connection() as conn:
            conn.execute(
                f"""
                INSERT INTO {self.schema}._schema_versions
                (version, applied_at, description, checksum)
                VALUES (%s, %s, %s, %s)
                """,
                (
                    version.version,
                    version.applied_at,
                    version.description,
                    version.checksum,
                ),
            )
            conn.commit()
        logger.debug(f"Recorded schema version {version.version}")

    def remove_version(self, version: str) -> None:
        """Remove a version record (for rollback)."""
        with self._get_connection() as conn:
            conn.execute(
                f"DELETE FROM {self.schema}._schema_versions WHERE version = %s",
                (version,),
            )
            conn.commit()
        logger.debug(f"Removed schema version {version}")

    def has_version_table(self) -> bool:
        """Check if the version table exists."""
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_schema = %s
                    AND table_name = '_schema_versions'
                )
            """, (self.schema,))
            row = cursor.fetchone()
            return row["exists"] if row else False


class FileBasedVersionStore:
    """
    Version store using a JSON file for tracking schema versions.

    Useful for file-based storage backends.
    """

    def __init__(self, storage_dir: Path):
        """
        Initialize file-based version store.

        Args:
            storage_dir: Directory to store version file
        """
        self.storage_dir = Path(storage_dir)
        self.version_file = self.storage_dir / "_schema_versions.json"
        self._ensure_file_exists()

    def _ensure_file_exists(self) -> None:
        """Create version file if it doesn't exist."""
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        if not self.version_file.exists():
            self._write_versions([])

    def _read_versions(self) -> List[dict]:
        """Read versions from file."""
        try:
            with open(self.version_file, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return []

    def _write_versions(self, versions: List[dict]) -> None:
        """Write versions to file."""
        with open(self.version_file, "w") as f:
            json.dump(versions, f, indent=2, default=str)

    def get_current_version(self) -> Optional[str]:
        """Get the current (latest) schema version."""
        versions = self._read_versions()
        if not versions:
            return None
        return versions[-1]["version"]

    def get_version_history(self) -> List[SchemaVersion]:
        """Get all applied versions in chronological order."""
        versions = self._read_versions()
        return [
            SchemaVersion(
                version=v["version"],
                applied_at=datetime.fromisoformat(v["applied_at"]),
                description=v.get("description", ""),
                checksum=v.get("checksum"),
            )
            for v in versions
        ]

    def record_version(self, version: SchemaVersion) -> None:
        """Record a new schema version."""
        versions = self._read_versions()
        versions.append(
            {
                "version": version.version,
                "applied_at": version.applied_at.isoformat(),
                "description": version.description,
                "checksum": version.checksum,
            }
        )
        self._write_versions(versions)
        logger.debug(f"Recorded schema version {version.version}")

    def remove_version(self, version: str) -> None:
        """Remove a version record (for rollback)."""
        versions = self._read_versions()
        versions = [v for v in versions if v["version"] != version]
        self._write_versions(versions)
        logger.debug(f"Removed schema version {version}")
