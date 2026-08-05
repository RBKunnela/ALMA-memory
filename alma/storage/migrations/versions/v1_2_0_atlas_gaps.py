"""
ALMA Schema Migration v1.2.0 — Atlas gaps (Chefe 561 / Code-Hub 1624).

G1: Persist VerificationStatus columns on core memory tables
G3: Same columns dual-backend (sqlite + postgresql)
G4: alma_forget_audit append-only table
"""

from typing import Any

from alma.storage.migrations.base import Migration, register_migration

VERIFICATION_COLUMNS_SQLITE = [
    ("verification_status", "TEXT"),
    ("verification_method", "TEXT"),
    ("verification_confidence", "REAL"),
    ("verification_reason", "TEXT"),
    ("verified_at", "TEXT"),
    ("contradicting_source", "TEXT"),
]

VERIFICATION_COLUMNS_PG = [
    ("verification_status", "TEXT"),
    ("verification_method", "TEXT"),
    ("verification_confidence", "DOUBLE PRECISION"),
    ("verification_reason", "TEXT"),
    ("verified_at", "TIMESTAMPTZ"),
    ("contradicting_source", "TEXT"),
]

MEMORY_TABLES = (
    "heuristics",
    "outcomes",
    "domain_knowledge",
    "preferences",
    "anti_patterns",
)


def _sqlite_add_column_if_missing(
    cursor: Any, table: str, column: str, coltype: str
) -> None:
    cursor.execute(f"PRAGMA table_info({table})")
    existing = {row[1] for row in cursor.fetchall()}
    if column not in existing:
        cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {coltype}")


def _pg_add_column_if_missing(
    cursor: Any, schema: str, table: str, column: str, coltype: str
) -> None:
    cursor.execute(
        """
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = %s AND table_name = %s AND column_name = %s
        """,
        (schema, table, column),
    )
    if cursor.fetchone() is None:
        cursor.execute(
            f"ALTER TABLE {schema}.{table} ADD COLUMN IF NOT EXISTS {column} {coltype}"
        )


@register_migration(backend="sqlite")
class SQLiteAtlasGapsMigration(Migration):
    """SQLite: verification columns + forget_audit (Atlas G1/G3/G4)."""

    version = "1.2.0"
    description = "Persist verification + forget_audit (Agent Memory Atlas gaps)"
    depends_on = "1.1.0"

    def upgrade(self, connection: Any) -> None:
        cursor = connection.cursor()
        for table in MEMORY_TABLES:
            # Table may not exist on partial DBs — create is owned by sqlite_local
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (table,),
            )
            if not cursor.fetchone():
                continue
            for col, coltype in VERIFICATION_COLUMNS_SQLITE:
                _sqlite_add_column_if_missing(cursor, table, col, coltype)

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS alma_forget_audit (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                memory_type TEXT NOT NULL,
                memory_id TEXT NOT NULL,
                agent TEXT,
                reason TEXT,
                strategy TEXT,
                pruned_at TEXT NOT NULL,
                metadata TEXT
            )
            """
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_forget_audit_project "
            "ON alma_forget_audit(project_id, pruned_at)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_forget_audit_type "
            "ON alma_forget_audit(memory_type, memory_id)"
        )

    def downgrade(self, connection: Any) -> None:
        # SQLite cannot DROP COLUMN portably pre-3.35; leave columns, drop audit only
        cursor = connection.cursor()
        cursor.execute("DROP TABLE IF EXISTS alma_forget_audit")


@register_migration(backend="postgresql")
class PostgreSQLAtlasGapsMigration(Migration):
    """PostgreSQL: verification columns + forget_audit (Atlas G1/G3/G4)."""

    version = "1.2.0"
    description = "Persist verification + forget_audit (Agent Memory Atlas gaps)"
    depends_on = "1.1.0"

    def upgrade(self, connection: Any) -> None:
        cursor = connection.cursor()
        schema = getattr(connection, "_schema", "public")
        for table in MEMORY_TABLES:
            for col, coltype in VERIFICATION_COLUMNS_PG:
                try:
                    _pg_add_column_if_missing(cursor, schema, table, col, coltype)
                except Exception:
                    # table may not exist yet
                    pass

        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {schema}.alma_forget_audit (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                memory_type TEXT NOT NULL,
                memory_id TEXT NOT NULL,
                agent TEXT,
                reason TEXT,
                strategy TEXT,
                pruned_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                metadata JSONB
            )
            """
        )
        cursor.execute(
            f"""
            CREATE INDEX IF NOT EXISTS idx_forget_audit_project
            ON {schema}.alma_forget_audit(project_id, pruned_at DESC)
            """
        )

    def downgrade(self, connection: Any) -> None:
        cursor = connection.cursor()
        schema = getattr(connection, "_schema", "public")
        cursor.execute(f"DROP TABLE IF EXISTS {schema}.alma_forget_audit")
