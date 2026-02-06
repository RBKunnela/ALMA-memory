"""
ALMA Schema Migration v1.0.0 - Initial Schema.

This migration establishes the baseline schema for existing databases.
For fresh installations, this creates the initial tables.
For existing databases, this records the current state.
"""

from typing import Any

from alma.storage.migrations.base import Migration, register_migration


@register_migration(backend="sqlite")
class SQLiteInitialSchema(Migration):
    """
    SQLite initial schema migration.

    This represents the baseline schema as of ALMA v0.5.x.
    For existing databases, this is a no-op that records the baseline.
    """

    version = "1.0.0"
    description = "Initial ALMA schema with core tables"

    def upgrade(self, connection: Any) -> None:
        """
        Create or verify initial schema.

        For SQLite, this creates all core tables if they don't exist.
        """
        cursor = connection.cursor()

        # Heuristics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS heuristics (
                id TEXT PRIMARY KEY,
                agent TEXT NOT NULL,
                project_id TEXT NOT NULL,
                condition TEXT NOT NULL,
                strategy TEXT NOT NULL,
                confidence REAL DEFAULT 0.0,
                occurrence_count INTEGER DEFAULT 0,
                success_count INTEGER DEFAULT 0,
                last_validated TEXT,
                created_at TEXT,
                metadata TEXT
            )
        """)
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_heuristics_project_agent "
            "ON heuristics(project_id, agent)"
        )

        # Outcomes table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS outcomes (
                id TEXT PRIMARY KEY,
                agent TEXT NOT NULL,
                project_id TEXT NOT NULL,
                task_type TEXT,
                task_description TEXT NOT NULL,
                success INTEGER DEFAULT 0,
                strategy_used TEXT,
                duration_ms INTEGER,
                error_message TEXT,
                user_feedback TEXT,
                timestamp TEXT,
                metadata TEXT
            )
        """)
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_outcomes_project_agent "
            "ON outcomes(project_id, agent)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_outcomes_task_type "
            "ON outcomes(project_id, agent, task_type)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_outcomes_timestamp "
            "ON outcomes(project_id, timestamp)"
        )

        # User preferences table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS preferences (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                category TEXT,
                preference TEXT NOT NULL,
                source TEXT,
                confidence REAL DEFAULT 1.0,
                timestamp TEXT,
                metadata TEXT
            )
        """)
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_preferences_user ON preferences(user_id)"
        )

        # Domain knowledge table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS domain_knowledge (
                id TEXT PRIMARY KEY,
                agent TEXT NOT NULL,
                project_id TEXT NOT NULL,
                domain TEXT,
                fact TEXT NOT NULL,
                source TEXT,
                confidence REAL DEFAULT 1.0,
                last_verified TEXT,
                metadata TEXT
            )
        """)
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_domain_knowledge_project_agent "
            "ON domain_knowledge(project_id, agent)"
        )

        # Anti-patterns table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS anti_patterns (
                id TEXT PRIMARY KEY,
                agent TEXT NOT NULL,
                project_id TEXT NOT NULL,
                pattern TEXT NOT NULL,
                why_bad TEXT,
                better_alternative TEXT,
                occurrence_count INTEGER DEFAULT 1,
                last_seen TEXT,
                created_at TEXT,
                metadata TEXT
            )
        """)
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_anti_patterns_project_agent "
            "ON anti_patterns(project_id, agent)"
        )

        # Embeddings table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_type TEXT NOT NULL,
                memory_id TEXT NOT NULL,
                embedding BLOB NOT NULL,
                UNIQUE(memory_type, memory_id)
            )
        """)
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_embeddings_type ON embeddings(memory_type)"
        )

        connection.commit()

    def downgrade(self, connection: Any) -> None:
        """
        Downgrade is not supported for initial schema.

        Rolling back the initial schema would destroy all data.
        """
        raise NotImplementedError(
            "Cannot roll back initial schema - this would destroy all data"
        )

    def pre_check(self, connection: Any) -> bool:
        """Verify we can connect to the database."""
        try:
            cursor = connection.cursor()
            cursor.execute("SELECT 1")
            return True
        except Exception:
            return False


@register_migration(backend="postgresql")
class PostgreSQLInitialSchema(Migration):
    """
    PostgreSQL initial schema migration.

    This represents the baseline schema as of ALMA v0.5.x.
    """

    version = "1.0.0"
    description = "Initial ALMA schema with core tables and pgvector support"

    def __init__(self, schema: str = "public", embedding_dim: int = 384):
        self._schema = schema
        self._embedding_dim = embedding_dim

    def upgrade(self, connection: Any) -> None:
        """Create or verify initial PostgreSQL schema."""
        schema = self._schema
        embedding_dim = self._embedding_dim

        # Try to enable pgvector
        pgvector_available = False
        try:
            connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
            connection.commit()
            pgvector_available = True
        except Exception:
            connection.rollback()

        vector_type = f"VECTOR({embedding_dim})" if pgvector_available else "BYTEA"

        # Heuristics table
        connection.execute(f"""
            CREATE TABLE IF NOT EXISTS {schema}.alma_heuristics (
                id TEXT PRIMARY KEY,
                agent TEXT NOT NULL,
                project_id TEXT NOT NULL,
                condition TEXT NOT NULL,
                strategy TEXT NOT NULL,
                confidence REAL DEFAULT 0.0,
                occurrence_count INTEGER DEFAULT 0,
                success_count INTEGER DEFAULT 0,
                last_validated TIMESTAMPTZ,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                metadata JSONB,
                embedding {vector_type}
            )
        """)
        connection.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_heuristics_project_agent
            ON {schema}.alma_heuristics(project_id, agent)
        """)
        connection.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_heuristics_confidence
            ON {schema}.alma_heuristics(project_id, confidence DESC)
        """)

        # Outcomes table
        connection.execute(f"""
            CREATE TABLE IF NOT EXISTS {schema}.alma_outcomes (
                id TEXT PRIMARY KEY,
                agent TEXT NOT NULL,
                project_id TEXT NOT NULL,
                task_type TEXT,
                task_description TEXT NOT NULL,
                success BOOLEAN DEFAULT FALSE,
                strategy_used TEXT,
                duration_ms INTEGER,
                error_message TEXT,
                user_feedback TEXT,
                timestamp TIMESTAMPTZ DEFAULT NOW(),
                metadata JSONB,
                embedding {vector_type}
            )
        """)
        connection.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_outcomes_project_agent
            ON {schema}.alma_outcomes(project_id, agent)
        """)
        connection.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_outcomes_task_type
            ON {schema}.alma_outcomes(project_id, agent, task_type)
        """)
        connection.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_outcomes_timestamp
            ON {schema}.alma_outcomes(project_id, timestamp DESC)
        """)

        # User preferences table
        connection.execute(f"""
            CREATE TABLE IF NOT EXISTS {schema}.alma_preferences (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                category TEXT,
                preference TEXT NOT NULL,
                source TEXT,
                confidence REAL DEFAULT 1.0,
                timestamp TIMESTAMPTZ DEFAULT NOW(),
                metadata JSONB
            )
        """)
        connection.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_preferences_user
            ON {schema}.alma_preferences(user_id)
        """)

        # Domain knowledge table
        connection.execute(f"""
            CREATE TABLE IF NOT EXISTS {schema}.alma_domain_knowledge (
                id TEXT PRIMARY KEY,
                agent TEXT NOT NULL,
                project_id TEXT NOT NULL,
                domain TEXT,
                fact TEXT NOT NULL,
                source TEXT,
                confidence REAL DEFAULT 1.0,
                last_verified TIMESTAMPTZ DEFAULT NOW(),
                metadata JSONB,
                embedding {vector_type}
            )
        """)
        connection.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_domain_knowledge_project_agent
            ON {schema}.alma_domain_knowledge(project_id, agent)
        """)
        connection.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_domain_knowledge_confidence
            ON {schema}.alma_domain_knowledge(project_id, confidence DESC)
        """)

        # Anti-patterns table
        connection.execute(f"""
            CREATE TABLE IF NOT EXISTS {schema}.alma_anti_patterns (
                id TEXT PRIMARY KEY,
                agent TEXT NOT NULL,
                project_id TEXT NOT NULL,
                pattern TEXT NOT NULL,
                why_bad TEXT,
                better_alternative TEXT,
                occurrence_count INTEGER DEFAULT 1,
                last_seen TIMESTAMPTZ DEFAULT NOW(),
                created_at TIMESTAMPTZ DEFAULT NOW(),
                metadata JSONB,
                embedding {vector_type}
            )
        """)
        connection.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_anti_patterns_project_agent
            ON {schema}.alma_anti_patterns(project_id, agent)
        """)

        # Create vector indexes if pgvector available
        if pgvector_available:
            for table in [
                "alma_heuristics",
                "alma_outcomes",
                "alma_domain_knowledge",
                "alma_anti_patterns",
            ]:
                try:
                    connection.execute(f"""
                        CREATE INDEX IF NOT EXISTS idx_{table}_embedding
                        ON {schema}.{table}
                        USING hnsw (embedding vector_cosine_ops)
                        WITH (m = 16, ef_construction = 64)
                    """)
                except Exception:
                    pass  # Index may already exist or not be possible

        connection.commit()

    def downgrade(self, connection: Any) -> None:
        """Cannot roll back initial schema."""
        raise NotImplementedError(
            "Cannot roll back initial schema - this would destroy all data"
        )


@register_migration()
class FileBasedInitialSchema(Migration):
    """
    File-based storage initial schema migration.

    This is mostly a no-op since file-based storage creates files on demand.
    """

    version = "1.0.0"
    description = "Initial ALMA file-based storage schema"

    def upgrade(self, storage: Any) -> None:
        """Verify file-based storage is properly initialized."""
        # File-based storage creates files on demand, so this is a no-op
        pass

    def downgrade(self, storage: Any) -> None:
        """Cannot roll back initial schema."""
        raise NotImplementedError("Cannot roll back initial schema")
