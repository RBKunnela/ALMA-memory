"""
ALMA Schema Migration v1.1.0 - Workflow Context Layer.

This migration adds support for AGtestari Workflow Studio integration:
- Checkpoints table: Crash recovery and state persistence
- Workflow Outcomes table: Learning from completed workflows
- Artifact Links table: Connecting artifacts to memories
- Workflow scope columns on existing tables

Sprint 0 Task 0.2, 0.3, 0.4, 0.7
Designed by: @data-analyst (Dara)
Reviewed by: @architect (Aria)
"""

from typing import Any

from alma.storage.migrations.base import Migration, register_migration


# =============================================================================
# POSTGRESQL MIGRATIONS
# =============================================================================

@register_migration(backend="postgresql")
class PostgreSQLWorkflowContextMigration(Migration):
    """
    PostgreSQL migration for workflow context layer.

    Includes pgvector support for semantic search on workflow outcomes.
    """

    version = "1.1.0"
    description = "Add workflow context layer (checkpoints, workflow_outcomes, artifact_links)"
    depends_on = "1.0.0"

    def upgrade(self, connection: Any) -> None:
        """Apply workflow context schema changes."""
        cursor = connection.cursor()

        # Get schema from connection or default to public
        schema = getattr(connection, '_schema', 'public')

        # Check if pgvector is available
        cursor.execute("""
            SELECT EXISTS (
                SELECT 1 FROM pg_extension WHERE extname = 'vector'
            )
        """)
        has_pgvector = cursor.fetchone()[0]

        # Determine embedding type
        # Default embedding dim for all-MiniLM-L6-v2
        embedding_dim = 384
        embedding_type = f"VECTOR({embedding_dim})" if has_pgvector else "BYTEA"

        # =====================================================================
        # TABLE 1: Checkpoints - Crash recovery and state persistence
        # =====================================================================
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {schema}.alma_checkpoints (
                -- Primary key
                id TEXT PRIMARY KEY,

                -- Workflow context (required)
                run_id TEXT NOT NULL,
                node_id TEXT NOT NULL,

                -- State data
                state_json JSONB NOT NULL,
                state_hash TEXT NOT NULL,  -- SHA256 hash for change detection

                -- Sequencing
                sequence_number INTEGER NOT NULL,

                -- Parallel execution support
                branch_id TEXT,  -- NULL for main branch
                parent_checkpoint_id TEXT REFERENCES {schema}.alma_checkpoints(id),

                -- Timestamps
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

                -- Extensibility
                metadata JSONB,

                -- Constraints
                CONSTRAINT uk_checkpoint_run_seq UNIQUE (run_id, sequence_number)
            )
        """)

        # Indexes for checkpoint queries
        cursor.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_checkpoints_run_seq
            ON {schema}.alma_checkpoints(run_id, sequence_number DESC)
        """)
        cursor.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_checkpoints_run_branch
            ON {schema}.alma_checkpoints(run_id, branch_id)
            WHERE branch_id IS NOT NULL
        """)
        cursor.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_checkpoints_created
            ON {schema}.alma_checkpoints(created_at)
        """)

        # Comment on table
        cursor.execute(f"""
            COMMENT ON TABLE {schema}.alma_checkpoints IS
            'Workflow state checkpoints for crash recovery. Each checkpoint captures state after a node completes.'
        """)

        # =====================================================================
        # TABLE 2: Workflow Outcomes - Learning from completed workflows
        # =====================================================================
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {schema}.alma_workflow_outcomes (
                -- Primary key
                id TEXT PRIMARY KEY,

                -- Multi-tenant hierarchy
                tenant_id TEXT NOT NULL DEFAULT 'default',
                workflow_id TEXT NOT NULL,
                workflow_version TEXT DEFAULT '1.0',
                run_id TEXT NOT NULL UNIQUE,  -- One outcome per run

                -- Outcome data
                success BOOLEAN NOT NULL,
                duration_ms INTEGER NOT NULL,

                -- Node statistics
                node_count INTEGER NOT NULL,
                nodes_succeeded INTEGER NOT NULL DEFAULT 0,
                nodes_failed INTEGER NOT NULL DEFAULT 0,

                -- Error tracking
                error_message TEXT,

                -- Artifacts (stored as JSON array of ArtifactRef)
                artifacts_json JSONB,

                -- Learning metrics
                learnings_extracted INTEGER DEFAULT 0,

                -- Timestamps
                timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),

                -- Semantic search (pgvector or fallback)
                embedding {embedding_type},

                -- Extensibility
                metadata JSONB,

                -- Constraints
                CONSTRAINT chk_nodes_count CHECK (
                    nodes_succeeded + nodes_failed <= node_count
                )
            )
        """)

        # Indexes for workflow outcome queries
        cursor.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_wo_tenant
            ON {schema}.alma_workflow_outcomes(tenant_id)
        """)
        cursor.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_wo_workflow
            ON {schema}.alma_workflow_outcomes(tenant_id, workflow_id)
        """)
        cursor.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_wo_success
            ON {schema}.alma_workflow_outcomes(tenant_id, success)
        """)
        cursor.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_wo_timestamp
            ON {schema}.alma_workflow_outcomes(timestamp DESC)
        """)

        # pgvector index for semantic search (if available)
        if has_pgvector:
            try:
                cursor.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_wo_embedding
                    ON {schema}.alma_workflow_outcomes
                    USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 100)
                """)
            except Exception:
                # IVFFlat requires data to build, skip if table is empty
                pass

        # Comment on table
        cursor.execute(f"""
            COMMENT ON TABLE {schema}.alma_workflow_outcomes IS
            'Aggregated outcomes from completed workflow runs. Used for learning patterns across workflows.'
        """)

        # =====================================================================
        # TABLE 3: Artifact Links - Connecting artifacts to memories
        # =====================================================================
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {schema}.alma_artifact_links (
                -- Primary key
                id TEXT PRIMARY KEY,

                -- Link to memory item
                memory_id TEXT NOT NULL,
                memory_type TEXT NOT NULL,  -- 'heuristic', 'outcome', 'domain_knowledge', etc.

                -- Artifact reference
                artifact_id TEXT NOT NULL,
                artifact_type TEXT NOT NULL,  -- 'screenshot', 'report', 'log', etc.

                -- Storage location (Cloudflare R2)
                storage_path TEXT NOT NULL,  -- e.g., 'r2://alma-artifacts/tenant/workflow/artifact.png'

                -- Integrity
                content_hash TEXT NOT NULL,  -- SHA256 for verification
                size_bytes INTEGER NOT NULL,

                -- Timestamps
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

                -- Extensibility
                metadata JSONB,

                -- Constraints
                CONSTRAINT chk_size_positive CHECK (size_bytes > 0)
            )
        """)

        # Indexes for artifact queries
        cursor.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_artifact_memory
            ON {schema}.alma_artifact_links(memory_id, memory_type)
        """)
        cursor.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_artifact_type
            ON {schema}.alma_artifact_links(artifact_type)
        """)
        cursor.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_artifact_created
            ON {schema}.alma_artifact_links(created_at)
        """)

        # Comment on table
        cursor.execute(f"""
            COMMENT ON TABLE {schema}.alma_artifact_links IS
            'Links between memory items and external artifacts stored in Cloudflare R2.'
        """)

        # =====================================================================
        # ALTER EXISTING TABLES: Add workflow scope columns
        # =====================================================================

        # Add workflow columns to heuristics (if they don't exist)
        self._add_column_if_not_exists(
            cursor, schema, 'alma_heuristics', 'tenant_id', "TEXT DEFAULT 'default'"
        )
        self._add_column_if_not_exists(
            cursor, schema, 'alma_heuristics', 'workflow_id', "TEXT"
        )
        self._add_column_if_not_exists(
            cursor, schema, 'alma_heuristics', 'run_id', "TEXT"
        )
        self._add_column_if_not_exists(
            cursor, schema, 'alma_heuristics', 'node_id', "TEXT"
        )

        # Add workflow columns to outcomes
        self._add_column_if_not_exists(
            cursor, schema, 'alma_outcomes', 'tenant_id', "TEXT DEFAULT 'default'"
        )
        self._add_column_if_not_exists(
            cursor, schema, 'alma_outcomes', 'workflow_id', "TEXT"
        )
        self._add_column_if_not_exists(
            cursor, schema, 'alma_outcomes', 'run_id', "TEXT"
        )
        self._add_column_if_not_exists(
            cursor, schema, 'alma_outcomes', 'node_id', "TEXT"
        )

        # Add workflow columns to domain_knowledge
        self._add_column_if_not_exists(
            cursor, schema, 'alma_domain_knowledge', 'tenant_id', "TEXT DEFAULT 'default'"
        )
        self._add_column_if_not_exists(
            cursor, schema, 'alma_domain_knowledge', 'workflow_id', "TEXT"
        )

        # Add workflow columns to anti_patterns
        self._add_column_if_not_exists(
            cursor, schema, 'alma_anti_patterns', 'tenant_id', "TEXT DEFAULT 'default'"
        )
        self._add_column_if_not_exists(
            cursor, schema, 'alma_anti_patterns', 'workflow_id', "TEXT"
        )

        # Add indexes for scope filtering
        cursor.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_heuristics_tenant
            ON {schema}.alma_heuristics(tenant_id)
            WHERE tenant_id IS NOT NULL
        """)
        cursor.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_heuristics_workflow
            ON {schema}.alma_heuristics(workflow_id)
            WHERE workflow_id IS NOT NULL
        """)
        cursor.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_outcomes_tenant
            ON {schema}.alma_outcomes(tenant_id)
            WHERE tenant_id IS NOT NULL
        """)
        cursor.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_outcomes_workflow
            ON {schema}.alma_outcomes(workflow_id)
            WHERE workflow_id IS NOT NULL
        """)

        connection.commit()

    def _add_column_if_not_exists(
        self, cursor: Any, schema: str, table: str, column: str, definition: str
    ) -> None:
        """Safely add a column if it doesn't exist."""
        cursor.execute(f"""
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_schema = '{schema}'
                    AND table_name = '{table}'
                    AND column_name = '{column}'
                ) THEN
                    ALTER TABLE {schema}.{table} ADD COLUMN {column} {definition};
                END IF;
            END
            $$;
        """)

    def downgrade(self, connection: Any) -> None:
        """Revert workflow context schema changes."""
        cursor = connection.cursor()
        schema = getattr(connection, '_schema', 'public')

        # Drop new tables (in reverse dependency order)
        cursor.execute(f"DROP TABLE IF EXISTS {schema}.alma_artifact_links CASCADE")
        cursor.execute(f"DROP TABLE IF EXISTS {schema}.alma_workflow_outcomes CASCADE")
        cursor.execute(f"DROP TABLE IF EXISTS {schema}.alma_checkpoints CASCADE")

        # Note: We don't remove columns from existing tables in downgrade
        # as it could cause data loss. They are nullable and won't affect existing code.

        connection.commit()


# =============================================================================
# SQLITE MIGRATIONS
# =============================================================================

@register_migration(backend="sqlite")
class SQLiteWorkflowContextMigration(Migration):
    """
    SQLite migration for workflow context layer.

    Uses BLOB for embeddings (no pgvector equivalent in SQLite).
    """

    version = "1.1.0"
    description = "Add workflow context layer (checkpoints, workflow_outcomes, artifact_links)"
    depends_on = "1.0.0"

    def upgrade(self, connection: Any) -> None:
        """Apply workflow context schema changes."""
        cursor = connection.cursor()

        # =====================================================================
        # TABLE 1: Checkpoints
        # Matches alma.workflow.checkpoint.Checkpoint dataclass
        # =====================================================================
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS checkpoints (
                id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                node_id TEXT NOT NULL,
                state TEXT NOT NULL,
                sequence_number INTEGER DEFAULT 0,
                branch_id TEXT,
                parent_checkpoint_id TEXT,
                state_hash TEXT,
                metadata TEXT,
                created_at TEXT NOT NULL
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_checkpoints_run
            ON checkpoints(run_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_checkpoints_run_branch
            ON checkpoints(run_id, branch_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_checkpoints_run_seq
            ON checkpoints(run_id, sequence_number DESC)
        """)

        # =====================================================================
        # TABLE 2: Workflow Outcomes
        # Matches alma.workflow.outcomes.WorkflowOutcome dataclass
        # =====================================================================
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS workflow_outcomes (
                id TEXT PRIMARY KEY,
                tenant_id TEXT,
                workflow_id TEXT NOT NULL,
                run_id TEXT NOT NULL,
                agent TEXT NOT NULL,
                project_id TEXT NOT NULL,
                result TEXT NOT NULL,
                summary TEXT,
                strategies_used TEXT,
                successful_patterns TEXT,
                failed_patterns TEXT,
                extracted_heuristics TEXT,
                extracted_anti_patterns TEXT,
                duration_seconds REAL,
                node_count INTEGER,
                error_message TEXT,
                metadata TEXT,
                created_at TEXT NOT NULL
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_wo_tenant
            ON workflow_outcomes(tenant_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_wo_workflow
            ON workflow_outcomes(workflow_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_wo_project_agent
            ON workflow_outcomes(project_id, agent)
        """)

        # =====================================================================
        # TABLE 3: Artifact Links
        # Matches alma.workflow.artifacts.ArtifactRef dataclass
        # =====================================================================
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS artifact_links (
                id TEXT PRIMARY KEY,
                memory_id TEXT NOT NULL,
                artifact_type TEXT NOT NULL,
                storage_url TEXT NOT NULL,
                filename TEXT,
                mime_type TEXT,
                size_bytes INTEGER,
                checksum TEXT,
                metadata TEXT,
                created_at TEXT NOT NULL
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_artifact_memory
            ON artifact_links(memory_id)
        """)

        # =====================================================================
        # ALTER EXISTING TABLES: Add workflow scope columns
        # =====================================================================
        # SQLite doesn't support IF NOT EXISTS for ALTER TABLE, so we need to check

        existing_tables = ['heuristics', 'outcomes', 'domain_knowledge', 'anti_patterns']
        new_columns = [
            ('tenant_id', "TEXT DEFAULT 'default'"),
            ('workflow_id', 'TEXT'),
            ('run_id', 'TEXT'),
            ('node_id', 'TEXT'),
        ]

        for table in existing_tables:
            # Check which columns already exist
            cursor.execute(f"PRAGMA table_info({table})")
            existing_cols = {row[1] for row in cursor.fetchall()}

            # Only add run_id and node_id to heuristics and outcomes
            cols_to_add = new_columns if table in ['heuristics', 'outcomes'] else new_columns[:2]

            for col_name, col_def in cols_to_add:
                if col_name not in existing_cols:
                    try:
                        cursor.execute(f"ALTER TABLE {table} ADD COLUMN {col_name} {col_def}")
                    except Exception:
                        pass  # Column might already exist

        # Add scope indexes
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_heuristics_tenant
            ON heuristics(tenant_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_heuristics_workflow
            ON heuristics(workflow_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_outcomes_tenant
            ON outcomes(tenant_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_outcomes_workflow
            ON outcomes(workflow_id)
        """)

        connection.commit()

    def downgrade(self, connection: Any) -> None:
        """Revert workflow context schema changes."""
        cursor = connection.cursor()

        # Drop new tables
        cursor.execute("DROP TABLE IF EXISTS artifact_links")
        cursor.execute("DROP TABLE IF EXISTS workflow_outcomes")
        cursor.execute("DROP TABLE IF EXISTS checkpoints")

        # Note: SQLite doesn't support DROP COLUMN easily
        # The added columns will remain but won't affect existing code

        connection.commit()
