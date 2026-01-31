"""
ALMA PostgreSQL Storage Backend.

Production-ready storage using PostgreSQL with pgvector extension for
native vector similarity search. Supports connection pooling.

Recommended for:
- Customer deployments (Azure PostgreSQL, AWS RDS, etc.)
- Self-hosted production environments
- High-availability requirements

v0.6.0 adds workflow context support:
- Checkpoint tables for crash recovery
- WorkflowOutcome tables for learning from workflows
- ArtifactRef tables for linking external files
- scope_filter parameter for workflow-scoped queries
"""

import json
import logging
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional

# numpy is optional - only needed for fallback similarity when pgvector unavailable
try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    np = None  # type: ignore
    NUMPY_AVAILABLE = False

from alma.storage.base import StorageBackend
from alma.storage.constants import POSTGRESQL_TABLE_NAMES, MemoryType
from alma.types import (
    AntiPattern,
    DomainKnowledge,
    Heuristic,
    Outcome,
    UserPreference,
)

if TYPE_CHECKING:
    from alma.workflow import ArtifactRef, Checkpoint, WorkflowOutcome

logger = logging.getLogger(__name__)

# Try to import psycopg (v3) with connection pooling
try:
    from psycopg.rows import dict_row
    from psycopg_pool import ConnectionPool

    PSYCOPG_AVAILABLE = True
except ImportError:
    PSYCOPG_AVAILABLE = False
    logger.warning(
        "psycopg not installed. Install with: pip install 'alma-memory[postgres]'"
    )


class PostgreSQLStorage(StorageBackend):
    """
    PostgreSQL storage backend with pgvector support.

    Uses native PostgreSQL vector operations for efficient similarity search.
    Falls back to application-level cosine similarity if pgvector is not installed.

    Database schema (uses canonical memory type names with alma_ prefix):
        - alma_heuristics: id, agent, project_id, condition, strategy, ...
        - alma_outcomes: id, agent, project_id, task_type, ...
        - alma_preferences: id, user_id, category, preference, ...
        - alma_domain_knowledge: id, agent, project_id, domain, fact, ...
        - alma_anti_patterns: id, agent, project_id, pattern, ...

    Vector search:
        - Uses pgvector extension if available
        - Embeddings stored as VECTOR type with cosine distance operator (<=>)

    Table names are derived from alma.storage.constants.POSTGRESQL_TABLE_NAMES
    for consistency across all storage backends.
    """

    # Table names from constants for consistent naming
    TABLE_NAMES = POSTGRESQL_TABLE_NAMES

    def __init__(
        self,
        host: str,
        port: int,
        database: str,
        user: str,
        password: str,
        embedding_dim: int = 384,
        pool_size: int = 10,
        schema: str = "public",
        ssl_mode: str = "prefer",
        auto_migrate: bool = True,
    ):
        """
        Initialize PostgreSQL storage.

        Args:
            host: Database host
            port: Database port
            database: Database name
            user: Database user
            password: Database password
            embedding_dim: Dimension of embedding vectors
            pool_size: Connection pool size
            schema: Database schema (default: public)
            ssl_mode: SSL mode (disable, allow, prefer, require, verify-ca, verify-full)
            auto_migrate: If True, automatically apply pending migrations on startup
        """
        if not PSYCOPG_AVAILABLE:
            raise ImportError(
                "psycopg not installed. Install with: pip install 'alma-memory[postgres]'"
            )

        self.embedding_dim = embedding_dim
        self.schema = schema
        self._pgvector_available = False

        # Migration support (lazy-loaded)
        self._migration_runner = None
        self._version_store = None

        # Build connection string
        conninfo = (
            f"host={host} port={port} dbname={database} "
            f"user={user} password={password} sslmode={ssl_mode}"
        )

        # Create connection pool
        self._pool = ConnectionPool(
            conninfo=conninfo,
            min_size=1,
            max_size=pool_size,
            kwargs={"row_factory": dict_row},
        )

        # Initialize database
        self._init_database()

        # Auto-migrate if enabled
        if auto_migrate:
            self._ensure_migrated()

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "PostgreSQLStorage":
        """Create instance from configuration."""
        pg_config = config.get("postgres", {})

        # Support environment variable expansion
        def get_value(key: str, default: Any = None) -> Any:
            value = pg_config.get(key, default)
            if (
                isinstance(value, str)
                and value.startswith("${")
                and value.endswith("}")
            ):
                env_var = value[2:-1]
                return os.environ.get(env_var, default)
            return value

        return cls(
            host=get_value("host", "localhost"),
            port=int(get_value("port", 5432)),
            database=get_value("database", "alma_memory"),
            user=get_value("user", "postgres"),
            password=get_value("password", ""),
            embedding_dim=int(config.get("embedding_dim", 384)),
            pool_size=int(get_value("pool_size", 10)),
            schema=get_value("schema", "public"),
            ssl_mode=get_value("ssl_mode", "prefer"),
        )

    @contextmanager
    def _get_connection(self):
        """Get database connection from pool."""
        with self._pool.connection() as conn:
            yield conn

    def _init_database(self):
        """Initialize database schema and pgvector extension."""
        with self._get_connection() as conn:
            # Try to enable pgvector extension
            try:
                conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
                conn.commit()
                self._pgvector_available = True
                logger.info("pgvector extension enabled")
            except Exception as e:
                conn.rollback()  # Important: rollback to clear aborted transaction
                logger.warning(f"pgvector not available: {e}. Using fallback search.")
                self._pgvector_available = False

            # Create tables
            vector_type = (
                f"VECTOR({self.embedding_dim})" if self._pgvector_available else "BYTEA"
            )

            # Heuristics table
            heuristics_table = self.TABLE_NAMES[MemoryType.HEURISTICS]
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.schema}.{heuristics_table} (
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
            conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_heuristics_project_agent
                ON {self.schema}.{heuristics_table}(project_id, agent)
            """)
            # Confidence index for efficient filtering by confidence score
            conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_heuristics_confidence
                ON {self.schema}.{heuristics_table}(project_id, confidence DESC)
            """)

            # Outcomes table
            outcomes_table = self.TABLE_NAMES[MemoryType.OUTCOMES]
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.schema}.{outcomes_table} (
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
            conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_outcomes_project_agent
                ON {self.schema}.{outcomes_table}(project_id, agent)
            """)
            conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_outcomes_task_type
                ON {self.schema}.{outcomes_table}(project_id, agent, task_type)
            """)
            conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_outcomes_timestamp
                ON {self.schema}.{outcomes_table}(project_id, timestamp DESC)
            """)

            # User preferences table
            preferences_table = self.TABLE_NAMES[MemoryType.PREFERENCES]
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.schema}.{preferences_table} (
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
            conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_preferences_user
                ON {self.schema}.{preferences_table}(user_id)
            """)

            # Domain knowledge table
            domain_knowledge_table = self.TABLE_NAMES[MemoryType.DOMAIN_KNOWLEDGE]
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.schema}.{domain_knowledge_table} (
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
            conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_domain_knowledge_project_agent
                ON {self.schema}.{domain_knowledge_table}(project_id, agent)
            """)
            # Confidence index for efficient filtering by confidence score
            conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_domain_knowledge_confidence
                ON {self.schema}.{domain_knowledge_table}(project_id, confidence DESC)
            """)

            # Anti-patterns table
            anti_patterns_table = self.TABLE_NAMES[MemoryType.ANTI_PATTERNS]
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.schema}.{anti_patterns_table} (
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
            conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_anti_patterns_project_agent
                ON {self.schema}.{anti_patterns_table}(project_id, agent)
            """)

            # Create vector indexes if pgvector available
            # Using HNSW instead of IVFFlat because HNSW can be built on empty tables
            # IVFFlat requires existing data to build, which causes silent failures on fresh databases
            if self._pgvector_available:
                # Vector-enabled tables use canonical memory type names
                vector_tables = [
                    self.TABLE_NAMES[mt] for mt in MemoryType.VECTOR_ENABLED
                ]
                for table in vector_tables:
                    try:
                        conn.execute(f"""
                            CREATE INDEX IF NOT EXISTS idx_{table}_embedding
                            ON {self.schema}.{table}
                            USING hnsw (embedding vector_cosine_ops)
                            WITH (m = 16, ef_construction = 64)
                        """)
                    except Exception as e:
                        logger.warning(f"Failed to create HNSW index for {table}: {e}")

            conn.commit()

    def _embedding_to_db(self, embedding: Optional[List[float]]) -> Any:
        """Convert embedding to database format."""
        if embedding is None:
            return None
        if self._pgvector_available:
            # pgvector expects string format: '[1.0, 2.0, 3.0]'
            return f"[{','.join(str(x) for x in embedding)}]"
        else:
            # Store as bytes (requires numpy)
            if not NUMPY_AVAILABLE:
                raise ImportError("numpy required for non-pgvector embedding storage")
            return np.array(embedding, dtype=np.float32).tobytes()

    def _embedding_from_db(self, value: Any) -> Optional[List[float]]:
        """Convert embedding from database format."""
        if value is None:
            return None
        if self._pgvector_available:
            # pgvector returns as string or list
            if isinstance(value, str):
                value = value.strip("[]")
                return [float(x) for x in value.split(",")]
            return list(value)
        else:
            # Stored as bytes (requires numpy)
            if not NUMPY_AVAILABLE or np is None:
                return None
            return np.frombuffer(value, dtype=np.float32).tolist()

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if not NUMPY_AVAILABLE or np is None:
            # Fallback to pure Python
            dot = sum(x * y for x, y in zip(a, b, strict=False))
            norm_a = sum(x * x for x in a) ** 0.5
            norm_b = sum(x * x for x in b) ** 0.5
            return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0
        a_arr = np.array(a)
        b_arr = np.array(b)
        return float(
            np.dot(a_arr, b_arr) / (np.linalg.norm(a_arr) * np.linalg.norm(b_arr))
        )

    # ==================== WRITE OPERATIONS ====================

    def save_heuristic(self, heuristic: Heuristic) -> str:
        """Save a heuristic."""
        with self._get_connection() as conn:
            conn.execute(
                f"""
                INSERT INTO {self.schema}.{self.TABLE_NAMES[MemoryType.HEURISTICS]}
                (id, agent, project_id, condition, strategy, confidence,
                 occurrence_count, success_count, last_validated, created_at, metadata, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    condition = EXCLUDED.condition,
                    strategy = EXCLUDED.strategy,
                    confidence = EXCLUDED.confidence,
                    occurrence_count = EXCLUDED.occurrence_count,
                    success_count = EXCLUDED.success_count,
                    last_validated = EXCLUDED.last_validated,
                    metadata = EXCLUDED.metadata,
                    embedding = EXCLUDED.embedding
                """,
                (
                    heuristic.id,
                    heuristic.agent,
                    heuristic.project_id,
                    heuristic.condition,
                    heuristic.strategy,
                    heuristic.confidence,
                    heuristic.occurrence_count,
                    heuristic.success_count,
                    heuristic.last_validated,
                    heuristic.created_at,
                    json.dumps(heuristic.metadata) if heuristic.metadata else None,
                    self._embedding_to_db(heuristic.embedding),
                ),
            )
            conn.commit()

        logger.debug(f"Saved heuristic: {heuristic.id}")
        return heuristic.id

    def save_outcome(self, outcome: Outcome) -> str:
        """Save an outcome."""
        with self._get_connection() as conn:
            conn.execute(
                f"""
                INSERT INTO {self.schema}.{self.TABLE_NAMES[MemoryType.OUTCOMES]}
                (id, agent, project_id, task_type, task_description, success,
                 strategy_used, duration_ms, error_message, user_feedback, timestamp, metadata, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    task_description = EXCLUDED.task_description,
                    success = EXCLUDED.success,
                    strategy_used = EXCLUDED.strategy_used,
                    duration_ms = EXCLUDED.duration_ms,
                    error_message = EXCLUDED.error_message,
                    user_feedback = EXCLUDED.user_feedback,
                    metadata = EXCLUDED.metadata,
                    embedding = EXCLUDED.embedding
                """,
                (
                    outcome.id,
                    outcome.agent,
                    outcome.project_id,
                    outcome.task_type,
                    outcome.task_description,
                    outcome.success,
                    outcome.strategy_used,
                    outcome.duration_ms,
                    outcome.error_message,
                    outcome.user_feedback,
                    outcome.timestamp,
                    json.dumps(outcome.metadata) if outcome.metadata else None,
                    self._embedding_to_db(outcome.embedding),
                ),
            )
            conn.commit()

        logger.debug(f"Saved outcome: {outcome.id}")
        return outcome.id

    def save_user_preference(self, preference: UserPreference) -> str:
        """Save a user preference."""
        with self._get_connection() as conn:
            conn.execute(
                f"""
                INSERT INTO {self.schema}.{self.TABLE_NAMES[MemoryType.PREFERENCES]}
                (id, user_id, category, preference, source, confidence, timestamp, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    preference = EXCLUDED.preference,
                    source = EXCLUDED.source,
                    confidence = EXCLUDED.confidence,
                    metadata = EXCLUDED.metadata
                """,
                (
                    preference.id,
                    preference.user_id,
                    preference.category,
                    preference.preference,
                    preference.source,
                    preference.confidence,
                    preference.timestamp,
                    json.dumps(preference.metadata) if preference.metadata else None,
                ),
            )
            conn.commit()

        logger.debug(f"Saved preference: {preference.id}")
        return preference.id

    def save_domain_knowledge(self, knowledge: DomainKnowledge) -> str:
        """Save domain knowledge."""
        with self._get_connection() as conn:
            conn.execute(
                f"""
                INSERT INTO {self.schema}.{self.TABLE_NAMES[MemoryType.DOMAIN_KNOWLEDGE]}
                (id, agent, project_id, domain, fact, source, confidence, last_verified, metadata, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    fact = EXCLUDED.fact,
                    source = EXCLUDED.source,
                    confidence = EXCLUDED.confidence,
                    last_verified = EXCLUDED.last_verified,
                    metadata = EXCLUDED.metadata,
                    embedding = EXCLUDED.embedding
                """,
                (
                    knowledge.id,
                    knowledge.agent,
                    knowledge.project_id,
                    knowledge.domain,
                    knowledge.fact,
                    knowledge.source,
                    knowledge.confidence,
                    knowledge.last_verified,
                    json.dumps(knowledge.metadata) if knowledge.metadata else None,
                    self._embedding_to_db(knowledge.embedding),
                ),
            )
            conn.commit()

        logger.debug(f"Saved domain knowledge: {knowledge.id}")
        return knowledge.id

    def save_anti_pattern(self, anti_pattern: AntiPattern) -> str:
        """Save an anti-pattern."""
        with self._get_connection() as conn:
            conn.execute(
                f"""
                INSERT INTO {self.schema}.{self.TABLE_NAMES[MemoryType.ANTI_PATTERNS]}
                (id, agent, project_id, pattern, why_bad, better_alternative,
                 occurrence_count, last_seen, created_at, metadata, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    pattern = EXCLUDED.pattern,
                    why_bad = EXCLUDED.why_bad,
                    better_alternative = EXCLUDED.better_alternative,
                    occurrence_count = EXCLUDED.occurrence_count,
                    last_seen = EXCLUDED.last_seen,
                    metadata = EXCLUDED.metadata,
                    embedding = EXCLUDED.embedding
                """,
                (
                    anti_pattern.id,
                    anti_pattern.agent,
                    anti_pattern.project_id,
                    anti_pattern.pattern,
                    anti_pattern.why_bad,
                    anti_pattern.better_alternative,
                    anti_pattern.occurrence_count,
                    anti_pattern.last_seen,
                    anti_pattern.created_at,
                    (
                        json.dumps(anti_pattern.metadata)
                        if anti_pattern.metadata
                        else None
                    ),
                    self._embedding_to_db(anti_pattern.embedding),
                ),
            )
            conn.commit()

        logger.debug(f"Saved anti-pattern: {anti_pattern.id}")
        return anti_pattern.id

    # ==================== BATCH WRITE OPERATIONS ====================

    def save_heuristics(self, heuristics: List[Heuristic]) -> List[str]:
        """Save multiple heuristics in a batch using executemany."""
        if not heuristics:
            return []

        with self._get_connection() as conn:
            conn.executemany(
                f"""
                INSERT INTO {self.schema}.{self.TABLE_NAMES[MemoryType.HEURISTICS]}
                (id, agent, project_id, condition, strategy, confidence,
                 occurrence_count, success_count, last_validated, created_at, metadata, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    condition = EXCLUDED.condition,
                    strategy = EXCLUDED.strategy,
                    confidence = EXCLUDED.confidence,
                    occurrence_count = EXCLUDED.occurrence_count,
                    success_count = EXCLUDED.success_count,
                    last_validated = EXCLUDED.last_validated,
                    metadata = EXCLUDED.metadata,
                    embedding = EXCLUDED.embedding
                """,
                [
                    (
                        h.id,
                        h.agent,
                        h.project_id,
                        h.condition,
                        h.strategy,
                        h.confidence,
                        h.occurrence_count,
                        h.success_count,
                        h.last_validated,
                        h.created_at,
                        json.dumps(h.metadata) if h.metadata else None,
                        self._embedding_to_db(h.embedding),
                    )
                    for h in heuristics
                ],
            )
            conn.commit()

        logger.debug(f"Batch saved {len(heuristics)} heuristics")
        return [h.id for h in heuristics]

    def save_outcomes(self, outcomes: List[Outcome]) -> List[str]:
        """Save multiple outcomes in a batch using executemany."""
        if not outcomes:
            return []

        with self._get_connection() as conn:
            conn.executemany(
                f"""
                INSERT INTO {self.schema}.{self.TABLE_NAMES[MemoryType.OUTCOMES]}
                (id, agent, project_id, task_type, task_description, success,
                 strategy_used, duration_ms, error_message, user_feedback, timestamp, metadata, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    task_description = EXCLUDED.task_description,
                    success = EXCLUDED.success,
                    strategy_used = EXCLUDED.strategy_used,
                    duration_ms = EXCLUDED.duration_ms,
                    error_message = EXCLUDED.error_message,
                    user_feedback = EXCLUDED.user_feedback,
                    metadata = EXCLUDED.metadata,
                    embedding = EXCLUDED.embedding
                """,
                [
                    (
                        o.id,
                        o.agent,
                        o.project_id,
                        o.task_type,
                        o.task_description,
                        o.success,
                        o.strategy_used,
                        o.duration_ms,
                        o.error_message,
                        o.user_feedback,
                        o.timestamp,
                        json.dumps(o.metadata) if o.metadata else None,
                        self._embedding_to_db(o.embedding),
                    )
                    for o in outcomes
                ],
            )
            conn.commit()

        logger.debug(f"Batch saved {len(outcomes)} outcomes")
        return [o.id for o in outcomes]

    def save_domain_knowledge_batch(
        self, knowledge_items: List[DomainKnowledge]
    ) -> List[str]:
        """Save multiple domain knowledge items in a batch using executemany."""
        if not knowledge_items:
            return []

        with self._get_connection() as conn:
            conn.executemany(
                f"""
                INSERT INTO {self.schema}.{self.TABLE_NAMES[MemoryType.DOMAIN_KNOWLEDGE]}
                (id, agent, project_id, domain, fact, source, confidence, last_verified, metadata, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    fact = EXCLUDED.fact,
                    source = EXCLUDED.source,
                    confidence = EXCLUDED.confidence,
                    last_verified = EXCLUDED.last_verified,
                    metadata = EXCLUDED.metadata,
                    embedding = EXCLUDED.embedding
                """,
                [
                    (
                        k.id,
                        k.agent,
                        k.project_id,
                        k.domain,
                        k.fact,
                        k.source,
                        k.confidence,
                        k.last_verified,
                        json.dumps(k.metadata) if k.metadata else None,
                        self._embedding_to_db(k.embedding),
                    )
                    for k in knowledge_items
                ],
            )
            conn.commit()

        logger.debug(f"Batch saved {len(knowledge_items)} domain knowledge items")
        return [k.id for k in knowledge_items]

    # ==================== READ OPERATIONS ====================

    def get_heuristics(
        self,
        project_id: str,
        agent: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        top_k: int = 5,
        min_confidence: float = 0.0,
    ) -> List[Heuristic]:
        """Get heuristics with optional vector search."""
        with self._get_connection() as conn:
            if embedding and self._pgvector_available:
                # Use pgvector similarity search
                query = f"""
                    SELECT *, 1 - (embedding <=> %s::vector) as similarity
                    FROM {self.schema}.{self.TABLE_NAMES[MemoryType.HEURISTICS]}
                    WHERE project_id = %s AND confidence >= %s
                """
                params: List[Any] = [
                    self._embedding_to_db(embedding),
                    project_id,
                    min_confidence,
                ]

                if agent:
                    query += " AND agent = %s"
                    params.append(agent)

                query += " ORDER BY similarity DESC LIMIT %s"
                params.append(top_k)
            else:
                # Standard query
                query = f"""
                    SELECT *
                    FROM {self.schema}.{self.TABLE_NAMES[MemoryType.HEURISTICS]}
                    WHERE project_id = %s AND confidence >= %s
                """
                params = [project_id, min_confidence]

                if agent:
                    query += " AND agent = %s"
                    params.append(agent)

                query += " ORDER BY confidence DESC LIMIT %s"
                params.append(top_k)

            cursor = conn.execute(query, params)
            rows = cursor.fetchall()

        results = [self._row_to_heuristic(row) for row in rows]

        # If embedding provided but pgvector not available, do app-level filtering
        if embedding and not self._pgvector_available and results:
            results = self._filter_by_similarity(results, embedding, top_k, "embedding")

        return results

    def get_outcomes(
        self,
        project_id: str,
        agent: Optional[str] = None,
        task_type: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        top_k: int = 5,
        success_only: bool = False,
    ) -> List[Outcome]:
        """Get outcomes with optional vector search."""
        with self._get_connection() as conn:
            if embedding and self._pgvector_available:
                query = f"""
                    SELECT *, 1 - (embedding <=> %s::vector) as similarity
                    FROM {self.schema}.{self.TABLE_NAMES[MemoryType.OUTCOMES]}
                    WHERE project_id = %s
                """
                params: List[Any] = [self._embedding_to_db(embedding), project_id]
            else:
                query = f"""
                    SELECT *
                    FROM {self.schema}.{self.TABLE_NAMES[MemoryType.OUTCOMES]}
                    WHERE project_id = %s
                """
                params = [project_id]

            if agent:
                query += " AND agent = %s"
                params.append(agent)

            if task_type:
                query += " AND task_type = %s"
                params.append(task_type)

            if success_only:
                query += " AND success = TRUE"

            if embedding and self._pgvector_available:
                query += " ORDER BY similarity DESC LIMIT %s"
            else:
                query += " ORDER BY timestamp DESC LIMIT %s"
            params.append(top_k)

            cursor = conn.execute(query, params)
            rows = cursor.fetchall()

        results = [self._row_to_outcome(row) for row in rows]

        if embedding and not self._pgvector_available and results:
            results = self._filter_by_similarity(results, embedding, top_k, "embedding")

        return results

    def get_user_preferences(
        self,
        user_id: str,
        category: Optional[str] = None,
    ) -> List[UserPreference]:
        """Get user preferences."""
        with self._get_connection() as conn:
            query = f"SELECT * FROM {self.schema}.{self.TABLE_NAMES[MemoryType.PREFERENCES]} WHERE user_id = %s"
            params: List[Any] = [user_id]

            if category:
                query += " AND category = %s"
                params.append(category)

            cursor = conn.execute(query, params)
            rows = cursor.fetchall()

        return [self._row_to_preference(row) for row in rows]

    def get_domain_knowledge(
        self,
        project_id: str,
        agent: Optional[str] = None,
        domain: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        top_k: int = 5,
    ) -> List[DomainKnowledge]:
        """Get domain knowledge with optional vector search."""
        with self._get_connection() as conn:
            if embedding and self._pgvector_available:
                query = f"""
                    SELECT *, 1 - (embedding <=> %s::vector) as similarity
                    FROM {self.schema}.{self.TABLE_NAMES[MemoryType.DOMAIN_KNOWLEDGE]}
                    WHERE project_id = %s
                """
                params: List[Any] = [self._embedding_to_db(embedding), project_id]
            else:
                query = f"""
                    SELECT *
                    FROM {self.schema}.{self.TABLE_NAMES[MemoryType.DOMAIN_KNOWLEDGE]}
                    WHERE project_id = %s
                """
                params = [project_id]

            if agent:
                query += " AND agent = %s"
                params.append(agent)

            if domain:
                query += " AND domain = %s"
                params.append(domain)

            if embedding and self._pgvector_available:
                query += " ORDER BY similarity DESC LIMIT %s"
            else:
                query += " ORDER BY confidence DESC LIMIT %s"
            params.append(top_k)

            cursor = conn.execute(query, params)
            rows = cursor.fetchall()

        results = [self._row_to_domain_knowledge(row) for row in rows]

        if embedding and not self._pgvector_available and results:
            results = self._filter_by_similarity(results, embedding, top_k, "embedding")

        return results

    def get_anti_patterns(
        self,
        project_id: str,
        agent: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        top_k: int = 5,
    ) -> List[AntiPattern]:
        """Get anti-patterns with optional vector search."""
        with self._get_connection() as conn:
            if embedding and self._pgvector_available:
                query = f"""
                    SELECT *, 1 - (embedding <=> %s::vector) as similarity
                    FROM {self.schema}.{self.TABLE_NAMES[MemoryType.ANTI_PATTERNS]}
                    WHERE project_id = %s
                """
                params: List[Any] = [self._embedding_to_db(embedding), project_id]
            else:
                query = f"""
                    SELECT *
                    FROM {self.schema}.{self.TABLE_NAMES[MemoryType.ANTI_PATTERNS]}
                    WHERE project_id = %s
                """
                params = [project_id]

            if agent:
                query += " AND agent = %s"
                params.append(agent)

            if embedding and self._pgvector_available:
                query += " ORDER BY similarity DESC LIMIT %s"
            else:
                query += " ORDER BY occurrence_count DESC LIMIT %s"
            params.append(top_k)

            cursor = conn.execute(query, params)
            rows = cursor.fetchall()

        results = [self._row_to_anti_pattern(row) for row in rows]

        if embedding and not self._pgvector_available and results:
            results = self._filter_by_similarity(results, embedding, top_k, "embedding")

        return results

    def _filter_by_similarity(
        self,
        items: List[Any],
        query_embedding: List[float],
        top_k: int,
        embedding_attr: str,
    ) -> List[Any]:
        """Filter items by cosine similarity (fallback when pgvector unavailable)."""
        scored = []
        for item in items:
            item_embedding = getattr(item, embedding_attr, None)
            if item_embedding:
                similarity = self._cosine_similarity(query_embedding, item_embedding)
                scored.append((item, similarity))
            else:
                scored.append((item, 0.0))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [item for item, _ in scored[:top_k]]

    # ==================== MULTI-AGENT MEMORY SHARING ====================

    def get_heuristics_for_agents(
        self,
        project_id: str,
        agents: List[str],
        embedding: Optional[List[float]] = None,
        top_k: int = 5,
        min_confidence: float = 0.0,
    ) -> List[Heuristic]:
        """Get heuristics from multiple agents using optimized ANY query."""
        if not agents:
            return []

        with self._get_connection() as conn:
            if embedding and self._pgvector_available:
                query = f"""
                    SELECT *, 1 - (embedding <=> %s::vector) as similarity
                    FROM {self.schema}.{self.TABLE_NAMES[MemoryType.HEURISTICS]}
                    WHERE project_id = %s AND confidence >= %s AND agent = ANY(%s)
                    ORDER BY similarity DESC LIMIT %s
                """
                params: List[Any] = [
                    self._embedding_to_db(embedding),
                    project_id,
                    min_confidence,
                    agents,
                    top_k * len(agents),
                ]
            else:
                query = f"""
                    SELECT *
                    FROM {self.schema}.{self.TABLE_NAMES[MemoryType.HEURISTICS]}
                    WHERE project_id = %s AND confidence >= %s AND agent = ANY(%s)
                    ORDER BY confidence DESC LIMIT %s
                """
                params = [project_id, min_confidence, agents, top_k * len(agents)]

            cursor = conn.execute(query, params)
            rows = cursor.fetchall()

        results = [self._row_to_heuristic(row) for row in rows]

        if embedding and not self._pgvector_available and results:
            results = self._filter_by_similarity(
                results, embedding, top_k * len(agents), "embedding"
            )

        return results

    def get_outcomes_for_agents(
        self,
        project_id: str,
        agents: List[str],
        task_type: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        top_k: int = 5,
        success_only: bool = False,
    ) -> List[Outcome]:
        """Get outcomes from multiple agents using optimized ANY query."""
        if not agents:
            return []

        with self._get_connection() as conn:
            if embedding and self._pgvector_available:
                query = f"""
                    SELECT *, 1 - (embedding <=> %s::vector) as similarity
                    FROM {self.schema}.{self.TABLE_NAMES[MemoryType.OUTCOMES]}
                    WHERE project_id = %s AND agent = ANY(%s)
                """
                params: List[Any] = [
                    self._embedding_to_db(embedding),
                    project_id,
                    agents,
                ]
            else:
                query = f"""
                    SELECT *
                    FROM {self.schema}.{self.TABLE_NAMES[MemoryType.OUTCOMES]}
                    WHERE project_id = %s AND agent = ANY(%s)
                """
                params = [project_id, agents]

            if task_type:
                query += " AND task_type = %s"
                params.append(task_type)

            if success_only:
                query += " AND success = TRUE"

            if embedding and self._pgvector_available:
                query += " ORDER BY similarity DESC LIMIT %s"
            else:
                query += " ORDER BY timestamp DESC LIMIT %s"
            params.append(top_k * len(agents))

            cursor = conn.execute(query, params)
            rows = cursor.fetchall()

        results = [self._row_to_outcome(row) for row in rows]

        if embedding and not self._pgvector_available and results:
            results = self._filter_by_similarity(
                results, embedding, top_k * len(agents), "embedding"
            )

        return results

    def get_domain_knowledge_for_agents(
        self,
        project_id: str,
        agents: List[str],
        domain: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        top_k: int = 5,
    ) -> List[DomainKnowledge]:
        """Get domain knowledge from multiple agents using optimized ANY query."""
        if not agents:
            return []

        with self._get_connection() as conn:
            if embedding and self._pgvector_available:
                query = f"""
                    SELECT *, 1 - (embedding <=> %s::vector) as similarity
                    FROM {self.schema}.{self.TABLE_NAMES[MemoryType.DOMAIN_KNOWLEDGE]}
                    WHERE project_id = %s AND agent = ANY(%s)
                """
                params: List[Any] = [
                    self._embedding_to_db(embedding),
                    project_id,
                    agents,
                ]
            else:
                query = f"""
                    SELECT *
                    FROM {self.schema}.{self.TABLE_NAMES[MemoryType.DOMAIN_KNOWLEDGE]}
                    WHERE project_id = %s AND agent = ANY(%s)
                """
                params = [project_id, agents]

            if domain:
                query += " AND domain = %s"
                params.append(domain)

            if embedding and self._pgvector_available:
                query += " ORDER BY similarity DESC LIMIT %s"
            else:
                query += " ORDER BY confidence DESC LIMIT %s"
            params.append(top_k * len(agents))

            cursor = conn.execute(query, params)
            rows = cursor.fetchall()

        results = [self._row_to_domain_knowledge(row) for row in rows]

        if embedding and not self._pgvector_available and results:
            results = self._filter_by_similarity(
                results, embedding, top_k * len(agents), "embedding"
            )

        return results

    def get_anti_patterns_for_agents(
        self,
        project_id: str,
        agents: List[str],
        embedding: Optional[List[float]] = None,
        top_k: int = 5,
    ) -> List[AntiPattern]:
        """Get anti-patterns from multiple agents using optimized ANY query."""
        if not agents:
            return []

        with self._get_connection() as conn:
            if embedding and self._pgvector_available:
                query = f"""
                    SELECT *, 1 - (embedding <=> %s::vector) as similarity
                    FROM {self.schema}.{self.TABLE_NAMES[MemoryType.ANTI_PATTERNS]}
                    WHERE project_id = %s AND agent = ANY(%s)
                """
                params: List[Any] = [
                    self._embedding_to_db(embedding),
                    project_id,
                    agents,
                ]
            else:
                query = f"""
                    SELECT *
                    FROM {self.schema}.{self.TABLE_NAMES[MemoryType.ANTI_PATTERNS]}
                    WHERE project_id = %s AND agent = ANY(%s)
                """
                params = [project_id, agents]

            if embedding and self._pgvector_available:
                query += " ORDER BY similarity DESC LIMIT %s"
            else:
                query += " ORDER BY occurrence_count DESC LIMIT %s"
            params.append(top_k * len(agents))

            cursor = conn.execute(query, params)
            rows = cursor.fetchall()

        results = [self._row_to_anti_pattern(row) for row in rows]

        if embedding and not self._pgvector_available and results:
            results = self._filter_by_similarity(
                results, embedding, top_k * len(agents), "embedding"
            )

        return results

    # ==================== UPDATE OPERATIONS ====================

    def update_heuristic(
        self,
        heuristic_id: str,
        updates: Dict[str, Any],
    ) -> bool:
        """Update a heuristic's fields."""
        if not updates:
            return False

        set_clauses = []
        params = []
        for key, value in updates.items():
            if key == "metadata" and value:
                value = json.dumps(value)
            set_clauses.append(f"{key} = %s")
            params.append(value)

        params.append(heuristic_id)

        with self._get_connection() as conn:
            cursor = conn.execute(
                f"UPDATE {self.schema}.{self.TABLE_NAMES[MemoryType.HEURISTICS]} SET {', '.join(set_clauses)} WHERE id = %s",
                params,
            )
            conn.commit()
            return cursor.rowcount > 0

    def increment_heuristic_occurrence(
        self,
        heuristic_id: str,
        success: bool,
    ) -> bool:
        """Increment heuristic occurrence count."""
        with self._get_connection() as conn:
            if success:
                cursor = conn.execute(
                    f"""
                    UPDATE {self.schema}.{self.TABLE_NAMES[MemoryType.HEURISTICS]}
                    SET occurrence_count = occurrence_count + 1,
                        success_count = success_count + 1,
                        last_validated = %s
                    WHERE id = %s
                    """,
                    (datetime.now(timezone.utc), heuristic_id),
                )
            else:
                cursor = conn.execute(
                    f"""
                    UPDATE {self.schema}.{self.TABLE_NAMES[MemoryType.HEURISTICS]}
                    SET occurrence_count = occurrence_count + 1,
                        last_validated = %s
                    WHERE id = %s
                    """,
                    (datetime.now(timezone.utc), heuristic_id),
                )
            conn.commit()
            return cursor.rowcount > 0

    def update_heuristic_confidence(
        self,
        heuristic_id: str,
        new_confidence: float,
    ) -> bool:
        """Update confidence score for a heuristic."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                f"UPDATE {self.schema}.{self.TABLE_NAMES[MemoryType.HEURISTICS]} SET confidence = %s WHERE id = %s",
                (new_confidence, heuristic_id),
            )
            conn.commit()
            return cursor.rowcount > 0

    def update_knowledge_confidence(
        self,
        knowledge_id: str,
        new_confidence: float,
    ) -> bool:
        """Update confidence score for domain knowledge."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                f"UPDATE {self.schema}.{self.TABLE_NAMES[MemoryType.DOMAIN_KNOWLEDGE]} SET confidence = %s WHERE id = %s",
                (new_confidence, knowledge_id),
            )
            conn.commit()
            return cursor.rowcount > 0

    # ==================== DELETE OPERATIONS ====================

    def delete_heuristic(self, heuristic_id: str) -> bool:
        """Delete a heuristic by ID."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                f"DELETE FROM {self.schema}.{self.TABLE_NAMES[MemoryType.HEURISTICS]} WHERE id = %s",
                (heuristic_id,),
            )
            conn.commit()
            return cursor.rowcount > 0

    def delete_outcome(self, outcome_id: str) -> bool:
        """Delete an outcome by ID."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                f"DELETE FROM {self.schema}.{self.TABLE_NAMES[MemoryType.OUTCOMES]} WHERE id = %s",
                (outcome_id,),
            )
            conn.commit()
            return cursor.rowcount > 0

    def delete_domain_knowledge(self, knowledge_id: str) -> bool:
        """Delete domain knowledge by ID."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                f"DELETE FROM {self.schema}.{self.TABLE_NAMES[MemoryType.DOMAIN_KNOWLEDGE]} WHERE id = %s",
                (knowledge_id,),
            )
            conn.commit()
            return cursor.rowcount > 0

    def delete_anti_pattern(self, anti_pattern_id: str) -> bool:
        """Delete an anti-pattern by ID."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                f"DELETE FROM {self.schema}.{self.TABLE_NAMES[MemoryType.ANTI_PATTERNS]} WHERE id = %s",
                (anti_pattern_id,),
            )
            conn.commit()
            return cursor.rowcount > 0

    def delete_outcomes_older_than(
        self,
        project_id: str,
        older_than: datetime,
        agent: Optional[str] = None,
    ) -> int:
        """Delete old outcomes."""
        with self._get_connection() as conn:
            query = f"DELETE FROM {self.schema}.{self.TABLE_NAMES[MemoryType.OUTCOMES]} WHERE project_id = %s AND timestamp < %s"
            params: List[Any] = [project_id, older_than]

            if agent:
                query += " AND agent = %s"
                params.append(agent)

            cursor = conn.execute(query, params)
            conn.commit()
            deleted = cursor.rowcount

        logger.info(f"Deleted {deleted} old outcomes")
        return deleted

    def delete_low_confidence_heuristics(
        self,
        project_id: str,
        below_confidence: float,
        agent: Optional[str] = None,
    ) -> int:
        """Delete low-confidence heuristics."""
        with self._get_connection() as conn:
            query = f"DELETE FROM {self.schema}.{self.TABLE_NAMES[MemoryType.HEURISTICS]} WHERE project_id = %s AND confidence < %s"
            params: List[Any] = [project_id, below_confidence]

            if agent:
                query += " AND agent = %s"
                params.append(agent)

            cursor = conn.execute(query, params)
            conn.commit()
            deleted = cursor.rowcount

        logger.info(f"Deleted {deleted} low-confidence heuristics")
        return deleted

    # ==================== STATS ====================

    def get_stats(
        self,
        project_id: str,
        agent: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get memory statistics."""
        stats = {
            "project_id": project_id,
            "agent": agent,
            "storage_type": "postgresql",
            "pgvector_available": self._pgvector_available,
        }

        with self._get_connection() as conn:
            # Use canonical memory types for stats
            for memory_type in MemoryType.ALL:
                table = self.TABLE_NAMES[memory_type]
                if memory_type == MemoryType.PREFERENCES:
                    # Preferences don't have project_id
                    cursor = conn.execute(
                        f"SELECT COUNT(*) as count FROM {self.schema}.{table}"
                    )
                    row = cursor.fetchone()
                    stats[f"{memory_type}_count"] = row["count"] if row else 0
                else:
                    query = f"SELECT COUNT(*) as count FROM {self.schema}.{table} WHERE project_id = %s"
                    params: List[Any] = [project_id]
                    if agent:
                        query += " AND agent = %s"
                        params.append(agent)
                    cursor = conn.execute(query, params)
                    row = cursor.fetchone()
                    stats[f"{memory_type}_count"] = row["count"] if row else 0

        stats["total_count"] = sum(
            stats.get(k, 0) for k in stats if k.endswith("_count")
        )

        return stats

    # ==================== HELPERS ====================

    def _parse_datetime(self, value: Any) -> Optional[datetime]:
        """Parse datetime from database value."""
        if value is None:
            return None
        if isinstance(value, datetime):
            return value
        try:
            return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            return None

    def _row_to_heuristic(self, row: Dict[str, Any]) -> Heuristic:
        """Convert database row to Heuristic."""
        return Heuristic(
            id=row["id"],
            agent=row["agent"],
            project_id=row["project_id"],
            condition=row["condition"],
            strategy=row["strategy"],
            confidence=row["confidence"] or 0.0,
            occurrence_count=row["occurrence_count"] or 0,
            success_count=row["success_count"] or 0,
            last_validated=self._parse_datetime(row["last_validated"])
            or datetime.now(timezone.utc),
            created_at=self._parse_datetime(row["created_at"])
            or datetime.now(timezone.utc),
            embedding=self._embedding_from_db(row.get("embedding")),
            metadata=row["metadata"] if row["metadata"] else {},
        )

    def _row_to_outcome(self, row: Dict[str, Any]) -> Outcome:
        """Convert database row to Outcome."""
        return Outcome(
            id=row["id"],
            agent=row["agent"],
            project_id=row["project_id"],
            task_type=row["task_type"] or "general",
            task_description=row["task_description"],
            success=bool(row["success"]),
            strategy_used=row["strategy_used"] or "",
            duration_ms=row["duration_ms"],
            error_message=row["error_message"],
            user_feedback=row["user_feedback"],
            timestamp=self._parse_datetime(row["timestamp"])
            or datetime.now(timezone.utc),
            embedding=self._embedding_from_db(row.get("embedding")),
            metadata=row["metadata"] if row["metadata"] else {},
        )

    def _row_to_preference(self, row: Dict[str, Any]) -> UserPreference:
        """Convert database row to UserPreference."""
        return UserPreference(
            id=row["id"],
            user_id=row["user_id"],
            category=row["category"] or "general",
            preference=row["preference"],
            source=row["source"] or "unknown",
            confidence=row["confidence"] or 1.0,
            timestamp=self._parse_datetime(row["timestamp"])
            or datetime.now(timezone.utc),
            metadata=row["metadata"] if row["metadata"] else {},
        )

    def _row_to_domain_knowledge(self, row: Dict[str, Any]) -> DomainKnowledge:
        """Convert database row to DomainKnowledge."""
        return DomainKnowledge(
            id=row["id"],
            agent=row["agent"],
            project_id=row["project_id"],
            domain=row["domain"] or "general",
            fact=row["fact"],
            source=row["source"] or "unknown",
            confidence=row["confidence"] or 1.0,
            last_verified=self._parse_datetime(row["last_verified"])
            or datetime.now(timezone.utc),
            embedding=self._embedding_from_db(row.get("embedding")),
            metadata=row["metadata"] if row["metadata"] else {},
        )

    def _row_to_anti_pattern(self, row: Dict[str, Any]) -> AntiPattern:
        """Convert database row to AntiPattern."""
        return AntiPattern(
            id=row["id"],
            agent=row["agent"],
            project_id=row["project_id"],
            pattern=row["pattern"],
            why_bad=row["why_bad"] or "",
            better_alternative=row["better_alternative"] or "",
            occurrence_count=row["occurrence_count"] or 1,
            last_seen=self._parse_datetime(row["last_seen"])
            or datetime.now(timezone.utc),
            created_at=self._parse_datetime(row["created_at"])
            or datetime.now(timezone.utc),
            embedding=self._embedding_from_db(row.get("embedding")),
            metadata=row["metadata"] if row["metadata"] else {},
        )

    def close(self):
        """Close connection pool."""
        if self._pool:
            self._pool.close()

    # ==================== MIGRATION SUPPORT ====================

    def _get_version_store(self):
        """Get or create the version store."""
        if self._version_store is None:
            from alma.storage.migrations.version_stores import PostgreSQLVersionStore

            self._version_store = PostgreSQLVersionStore(self._pool, self.schema)
        return self._version_store

    def _get_migration_runner(self):
        """Get or create the migration runner."""
        if self._migration_runner is None:
            from alma.storage.migrations.runner import MigrationRunner
            from alma.storage.migrations.versions import v1_0_0  # noqa: F401

            self._migration_runner = MigrationRunner(
                version_store=self._get_version_store(),
                backend="postgresql",
            )
        return self._migration_runner

    def _ensure_migrated(self) -> None:
        """Ensure database is migrated to latest version."""
        runner = self._get_migration_runner()
        if runner.needs_migration():
            with self._get_connection() as conn:
                applied = runner.migrate(conn)
                if applied:
                    logger.info(f"Applied {len(applied)} migrations: {applied}")

    def get_schema_version(self) -> Optional[str]:
        """Get the current schema version."""
        return self._get_version_store().get_current_version()

    def get_migration_status(self) -> Dict[str, Any]:
        """Get migration status information."""
        runner = self._get_migration_runner()
        status = runner.get_status()
        status["migration_supported"] = True
        return status

    def migrate(
        self,
        target_version: Optional[str] = None,
        dry_run: bool = False,
    ) -> List[str]:
        """
        Apply pending schema migrations.

        Args:
            target_version: Optional target version (applies all if not specified)
            dry_run: If True, show what would be done without making changes

        Returns:
            List of applied migration versions
        """
        runner = self._get_migration_runner()
        with self._get_connection() as conn:
            return runner.migrate(conn, target_version=target_version, dry_run=dry_run)

    def rollback(
        self,
        target_version: str,
        dry_run: bool = False,
    ) -> List[str]:
        """
        Roll back schema to a previous version.

        Args:
            target_version: Version to roll back to
            dry_run: If True, show what would be done without making changes

        Returns:
            List of rolled back migration versions
        """
        runner = self._get_migration_runner()
        with self._get_connection() as conn:
            return runner.rollback(conn, target_version=target_version, dry_run=dry_run)

    # ==================== CHECKPOINT OPERATIONS (v0.6.0+) ====================

    def save_checkpoint(self, checkpoint: "Checkpoint") -> str:
        """Save a workflow checkpoint."""
        with self._get_connection() as conn:
            conn.execute(
                f"""
                INSERT INTO {self.schema}.alma_checkpoints
                (id, run_id, node_id, state_json, state_hash, sequence_number,
                 branch_id, parent_checkpoint_id, metadata, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    state_json = EXCLUDED.state_json,
                    state_hash = EXCLUDED.state_hash,
                    sequence_number = EXCLUDED.sequence_number,
                    metadata = EXCLUDED.metadata
                """,
                (
                    checkpoint.id,
                    checkpoint.run_id,
                    checkpoint.node_id,
                    json.dumps(checkpoint.state),
                    checkpoint.state_hash,
                    checkpoint.sequence_number,
                    checkpoint.branch_id,
                    checkpoint.parent_checkpoint_id,
                    json.dumps(checkpoint.metadata) if checkpoint.metadata else None,
                    checkpoint.created_at,
                ),
            )
            conn.commit()

        logger.debug(f"Saved checkpoint: {checkpoint.id}")
        return checkpoint.id

    def get_checkpoint(self, checkpoint_id: str) -> Optional["Checkpoint"]:
        """Get a checkpoint by ID."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                f"SELECT * FROM {self.schema}.alma_checkpoints WHERE id = %s",
                (checkpoint_id,),
            )
            row = cursor.fetchone()

        if row is None:
            return None
        return self._row_to_checkpoint(row)

    def get_latest_checkpoint(
        self,
        run_id: str,
        branch_id: Optional[str] = None,
    ) -> Optional["Checkpoint"]:
        """Get the most recent checkpoint for a workflow run."""
        with self._get_connection() as conn:
            if branch_id is not None:
                cursor = conn.execute(
                    f"""
                    SELECT * FROM {self.schema}.alma_checkpoints
                    WHERE run_id = %s AND branch_id = %s
                    ORDER BY sequence_number DESC LIMIT 1
                    """,
                    (run_id, branch_id),
                )
            else:
                cursor = conn.execute(
                    f"""
                    SELECT * FROM {self.schema}.alma_checkpoints
                    WHERE run_id = %s
                    ORDER BY sequence_number DESC LIMIT 1
                    """,
                    (run_id,),
                )
            row = cursor.fetchone()

        if row is None:
            return None
        return self._row_to_checkpoint(row)

    def get_checkpoints_for_run(
        self,
        run_id: str,
        branch_id: Optional[str] = None,
        limit: int = 100,
    ) -> List["Checkpoint"]:
        """Get all checkpoints for a workflow run."""
        with self._get_connection() as conn:
            if branch_id is not None:
                cursor = conn.execute(
                    f"""
                    SELECT * FROM {self.schema}.alma_checkpoints
                    WHERE run_id = %s AND branch_id = %s
                    ORDER BY sequence_number ASC LIMIT %s
                    """,
                    (run_id, branch_id, limit),
                )
            else:
                cursor = conn.execute(
                    f"""
                    SELECT * FROM {self.schema}.alma_checkpoints
                    WHERE run_id = %s
                    ORDER BY sequence_number ASC LIMIT %s
                    """,
                    (run_id, limit),
                )
            rows = cursor.fetchall()

        return [self._row_to_checkpoint(row) for row in rows]

    def cleanup_checkpoints(
        self,
        run_id: str,
        keep_latest: int = 1,
    ) -> int:
        """Clean up old checkpoints for a completed run."""
        with self._get_connection() as conn:
            # Delete all but the latest N checkpoints
            cursor = conn.execute(
                f"""
                DELETE FROM {self.schema}.alma_checkpoints
                WHERE run_id = %s AND id NOT IN (
                    SELECT id FROM {self.schema}.alma_checkpoints
                    WHERE run_id = %s
                    ORDER BY sequence_number DESC
                    LIMIT %s
                )
                """,
                (run_id, run_id, keep_latest),
            )
            conn.commit()
            deleted = cursor.rowcount

        logger.info(f"Cleaned up {deleted} checkpoints for run {run_id}")
        return deleted

    def _row_to_checkpoint(self, row: Dict[str, Any]) -> "Checkpoint":
        """Convert database row to Checkpoint."""
        from alma.workflow import Checkpoint

        return Checkpoint(
            id=row["id"],
            run_id=row["run_id"],
            node_id=row["node_id"],
            state=json.loads(row["state_json"]) if row["state_json"] else {},
            sequence_number=row["sequence_number"] or 0,
            branch_id=row["branch_id"],
            parent_checkpoint_id=row["parent_checkpoint_id"],
            state_hash=row["state_hash"] or "",
            metadata=row["metadata"] if row["metadata"] else {},
            created_at=self._parse_datetime(row["created_at"])
            or datetime.now(timezone.utc),
        )

    # ==================== WORKFLOW OUTCOME OPERATIONS (v0.6.0+) ====================

    def save_workflow_outcome(self, outcome: "WorkflowOutcome") -> str:
        """Save a workflow outcome."""
        with self._get_connection() as conn:
            conn.execute(
                f"""
                INSERT INTO {self.schema}.alma_workflow_outcomes
                (id, tenant_id, workflow_id, run_id, agent, project_id, result,
                 summary, strategies_used, successful_patterns, failed_patterns,
                 extracted_heuristics, extracted_anti_patterns, duration_seconds,
                 node_count, error_message, metadata, embedding, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    result = EXCLUDED.result,
                    summary = EXCLUDED.summary,
                    strategies_used = EXCLUDED.strategies_used,
                    successful_patterns = EXCLUDED.successful_patterns,
                    failed_patterns = EXCLUDED.failed_patterns,
                    extracted_heuristics = EXCLUDED.extracted_heuristics,
                    extracted_anti_patterns = EXCLUDED.extracted_anti_patterns,
                    duration_seconds = EXCLUDED.duration_seconds,
                    node_count = EXCLUDED.node_count,
                    error_message = EXCLUDED.error_message,
                    metadata = EXCLUDED.metadata,
                    embedding = EXCLUDED.embedding
                """,
                (
                    outcome.id,
                    outcome.tenant_id,
                    outcome.workflow_id,
                    outcome.run_id,
                    outcome.agent,
                    outcome.project_id,
                    outcome.result.value,
                    outcome.summary,
                    outcome.strategies_used,
                    outcome.successful_patterns,
                    outcome.failed_patterns,
                    outcome.extracted_heuristics,
                    outcome.extracted_anti_patterns,
                    outcome.duration_seconds,
                    outcome.node_count,
                    outcome.error_message,
                    outcome.metadata,
                    self._embedding_to_db(outcome.embedding),
                    outcome.created_at,
                ),
            )
            conn.commit()

        logger.debug(f"Saved workflow outcome: {outcome.id}")
        return outcome.id

    def get_workflow_outcome(self, outcome_id: str) -> Optional["WorkflowOutcome"]:
        """Get a workflow outcome by ID."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                f"SELECT * FROM {self.schema}.alma_workflow_outcomes WHERE id = %s",
                (outcome_id,),
            )
            row = cursor.fetchone()

        if row is None:
            return None
        return self._row_to_workflow_outcome(row)

    def get_workflow_outcomes(
        self,
        project_id: str,
        agent: Optional[str] = None,
        workflow_id: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        top_k: int = 10,
        scope_filter: Optional[Dict[str, Any]] = None,
    ) -> List["WorkflowOutcome"]:
        """Get workflow outcomes with optional filtering."""
        with self._get_connection() as conn:
            if embedding and self._pgvector_available:
                query = f"""
                    SELECT *, 1 - (embedding <=> %s::vector) as similarity
                    FROM {self.schema}.alma_workflow_outcomes
                    WHERE project_id = %s
                """
                params: List[Any] = [self._embedding_to_db(embedding), project_id]
            else:
                query = f"""
                    SELECT *
                    FROM {self.schema}.alma_workflow_outcomes
                    WHERE project_id = %s
                """
                params = [project_id]

            if agent:
                query += " AND agent = %s"
                params.append(agent)

            if workflow_id:
                query += " AND workflow_id = %s"
                params.append(workflow_id)

            # Apply scope filter
            if scope_filter:
                if scope_filter.get("tenant_id"):
                    query += " AND tenant_id = %s"
                    params.append(scope_filter["tenant_id"])
                if scope_filter.get("workflow_id"):
                    query += " AND workflow_id = %s"
                    params.append(scope_filter["workflow_id"])
                if scope_filter.get("run_id"):
                    query += " AND run_id = %s"
                    params.append(scope_filter["run_id"])

            if embedding and self._pgvector_available:
                query += " ORDER BY similarity DESC LIMIT %s"
            else:
                query += " ORDER BY created_at DESC LIMIT %s"
            params.append(top_k)

            cursor = conn.execute(query, params)
            rows = cursor.fetchall()

        return [self._row_to_workflow_outcome(row) for row in rows]

    def _row_to_workflow_outcome(self, row: Dict[str, Any]) -> "WorkflowOutcome":
        """Convert database row to WorkflowOutcome."""
        from alma.workflow import WorkflowOutcome, WorkflowResult

        return WorkflowOutcome(
            id=row["id"],
            tenant_id=row["tenant_id"],
            workflow_id=row["workflow_id"],
            run_id=row["run_id"],
            agent=row["agent"],
            project_id=row["project_id"],
            result=WorkflowResult(row["result"]),
            summary=row["summary"] or "",
            strategies_used=row["strategies_used"] or [],
            successful_patterns=row["successful_patterns"] or [],
            failed_patterns=row["failed_patterns"] or [],
            extracted_heuristics=row["extracted_heuristics"] or [],
            extracted_anti_patterns=row["extracted_anti_patterns"] or [],
            duration_seconds=row["duration_seconds"],
            node_count=row["node_count"],
            error_message=row["error_message"],
            embedding=self._embedding_from_db(row.get("embedding")),
            metadata=row["metadata"] if row["metadata"] else {},
            created_at=self._parse_datetime(row["created_at"])
            or datetime.now(timezone.utc),
        )

    # ==================== ARTIFACT LINK OPERATIONS (v0.6.0+) ====================

    def save_artifact_link(self, artifact_ref: "ArtifactRef") -> str:
        """Save an artifact reference linked to a memory."""
        with self._get_connection() as conn:
            conn.execute(
                f"""
                INSERT INTO {self.schema}.alma_artifact_links
                (id, memory_id, artifact_type, storage_url, filename,
                 mime_type, size_bytes, checksum, metadata, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    storage_url = EXCLUDED.storage_url,
                    filename = EXCLUDED.filename,
                    mime_type = EXCLUDED.mime_type,
                    size_bytes = EXCLUDED.size_bytes,
                    checksum = EXCLUDED.checksum,
                    metadata = EXCLUDED.metadata
                """,
                (
                    artifact_ref.id,
                    artifact_ref.memory_id,
                    artifact_ref.artifact_type.value,
                    artifact_ref.storage_url,
                    artifact_ref.filename,
                    artifact_ref.mime_type,
                    artifact_ref.size_bytes,
                    artifact_ref.checksum,
                    artifact_ref.metadata,
                    artifact_ref.created_at,
                ),
            )
            conn.commit()

        logger.debug(f"Saved artifact link: {artifact_ref.id}")
        return artifact_ref.id

    def get_artifact_links(self, memory_id: str) -> List["ArtifactRef"]:
        """Get all artifact references linked to a memory."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                f"SELECT * FROM {self.schema}.alma_artifact_links WHERE memory_id = %s",
                (memory_id,),
            )
            rows = cursor.fetchall()

        return [self._row_to_artifact_ref(row) for row in rows]

    def delete_artifact_link(self, artifact_id: str) -> bool:
        """Delete an artifact reference."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                f"DELETE FROM {self.schema}.alma_artifact_links WHERE id = %s",
                (artifact_id,),
            )
            conn.commit()
            return cursor.rowcount > 0

    def _row_to_artifact_ref(self, row: Dict[str, Any]) -> "ArtifactRef":
        """Convert database row to ArtifactRef."""
        from alma.workflow import ArtifactRef, ArtifactType

        return ArtifactRef(
            id=row["id"],
            memory_id=row["memory_id"],
            artifact_type=ArtifactType(row["artifact_type"]),
            storage_url=row["storage_url"],
            filename=row["filename"],
            mime_type=row["mime_type"],
            size_bytes=row["size_bytes"],
            checksum=row["checksum"],
            metadata=row["metadata"] if row["metadata"] else {},
            created_at=self._parse_datetime(row["created_at"])
            or datetime.now(timezone.utc),
        )
