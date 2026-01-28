"""
ALMA SQLite + FAISS Storage Backend.

Local storage using SQLite for structured data and FAISS for vector search.
This is the recommended backend for local development and testing.
"""

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from alma.storage.base import StorageBackend
from alma.types import (
    AntiPattern,
    DomainKnowledge,
    Heuristic,
    Outcome,
    UserPreference,
)

logger = logging.getLogger(__name__)

# Try to import FAISS, fall back to numpy-based search if not available
try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS not available, falling back to numpy-based vector search")


class SQLiteStorage(StorageBackend):
    """
    SQLite + FAISS storage backend.

    Uses SQLite for structured data and FAISS for efficient vector similarity search.
    Falls back to numpy cosine similarity if FAISS is not installed.

    Database schema:
        - heuristics: id, agent, project_id, condition, strategy, confidence, ...
        - outcomes: id, agent, project_id, task_type, task_description, success, ...
        - preferences: id, user_id, category, preference, source, ...
        - domain_knowledge: id, agent, project_id, domain, fact, ...
        - anti_patterns: id, agent, project_id, pattern, why_bad, ...
        - embeddings: id, memory_type, memory_id, embedding (blob)
    """

    def __init__(
        self,
        db_path: Path,
        embedding_dim: int = 384,  # Default for all-MiniLM-L6-v2
    ):
        """
        Initialize SQLite storage.

        Args:
            db_path: Path to SQLite database file
            embedding_dim: Dimension of embedding vectors
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.embedding_dim = embedding_dim

        # Initialize database
        self._init_database()

        # Initialize FAISS indices (one per memory type)
        self._indices: Dict[str, Any] = {}
        self._id_maps: Dict[str, List[str]] = {}  # memory_type -> [memory_ids]
        self._index_dirty: Dict[str, bool] = {}  # Track which indexes need rebuilding
        self._load_faiss_indices()

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SQLiteStorage":
        """Create instance from configuration."""
        storage_dir = config.get("storage_dir", ".alma")
        db_name = config.get("db_name", "alma.db")
        embedding_dim = config.get("embedding_dim", 384)

        db_path = Path(storage_dir) / db_name
        return cls(db_path=db_path, embedding_dim=embedding_dim)

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

    def _init_database(self):
        """Initialize database schema."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

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
                "CREATE INDEX IF NOT EXISTS idx_preferences_user "
                "ON preferences(user_id)"
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

            # Embeddings table (stores vectors as blobs)
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
                "CREATE INDEX IF NOT EXISTS idx_embeddings_type "
                "ON embeddings(memory_type)"
            )

    def _load_faiss_indices(self, memory_types: Optional[List[str]] = None):
        """Load or create FAISS indices for specified memory types.

        Args:
            memory_types: List of memory types to load. If None, loads all types.
        """
        if memory_types is None:
            memory_types = [
                "heuristics",
                "outcomes",
                "domain_knowledge",
                "anti_patterns",
            ]

        for memory_type in memory_types:
            if FAISS_AVAILABLE:
                # Use FAISS index
                self._indices[memory_type] = faiss.IndexFlatIP(self.embedding_dim)
            else:
                # Use list for numpy fallback
                self._indices[memory_type] = []

            self._id_maps[memory_type] = []
            self._index_dirty[memory_type] = False  # Mark as fresh after rebuild

            # Load existing embeddings
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT memory_id, embedding FROM embeddings WHERE memory_type = ?",
                    (memory_type,),
                )
                rows = cursor.fetchall()

                for row in rows:
                    memory_id = row["memory_id"]
                    embedding = np.frombuffer(row["embedding"], dtype=np.float32)

                    self._id_maps[memory_type].append(memory_id)
                    if FAISS_AVAILABLE:
                        self._indices[memory_type].add(
                            embedding.reshape(1, -1).astype(np.float32)
                        )
                    else:
                        self._indices[memory_type].append(embedding)

    def _ensure_index_fresh(self, memory_type: str) -> None:
        """Rebuild index for a memory type if it has been marked dirty.

        This implements lazy rebuilding - indexes are only rebuilt when
        actually needed for search, not immediately on every delete.

        Args:
            memory_type: The type of memory index to check/rebuild.
        """
        if self._index_dirty.get(memory_type, False):
            logger.debug(f"Rebuilding dirty index for {memory_type}")
            self._load_faiss_indices([memory_type])

    def _add_to_index(
        self,
        memory_type: str,
        memory_id: str,
        embedding: Optional[List[float]],
    ):
        """Add embedding to FAISS index."""
        if embedding is None:
            return

        embedding_array = np.array(embedding, dtype=np.float32)

        # Store in database
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO embeddings (memory_type, memory_id, embedding)
                VALUES (?, ?, ?)
                """,
                (memory_type, memory_id, embedding_array.tobytes()),
            )

        # Add to index
        self._id_maps[memory_type].append(memory_id)
        if FAISS_AVAILABLE:
            self._indices[memory_type].add(
                embedding_array.reshape(1, -1).astype(np.float32)
            )
        else:
            self._indices[memory_type].append(embedding_array)

    def _search_index(
        self,
        memory_type: str,
        query_embedding: List[float],
        top_k: int,
    ) -> List[Tuple[str, float]]:
        """Search FAISS index for similar embeddings."""
        # Ensure index is up-to-date before searching (lazy rebuild)
        self._ensure_index_fresh(memory_type)

        if not self._id_maps[memory_type]:
            return []

        query = np.array(query_embedding, dtype=np.float32).reshape(1, -1)

        if FAISS_AVAILABLE:
            # Normalize for cosine similarity (IndexFlatIP)
            faiss.normalize_L2(query)
            scores, indices = self._indices[memory_type].search(
                query, min(top_k, len(self._id_maps[memory_type]))
            )

            results = []
            for score, idx in zip(scores[0], indices[0], strict=False):
                if idx >= 0 and idx < len(self._id_maps[memory_type]):
                    results.append((self._id_maps[memory_type][idx], float(score)))
            return results
        else:
            # Numpy fallback with cosine similarity
            embeddings = np.array(self._indices[memory_type])
            if len(embeddings) == 0:
                return []

            # Normalize
            query_norm = query / np.linalg.norm(query)
            emb_norms = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

            # Cosine similarity
            similarities = np.dot(emb_norms, query_norm.T).flatten()

            # Get top k
            top_indices = np.argsort(similarities)[::-1][:top_k]

            return [
                (self._id_maps[memory_type][i], float(similarities[i]))
                for i in top_indices
            ]

    # ==================== WRITE OPERATIONS ====================

    def save_heuristic(self, heuristic: Heuristic) -> str:
        """Save a heuristic."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO heuristics
                (id, agent, project_id, condition, strategy, confidence,
                 occurrence_count, success_count, last_validated, created_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    (
                        heuristic.last_validated.isoformat()
                        if heuristic.last_validated
                        else None
                    ),
                    heuristic.created_at.isoformat() if heuristic.created_at else None,
                    json.dumps(heuristic.metadata) if heuristic.metadata else None,
                ),
            )

        # Add embedding to index
        self._add_to_index("heuristics", heuristic.id, heuristic.embedding)
        logger.debug(f"Saved heuristic: {heuristic.id}")
        return heuristic.id

    def save_outcome(self, outcome: Outcome) -> str:
        """Save an outcome."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO outcomes
                (id, agent, project_id, task_type, task_description, success,
                 strategy_used, duration_ms, error_message, user_feedback, timestamp, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    outcome.id,
                    outcome.agent,
                    outcome.project_id,
                    outcome.task_type,
                    outcome.task_description,
                    1 if outcome.success else 0,
                    outcome.strategy_used,
                    outcome.duration_ms,
                    outcome.error_message,
                    outcome.user_feedback,
                    outcome.timestamp.isoformat() if outcome.timestamp else None,
                    json.dumps(outcome.metadata) if outcome.metadata else None,
                ),
            )

        # Add embedding to index
        self._add_to_index("outcomes", outcome.id, outcome.embedding)
        logger.debug(f"Saved outcome: {outcome.id}")
        return outcome.id

    def save_user_preference(self, preference: UserPreference) -> str:
        """Save a user preference."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO preferences
                (id, user_id, category, preference, source, confidence, timestamp, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    preference.id,
                    preference.user_id,
                    preference.category,
                    preference.preference,
                    preference.source,
                    preference.confidence,
                    preference.timestamp.isoformat() if preference.timestamp else None,
                    json.dumps(preference.metadata) if preference.metadata else None,
                ),
            )
        logger.debug(f"Saved preference: {preference.id}")
        return preference.id

    def save_domain_knowledge(self, knowledge: DomainKnowledge) -> str:
        """Save domain knowledge."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO domain_knowledge
                (id, agent, project_id, domain, fact, source, confidence, last_verified, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    knowledge.id,
                    knowledge.agent,
                    knowledge.project_id,
                    knowledge.domain,
                    knowledge.fact,
                    knowledge.source,
                    knowledge.confidence,
                    (
                        knowledge.last_verified.isoformat()
                        if knowledge.last_verified
                        else None
                    ),
                    json.dumps(knowledge.metadata) if knowledge.metadata else None,
                ),
            )

        # Add embedding to index
        self._add_to_index("domain_knowledge", knowledge.id, knowledge.embedding)
        logger.debug(f"Saved domain knowledge: {knowledge.id}")
        return knowledge.id

    def save_anti_pattern(self, anti_pattern: AntiPattern) -> str:
        """Save an anti-pattern."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO anti_patterns
                (id, agent, project_id, pattern, why_bad, better_alternative,
                 occurrence_count, last_seen, created_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    anti_pattern.id,
                    anti_pattern.agent,
                    anti_pattern.project_id,
                    anti_pattern.pattern,
                    anti_pattern.why_bad,
                    anti_pattern.better_alternative,
                    anti_pattern.occurrence_count,
                    (
                        anti_pattern.last_seen.isoformat()
                        if anti_pattern.last_seen
                        else None
                    ),
                    (
                        anti_pattern.created_at.isoformat()
                        if anti_pattern.created_at
                        else None
                    ),
                    (
                        json.dumps(anti_pattern.metadata)
                        if anti_pattern.metadata
                        else None
                    ),
                ),
            )

        # Add embedding to index
        self._add_to_index("anti_patterns", anti_pattern.id, anti_pattern.embedding)
        logger.debug(f"Saved anti-pattern: {anti_pattern.id}")
        return anti_pattern.id

    # ==================== BATCH WRITE OPERATIONS ====================

    def save_heuristics(self, heuristics: List[Heuristic]) -> List[str]:
        """Save multiple heuristics in a batch using executemany."""
        if not heuristics:
            return []

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(
                """
                INSERT OR REPLACE INTO heuristics
                (id, agent, project_id, condition, strategy, confidence,
                 occurrence_count, success_count, last_validated, created_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                        h.last_validated.isoformat() if h.last_validated else None,
                        h.created_at.isoformat() if h.created_at else None,
                        json.dumps(h.metadata) if h.metadata else None,
                    )
                    for h in heuristics
                ],
            )

        # Add embeddings to index
        for h in heuristics:
            self._add_to_index("heuristics", h.id, h.embedding)

        logger.debug(f"Batch saved {len(heuristics)} heuristics")
        return [h.id for h in heuristics]

    def save_outcomes(self, outcomes: List[Outcome]) -> List[str]:
        """Save multiple outcomes in a batch using executemany."""
        if not outcomes:
            return []

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(
                """
                INSERT OR REPLACE INTO outcomes
                (id, agent, project_id, task_type, task_description, success,
                 strategy_used, duration_ms, error_message, user_feedback, timestamp, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        o.id,
                        o.agent,
                        o.project_id,
                        o.task_type,
                        o.task_description,
                        1 if o.success else 0,
                        o.strategy_used,
                        o.duration_ms,
                        o.error_message,
                        o.user_feedback,
                        o.timestamp.isoformat() if o.timestamp else None,
                        json.dumps(o.metadata) if o.metadata else None,
                    )
                    for o in outcomes
                ],
            )

        # Add embeddings to index
        for o in outcomes:
            self._add_to_index("outcomes", o.id, o.embedding)

        logger.debug(f"Batch saved {len(outcomes)} outcomes")
        return [o.id for o in outcomes]

    def save_domain_knowledge_batch(
        self, knowledge_items: List[DomainKnowledge]
    ) -> List[str]:
        """Save multiple domain knowledge items in a batch using executemany."""
        if not knowledge_items:
            return []

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(
                """
                INSERT OR REPLACE INTO domain_knowledge
                (id, agent, project_id, domain, fact, source, confidence, last_verified, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                        k.last_verified.isoformat() if k.last_verified else None,
                        json.dumps(k.metadata) if k.metadata else None,
                    )
                    for k in knowledge_items
                ],
            )

        # Add embeddings to index
        for k in knowledge_items:
            self._add_to_index("domain_knowledge", k.id, k.embedding)

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
        # If embedding provided, use vector search to get candidate IDs
        candidate_ids = None
        if embedding:
            search_results = self._search_index("heuristics", embedding, top_k * 2)
            candidate_ids = [id for id, _ in search_results]

        with self._get_connection() as conn:
            cursor = conn.cursor()

            query = "SELECT * FROM heuristics WHERE project_id = ? AND confidence >= ?"
            params: List[Any] = [project_id, min_confidence]

            if agent:
                query += " AND agent = ?"
                params.append(agent)

            if candidate_ids is not None:
                placeholders = ",".join("?" * len(candidate_ids))
                query += f" AND id IN ({placeholders})"
                params.extend(candidate_ids)

            query += " ORDER BY confidence DESC LIMIT ?"
            params.append(top_k)

            cursor.execute(query, params)
            rows = cursor.fetchall()

        return [self._row_to_heuristic(row) for row in rows]

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
        candidate_ids = None
        if embedding:
            search_results = self._search_index("outcomes", embedding, top_k * 2)
            candidate_ids = [id for id, _ in search_results]

        with self._get_connection() as conn:
            cursor = conn.cursor()

            query = "SELECT * FROM outcomes WHERE project_id = ?"
            params: List[Any] = [project_id]

            if agent:
                query += " AND agent = ?"
                params.append(agent)

            if task_type:
                query += " AND task_type = ?"
                params.append(task_type)

            if success_only:
                query += " AND success = 1"

            if candidate_ids is not None:
                placeholders = ",".join("?" * len(candidate_ids))
                query += f" AND id IN ({placeholders})"
                params.extend(candidate_ids)

            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(top_k)

            cursor.execute(query, params)
            rows = cursor.fetchall()

        return [self._row_to_outcome(row) for row in rows]

    def get_user_preferences(
        self,
        user_id: str,
        category: Optional[str] = None,
    ) -> List[UserPreference]:
        """Get user preferences."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            query = "SELECT * FROM preferences WHERE user_id = ?"
            params: List[Any] = [user_id]

            if category:
                query += " AND category = ?"
                params.append(category)

            cursor.execute(query, params)
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
        candidate_ids = None
        if embedding:
            search_results = self._search_index(
                "domain_knowledge", embedding, top_k * 2
            )
            candidate_ids = [id for id, _ in search_results]

        with self._get_connection() as conn:
            cursor = conn.cursor()

            query = "SELECT * FROM domain_knowledge WHERE project_id = ?"
            params: List[Any] = [project_id]

            if agent:
                query += " AND agent = ?"
                params.append(agent)

            if domain:
                query += " AND domain = ?"
                params.append(domain)

            if candidate_ids is not None:
                placeholders = ",".join("?" * len(candidate_ids))
                query += f" AND id IN ({placeholders})"
                params.extend(candidate_ids)

            query += " ORDER BY confidence DESC LIMIT ?"
            params.append(top_k)

            cursor.execute(query, params)
            rows = cursor.fetchall()

        return [self._row_to_domain_knowledge(row) for row in rows]

    def get_anti_patterns(
        self,
        project_id: str,
        agent: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        top_k: int = 5,
    ) -> List[AntiPattern]:
        """Get anti-patterns with optional vector search."""
        candidate_ids = None
        if embedding:
            search_results = self._search_index("anti_patterns", embedding, top_k * 2)
            candidate_ids = [id for id, _ in search_results]

        with self._get_connection() as conn:
            cursor = conn.cursor()

            query = "SELECT * FROM anti_patterns WHERE project_id = ?"
            params: List[Any] = [project_id]

            if agent:
                query += " AND agent = ?"
                params.append(agent)

            if candidate_ids is not None:
                placeholders = ",".join("?" * len(candidate_ids))
                query += f" AND id IN ({placeholders})"
                params.extend(candidate_ids)

            query += " ORDER BY occurrence_count DESC LIMIT ?"
            params.append(top_k)

            cursor.execute(query, params)
            rows = cursor.fetchall()

        return [self._row_to_anti_pattern(row) for row in rows]

    # ==================== MULTI-AGENT MEMORY SHARING ====================

    def get_heuristics_for_agents(
        self,
        project_id: str,
        agents: List[str],
        embedding: Optional[List[float]] = None,
        top_k: int = 5,
        min_confidence: float = 0.0,
    ) -> List[Heuristic]:
        """Get heuristics from multiple agents using optimized IN query."""
        if not agents:
            return []

        candidate_ids = None
        if embedding:
            search_results = self._search_index(
                "heuristics", embedding, top_k * 2 * len(agents)
            )
            candidate_ids = [id for id, _ in search_results]

        with self._get_connection() as conn:
            cursor = conn.cursor()

            placeholders = ",".join("?" * len(agents))
            query = f"SELECT * FROM heuristics WHERE project_id = ? AND confidence >= ? AND agent IN ({placeholders})"
            params: List[Any] = [project_id, min_confidence] + list(agents)

            if candidate_ids is not None:
                id_placeholders = ",".join("?" * len(candidate_ids))
                query += f" AND id IN ({id_placeholders})"
                params.extend(candidate_ids)

            query += " ORDER BY confidence DESC LIMIT ?"
            params.append(top_k * len(agents))

            cursor.execute(query, params)
            rows = cursor.fetchall()

        return [self._row_to_heuristic(row) for row in rows]

    def get_outcomes_for_agents(
        self,
        project_id: str,
        agents: List[str],
        task_type: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        top_k: int = 5,
        success_only: bool = False,
    ) -> List[Outcome]:
        """Get outcomes from multiple agents using optimized IN query."""
        if not agents:
            return []

        candidate_ids = None
        if embedding:
            search_results = self._search_index(
                "outcomes", embedding, top_k * 2 * len(agents)
            )
            candidate_ids = [id for id, _ in search_results]

        with self._get_connection() as conn:
            cursor = conn.cursor()

            placeholders = ",".join("?" * len(agents))
            query = f"SELECT * FROM outcomes WHERE project_id = ? AND agent IN ({placeholders})"
            params: List[Any] = [project_id] + list(agents)

            if task_type:
                query += " AND task_type = ?"
                params.append(task_type)

            if success_only:
                query += " AND success = 1"

            if candidate_ids is not None:
                id_placeholders = ",".join("?" * len(candidate_ids))
                query += f" AND id IN ({id_placeholders})"
                params.extend(candidate_ids)

            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(top_k * len(agents))

            cursor.execute(query, params)
            rows = cursor.fetchall()

        return [self._row_to_outcome(row) for row in rows]

    def get_domain_knowledge_for_agents(
        self,
        project_id: str,
        agents: List[str],
        domain: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        top_k: int = 5,
    ) -> List[DomainKnowledge]:
        """Get domain knowledge from multiple agents using optimized IN query."""
        if not agents:
            return []

        candidate_ids = None
        if embedding:
            search_results = self._search_index(
                "domain_knowledge", embedding, top_k * 2 * len(agents)
            )
            candidate_ids = [id for id, _ in search_results]

        with self._get_connection() as conn:
            cursor = conn.cursor()

            placeholders = ",".join("?" * len(agents))
            query = f"SELECT * FROM domain_knowledge WHERE project_id = ? AND agent IN ({placeholders})"
            params: List[Any] = [project_id] + list(agents)

            if domain:
                query += " AND domain = ?"
                params.append(domain)

            if candidate_ids is not None:
                id_placeholders = ",".join("?" * len(candidate_ids))
                query += f" AND id IN ({id_placeholders})"
                params.extend(candidate_ids)

            query += " ORDER BY confidence DESC LIMIT ?"
            params.append(top_k * len(agents))

            cursor.execute(query, params)
            rows = cursor.fetchall()

        return [self._row_to_domain_knowledge(row) for row in rows]

    def get_anti_patterns_for_agents(
        self,
        project_id: str,
        agents: List[str],
        embedding: Optional[List[float]] = None,
        top_k: int = 5,
    ) -> List[AntiPattern]:
        """Get anti-patterns from multiple agents using optimized IN query."""
        if not agents:
            return []

        candidate_ids = None
        if embedding:
            search_results = self._search_index(
                "anti_patterns", embedding, top_k * 2 * len(agents)
            )
            candidate_ids = [id for id, _ in search_results]

        with self._get_connection() as conn:
            cursor = conn.cursor()

            placeholders = ",".join("?" * len(agents))
            query = f"SELECT * FROM anti_patterns WHERE project_id = ? AND agent IN ({placeholders})"
            params: List[Any] = [project_id] + list(agents)

            if candidate_ids is not None:
                id_placeholders = ",".join("?" * len(candidate_ids))
                query += f" AND id IN ({id_placeholders})"
                params.extend(candidate_ids)

            query += " ORDER BY occurrence_count DESC LIMIT ?"
            params.append(top_k * len(agents))

            cursor.execute(query, params)
            rows = cursor.fetchall()

        return [self._row_to_anti_pattern(row) for row in rows]

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
            elif isinstance(value, datetime):
                value = value.isoformat()
            set_clauses.append(f"{key} = ?")
            params.append(value)

        params.append(heuristic_id)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"UPDATE heuristics SET {', '.join(set_clauses)} WHERE id = ?",
                params,
            )
            return cursor.rowcount > 0

    def increment_heuristic_occurrence(
        self,
        heuristic_id: str,
        success: bool,
    ) -> bool:
        """Increment heuristic occurrence count."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            if success:
                cursor.execute(
                    """
                    UPDATE heuristics
                    SET occurrence_count = occurrence_count + 1,
                        success_count = success_count + 1,
                        last_validated = ?
                    WHERE id = ?
                    """,
                    (datetime.now(timezone.utc).isoformat(), heuristic_id),
                )
            else:
                cursor.execute(
                    """
                    UPDATE heuristics
                    SET occurrence_count = occurrence_count + 1,
                        last_validated = ?
                    WHERE id = ?
                    """,
                    (datetime.now(timezone.utc).isoformat(), heuristic_id),
                )

            return cursor.rowcount > 0

    # ==================== DELETE OPERATIONS ====================

    def delete_outcomes_older_than(
        self,
        project_id: str,
        older_than: datetime,
        agent: Optional[str] = None,
    ) -> int:
        """Delete old outcomes."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            query = "DELETE FROM outcomes WHERE project_id = ? AND timestamp < ?"
            params: List[Any] = [project_id, older_than.isoformat()]

            if agent:
                query += " AND agent = ?"
                params.append(agent)

            cursor.execute(query, params)
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
            cursor = conn.cursor()

            query = "DELETE FROM heuristics WHERE project_id = ? AND confidence < ?"
            params: List[Any] = [project_id, below_confidence]

            if agent:
                query += " AND agent = ?"
                params.append(agent)

            cursor.execute(query, params)
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
            "storage_type": "sqlite",
            "faiss_available": FAISS_AVAILABLE,
        }

        with self._get_connection() as conn:
            cursor = conn.cursor()

            tables = ["heuristics", "outcomes", "domain_knowledge", "anti_patterns"]
            for table in tables:
                query = f"SELECT COUNT(*) FROM {table} WHERE project_id = ?"
                params: List[Any] = [project_id]
                if agent:
                    query += " AND agent = ?"
                    params.append(agent)
                cursor.execute(query, params)
                stats[f"{table}_count"] = cursor.fetchone()[0]

            # Preferences don't have project_id
            cursor.execute("SELECT COUNT(*) FROM preferences")
            stats["preferences_count"] = cursor.fetchone()[0]

            # Embedding counts
            cursor.execute("SELECT COUNT(*) FROM embeddings")
            stats["embeddings_count"] = cursor.fetchone()[0]

        stats["total_count"] = sum(
            stats.get(k, 0) for k in stats if k.endswith("_count")
        )

        return stats

    # ==================== HELPERS ====================

    def _parse_datetime(self, value: Any) -> Optional[datetime]:
        """Parse datetime from string."""
        if value is None:
            return None
        if isinstance(value, datetime):
            return value
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            return None

    def _row_to_heuristic(self, row: sqlite3.Row) -> Heuristic:
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
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
        )

    def _row_to_outcome(self, row: sqlite3.Row) -> Outcome:
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
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
        )

    def _row_to_preference(self, row: sqlite3.Row) -> UserPreference:
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
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
        )

    def _row_to_domain_knowledge(self, row: sqlite3.Row) -> DomainKnowledge:
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
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
        )

    def _row_to_anti_pattern(self, row: sqlite3.Row) -> AntiPattern:
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
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
        )

    # ===== Additional abstract method implementations =====

    def update_heuristic_confidence(
        self,
        heuristic_id: str,
        new_confidence: float,
    ) -> bool:
        """Update confidence score for a heuristic."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "UPDATE heuristics SET confidence = ? WHERE id = ?",
                (new_confidence, heuristic_id),
            )
            return cursor.rowcount > 0

    def update_knowledge_confidence(
        self,
        knowledge_id: str,
        new_confidence: float,
    ) -> bool:
        """Update confidence score for domain knowledge."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "UPDATE domain_knowledge SET confidence = ? WHERE id = ?",
                (new_confidence, knowledge_id),
            )
            return cursor.rowcount > 0

    def delete_heuristic(self, heuristic_id: str) -> bool:
        """Delete a heuristic by ID."""
        with self._get_connection() as conn:
            # Also remove from embedding index
            conn.execute(
                "DELETE FROM embeddings WHERE memory_type = 'heuristics' AND memory_id = ?",
                (heuristic_id,),
            )
            cursor = conn.execute(
                "DELETE FROM heuristics WHERE id = ?",
                (heuristic_id,),
            )
            if cursor.rowcount > 0:
                # Mark index as dirty for lazy rebuild on next search
                self._index_dirty["heuristics"] = True
                return True
            return False

    def delete_outcome(self, outcome_id: str) -> bool:
        """Delete an outcome by ID."""
        with self._get_connection() as conn:
            # Also remove from embedding index
            conn.execute(
                "DELETE FROM embeddings WHERE memory_type = 'outcomes' AND memory_id = ?",
                (outcome_id,),
            )
            cursor = conn.execute(
                "DELETE FROM outcomes WHERE id = ?",
                (outcome_id,),
            )
            if cursor.rowcount > 0:
                # Mark index as dirty for lazy rebuild on next search
                self._index_dirty["outcomes"] = True
                return True
            return False

    def delete_domain_knowledge(self, knowledge_id: str) -> bool:
        """Delete domain knowledge by ID."""
        with self._get_connection() as conn:
            # Also remove from embedding index
            conn.execute(
                "DELETE FROM embeddings WHERE memory_type = 'domain_knowledge' AND memory_id = ?",
                (knowledge_id,),
            )
            cursor = conn.execute(
                "DELETE FROM domain_knowledge WHERE id = ?",
                (knowledge_id,),
            )
            if cursor.rowcount > 0:
                # Mark index as dirty for lazy rebuild on next search
                self._index_dirty["domain_knowledge"] = True
                return True
            return False

    def delete_anti_pattern(self, anti_pattern_id: str) -> bool:
        """Delete an anti-pattern by ID."""
        with self._get_connection() as conn:
            # Also remove from embedding index
            conn.execute(
                "DELETE FROM embeddings WHERE memory_type = 'anti_patterns' AND memory_id = ?",
                (anti_pattern_id,),
            )
            cursor = conn.execute(
                "DELETE FROM anti_patterns WHERE id = ?",
                (anti_pattern_id,),
            )
            if cursor.rowcount > 0:
                # Mark index as dirty for lazy rebuild on next search
                self._index_dirty["anti_patterns"] = True
                return True
            return False
