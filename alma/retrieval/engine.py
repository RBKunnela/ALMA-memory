"""
ALMA Retrieval Engine.

Handles semantic search and memory retrieval with scoring.
"""

import time
import logging
from typing import Optional, List, Dict, Any

from alma.types import MemorySlice, MemoryScope
from alma.storage.base import StorageBackend

logger = logging.getLogger(__name__)


class RetrievalEngine:
    """
    Retrieves relevant memories for task context injection.

    Features:
    - Semantic search via embeddings
    - Recency weighting
    - Success rate weighting
    - Caching for repeated queries
    """

    def __init__(
        self,
        storage: StorageBackend,
        embedding_provider: str = "local",
        cache_ttl_seconds: int = 300,
    ):
        """
        Initialize retrieval engine.

        Args:
            storage: Storage backend to query
            embedding_provider: "local" (sentence-transformers) or "azure" (Azure OpenAI)
            cache_ttl_seconds: How long to cache query results
        """
        self.storage = storage
        self.embedding_provider = embedding_provider
        self.cache_ttl = cache_ttl_seconds
        self._cache: Dict[str, tuple] = {}  # query_hash -> (result, timestamp)
        self._embedder = None

    def retrieve(
        self,
        query: str,
        agent: str,
        project_id: str,
        user_id: Optional[str] = None,
        top_k: int = 5,
        scope: Optional[MemoryScope] = None,
    ) -> MemorySlice:
        """
        Retrieve relevant memories for a task.

        Args:
            query: Task description to find relevant memories for
            agent: Agent requesting memories
            project_id: Project context
            user_id: Optional user for preference retrieval
            top_k: Max items per memory type
            scope: Agent's learning scope for filtering

        Returns:
            MemorySlice with relevant memories
        """
        start_time = time.time()

        # Generate embedding for query
        query_embedding = self._get_embedding(query)

        # Retrieve from each memory type
        heuristics = self.storage.get_heuristics(
            project_id=project_id,
            agent=agent,
            embedding=query_embedding,
            top_k=top_k,
            min_confidence=0.3,
        )

        outcomes = self.storage.get_outcomes(
            project_id=project_id,
            agent=agent,
            embedding=query_embedding,
            top_k=top_k,
            success_only=False,  # Include failures to learn from
        )

        domain_knowledge = self.storage.get_domain_knowledge(
            project_id=project_id,
            agent=agent,
            embedding=query_embedding,
            top_k=top_k,
        )

        anti_patterns = self.storage.get_anti_patterns(
            project_id=project_id,
            agent=agent,
            embedding=query_embedding,
            top_k=top_k,
        )

        # Get user preferences if user_id provided
        preferences = []
        if user_id:
            preferences = self.storage.get_user_preferences(user_id=user_id)

        retrieval_time_ms = int((time.time() - start_time) * 1000)

        return MemorySlice(
            heuristics=heuristics,
            outcomes=outcomes,
            preferences=preferences,
            domain_knowledge=domain_knowledge,
            anti_patterns=anti_patterns,
            query=query,
            agent=agent,
            retrieval_time_ms=retrieval_time_ms,
        )

    def _get_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for text.

        Uses lazy initialization of embedding model.
        """
        if self._embedder is None:
            self._embedder = self._init_embedder()

        return self._embedder.encode(text)

    def _init_embedder(self):
        """Initialize the embedding model based on provider config."""
        if self.embedding_provider == "azure":
            # Azure OpenAI embeddings
            from alma.retrieval.embeddings import AzureEmbedder
            return AzureEmbedder()
        else:
            # Local sentence-transformers
            from alma.retrieval.embeddings import LocalEmbedder
            return LocalEmbedder()

    def clear_cache(self):
        """Clear the query cache."""
        self._cache.clear()
