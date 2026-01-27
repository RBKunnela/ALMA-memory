"""
ALMA Retrieval Engine.

Handles semantic search and memory retrieval with scoring and caching.
"""

import logging
import time
from typing import Any, Dict, List, Optional

from alma.retrieval.cache import NullCache, RetrievalCache
from alma.retrieval.scoring import MemoryScorer, ScoredItem, ScoringWeights
from alma.storage.base import StorageBackend
from alma.types import MemoryScope, MemorySlice

logger = logging.getLogger(__name__)


class RetrievalEngine:
    """
    Retrieves relevant memories for task context injection.

    Features:
    - Semantic search via embeddings
    - Recency weighting (newer memories preferred)
    - Success rate weighting (proven strategies ranked higher)
    - Caching for repeated queries
    - Configurable scoring weights
    """

    def __init__(
        self,
        storage: StorageBackend,
        embedding_provider: str = "local",
        cache_ttl_seconds: int = 300,
        enable_cache: bool = True,
        max_cache_entries: int = 1000,
        scoring_weights: Optional[ScoringWeights] = None,
        recency_half_life_days: float = 30.0,
        min_score_threshold: float = 0.2,
    ):
        """
        Initialize retrieval engine.

        Args:
            storage: Storage backend to query
            embedding_provider: "local" (sentence-transformers) or "azure" (Azure OpenAI)
            cache_ttl_seconds: How long to cache query results
            enable_cache: Whether to enable caching
            max_cache_entries: Maximum cache entries before eviction
            scoring_weights: Custom weights for similarity/recency/success/confidence
            recency_half_life_days: Days after which recency score halves
            min_score_threshold: Minimum score to include in results
        """
        self.storage = storage
        self.embedding_provider = embedding_provider
        self.min_score_threshold = min_score_threshold
        self._embedder = None

        # Initialize scorer
        self.scorer = MemoryScorer(
            weights=scoring_weights or ScoringWeights(),
            recency_half_life_days=recency_half_life_days,
        )

        # Initialize cache
        if enable_cache:
            self.cache = RetrievalCache(
                ttl_seconds=cache_ttl_seconds,
                max_entries=max_cache_entries,
            )
        else:
            self.cache = NullCache()

    def retrieve(
        self,
        query: str,
        agent: str,
        project_id: str,
        user_id: Optional[str] = None,
        top_k: int = 5,
        scope: Optional[MemoryScope] = None,
        bypass_cache: bool = False,
        include_shared: bool = True,
    ) -> MemorySlice:
        """
        Retrieve relevant memories for a task.

        Supports multi-agent memory sharing: if a scope is provided with
        inherit_from agents, memories from those agents will also be included.
        Shared memories have their origin tracked in the metadata['shared_from'] field.

        Args:
            query: Task description to find relevant memories for
            agent: Agent requesting memories
            project_id: Project context
            user_id: Optional user for preference retrieval
            top_k: Max items per memory type
            scope: Agent's learning scope for filtering (enables multi-agent sharing)
            bypass_cache: Skip cache lookup/storage
            include_shared: If True and scope has inherit_from, include shared memories

        Returns:
            MemorySlice with relevant memories, scored and ranked
        """
        start_time = time.time()

        # Check cache first
        if not bypass_cache:
            cached = self.cache.get(query, agent, project_id, user_id, top_k)
            if cached is not None:
                cached.retrieval_time_ms = int((time.time() - start_time) * 1000)
                logger.debug(f"Cache hit for query: {query[:50]}...")
                return cached

        # Generate embedding for query
        query_embedding = self._get_embedding(query)

        # Determine which agents to query based on scope
        agents_to_query = [agent]
        if include_shared and scope and scope.inherit_from:
            agents_to_query = scope.get_readable_agents()
            logger.debug(
                f"Multi-agent retrieval for {agent}: querying {agents_to_query}"
            )

        # Retrieve raw items from storage (with vector search)
        if len(agents_to_query) > 1:
            # Use multi-agent query methods
            raw_heuristics = self.storage.get_heuristics_for_agents(
                project_id=project_id,
                agents=agents_to_query,
                embedding=query_embedding,
                top_k=top_k * 2,
                min_confidence=0.0,
            )
            raw_outcomes = self.storage.get_outcomes_for_agents(
                project_id=project_id,
                agents=agents_to_query,
                embedding=query_embedding,
                top_k=top_k * 2,
                success_only=False,
            )
            raw_domain_knowledge = self.storage.get_domain_knowledge_for_agents(
                project_id=project_id,
                agents=agents_to_query,
                embedding=query_embedding,
                top_k=top_k * 2,
            )
            raw_anti_patterns = self.storage.get_anti_patterns_for_agents(
                project_id=project_id,
                agents=agents_to_query,
                embedding=query_embedding,
                top_k=top_k * 2,
            )

            # Mark shared memories with origin tracking
            raw_heuristics = self._mark_shared_memories(raw_heuristics, agent)
            raw_outcomes = self._mark_shared_memories(raw_outcomes, agent)
            raw_domain_knowledge = self._mark_shared_memories(
                raw_domain_knowledge, agent
            )
            raw_anti_patterns = self._mark_shared_memories(raw_anti_patterns, agent)
        else:
            # Single agent query (original behavior)
            raw_heuristics = self.storage.get_heuristics(
                project_id=project_id,
                agent=agent,
                embedding=query_embedding,
                top_k=top_k * 2,
                min_confidence=0.0,
            )
            raw_outcomes = self.storage.get_outcomes(
                project_id=project_id,
                agent=agent,
                embedding=query_embedding,
                top_k=top_k * 2,
                success_only=False,
            )
            raw_domain_knowledge = self.storage.get_domain_knowledge(
                project_id=project_id,
                agent=agent,
                embedding=query_embedding,
                top_k=top_k * 2,
            )
            raw_anti_patterns = self.storage.get_anti_patterns(
                project_id=project_id,
                agent=agent,
                embedding=query_embedding,
                top_k=top_k * 2,
            )

        # Score and rank each type
        scored_heuristics = self.scorer.score_heuristics(raw_heuristics)
        scored_outcomes = self.scorer.score_outcomes(raw_outcomes)
        scored_knowledge = self.scorer.score_domain_knowledge(raw_domain_knowledge)
        scored_anti_patterns = self.scorer.score_anti_patterns(raw_anti_patterns)

        # Apply threshold and limit
        final_heuristics = self._extract_top_k(scored_heuristics, top_k)
        final_outcomes = self._extract_top_k(scored_outcomes, top_k)
        final_knowledge = self._extract_top_k(scored_knowledge, top_k)
        final_anti_patterns = self._extract_top_k(scored_anti_patterns, top_k)

        # Get user preferences (not scored, just retrieved)
        preferences = []
        if user_id:
            preferences = self.storage.get_user_preferences(user_id=user_id)

        retrieval_time_ms = int((time.time() - start_time) * 1000)

        result = MemorySlice(
            heuristics=final_heuristics,
            outcomes=final_outcomes,
            preferences=preferences,
            domain_knowledge=final_knowledge,
            anti_patterns=final_anti_patterns,
            query=query,
            agent=agent,
            retrieval_time_ms=retrieval_time_ms,
        )

        # Cache result
        if not bypass_cache:
            self.cache.set(query, agent, project_id, result, user_id, top_k)

        logger.info(
            f"Retrieved {result.total_items} memories for '{query[:50]}...' "
            f"in {retrieval_time_ms}ms"
        )

        return result

    def _mark_shared_memories(
        self,
        memories: List[Any],
        requesting_agent: str,
    ) -> List[Any]:
        """
        Mark memories that came from other agents with their origin.

        Adds 'shared_from' to metadata for memories not owned by requesting_agent.
        This maintains write isolation - only the owning agent can modify their memories.

        Args:
            memories: List of memory objects (Heuristic, Outcome, etc.)
            requesting_agent: The agent that requested the memories

        Returns:
            Same memories with shared_from metadata added where applicable
        """
        for memory in memories:
            if hasattr(memory, "agent") and memory.agent != requesting_agent:
                if not hasattr(memory, "metadata") or memory.metadata is None:
                    memory.metadata = {}
                memory.metadata["shared_from"] = memory.agent
        return memories

    def _extract_top_k(
        self,
        scored_items: List[ScoredItem],
        top_k: int,
    ) -> List[Any]:
        """
        Extract top-k items after filtering by score threshold.

        Args:
            scored_items: Scored and sorted items
            top_k: Maximum number to return

        Returns:
            List of original items (unwrapped from ScoredItem)
        """
        filtered = self.scorer.apply_score_threshold(
            scored_items, self.min_score_threshold
        )
        return [item.item for item in filtered[:top_k]]

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
            from alma.retrieval.embeddings import AzureEmbedder

            return AzureEmbedder()
        elif self.embedding_provider == "mock":
            from alma.retrieval.embeddings import MockEmbedder

            return MockEmbedder()
        else:
            from alma.retrieval.embeddings import LocalEmbedder

            return LocalEmbedder()

    def invalidate_cache(
        self,
        agent: Optional[str] = None,
        project_id: Optional[str] = None,
    ):
        """
        Invalidate cache entries.

        Should be called after memory updates to ensure fresh results.

        Args:
            agent: Invalidate entries for this agent
            project_id: Invalidate entries for this project
        """
        self.cache.invalidate(agent=agent, project_id=project_id)

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        stats = self.cache.get_stats()
        return stats.to_dict()

    def clear_cache(self):
        """Clear all cached results."""
        self.cache.clear()

    def get_scorer_weights(self) -> Dict[str, float]:
        """Get current scoring weights."""
        w = self.scorer.weights
        return {
            "similarity": w.similarity,
            "recency": w.recency,
            "success_rate": w.success_rate,
            "confidence": w.confidence,
        }

    def update_scorer_weights(
        self,
        similarity: Optional[float] = None,
        recency: Optional[float] = None,
        success_rate: Optional[float] = None,
        confidence: Optional[float] = None,
    ):
        """
        Update scoring weights (will be normalized to sum to 1.0).

        Args:
            similarity: Weight for semantic similarity
            recency: Weight for recency
            success_rate: Weight for success rate
            confidence: Weight for stored confidence
        """
        current = self.scorer.weights
        self.scorer.weights = ScoringWeights(
            similarity=similarity if similarity is not None else current.similarity,
            recency=recency if recency is not None else current.recency,
            success_rate=success_rate
            if success_rate is not None
            else current.success_rate,
            confidence=confidence if confidence is not None else current.confidence,
        )
        # Clear cache since scoring changed
        self.cache.clear()
