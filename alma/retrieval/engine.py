"""
ALMA Retrieval Engine.

Handles semantic search and memory retrieval with scoring and caching.
Supports mode-aware retrieval for different cognitive tasks.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from alma.observability.logging import get_logger
from alma.observability.metrics import get_metrics
from alma.observability.tracing import get_tracer
from alma.retrieval.cache import NullCache, RetrievalCache
from alma.retrieval.modes import (
    RetrievalMode,
    get_mode_config,
    get_mode_reason,
    infer_mode_from_query,
)
from alma.retrieval.scoring import MemoryScorer, ScoredItem, ScoringWeights
from alma.storage.base import StorageBackend
from alma.types import MemoryScope, MemorySlice

logger = logging.getLogger(__name__)
structured_logger = get_logger(__name__)
tracer = get_tracer(__name__)


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

    def retrieve_with_mode(
        self,
        query: str,
        agent: str,
        project_id: str,
        mode: Optional[RetrievalMode] = None,
        user_id: Optional[str] = None,
        top_k: Optional[int] = None,
        min_confidence: Optional[float] = None,
        scope: Optional[MemoryScope] = None,
        bypass_cache: bool = False,
        include_shared: bool = True,
    ) -> Tuple[MemorySlice, RetrievalMode, str]:
        """
        Retrieve memories using mode-aware strategy.

        Different cognitive tasks require different retrieval approaches:
        - BROAD: Planning, brainstorming - diverse, exploratory results
        - PRECISE: Execution, implementation - high-confidence matches
        - DIAGNOSTIC: Debugging, troubleshooting - anti-patterns and failures
        - LEARNING: Pattern finding - similar memories for consolidation
        - RECALL: Exact lookup - prioritizes exact matches

        Args:
            query: Task description to find relevant memories for
            agent: Agent requesting memories
            project_id: Project context
            mode: Retrieval mode (auto-inferred if None)
            user_id: Optional user for preference retrieval
            top_k: Override mode's default top_k
            min_confidence: Override mode's default min_confidence
            scope: Agent's learning scope for filtering
            bypass_cache: Skip cache lookup/storage
            include_shared: Include memories from inherit_from agents

        Returns:
            Tuple of (MemorySlice, detected_mode, mode_reason)
        """
        start_time = time.time()

        # Auto-infer mode if not specified
        if mode is None:
            mode = infer_mode_from_query(query)

        mode_reason = get_mode_reason(query, mode)
        config = get_mode_config(mode)

        # Apply overrides
        effective_top_k = top_k if top_k is not None else config.top_k
        effective_min_confidence = (
            min_confidence if min_confidence is not None else config.min_confidence
        )

        # Store original settings
        original_weights = self.scorer.weights
        original_threshold = self.min_score_threshold

        try:
            # Apply mode-specific scoring weights
            if config.weights:
                self.scorer.weights = ScoringWeights(
                    similarity=config.weights.get("similarity", 0.4),
                    recency=config.weights.get("recency", 0.3),
                    success_rate=config.weights.get("success_rate", 0.2),
                    confidence=config.weights.get("confidence", 0.1),
                )
            self.min_score_threshold = effective_min_confidence

            # Get extra candidates for diversity filtering
            retrieval_k = effective_top_k
            if config.diversity_factor > 0:
                retrieval_k = effective_top_k * 3  # Get more for filtering

            # Check cache (with mode in key would be ideal, but use bypass for now)
            # Mode-aware caching could be added in future
            if not bypass_cache:
                cache_key_suffix = f"_mode_{mode.value}"
                cached = self.cache.get(
                    query + cache_key_suffix,
                    agent,
                    project_id,
                    user_id,
                    effective_top_k,
                )
                if cached is not None:
                    cached.retrieval_time_ms = int((time.time() - start_time) * 1000)
                    logger.debug(f"Cache hit for mode-aware query: {query[:50]}...")
                    return cached, mode, mode_reason

            # Generate embedding
            query_embedding = self._get_embedding(query)

            # Determine agents to query
            agents_to_query = [agent]
            if include_shared and scope and scope.inherit_from:
                agents_to_query = scope.get_readable_agents()

            # Retrieve raw items
            if len(agents_to_query) > 1:
                raw_heuristics = self.storage.get_heuristics_for_agents(
                    project_id=project_id,
                    agents=agents_to_query,
                    embedding=query_embedding,
                    top_k=retrieval_k * 2,
                    min_confidence=0.0,
                )
                raw_outcomes = self.storage.get_outcomes_for_agents(
                    project_id=project_id,
                    agents=agents_to_query,
                    embedding=query_embedding,
                    top_k=retrieval_k * 2,
                    success_only=False,
                )
                raw_domain_knowledge = self.storage.get_domain_knowledge_for_agents(
                    project_id=project_id,
                    agents=agents_to_query,
                    embedding=query_embedding,
                    top_k=retrieval_k * 2,
                )
                raw_anti_patterns = []
                if config.include_anti_patterns:
                    raw_anti_patterns = self.storage.get_anti_patterns_for_agents(
                        project_id=project_id,
                        agents=agents_to_query,
                        embedding=query_embedding,
                        top_k=retrieval_k * 2,
                    )

                # Mark shared memories
                raw_heuristics = self._mark_shared_memories(raw_heuristics, agent)
                raw_outcomes = self._mark_shared_memories(raw_outcomes, agent)
                raw_domain_knowledge = self._mark_shared_memories(
                    raw_domain_knowledge, agent
                )
                raw_anti_patterns = self._mark_shared_memories(raw_anti_patterns, agent)
            else:
                raw_heuristics = self.storage.get_heuristics(
                    project_id=project_id,
                    agent=agent,
                    embedding=query_embedding,
                    top_k=retrieval_k * 2,
                    min_confidence=0.0,
                )
                raw_outcomes = self.storage.get_outcomes(
                    project_id=project_id,
                    agent=agent,
                    embedding=query_embedding,
                    top_k=retrieval_k * 2,
                    success_only=False,
                )
                raw_domain_knowledge = self.storage.get_domain_knowledge(
                    project_id=project_id,
                    agent=agent,
                    embedding=query_embedding,
                    top_k=retrieval_k * 2,
                )
                raw_anti_patterns = []
                if config.include_anti_patterns:
                    raw_anti_patterns = self.storage.get_anti_patterns(
                        project_id=project_id,
                        agent=agent,
                        embedding=query_embedding,
                        top_k=retrieval_k * 2,
                    )

            # Score items
            scored_heuristics = self.scorer.score_heuristics(raw_heuristics)
            scored_outcomes = self.scorer.score_outcomes(raw_outcomes)
            scored_knowledge = self.scorer.score_domain_knowledge(raw_domain_knowledge)
            scored_anti_patterns = self.scorer.score_anti_patterns(raw_anti_patterns)

            # Apply mode-specific processing
            if config.prioritize_failures:
                scored_outcomes = self._boost_failures(scored_outcomes)

            if config.exact_match_boost > 1.0:
                scored_heuristics = self._apply_exact_match_boost(
                    scored_heuristics, config.exact_match_boost
                )
                scored_outcomes = self._apply_exact_match_boost(
                    scored_outcomes, config.exact_match_boost
                )
                scored_knowledge = self._apply_exact_match_boost(
                    scored_knowledge, config.exact_match_boost
                )

            # Apply diversity filtering if enabled
            if config.diversity_factor > 0:
                scored_heuristics = self._diversify_results(
                    scored_heuristics, config.diversity_factor
                )
                scored_outcomes = self._diversify_results(
                    scored_outcomes, config.diversity_factor
                )
                scored_knowledge = self._diversify_results(
                    scored_knowledge, config.diversity_factor
                )
                scored_anti_patterns = self._diversify_results(
                    scored_anti_patterns, config.diversity_factor
                )

            # Extract top-k with threshold
            final_heuristics = self._extract_top_k(scored_heuristics, effective_top_k)
            final_outcomes = self._extract_top_k(scored_outcomes, effective_top_k)
            final_knowledge = self._extract_top_k(scored_knowledge, effective_top_k)
            final_anti_patterns = self._extract_top_k(
                scored_anti_patterns, effective_top_k
            )

            # Get user preferences
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
                cache_key_suffix = f"_mode_{mode.value}"
                self.cache.set(
                    query + cache_key_suffix,
                    agent,
                    project_id,
                    result,
                    user_id,
                    effective_top_k,
                )

            logger.info(
                f"Mode-aware retrieval ({mode.value}): {result.total_items} memories "
                f"for '{query[:50]}...' in {retrieval_time_ms}ms"
            )

            return result, mode, mode_reason

        finally:
            # Restore original settings
            self.scorer.weights = original_weights
            self.min_score_threshold = original_threshold

    def _diversify_results(
        self,
        scored_items: List[ScoredItem],
        diversity_factor: float,
    ) -> List[ScoredItem]:
        """
        Apply MMR-style diversity filtering to reduce redundancy.

        Maximal Marginal Relevance balances relevance with diversity
        by penalizing items too similar to already-selected items.

        Args:
            scored_items: Scored and sorted items
            diversity_factor: 0.0 = pure relevance, 1.0 = max diversity

        Returns:
            Reordered list with diversity applied
        """
        if not scored_items or len(scored_items) <= 1 or diversity_factor == 0:
            return scored_items

        # Start with highest-scored item
        selected = [scored_items[0]]
        remaining = list(scored_items[1:])

        while remaining:
            best_idx = 0
            best_mmr_score = float("-inf")

            for i, candidate in enumerate(remaining):
                # Find max similarity to any selected item
                max_sim_to_selected = 0.0
                for selected_item in selected:
                    # Use similarity scores as proxy for semantic similarity
                    # In a full implementation, we'd recompute embeddings
                    sim = self._estimate_similarity(candidate, selected_item)
                    max_sim_to_selected = max(max_sim_to_selected, sim)

                # MMR score: relevance - diversity_factor * max_similarity
                mmr_score = candidate.score - (diversity_factor * max_sim_to_selected)

                if mmr_score > best_mmr_score:
                    best_mmr_score = mmr_score
                    best_idx = i

            selected.append(remaining.pop(best_idx))

        return selected

    def _estimate_similarity(
        self,
        item1: ScoredItem,
        item2: ScoredItem,
    ) -> float:
        """
        Estimate semantic similarity between two items.

        Uses similarity scores as a proxy. For more accurate results,
        would need to recompute embeddings (expensive).

        Args:
            item1: First scored item
            item2: Second scored item

        Returns:
            Estimated similarity (0-1)
        """
        # Use the geometric mean of similarity scores as a rough estimate
        # Items with similar relevance to query are likely similar to each other
        sim1 = item1.similarity_score
        sim2 = item2.similarity_score

        # If both are highly similar to query, they're likely similar to each other
        # This is a heuristic - proper MMR would use actual pairwise similarity
        if sim1 > 0 and sim2 > 0:
            return (sim1 * sim2) ** 0.5
        return 0.0

    def _boost_failures(
        self,
        scored_outcomes: List[ScoredItem],
    ) -> List[ScoredItem]:
        """
        Boost failed outcomes for diagnostic mode.

        In debugging contexts, failures are valuable learning opportunities.

        Args:
            scored_outcomes: Scored outcomes

        Returns:
            Outcomes with failures boosted
        """
        boosted = []
        for item in scored_outcomes:
            outcome = item.item
            # Check if this is a failed outcome
            if hasattr(outcome, "success") and not outcome.success:
                # Boost the score by 50%
                boosted.append(
                    ScoredItem(
                        item=item.item,
                        score=item.score * 1.5,
                        similarity_score=item.similarity_score,
                        recency_score=item.recency_score,
                        success_score=item.success_score,
                        confidence_score=item.confidence_score,
                    )
                )
            else:
                boosted.append(item)

        # Re-sort after boosting
        return sorted(boosted, key=lambda x: -x.score)

    def _apply_exact_match_boost(
        self,
        scored_items: List[ScoredItem],
        boost_factor: float,
    ) -> List[ScoredItem]:
        """
        Boost items with very high similarity scores.

        For PRECISE and RECALL modes, we want to strongly prefer
        near-exact matches.

        Args:
            scored_items: Scored items
            boost_factor: Multiplier for high-similarity items

        Returns:
            Items with exact matches boosted
        """
        boosted = []
        for item in scored_items:
            # Boost items with >0.9 similarity
            if item.similarity_score > 0.9:
                boosted.append(
                    ScoredItem(
                        item=item.item,
                        score=item.score * boost_factor,
                        similarity_score=item.similarity_score,
                        recency_score=item.recency_score,
                        success_score=item.success_score,
                        confidence_score=item.confidence_score,
                    )
                )
            # Smaller boost for >0.8 similarity
            elif item.similarity_score > 0.8:
                boosted.append(
                    ScoredItem(
                        item=item.item,
                        score=item.score * (1 + (boost_factor - 1) / 2),
                        similarity_score=item.similarity_score,
                        recency_score=item.recency_score,
                        success_score=item.success_score,
                        confidence_score=item.confidence_score,
                    )
                )
            else:
                boosted.append(item)

        # Re-sort after boosting
        return sorted(boosted, key=lambda x: -x.score)

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

        start_time = time.time()
        embedding = self._embedder.encode(text)
        duration_ms = (time.time() - start_time) * 1000

        # Record embedding generation metrics
        metrics = get_metrics()
        metrics.record_embedding_latency(
            duration_ms=duration_ms,
            provider=self.embedding_provider,
            batch_size=1,
        )

        return embedding

    def _init_embedder(self):
        """Initialize the embedding model based on provider config."""
        if self.embedding_provider == "azure":
            from alma.retrieval.embeddings import AzureEmbedder

            embedder = AzureEmbedder()
        elif self.embedding_provider == "mock":
            from alma.retrieval.embeddings import MockEmbedder

            embedder = MockEmbedder()
        else:
            from alma.retrieval.embeddings import LocalEmbedder

            embedder = LocalEmbedder()

        # Validate embedding dimension matches storage configuration
        self._validate_embedding_dimension(embedder)
        return embedder

    def _validate_embedding_dimension(self, embedder) -> None:
        """
        Validate that embedding provider dimension matches storage configuration.

        Raises:
            ValueError: If dimensions don't match
        """
        provider_dim = embedder.dimension

        # Check if storage has embedding_dim attribute
        storage_dim = getattr(self.storage, "embedding_dim", None)
        if storage_dim is None:
            logger.debug(
                "Storage backend doesn't specify embedding_dim, skipping validation"
            )
            return

        # Skip validation if storage_dim is not an integer (e.g., mock objects)
        if not isinstance(storage_dim, int):
            logger.debug(
                f"Storage embedding_dim is not an integer ({type(storage_dim)}), "
                "skipping validation"
            )
            return

        if provider_dim != storage_dim:
            raise ValueError(
                f"Embedding dimension mismatch: provider '{self.embedding_provider}' "
                f"outputs {provider_dim} dimensions, but storage is configured for "
                f"{storage_dim} dimensions. Update your config's embedding_dim to "
                f"match the provider, or use a different embedding provider.\n"
                f"  - local (all-MiniLM-L6-v2): 384 dimensions\n"
                f"  - azure (text-embedding-3-small): 1536 dimensions"
            )

        logger.info(
            f"Embedding dimension validated: {provider_dim} "
            f"(provider: {self.embedding_provider})"
        )

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
            success_rate=(
                success_rate if success_rate is not None else current.success_rate
            ),
            confidence=confidence if confidence is not None else current.confidence,
        )
        # Clear cache since scoring changed
        self.cache.clear()
