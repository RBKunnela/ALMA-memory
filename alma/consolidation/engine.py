"""
ALMA Memory Consolidation Engine.

Implements LLM-powered deduplication that merges similar memories,
inspired by Mem0's core innovation.
"""

import asyncio
import json
import logging
import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from alma.consolidation.prompts import (
    MERGE_ANTI_PATTERNS_PROMPT,
    MERGE_DOMAIN_KNOWLEDGE_PROMPT,
    MERGE_HEURISTICS_PROMPT,
)
from alma.retrieval.embeddings import EmbeddingProvider, LocalEmbedder
from alma.storage.base import StorageBackend
from alma.types import (
    AntiPattern,
    DomainKnowledge,
    Heuristic,
)

logger = logging.getLogger(__name__)


@dataclass
class ConsolidationResult:
    """Result of a consolidation operation."""

    merged_count: int  # Number of memories that were merged (deleted and replaced)
    groups_found: int  # Number of similar memory groups identified
    memories_processed: int  # Total memories analyzed
    errors: List[str] = field(default_factory=list)

    # Detailed merge information
    merge_details: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def success(self) -> bool:
        """Check if consolidation completed without critical errors."""
        return len(self.errors) == 0 or self.merged_count > 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "merged_count": self.merged_count,
            "groups_found": self.groups_found,
            "memories_processed": self.memories_processed,
            "errors": self.errors,
            "merge_details": self.merge_details,
            "success": self.success,
        }


class ConsolidationEngine:
    """
    Memory consolidation engine for deduplicating and merging similar memories.

    Key features:
    - Cosine similarity-based grouping
    - Optional LLM-powered intelligent merging
    - Provenance tracking (merged_from metadata)
    - Dry-run mode for safety
    """

    def __init__(
        self,
        storage: StorageBackend,
        embedder: Optional[EmbeddingProvider] = None,
        llm_client: Optional[Any] = None,
    ):
        """
        Initialize the consolidation engine.

        Args:
            storage: Storage backend for memory operations
            embedder: Embedding provider (defaults to LocalEmbedder)
            llm_client: Optional LLM client for intelligent merging
                       Should have a method `complete(prompt: str) -> str`
        """
        self.storage = storage
        self.embedder = embedder or LocalEmbedder()
        self.llm_client = llm_client

    async def consolidate(
        self,
        agent: str,
        project_id: str,
        memory_type: str = "heuristics",
        similarity_threshold: float = 0.85,
        use_llm: bool = True,
        dry_run: bool = False,
    ) -> ConsolidationResult:
        """
        Merge similar memories to reduce redundancy.

        Algorithm:
        1. Get all memories for agent of the specified type
        2. Compute embeddings if not present
        3. Cluster by similarity (pairwise comparison)
        4. For each cluster > 1 memory:
           - If use_llm: Generate merged summary via LLM
           - Else: Keep highest confidence/most recent
        5. Delete originals, save merged (unless dry_run)

        Args:
            agent: Agent name whose memories to consolidate
            project_id: Project ID
            memory_type: Type of memory to consolidate
                        ("heuristics", "outcomes", "domain_knowledge", "anti_patterns")
            similarity_threshold: Minimum cosine similarity to group (0.0 to 1.0)
            use_llm: Whether to use LLM for intelligent merging
            dry_run: If True, don't actually modify storage

        Returns:
            ConsolidationResult with details of the operation
        """
        result = ConsolidationResult(
            merged_count=0,
            groups_found=0,
            memories_processed=0,
            errors=[],
            merge_details=[],
        )

        try:
            # 1. Get memories based on type
            memories = self._get_memories_by_type(
                agent=agent,
                project_id=project_id,
                memory_type=memory_type,
            )

            result.memories_processed = len(memories)

            if len(memories) < 2:
                logger.info(f"Not enough memories to consolidate: {len(memories)}")
                return result

            # 2. Ensure embeddings are present
            memories = self._ensure_embeddings(memories, memory_type)

            # 3. Find similar groups
            groups = self._find_similar_groups(memories, similarity_threshold)
            result.groups_found = len([g for g in groups if len(g) > 1])

            # 4. Process each group
            for group in groups:
                if len(group) <= 1:
                    continue  # Skip singletons

                try:
                    # Merge the group
                    merged, original_ids = await self._merge_group(
                        group=group,
                        memory_type=memory_type,
                        use_llm=use_llm,
                        project_id=project_id,
                        agent=agent,
                    )

                    if not dry_run:
                        # Save merged memory
                        self._save_merged(merged, memory_type)

                        # Delete originals
                        for original_id in original_ids:
                            self._delete_memory(original_id, memory_type)

                    result.merged_count += len(original_ids) - 1  # N merged into 1
                    result.merge_details.append({
                        "merged_from": original_ids,
                        "merged_into": merged.id if hasattr(merged, 'id') else str(merged),
                        "count": len(original_ids),
                    })

                except Exception as e:
                    error_msg = f"Failed to merge group: {str(e)}"
                    logger.exception(error_msg)
                    result.errors.append(error_msg)

        except Exception as e:
            error_msg = f"Consolidation failed: {str(e)}"
            logger.exception(error_msg)
            result.errors.append(error_msg)

        return result

    def _get_memories_by_type(
        self,
        agent: str,
        project_id: str,
        memory_type: str,
    ) -> List[Any]:
        """Get all memories of a specific type for an agent."""
        if memory_type == "heuristics":
            return self.storage.get_heuristics(
                project_id=project_id,
                agent=agent,
                top_k=1000,  # Get all
            )
        elif memory_type == "outcomes":
            return self.storage.get_outcomes(
                project_id=project_id,
                agent=agent,
                top_k=1000,
            )
        elif memory_type == "domain_knowledge":
            return self.storage.get_domain_knowledge(
                project_id=project_id,
                agent=agent,
                top_k=1000,
            )
        elif memory_type == "anti_patterns":
            return self.storage.get_anti_patterns(
                project_id=project_id,
                agent=agent,
                top_k=1000,
            )
        else:
            raise ValueError(f"Unknown memory type: {memory_type}")

    def _ensure_embeddings(
        self,
        memories: List[Any],
        memory_type: str,
    ) -> List[Any]:
        """Ensure all memories have embeddings, computing if needed."""
        needs_embedding = []
        needs_embedding_indices = []

        for i, memory in enumerate(memories):
            if not hasattr(memory, 'embedding') or memory.embedding is None:
                needs_embedding.append(self._get_embedding_text(memory, memory_type))
                needs_embedding_indices.append(i)

        if needs_embedding:
            logger.info(f"Computing embeddings for {len(needs_embedding)} memories")
            embeddings = self.embedder.encode_batch(needs_embedding)

            for i, embedding in zip(needs_embedding_indices, embeddings, strict=False):
                memories[i].embedding = embedding

        return memories

    def _get_embedding_text(self, memory: Any, memory_type: str) -> str:
        """Get the text to embed for a memory."""
        if memory_type == "heuristics":
            return f"{memory.condition} {memory.strategy}"
        elif memory_type == "outcomes":
            return f"{memory.task_description} {memory.strategy_used}"
        elif memory_type == "domain_knowledge":
            return f"{memory.domain} {memory.fact}"
        elif memory_type == "anti_patterns":
            return f"{memory.pattern} {memory.why_bad} {memory.better_alternative}"
        else:
            return str(memory)

    def _find_similar_groups(
        self,
        memories: List[Any],
        threshold: float,
    ) -> List[List[Any]]:
        """
        Group memories by embedding similarity using union-find.

        Args:
            memories: List of memories with embeddings
            threshold: Minimum cosine similarity to group

        Returns:
            List of groups (each group is a list of memories)
        """
        n = len(memories)
        if n == 0:
            return []

        # Union-Find data structure
        parent = list(range(n))
        rank = [0] * n

        def find(x: int) -> int:
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x: int, y: int) -> None:
            px, py = find(x), find(y)
            if px == py:
                return
            if rank[px] < rank[py]:
                px, py = py, px
            parent[py] = px
            if rank[px] == rank[py]:
                rank[px] += 1

        # Compare all pairs
        for i in range(n):
            for j in range(i + 1, n):
                emb1 = memories[i].embedding
                emb2 = memories[j].embedding

                if emb1 is not None and emb2 is not None:
                    similarity = self._compute_similarity(emb1, emb2)
                    if similarity >= threshold:
                        union(i, j)

        # Build groups
        groups_dict: Dict[int, List[Any]] = {}
        for i in range(n):
            root = find(i)
            if root not in groups_dict:
                groups_dict[root] = []
            groups_dict[root].append(memories[i])

        return list(groups_dict.values())

    def _compute_similarity(
        self,
        emb1: List[float],
        emb2: List[float],
    ) -> float:
        """
        Compute cosine similarity between two embeddings.

        Args:
            emb1: First embedding vector
            emb2: Second embedding vector

        Returns:
            Cosine similarity (0.0 to 1.0)
        """
        if len(emb1) != len(emb2):
            return 0.0

        dot_product = sum(a * b for a, b in zip(emb1, emb2, strict=False))
        norm1 = math.sqrt(sum(a * a for a in emb1))
        norm2 = math.sqrt(sum(b * b for b in emb2))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    async def _merge_group(
        self,
        group: List[Any],
        memory_type: str,
        use_llm: bool,
        project_id: str,
        agent: str,
    ) -> Tuple[Any, List[str]]:
        """
        Merge a group of similar memories into one.

        Args:
            group: List of similar memories
            memory_type: Type of memories
            use_llm: Whether to use LLM for intelligent merging
            project_id: Project ID
            agent: Agent name

        Returns:
            Tuple of (merged_memory, list_of_original_ids)
        """
        original_ids = [m.id for m in group]

        if memory_type == "heuristics":
            merged = await self._merge_heuristics(group, use_llm, project_id, agent)
        elif memory_type == "domain_knowledge":
            merged = await self._merge_domain_knowledge(group, use_llm, project_id, agent)
        elif memory_type == "anti_patterns":
            merged = await self._merge_anti_patterns(group, use_llm, project_id, agent)
        elif memory_type == "outcomes":
            # Outcomes typically aren't merged - just keep the most recent
            merged = self._keep_most_recent(group)
        else:
            raise ValueError(f"Unknown memory type: {memory_type}")

        return merged, original_ids

    async def _merge_heuristics(
        self,
        group: List[Heuristic],
        use_llm: bool,
        project_id: str,
        agent: str,
    ) -> Heuristic:
        """
        Merge a group of similar heuristics into one.

        Args:
            group: List of similar heuristics
            use_llm: Whether to use LLM for intelligent merging
            project_id: Project ID
            agent: Agent name

        Returns:
            Merged heuristic
        """
        # Collect metadata
        original_ids = [h.id for h in group]
        total_occurrences = sum(h.occurrence_count for h in group)
        total_successes = sum(h.success_count for h in group)
        avg_confidence = sum(h.confidence for h in group) / len(group)

        # Find the highest confidence heuristic as base
        base = max(group, key=lambda h: h.confidence)

        if use_llm and self.llm_client:
            # Use LLM for intelligent merging
            try:
                merged_data = await self._llm_merge_heuristics(group)
                condition = merged_data.get("condition", base.condition)
                strategy = merged_data.get("strategy", base.strategy)
                confidence = merged_data.get("confidence", avg_confidence)
            except Exception as e:
                logger.warning(f"LLM merge failed, using base: {e}")
                condition = base.condition
                strategy = base.strategy
                confidence = avg_confidence
        else:
            # Without LLM, use the highest confidence heuristic
            condition = base.condition
            strategy = base.strategy
            confidence = avg_confidence

        # Create embedding for merged heuristic
        embedding_text = f"{condition} {strategy}"
        embedding = self.embedder.encode(embedding_text)

        now = datetime.now(timezone.utc)

        return Heuristic(
            id=f"heuristic_{uuid.uuid4().hex[:12]}",
            agent=agent,
            project_id=project_id,
            condition=condition,
            strategy=strategy,
            confidence=min(confidence, 1.0),
            occurrence_count=total_occurrences,
            success_count=total_successes,
            last_validated=now,
            created_at=now,
            embedding=embedding,
            metadata={
                "merged_from": original_ids,
                "merge_timestamp": now.isoformat(),
                "original_count": len(group),
            },
        )

    async def _merge_domain_knowledge(
        self,
        group: List[DomainKnowledge],
        use_llm: bool,
        project_id: str,
        agent: str,
    ) -> DomainKnowledge:
        """Merge a group of similar domain knowledge items."""
        original_ids = [dk.id for dk in group]
        avg_confidence = sum(dk.confidence for dk in group) / len(group)
        base = max(group, key=lambda dk: dk.confidence)

        if use_llm and self.llm_client:
            try:
                merged_data = await self._llm_merge_domain_knowledge(group)
                fact = merged_data.get("fact", base.fact)
                confidence = merged_data.get("confidence", avg_confidence)
            except Exception as e:
                logger.warning(f"LLM merge failed, using base: {e}")
                fact = base.fact
                confidence = avg_confidence
        else:
            fact = base.fact
            confidence = avg_confidence

        embedding_text = f"{base.domain} {fact}"
        embedding = self.embedder.encode(embedding_text)

        now = datetime.now(timezone.utc)

        return DomainKnowledge(
            id=f"dk_{uuid.uuid4().hex[:12]}",
            agent=agent,
            project_id=project_id,
            domain=base.domain,
            fact=fact,
            source="consolidation",
            confidence=min(confidence, 1.0),
            last_verified=now,
            embedding=embedding,
            metadata={
                "merged_from": original_ids,
                "merge_timestamp": now.isoformat(),
                "original_count": len(group),
            },
        )

    async def _merge_anti_patterns(
        self,
        group: List[AntiPattern],
        use_llm: bool,
        project_id: str,
        agent: str,
    ) -> AntiPattern:
        """Merge a group of similar anti-patterns."""
        original_ids = [ap.id for ap in group]
        total_occurrences = sum(ap.occurrence_count for ap in group)
        base = max(group, key=lambda ap: ap.occurrence_count)

        if use_llm and self.llm_client:
            try:
                merged_data = await self._llm_merge_anti_patterns(group)
                pattern = merged_data.get("pattern", base.pattern)
                why_bad = merged_data.get("why_bad", base.why_bad)
                better_alternative = merged_data.get("better_alternative", base.better_alternative)
            except Exception as e:
                logger.warning(f"LLM merge failed, using base: {e}")
                pattern = base.pattern
                why_bad = base.why_bad
                better_alternative = base.better_alternative
        else:
            pattern = base.pattern
            why_bad = base.why_bad
            better_alternative = base.better_alternative

        embedding_text = f"{pattern} {why_bad} {better_alternative}"
        embedding = self.embedder.encode(embedding_text)

        now = datetime.now(timezone.utc)

        return AntiPattern(
            id=f"ap_{uuid.uuid4().hex[:12]}",
            agent=agent,
            project_id=project_id,
            pattern=pattern,
            why_bad=why_bad,
            better_alternative=better_alternative,
            occurrence_count=total_occurrences,
            last_seen=now,
            created_at=now,
            embedding=embedding,
            metadata={
                "merged_from": original_ids,
                "merge_timestamp": now.isoformat(),
                "original_count": len(group),
            },
        )

    def _keep_most_recent(self, group: List[Any]) -> Any:
        """Keep the most recent memory from a group."""
        return max(group, key=lambda m: getattr(m, 'timestamp', getattr(m, 'created_at', datetime.min)))

    async def _llm_merge_heuristics(self, group: List[Heuristic]) -> Dict[str, Any]:
        """Use LLM to intelligently merge heuristics."""
        heuristics_text = "\n\n".join([
            f"Heuristic {i+1}:\n"
            f"  Condition: {h.condition}\n"
            f"  Strategy: {h.strategy}\n"
            f"  Confidence: {h.confidence:.2f}\n"
            f"  Occurrences: {h.occurrence_count}"
            for i, h in enumerate(group)
        ])

        prompt = MERGE_HEURISTICS_PROMPT.format(heuristics=heuristics_text)

        response = await self._call_llm(prompt)
        return json.loads(response)

    async def _llm_merge_domain_knowledge(self, group: List[DomainKnowledge]) -> Dict[str, Any]:
        """Use LLM to intelligently merge domain knowledge."""
        knowledge_text = "\n\n".join([
            f"Knowledge {i+1}:\n"
            f"  Domain: {dk.domain}\n"
            f"  Fact: {dk.fact}\n"
            f"  Confidence: {dk.confidence:.2f}"
            for i, dk in enumerate(group)
        ])

        prompt = MERGE_DOMAIN_KNOWLEDGE_PROMPT.format(knowledge_items=knowledge_text)

        response = await self._call_llm(prompt)
        return json.loads(response)

    async def _llm_merge_anti_patterns(self, group: List[AntiPattern]) -> Dict[str, Any]:
        """Use LLM to intelligently merge anti-patterns."""
        patterns_text = "\n\n".join([
            f"Anti-Pattern {i+1}:\n"
            f"  Pattern: {ap.pattern}\n"
            f"  Why Bad: {ap.why_bad}\n"
            f"  Alternative: {ap.better_alternative}\n"
            f"  Occurrences: {ap.occurrence_count}"
            for i, ap in enumerate(group)
        ])

        prompt = MERGE_ANTI_PATTERNS_PROMPT.format(anti_patterns=patterns_text)

        response = await self._call_llm(prompt)
        return json.loads(response)

    async def _call_llm(self, prompt: str) -> str:
        """Call the LLM client."""
        if not self.llm_client:
            raise ValueError("LLM client not configured")

        # Support different LLM client interfaces
        if hasattr(self.llm_client, 'complete'):
            result = self.llm_client.complete(prompt)
            if asyncio.iscoroutine(result):
                return await result
            return result
        elif hasattr(self.llm_client, 'chat'):
            result = self.llm_client.chat([{"role": "user", "content": prompt}])
            if asyncio.iscoroutine(result):
                return await result
            return result
        elif callable(self.llm_client):
            result = self.llm_client(prompt)
            if asyncio.iscoroutine(result):
                return await result
            return result
        else:
            raise ValueError("LLM client must have 'complete', 'chat', or '__call__' method")

    def _save_merged(self, memory: Any, memory_type: str) -> None:
        """Save a merged memory to storage."""
        if memory_type == "heuristics":
            self.storage.save_heuristic(memory)
        elif memory_type == "domain_knowledge":
            self.storage.save_domain_knowledge(memory)
        elif memory_type == "anti_patterns":
            self.storage.save_anti_pattern(memory)
        elif memory_type == "outcomes":
            self.storage.save_outcome(memory)

    def _delete_memory(self, memory_id: str, memory_type: str) -> None:
        """Delete a memory from storage."""
        if memory_type == "heuristics":
            self.storage.delete_heuristic(memory_id)
        elif memory_type == "domain_knowledge":
            self.storage.delete_domain_knowledge(memory_id)
        elif memory_type == "anti_patterns":
            self.storage.delete_anti_pattern(memory_id)
        elif memory_type == "outcomes":
            self.storage.delete_outcome(memory_id)
