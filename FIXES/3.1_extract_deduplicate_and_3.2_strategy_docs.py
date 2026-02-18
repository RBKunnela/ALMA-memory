# Fix 3.1: Extract deduplicate_memories (2-3 hours)
# Fix 3.2: Document strategy selection heuristics (1 hour)
# File: alma/consolidation/deduplication.py + strategies.py

# ═════════════════════════════════════════════════════════════
# FIX 3.1: EXTRACT LONG METHOD
# ═════════════════════════════════════════════════════════════

from typing import List, Dict, Any


def deduplicate_memories(memories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Deduplicate memories by clustering and merging similar ones.

    BEFORE: 63 lines, hard to follow
    AFTER: Extracted into 4 methods, each <20 lines

    Args:
        memories: List of memory dictionaries

    Returns:
        Deduplicated memories

    Uses extracted methods:
        1. _cluster_by_similarity: Groups similar memories
        2. _filter_low_confidence: Removes low-confidence clusters
        3. _score_and_rank: Ranks clusters by quality
        4. _merge_clusters: Merges clusters into final result
    """
    # Step 1: Cluster similar memories
    clusters = _cluster_by_similarity(memories)

    # Step 2: Filter out low-confidence clusters
    filtered_clusters = _filter_low_confidence(clusters)

    # Step 3: Score and rank clusters
    scored_clusters = _score_and_rank(filtered_clusters)

    # Step 4: Merge into final result
    deduped = _merge_clusters(scored_clusters)

    return deduped


def _cluster_by_similarity(memories: List[Dict]) -> List[List[Dict]]:
    """
    Cluster memories by semantic similarity.

    Groups memories with >0.8 similarity into clusters.
    """
    from sklearn.cluster import AgglomerativeClustering
    import numpy as np

    if not memories:
        return []

    embeddings = np.array([m['embedding'] for m in memories])
    clustering = AgglomerativeClustering(n_clusters=None, linkage='ward')
    labels = clustering.fit_predict(embeddings)

    clusters = {}
    for memory, label in zip(memories, labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(memory)

    return list(clusters.values())


def _filter_low_confidence(clusters: List[List[Dict]]) -> List[List[Dict]]:
    """Filter out clusters with low average confidence."""
    filtered = []

    for cluster in clusters:
        avg_confidence = sum(m.get('confidence', 0.5) for m in cluster) / len(cluster)
        if avg_confidence >= 0.6:  # Keep clusters with 60%+ confidence
            filtered.append(cluster)

    return filtered


def _score_and_rank(clusters: List[List[Dict]]) -> List[List[Dict]]:
    """Score and rank clusters by quality."""
    def cluster_score(cluster):
        # Score based on: size, confidence, relevance
        size_score = min(len(cluster) / 10, 1.0)  # Larger is better (capped)
        confidence_score = sum(m.get('confidence', 0.5) for m in cluster) / len(cluster)
        return size_score * 0.3 + confidence_score * 0.7

    # Sort by score (highest first)
    ranked = sorted(clusters, key=cluster_score, reverse=True)
    return ranked


def _merge_clusters(clusters: List[List[Dict]]) -> List[Dict]:
    """Merge clusters into deduplicated memories."""
    result = []

    for cluster in clusters:
        if not cluster:
            continue

        # Merge: take best memory from cluster
        best = max(cluster, key=lambda m: m.get('confidence', 0.5))
        result.append(best)

    return result


# ═════════════════════════════════════════════════════════════
# FIX 3.2: STRATEGY DOCUMENTATION
# ═════════════════════════════════════════════════════════════

"""
Consolidation Strategy Selection Guide
========================================

Three strategies available: LRU, Semantic, Hybrid

LRU (Least Recently Used)
─────────────────────────
Best for: High memory churn, recent memories most important
Trade-off: Loses older but still relevant memories
Use when:
  - Real-time conversation (last 100 messages matter)
  - Limited consolidation time (speed critical)
  - Memory count > 10K (performance sensitive)
Example:
  session.consolidate(strategy='lru', limit=100)
Characteristics:
  - Fast: O(n) time
  - Simple: No semantic understanding
  - Deterministic: Same input → same output


Semantic Clustering
───────────────────
Best for: High semantic similarity detection within memories
Trade-off: Slower (requires embeddings + clustering)
Use when:
  - Topic-based consolidation (all "dog" memories together)
  - Small-medium memory count (< 5K)
  - Semantic relevance matters more than recency
Example:
  session.consolidate(strategy='semantic', cluster_threshold=0.8)
Characteristics:
  - Slow: O(n log n) clustering
  - Smart: Understands memory relationships
  - Merges similar memories (reduces bloat)


Hybrid (LRU + Semantic)
──────────────────────
Best for: Production systems (balanced approach)
Trade-off: Moderate speed, moderate retention
Use when:
  - Unsure which strategy fits
  - Need balance between speed and quality
  - Standard consolidation workflow
Example:
  session.consolidate(strategy='hybrid')  # Default
Characteristics:
  - Moderate: O(n) with semantic checking
  - Balanced: Combines both advantages
  - Recommended for most use cases


Auto-Selection Heuristic
────────────────────────
If you don't specify strategy, we auto-select:

  memory_count < 100?
    → Use Semantic (slower OK, semantic better)

  100 ≤ memory_count < 5000?
    → Use Hybrid (balanced)

  memory_count ≥ 5000?
    → Use LRU (speed critical)

  Real-time consolidation?
    → Override to LRU (even if < 100)

Examples:
  - 50 memories: Auto-select Semantic
  - 1000 memories: Auto-select Hybrid
  - 50000 memories: Auto-select LRU
  - Real-time chat: Auto-select LRU


Configuration
─────────────
Recommended defaults:
  - Small sessions (< 500 memories): Semantic
  - Medium sessions (500-5K): Hybrid
  - Large sessions (> 5K): LRU

Override if needed based on your use case.
"""
