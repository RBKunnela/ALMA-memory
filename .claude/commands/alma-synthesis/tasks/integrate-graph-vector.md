---
task: Integrate Graph Traversal with Vector Search
agent: "@graph-integrator"
persona: Bridge
squad: alma-synthesis
phase: "2 — Intelligence"
version: 1.0.0
---

# Integrate Graph Traversal with Vector Search

## Goal

Build a unified retrieval module that combines ALMA's vector search engine (`alma/retrieval/engine.py`) with its graph backends (`alma/graph/`) into a single hybrid query interface. Graph edges should enhance relevance scoring so that memories connected via relationships rank higher than isolated matches. The module must work with all 4 graph backends (Neo4j, Memgraph, Kuzu, In-Memory) and degrade gracefully when no graph backend is configured.

## Agent

`@graph-integrator` (Bridge) — Graph + vector search integration specialist.

## Requires

- **build-weekly-review** (completed) — The synthesis package structure (`alma/synthesis/`) and types (`alma/synthesis/types.py`) must exist.
- Existing ALMA graph backends operational (`alma/graph/base.py` ABC with `GraphBackend`).
- Existing ALMA retrieval engine operational (`alma/retrieval/engine.py` with `RetrievalEngine`).
- Existing ALMA scoring module (`alma/retrieval/scoring.py` with `MemoryScorer`).

## Steps

### Step 1: Analyze ALMA's 4 Graph Backends

Study the existing graph infrastructure to understand available APIs:

| Backend | Path | Key Features |
|---------|------|--------------|
| Neo4j | `alma/graph/backends/neo4j.py` | Full Cypher queries, production-grade |
| Memgraph | `alma/graph/backends/memgraph.py` | Cypher-compatible, in-memory optimized |
| Kuzu | `alma/graph/backends/kuzu.py` | Embedded graph, no server needed |
| In-Memory | `alma/graph/backends/memory.py` | Dictionary-based, perfect for testing |

Key APIs from `alma/graph/base.py` (`GraphBackend` ABC):
- `search_entities(query, embedding, top_k)` — Find entities matching a query
- `get_relationships(entity_id)` — Get all edges for an entity
- `traverse(start_entity_id, max_hops, relation_types)` — Multi-hop traversal

Key types from `alma/graph/store.py`:
- `Entity` — graph node with id, type, properties
- `Relationship` — graph edge with source, target, type, weight
- `GraphQuery` — structured query with max_hops, relation_type filters

### Step 2: Analyze ALMA's Hybrid Search Engine

Study the existing retrieval infrastructure:

| Module | Path | Key Features |
|--------|------|--------------|
| Retrieval engine | `alma/retrieval/engine.py` | `RetrievalEngine.retrieve()` — vector search with scoring |
| Scoring | `alma/retrieval/scoring.py` | `MemoryScorer.score()` — similarity + recency + success + confidence |
| Cache | `alma/retrieval/cache.py` | Query result caching |

The current scoring formula in `MemoryScorer`:
```
final_score = (similarity * w_sim) + (recency * w_rec) + (success * w_suc) + (confidence * w_conf)
```

### Step 3: Design Unified Query Architecture

Design a query that combines both signals:

1. **Vector search phase**: Use `RetrievalEngine.retrieve()` to get top-K results by embedding similarity
2. **Entity resolution phase**: For each vector result, find matching entities in the graph via `GraphBackend.search_entities()`
3. **Graph expansion phase**: For matched entities, traverse relationships using `GraphBackend.get_relationships()` up to `max_hops` (default: 2)
4. **Result enrichment phase**: Attach related entities and relationships to each search result
5. **Re-scoring phase**: Combine vector similarity score with graph relationship weight:

```
unified_score = (vector_score * w_vector) + (graph_score * w_graph) + (recency * w_recency)
```

Where:
- `vector_score` = original retrieval engine score
- `graph_score` = function of (edge count, edge weights, hop distance)
- `w_vector` default = 0.6, `w_graph` default = 0.25, `w_recency` default = 0.15

6. **Deduplication phase**: Merge results that appear via both vector and graph paths (keep highest score)

### Step 4: Build `alma/retrieval/unified.py`

Implement the `UnifiedRetriever` class:

```python
# alma/retrieval/unified.py
from alma.graph.base import GraphBackend
from alma.retrieval.engine import RetrievalEngine
from alma.retrieval.scoring import MemoryScorer
from alma.storage.base import StorageBackend

class UnifiedRetriever:
    """Combines vector search with graph traversal for enriched retrieval."""

    def __init__(
        self,
        storage: StorageBackend,
        retrieval_engine: RetrievalEngine,
        graph: Optional[GraphBackend] = None,
        config: Optional[UnifiedRetrievalConfig] = None,
    ): ...

    def search(
        self,
        query: str,
        project_id: str,
        agent: str,
        top_k: int = 10,
        include_graph: bool = True,
        max_hops: int = 2,
        min_similarity: float = 0.5,
    ) -> List[UnifiedResult]: ...

    async def async_search(
        self,
        query: str,
        project_id: str,
        agent: str,
        top_k: int = 10,
        include_graph: bool = True,
        max_hops: int = 2,
        min_similarity: float = 0.5,
    ) -> List[UnifiedResult]: ...
```

Supporting types:

```python
@dataclass
class UnifiedResult:
    """A search result combining vector and graph signals."""
    memory_id: str
    memory_type: str
    content: str
    vector_score: float
    graph_score: float
    unified_score: float
    related_entities: List[Entity]
    relationships: List[Relationship]
    retrieval_path: str  # "vector", "graph", or "both"
```

### Step 5: Graph Edges Enhance Relevance Scoring

Implement the graph scoring function:

```python
def compute_graph_score(
    entity: Entity,
    relationships: List[Relationship],
    max_hops: int,
) -> float:
    """Score based on graph connectivity.

    Factors:
    - edge_count: More connections = higher score
    - edge_weights: Weighted relationships score higher
    - hop_distance: Closer connections (1 hop) score higher than distant (2+ hops)
    - relationship_diversity: Connections of different types score higher
    """
```

Scoring formula:
- `edge_score = sum(edge.weight / (hop_distance ** decay_factor) for edge in edges)`
- `diversity_bonus = len(unique_relation_types) / len(edges)` (if > 1 type)
- `graph_score = normalize(edge_score * (1 + diversity_bonus), 0.0, 1.0)`

### Step 6: Works with All 4 Graph Backends

Ensure the unified retriever works identically with all backends:

- Use only `GraphBackend` ABC methods — never backend-specific APIs
- Test with `InMemoryGraphBackend` in unit tests
- Integration tests should cover at least Neo4j and Kuzu if available
- Graceful degradation: if graph backend is `None`, fall back to pure vector search
- Graceful degradation: if graph traversal times out, return vector-only results with a warning
- Log all degradation events via ALMA's observability module (`alma/observability/logging`)

## Output

| Artifact | Path | Description |
|----------|------|-------------|
| Unified retrieval | `alma/retrieval/unified.py` | `UnifiedRetriever` class |
| Config | `alma/retrieval/unified_config.py` or added to existing config | `UnifiedRetrievalConfig` dataclass |
| Types (additions) | `alma/synthesis/types.py` or `alma/retrieval/unified.py` | `UnifiedResult` dataclass |
| Unit tests | `tests/unit/test_unified_retrieval.py` | Tests for unified retrieval |

## Gate

- [ ] `UnifiedRetriever.search()` returns results from vector search
- [ ] When graph backend is configured, results include graph-expanded entities and relationships
- [ ] Graph edges measurably improve retrieval relevance: connected memories score higher than isolated ones with equal vector similarity
- [ ] Works with `InMemoryGraphBackend` (unit tests)
- [ ] Graceful degradation when `graph=None` (falls back to pure vector search)
- [ ] Graceful degradation when graph traversal returns no results (vector-only results returned)
- [ ] Deduplication merges results found via both vector and graph paths
- [ ] Uses only `GraphBackend` ABC methods (no backend-specific code)
- [ ] All unit tests pass with >80% coverage on new code
- [ ] Code passes `ruff check` and `ruff format --check`
- [ ] Type hints on all public APIs, Google-style docstrings

---

*Task: Integrate Graph + Vector Search — alma-synthesis squad v1.0.0*
