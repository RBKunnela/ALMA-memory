---
task: Implement Connection Finder MCP Tool
agent: "@graph-integrator"
persona: Bridge
squad: alma-synthesis
phase: "2 — Intelligence"
version: 1.0.0
---

# Implement Connection Finder MCP Tool

## Goal

Build the `alma_find_connections` MCP tool that surfaces non-obvious links between memories by combining vector similarity search with graph relationship traversal. Given a memory (or a natural language query), the tool finds related memories through both embedding proximity and graph edge paths, then generates human-readable explanations of why those memories are connected.

## Agent

`@graph-integrator` (Bridge) — Graph + vector search integration specialist.

## Requires

- **integrate-graph-vector** (completed) — The `UnifiedRetriever` class in `alma/retrieval/unified.py` must be functional, providing combined vector + graph search.
- `alma/synthesis/types.py` exists with `UnifiedResult` and supporting types.
- `alma/synthesis/connections.py` or equivalent module for connection-specific logic.

## Steps

### Step 1: Use Unified Retrieval to Find Semantically Similar Memories

Leverage the `UnifiedRetriever` from the previous task:

```python
# For a given memory_id, get its content and embedding
source_memory = storage.get_memory(memory_id)

# Search for similar memories via vector similarity
vector_results = unified_retriever.search(
    query=source_memory.content,
    project_id=project_id,
    agent=agent,
    top_k=top_k * 2,  # Over-fetch to allow filtering
    include_graph=False,  # Vector-only for this phase
    min_similarity=min_similarity,
)
```

Filter out the source memory itself from results. Rank by vector similarity score.

### Step 2: Use Graph Traversal to Find Relationship Paths

For the source memory, traverse the graph to discover relationship-based connections:

```python
# Find the entity corresponding to this memory in the graph
entities = graph.search_entities(
    query=source_memory.content,
    embedding=source_memory.embedding,
    top_k=5,
)

# For each matched entity, traverse relationships
for entity in entities:
    relationships = graph.get_relationships(entity.id)
    # Multi-hop traversal
    paths = graph.traverse(
        start_entity_id=entity.id,
        max_hops=max_hops,
        relation_types=None,  # All types
    )
```

Build a `ConnectionPath` for each discovered route:
- Direct connections (1 hop): `Memory A --[relates_to]--> Memory B`
- Indirect connections (2+ hops): `Memory A --[mentions]--> Entity X --[referenced_in]--> Memory B`

### Step 3: Combine Both Signals to Surface Non-Obvious Connections

The most valuable connections are those that are:
- **Non-obvious**: Not immediately apparent from the content alone
- **Multi-signal**: Found via both vector similarity AND graph edges
- **Cross-domain**: Connect memories from different topics, time periods, or types

Scoring for connection strength:

```python
@dataclass
class Connection:
    """A discovered connection between two memories."""
    source_memory_id: str
    target_memory_id: str
    vector_similarity: float       # 0.0 to 1.0
    graph_path: Optional[List[str]]  # Entity path through graph
    graph_hops: int                 # Number of hops (0 if no graph path)
    connection_strength: float      # Combined score
    connection_type: str            # "semantic", "relational", "both"
    explanation: str                # Human-readable explanation
```

Connection strength formula:
```
# Connections found via BOTH paths are most valuable
if found_via_vector AND found_via_graph:
    strength = 0.5 * vector_sim + 0.3 * graph_weight + 0.2 * novelty_bonus
elif found_via_graph_only:
    strength = 0.7 * graph_weight + 0.3 * novelty_bonus
elif found_via_vector_only:
    strength = 0.7 * vector_sim + 0.3 * novelty_bonus

# Novelty bonus: penalize obvious connections (same day, same tags, same type)
novelty_bonus = 1.0 - overlap_score(source, target)
```

### Step 4: Generate Explanations of Why Memories Are Connected

For each discovered connection, generate a human-readable explanation:

**Template-based explanations** (no LLM required):

| Connection Type | Explanation Template |
|----------------|---------------------|
| Semantic + same topic | "Both memories discuss **{topic}**, captured {days_apart} days apart" |
| Semantic + different topic | "These memories share similar concepts despite being about different topics (**{topic_a}** and **{topic_b}**)" |
| Graph direct | "Connected via **{relationship_type}**: {source_summary} --> {target_summary}" |
| Graph indirect | "Linked through **{entity_name}** ({entity_type}): {path_description}" |
| Both signals | "Strong connection: semantically similar ({similarity}%) AND linked via **{entity_name}** in the knowledge graph" |
| Cross-domain | "Cross-domain insight: a pattern in **{domain_a}** may relate to **{domain_b}**" |

**LLM-enhanced explanations** (optional, if LLM provider is configured):
- Send source and target memory content to LLM with prompt: "Explain the connection between these two memories in 1-2 sentences"
- Cache explanations to avoid re-generating for known pairs

### Step 5: Implement as `alma_find_connections` MCP Tool

```python
# In alma/mcp/tools/ (synthesis tools)
def alma_find_connections(
    memory_id: Optional[str] = None,
    query: Optional[str] = None,
    project_id: str = "",
    agent: str = "",
    max_connections: int = 10,
    min_strength: float = 0.3,
    max_hops: int = 2,
    include_explanations: bool = True,
) -> dict:
    """Find non-obvious connections between memories.

    Provide either memory_id (find connections for a specific memory)
    or query (find connections for a natural language query).

    Returns:
        dict with:
        - connections: List of Connection objects with explanations
        - source: The source memory or query
        - stats: Summary statistics (total found, avg strength, connection types)
    """
```

Return format:
```json
{
  "source": {"id": "...", "content": "..."},
  "connections": [
    {
      "memory_id": "...",
      "content": "...",
      "strength": 0.85,
      "type": "both",
      "explanation": "Strong connection: semantically similar (82%) AND linked via 'API design' in the knowledge graph",
      "graph_path": ["memory_a", "api_design_entity", "memory_b"]
    }
  ],
  "stats": {
    "total_found": 7,
    "avg_strength": 0.62,
    "types": {"semantic": 3, "relational": 2, "both": 2}
  }
}
```

Register the tool in the MCP server with proper parameter schemas and descriptions.

### Step 6: Test with Diverse Memory Corpus

Build comprehensive tests:

- **Planted connections**: Create memories with known semantic overlaps and graph relationships; verify the tool finds them
- **Cross-domain connections**: Memories from different domains that share an entity; verify cross-domain discovery
- **No connections**: Isolated memory with no similar content or graph edges; verify empty result
- **Graph-only connections**: Memories linked via graph but with low vector similarity; verify graph path discovery
- **Vector-only connections**: Semantically similar memories with no graph edges; verify vector discovery
- **Explanation quality**: Verify explanations are meaningful and match the connection type
- **Performance**: Test with 100+ memories to verify reasonable response time

Use `alma.testing.MockStorage`, `alma.testing.MockEmbedder`, and `alma.graph.backends.memory.InMemoryGraphBackend` for all tests.

## Output

| Artifact | Path | Description |
|----------|------|-------------|
| Connection finder | `alma/synthesis/connections.py` | `ConnectionFinder` class |
| Types (additions) | `alma/synthesis/types.py` | `Connection`, `ConnectionResult`, `ConnectionPath` dataclasses |
| MCP tool | `alma/mcp/tools.py` or `alma/mcp/tools/synthesis.py` | `alma_find_connections` function |
| Unit tests | `tests/unit/test_connection_finder.py` | Tests for connection finding |

## Gate

- [ ] `ConnectionFinder.find_connections()` discovers planted connections in test data
- [ ] Connections found via both vector and graph paths are ranked highest
- [ ] Non-obvious cross-domain connections are surfaced (different topics linked through shared entities)
- [ ] Each connection includes a meaningful human-readable explanation
- [ ] Template-based explanations work without LLM provider
- [ ] `alma_find_connections` MCP tool is registered, callable, and returns structured results
- [ ] Tool accepts both `memory_id` and `query` as input modes
- [ ] Graceful degradation when graph backend is not configured (vector-only connections)
- [ ] Empty result returned (not error) when no connections found
- [ ] All unit tests pass with >80% coverage on new code
- [ ] Code passes `ruff check` and `ruff format --check`
- [ ] Type hints on all public APIs, Google-style docstrings

---

*Task: Implement Connection Finder — alma-synthesis squad v1.0.0*
