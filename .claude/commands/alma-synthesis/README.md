# ALMA Synthesis Squad

> Builds the intelligence and review layer for ALMA-memory -- weekly reviews, pattern detection, connection surfacing, and knowledge graph integration.

## Purpose

ALMA stores memories (heuristics, outcomes, domain knowledge, anti-patterns, user preferences) across 7 storage backends and 4 graph backends. But raw storage is only half the value. The Synthesis Squad builds the **review and intelligence engine** that turns stored memories into actionable insights:

- **Weekly Reviews** -- Batch-retrieve recent memories, cluster by semantic similarity, summarize per cluster, detect open loops, find gaps.
- **Pattern Detection** -- Cross-session recurring theme analysis, topic evolution tracking, anomaly detection.
- **Graph-Vector Integration** -- Unified retrieval that traverses both embeddings (vector search) and relationships (graph traversal) simultaneously.
- **Connection Finding** -- Given any memory, discover non-obvious connections via combined vector similarity and graph edges.

## Agents

| Agent | Persona | Role |
|-------|---------|------|
| `@synthesis-chief` | Synth | Squad leader. Orchestrates reviews, delegates tasks, tracks progress. |
| `@review-engine-dev` | Recap | Builds the weekly review synthesis engine. Clustering, summarization, open loop detection. |
| `@graph-integrator` | Bridge | Connects vector search with graph backends for unified hybrid retrieval. |
| `@pattern-detector` | Pulse | Builds recurring theme detection, topic evolution, anomaly detection. |

## Structure

```
alma-synthesis/
  README.md                              # This file
  agents/
    synthesis-chief.md                   # Squad leader
    review-engine-dev.md                 # Weekly review engine builder
    graph-integrator.md                  # Graph + vector integration
    pattern-detector.md                  # Pattern and anomaly detection
  tasks/
    build-weekly-review.md               # Implement weekly review synthesis
    implement-pattern-detection.md       # Build pattern detection system
    integrate-graph-vector.md            # Unify graph traversal + vector search
    implement-connection-finder.md       # Build connection finder MCP tool
  checklists/
    synthesis-quality.md                 # Quality gates for all synthesis features
  data/
    synthesis-kb.md                      # Knowledge base and reference material
  workflows/
    weekly-review-pipeline.yaml          # Automated weekly review pipeline
```

## ALMA Infrastructure This Squad Builds On

| Module | Path | What It Provides |
|--------|------|------------------|
| Storage backends | `alma/storage/` | 7 backends (SQLite+FAISS, PostgreSQL+pgvector, Azure Cosmos, Qdrant, Chroma, Pinecone, File) |
| Graph backends | `alma/graph/` | 4 backends (Neo4j, Memgraph, Kuzu, In-Memory) with Entity/Relationship types |
| Retrieval engine | `alma/retrieval/engine.py` | Semantic search, scoring (similarity + recency + success + confidence) |
| Consolidation | `alma/consolidation/` | LLM-powered deduplication and merging |
| Compression | `alma/compression/` | LLM + rule-based memory compression |
| MCP tools | `alma/mcp/tools/` | Existing tools: retrieval, learning, workflow, admin |
| Events | `alma/events/` | Pub/sub emitter + webhooks |
| Types | `alma/types.py` | Heuristic, Outcome, DomainKnowledge, AntiPattern, UserPreference |

## Workflows

### Weekly Review Pipeline

Triggered weekly (or on-demand). Steps:

1. Batch retrieve last 7 days of memories
2. Cluster by semantic similarity
3. Generate LLM summary per cluster
4. Detect open loops (actions without completion)
5. Find cross-cluster connections
6. Identify knowledge gaps
7. Store synthesis result as a new memory
8. Emit event via `alma.events`

See `workflows/weekly-review-pipeline.yaml` for the full definition.

## Development Standards

- Python 3.10+, ruff formatting, Google-style docstrings
- All public APIs need type hints
- Tests use `alma.testing.MockStorage` and `alma.testing.MockEmbedder`
- New modules go under `alma/synthesis/` package
- New MCP tools go under `alma/mcp/tools/synthesis.py`
- Follow existing ABC patterns for extensibility

---

*ALMA Synthesis Squad v1.0.0*
