# Roadmap

## Completed

### v0.8.0 - RAG Integration (Current)

- [x] RAG Bridge - Accept external RAG chunks, enhance with memory signals
- [x] Memory Enhancer - Enrich retrieval with heuristics, anti-patterns, domain knowledge
- [x] Retrieval Feedback Loop - Learn which retrieval strategies work
- [x] IR Metrics - MRR, NDCG, Recall, Precision, MAP (pure Python)
- [x] Hybrid Search - Vector + keyword (BM25/TF-IDF) with RRF fusion
- [x] Cross-Encoder Reranking - Pluggable reranker with optional `rerankers` library
- [x] Text Search Providers - BM25S + pure-Python TF-IDF fallback
- [x] Optional dependency group: `pip install alma-memory[rag]`

### v0.7.0 - Intelligence Layer

- [x] Memory Decay - Time-based confidence decay for aging memories
- [x] Memory Compression - LLM + rule-based memory summarization
- [x] Retrieval Verification - Two-stage verified retrieval pipeline
- [x] Retrieval Modes - 7 cognitive task modes (standard, progressive, verified, budget, etc.)
- [x] Trust-Integrated Scoring - Agent trust weighting for memory retrieval
- [x] Token Budget Management - Context window management
- [x] Progressive Disclosure - Summary, highlights, full detail levels
- [x] 6 New MCP Tools - Memory Wall tools for Claude Code
- [x] Archive System - Soft-delete with recovery

### v0.6.0 - Workflow Context Layer

- [x] Checkpoint & Resume - Save and restore workflow state
- [x] State Reducers - Merge outputs from parallel agents
- [x] Artifact Linking - Track code, tests, and documents per workflow
- [x] Scoped Retrieval - Filter memories by workflow context
- [x] Session Persistence - Handoffs survive process restarts
- [x] 8 New MCP Tools - Full workflow support in Claude Code
- [x] TypeScript SDK v0.6.0 - Full workflow API parity

### v0.5.0 - Vector Database Backends

- [x] Qdrant Backend - Managed vector database support
- [x] Pinecone Backend - Serverless vector database
- [x] Chroma Backend - Lightweight local option
- [x] Graph Database Abstraction - Neo4j, Memgraph, Kuzu backends
- [x] Testing Module - MockStorage, MockEmbedder, factories
- [x] Memory Consolidation - LLM-powered deduplication
- [x] Event System - Webhooks and in-process callbacks
- [x] TypeScript SDK - Initial release
- [x] Multi-Agent Memory Sharing - inherit_from, share_with

### v0.4.0 and Earlier

- [x] Core memory types (Heuristic, Outcome, Preference, Knowledge, Anti-pattern)
- [x] Scoped learning (can_learn, cannot_learn)
- [x] SQLite + FAISS storage
- [x] PostgreSQL + pgvector
- [x] Azure Cosmos DB
- [x] MCP Server integration
- [x] Progress tracking
- [x] Session handoff
- [x] Domain memory factory
- [x] Confidence engine
- [x] Cache layer
- [x] Forgetting mechanism

## In Progress

### v0.9.0 - Visualization & Temporal (Planned)

- [ ] Visual memory explorer dashboard
- [ ] Memory graph visualization
- [ ] Analytics and insights
- [ ] Temporal reasoning (time-aware retrieval)
- [ ] Proactive memory suggestions

## Future

- [ ] Additional graph backends (Neptune, TigerGraph)
- [ ] Ollama embedding support
- [ ] Groq LLM integration
- [ ] Memory import/export
- [ ] Team collaboration features

## Contributing

Want to help? Check out our [Contributing Guide](contributing.md) and [Good First Issues](https://github.com/RBKunnela/ALMA-memory/labels/good%20first%20issue).
