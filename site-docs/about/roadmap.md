# Roadmap

## Completed

### v0.6.0 - Workflow Context Layer (Current)

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

### v0.7.0 - Intelligence Layer (Planned)

- [ ] Memory compression / summarization
- [ ] Temporal reasoning (time-aware retrieval)
- [ ] Proactive memory suggestions
- [ ] Memory importance scoring

### v0.8.0 - Visualization (Planned)

- [ ] Visual memory explorer dashboard
- [ ] Memory graph visualization
- [ ] Analytics and insights

## Future

- [ ] Additional graph backends (Neptune, TigerGraph)
- [ ] Ollama embedding support
- [ ] Groq LLM integration
- [ ] Memory import/export
- [ ] Team collaboration features

## Contributing

Want to help? Check out our [Contributing Guide](contributing.md) and [Good First Issues](https://github.com/RBKunnela/ALMA-memory/labels/good%20first%20issue).
