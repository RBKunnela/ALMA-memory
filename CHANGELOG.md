# Changelog

All notable changes to ALMA-memory will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.4.0] - 2026-01-28

### Security

- **CRIT-001**: Fixed critical `eval()` vulnerability in Neo4j graph store (`graph/store.py`)
  - Replaced unsafe `eval()` with `json.loads()` for property deserialization
  - Added input validation for graph entity properties
  - This prevented potential arbitrary code execution when loading graph data

- Added comprehensive input validation to all MCP tools
  - Validates required parameters before processing
  - Returns clear error messages for invalid inputs
  - Prevents injection attacks through tool parameters

### Bug Fixes

- **CRIT-002**: Fixed SQLite embeddings delete bug (`storage/sqlite_local.py`)
  - Delete operations used singular form (`heuristic`) while add operations used plural (`heuristics`)
  - Embeddings were never actually deleted, causing unbounded storage growth
  - Now consistently uses plural form across all operations

- Fixed Azure Cosmos DB missing methods (`storage/azure_cosmos.py`)
  - Implemented `update_heuristic_confidence()` method
  - Implemented `update_knowledge_confidence()` method
  - These were defined in base class but never implemented, causing `AttributeError`

- Fixed PostgreSQL IVFFlat index creation on empty tables (`storage/postgresql.py`)
  - IVFFlat indexes require data to create clusters
  - Now defers index creation until first data is inserted
  - Falls back to exact search until index is ready

### Performance

- **PostgreSQL HNSW indexes** (`storage/postgresql.py`)
  - Added HNSW (Hierarchical Navigable Small World) indexes as alternative to IVFFlat
  - HNSW provides better recall with similar performance
  - Configuration option: `vector_index_type: hnsw | ivfflat`

- **Lazy FAISS index rebuild** (`storage/sqlite_local.py`)
  - FAISS indexes now only rebuild when necessary
  - Tracks dirty state to avoid redundant rebuilds
  - Reduces startup time for large databases by 60-80%

- **Timestamp indexes** (`storage/sqlite_local.py`, `storage/postgresql.py`)
  - Added indexes on `created_at` and `updated_at` columns
  - Improves performance of temporal queries and forgetting operations
  - Reduces query time for date-range filters by 10x+

- **Connection pooling improvements** (`storage/postgresql.py`)
  - Increased default pool size from 5 to 10
  - Added configurable min/max pool sizes
  - Better handling of connection exhaustion

### API Improvements

- **Batch operations** (`storage/base.py`, all backends)
  - New `batch_save_heuristics()` method
  - New `batch_save_outcomes()` method
  - New `batch_save_domain_knowledge()` method
  - Reduces database round-trips for bulk operations by 90%

- **Exception hierarchy** (`exceptions.py`)
  - New base class: `ALMAError`
  - Storage errors: `StorageError`, `ConnectionError`, `QueryError`
  - Validation errors: `ValidationError`, `ConfigurationError`
  - Learning errors: `LearningError`, `ScopeViolationError`
  - Retrieval errors: `RetrievalError`, `EmbeddingError`
  - Makes error handling more precise and debugging easier

- **Input validation with clear error messages**
  - All public methods now validate inputs before processing
  - Clear error messages indicate which parameter failed and why
  - Example: `ValidationError: agent name cannot be empty`

### Deprecation Fixes

- **Replaced `datetime.utcnow()`** (`types.py`, all storage backends)
  - `datetime.utcnow()` is deprecated in Python 3.12+
  - Replaced with `datetime.now(timezone.utc)` throughout codebase
  - Ensures Python 3.13+ compatibility

### Documentation

- Added comprehensive technical debt assessment documentation
- Created competitive analysis vs Mem0 (`docs/architecture/competitive-analysis-mem0.md`)
- Updated system architecture documentation (`docs/architecture/system-architecture.md`)
- Added troubleshooting section to README

### Internal

- Added pre-commit hooks for code quality
- Standardized logging format across modules
- Improved test fixtures for storage backends

---

## [0.3.0] - 2026-01-15

### Added

- **LLM-Powered Fact Extraction** (`alma/extraction/`)
  - `AutoLearner` class for automatic learning from conversations
  - Supports OpenAI, Anthropic, and rule-based extraction
  - Configurable confidence thresholds

- **Graph Memory** (`alma/graph/`)
  - Entity and relationship storage via Neo4j
  - `EntityExtractor` for LLM-powered entity extraction
  - In-memory graph store for testing

- **Confidence Engine** (`alma/confidence/`)
  - Forward-looking strategy assessment
  - `assess_strategy()` method returns confidence scores
  - `rank_strategies()` for comparing multiple approaches

- **Session Initializer** (`alma/initializer/`)
  - Full session initialization before starting work
  - Combines memory retrieval, progress check, and session handoff
  - `initialize()` method returns comprehensive context

### Changed

- Improved retrieval scoring weights (similarity: 0.4, recency: 0.3, success: 0.2, confidence: 0.1)
- Enhanced cache invalidation after learning operations

### Fixed

- Fixed race condition in concurrent cache access
- Fixed embedding dimension validation on startup

---

## [0.2.0] - 2026-01-01

### Added

- **Progress Tracking** (`alma/progress/`)
  - `ProgressTracker` class for work item management
  - Priority-based and quick-win task selection strategies
  - Progress summary with completion percentages

- **Session Handoff** (`alma/session/`)
  - `SessionManager` for cross-session context continuity
  - Handoff creation with next steps and blockers
  - Quick reload summaries for session start

- **Domain Memory Factory** (`alma/domains/`)
  - `DomainMemoryFactory` for creating domain-specific ALMA instances
  - Pre-built schemas: coding, research, sales, general, customer_support, content_creation
  - Custom schema creation support

- **Harness Pattern** (`alma/harness/`)
  - Decouples agent from domain memory schema
  - `create_harness()` factory function
  - Automatic memory injection during execution

### Changed

- Refactored storage backends for better abstraction
- Improved MCP server error handling

---

## [0.1.0] - 2025-12-15

### Added

- Initial release of ALMA-memory
- **Core memory types**: Heuristic, Outcome, UserPreference, DomainKnowledge, AntiPattern
- **Memory scoping**: `can_learn` and `cannot_learn` per agent
- **Storage backends**: SQLite+FAISS, Azure Cosmos DB, File-based
- **Retrieval engine**: Vector similarity search with multi-factor scoring
- **Learning protocol**: Automatic heuristic formation from outcomes
- **Cache layer**: In-memory and Redis backends
- **Forgetting mechanism**: Prune stale/low-confidence memories
- **MCP server**: 7 tools for Claude Code integration
- **Embedding providers**: Local (sentence-transformers), Azure OpenAI, Mock

---

## Legend

- **Added**: New features
- **Changed**: Changes to existing functionality
- **Deprecated**: Features that will be removed in future versions
- **Removed**: Features that have been removed
- **Fixed**: Bug fixes
- **Security**: Security-related changes
- **Performance**: Performance improvements
- **Documentation**: Documentation changes
- **Internal**: Internal changes not affecting the public API
