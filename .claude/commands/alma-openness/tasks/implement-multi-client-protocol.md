# Task: Implement Multi-Client Protocol

**Agent:** `@protocol-architect` (Sync)
**Squad:** alma-openness
**Priority:** P1
**Requires:** `design-personal-brain-schema` (completed)

---

## Goal

Handle concurrent MCP connections from multiple AI tools (Claude Code, ChatGPT, Cursor, Obsidian, etc.) writing to the same ALMA memory backend simultaneously. Implement source_tool tracking, cross-tool deduplication, and concurrent write safety so that when five AI tools talk to the same brain at the same time, the brain gets smarter -- not corrupted.

## Output

- `alma/mcp/multi_client.py` -- Multi-client connection handler with source tracking and deduplication
- `alma/mcp/conflict.py` -- Conflict resolution strategies for concurrent writes
- Updates to `alma/mcp/server.py` -- Integration with multi-client handler
- `tests/unit/test_multi_client.py` -- Unit tests for concurrent write handling

## Steps

### 1. Analyze ALMA's Existing MCP Server

Study the current MCP server architecture:

- **Server**: `alma/mcp/server.py` -- `ALMAMCPServer` class with 22 tools and 2 resources
- **Tools**: `alma/mcp/tools.py` -- Tool implementations (learn, retrieve, checkpoint, etc.)
- **Entry point**: `alma/mcp/__main__.py` -- stdio and HTTP mode startup
- **Resources**: `alma/mcp/resources.py` -- Agent and config resources

Key observations:
- Current server creates a single `ALMA` instance shared across all tool calls
- No client identification mechanism exists today
- No source_tool tracking on memories
- Storage backends have varying concurrency support (see below)

### 2. Design Client Identification

Each MCP client must identify itself so ALMA can track which tool created which memory.

**Protocol:**

```python
@dataclass
class ClientIdentity:
    """Identifies an MCP client connection."""
    tool_name: str        # "claude-code", "chatgpt", "cursor", "obsidian"
    tool_version: str     # "1.2.3"
    session_id: str       # Unique per connection session
    user_id: Optional[str] = None  # Optional user identifier
```

**How clients identify themselves:**

Option A: **MCP initialization params** -- Client passes identity in the `initialize` request
Option B: **Custom header** -- HTTP mode clients pass `X-ALMA-Client` header
Option C: **Tool parameter** -- Add optional `source_tool` param to all MCP tools

Recommended: Option C (most compatible, works with all MCP clients without protocol changes).

### 3. Implement Source Tracking

Add `source_tool` to the metadata of every memory operation:

```python
class MultiClientHandler:
    """Handles multi-client access to ALMA."""

    def __init__(self, alma: ALMA):
        self.alma = alma
        self._active_clients: Dict[str, ClientIdentity] = {}
        self._write_log: List[WriteRecord] = []

    def track_write(self, source_tool: str, operation: str, memory_key: str):
        """Record which tool performed which write operation."""
        record = WriteRecord(
            source_tool=source_tool,
            operation=operation,
            memory_key=memory_key,
            timestamp=datetime.now(timezone.utc),
        )
        self._write_log.append(record)

    def get_source_tools_for_memory(self, memory_key: str) -> List[str]:
        """Get all tools that have written to a specific memory."""
        return [r.source_tool for r in self._write_log if r.memory_key == memory_key]
```

The `source_tool` field should be stored in the memory's metadata dict, which all storage backends already support.

### 4. Design Concurrent Write Handling

Different storage backends have different concurrency characteristics:

| Backend | Concurrency Model | Safety |
|---------|-------------------|--------|
| SQLite+FAISS | File-level locking, single writer | Use write queue |
| PostgreSQL+pgvector | Row-level locking, MVCC | Safe for concurrent writes |
| Azure Cosmos | Optimistic concurrency with ETags | Use ETag-based retry |
| Qdrant | API-level, eventual consistency | Safe with idempotent writes |
| Chroma | API-level | Safe with idempotent writes |
| Pinecone | API-level, eventual consistency | Safe with idempotent writes |
| File-based | No locking | Use write queue |

**Strategy by backend type:**

```python
class ConcurrencyStrategy(ABC):
    """Base class for backend-specific concurrency handling."""

    @abstractmethod
    async def acquire_write_lock(self, key: str) -> bool: ...

    @abstractmethod
    async def release_write_lock(self, key: str): ...


class DatabaseConcurrency(ConcurrencyStrategy):
    """For PostgreSQL -- relies on database-level locking."""
    # No application-level locking needed; DB handles it

class OptimisticConcurrency(ConcurrencyStrategy):
    """For Cosmos DB -- retry on ETag conflict."""
    # Read ETag, write with ETag, retry on 412

class QueuedConcurrency(ConcurrencyStrategy):
    """For SQLite and File-based -- serialize writes through a queue."""
    # asyncio.Queue to serialize write operations
```

### 5. Implement Deduplication Layer

Prevent duplicate saves when multiple tools capture the same thought simultaneously.

**Deduplication strategies:**

1. **Content hash dedup** -- Hash the content and check for recent duplicates (fast, exact match)
2. **Semantic dedup** -- Use ALMA's existing `DeduplicationEngine` from `alma/consolidation/deduplication.py` for near-duplicate detection (slower, semantic match)
3. **Time-window dedup** -- If the same content arrives from different tools within N seconds, treat as duplicate

```python
class CrossToolDeduplicator:
    """Prevents duplicate memories from multiple tools."""

    def __init__(
        self,
        time_window_seconds: int = 60,
        similarity_threshold: float = 0.85,
    ):
        self.time_window = time_window_seconds
        self.similarity_threshold = similarity_threshold
        self._recent_writes: Dict[str, WriteRecord] = {}  # content_hash -> record

    def is_duplicate(self, content: str, source_tool: str) -> DuplicateCheckResult:
        """
        Check if content is a duplicate from another tool.

        Returns:
            DuplicateCheckResult with is_duplicate flag and the original source_tool
        """
        content_hash = self._hash_content(content)

        # Check exact match within time window
        if content_hash in self._recent_writes:
            original = self._recent_writes[content_hash]
            if original.source_tool != source_tool:
                elapsed = (datetime.now(timezone.utc) - original.timestamp).total_seconds()
                if elapsed < self.time_window:
                    return DuplicateCheckResult(
                        is_duplicate=True,
                        original_source_tool=original.source_tool,
                        match_type="exact",
                    )

        return DuplicateCheckResult(is_duplicate=False)

    def record_write(self, content: str, source_tool: str):
        """Record a write for future dedup checks."""
        content_hash = self._hash_content(content)
        self._recent_writes[content_hash] = WriteRecord(
            source_tool=source_tool,
            content_hash=content_hash,
            timestamp=datetime.now(timezone.utc),
        )
```

### 6. Build alma/mcp/multi_client.py

Combine all components into the multi-client handler:

```python
"""
Multi-Client MCP Handler for ALMA.

Handles concurrent connections from multiple AI tools,
tracks source_tool per memory, and prevents cross-tool duplicates.
"""

class MultiClientManager:
    """
    Manages multi-client access to a single ALMA instance.

    Responsibilities:
    - Track connected clients and their identities
    - Route writes through appropriate concurrency strategy
    - Deduplicate cross-tool memory captures
    - Record source_tool metadata on all writes
    - Emit events when one tool's write may interest another
    """

    def __init__(
        self,
        alma: ALMA,
        dedup_time_window: int = 60,
        dedup_similarity_threshold: float = 0.85,
    ):
        self.alma = alma
        self.deduplicator = CrossToolDeduplicator(
            time_window_seconds=dedup_time_window,
            similarity_threshold=dedup_similarity_threshold,
        )
        self.concurrency = self._detect_concurrency_strategy()
        self._clients: Dict[str, ClientIdentity] = {}

    def _detect_concurrency_strategy(self) -> ConcurrencyStrategy:
        """Auto-detect the right concurrency strategy from the storage backend."""
        backend_name = type(self.alma.storage).__name__
        if "PostgreSQL" in backend_name or "Cosmos" in backend_name:
            return DatabaseConcurrency()
        elif "Qdrant" in backend_name or "Chroma" in backend_name or "Pinecone" in backend_name:
            return DatabaseConcurrency()  # API-level safety
        else:
            return QueuedConcurrency()  # SQLite, File-based

    async def handle_learn(self, content: str, source_tool: str, **kwargs):
        """Handle a learn operation from a specific tool."""
        # 1. Check for duplicates
        dedup_result = self.deduplicator.is_duplicate(content, source_tool)
        if dedup_result.is_duplicate:
            logger.info(f"Duplicate from {source_tool}, original from {dedup_result.original_source_tool}")
            return {"status": "duplicate", "original_tool": dedup_result.original_source_tool}

        # 2. Add source_tool to metadata
        metadata = kwargs.get("metadata", {})
        metadata["source_tool"] = source_tool

        # 3. Acquire lock if needed
        await self.concurrency.acquire_write_lock(content)
        try:
            result = self.alma.learn(content=content, metadata=metadata, **kwargs)
        finally:
            await self.concurrency.release_write_lock(content)

        # 4. Record for future dedup
        self.deduplicator.record_write(content, source_tool)

        return result
```

### 7. Test with Simulated Concurrent Writes

```python
import asyncio

async def test_concurrent_writes_from_two_clients():
    """Two tools writing simultaneously should not corrupt data."""
    manager = MultiClientManager(alma=mock_alma)

    # Simulate Claude Code and Cursor writing at the same time
    results = await asyncio.gather(
        manager.handle_learn("Python typing is great", source_tool="claude-code"),
        manager.handle_learn("Python typing is great", source_tool="cursor"),
    )

    # One should succeed, one should be detected as duplicate
    statuses = [r.get("status") for r in results]
    assert "duplicate" in statuses or len(set(statuses)) == 1


async def test_different_content_concurrent():
    """Different content from different tools should both succeed."""
    manager = MultiClientManager(alma=mock_alma)

    results = await asyncio.gather(
        manager.handle_learn("I prefer dark themes", source_tool="claude-code"),
        manager.handle_learn("Always use type hints", source_tool="chatgpt"),
    )

    # Both should succeed (different content)
    assert all(r.get("status") != "duplicate" for r in results)


def test_source_tool_tracked():
    """source_tool should be recorded in memory metadata."""
    manager = MultiClientManager(alma=mock_alma)
    manager.handle_learn("Test thought", source_tool="claude-code")
    # Verify metadata contains source_tool="claude-code"
```

## Gate (Definition of Done)

- [ ] `MultiClientManager` class implemented in `alma/mcp/multi_client.py`
- [ ] Client identification works (source_tool tracked per write)
- [ ] `source_tool` stored in memory metadata for every write operation
- [ ] Cross-tool deduplication prevents duplicate saves within configurable time window
- [ ] Concurrent writes from 2+ simulated clients handled without data corruption
- [ ] Concurrency strategy auto-detected from storage backend type
- [ ] `QueuedConcurrency` serializes writes for SQLite and file-based backends
- [ ] `DatabaseConcurrency` relies on DB-level locking for PostgreSQL/Cosmos
- [ ] Integration with `ALMAMCPServer` in `alma/mcp/server.py`
- [ ] All unit tests pass: `.venv/bin/python -m pytest tests/unit/test_multi_client.py -v`
- [ ] Linting passes: `.venv/bin/python -m ruff check alma/mcp/multi_client.py alma/mcp/conflict.py`

## References

- `alma/mcp/server.py` -- ALMAMCPServer (22 tools, 2 resources)
- `alma/mcp/tools.py` -- Existing MCP tool implementations
- `alma/mcp/__main__.py` -- Server entry point (stdio + HTTP modes)
- `alma/consolidation/deduplication.py` -- DeduplicationEngine for semantic dedup
- `alma/storage/base.py` -- StorageBackend ABC (concurrency characteristics)
- `alma/events/emitter.py` -- Event emitter for cross-tool notifications
- `data/openness-patterns-kb.md` -- Concurrent write patterns and MCP best practices
