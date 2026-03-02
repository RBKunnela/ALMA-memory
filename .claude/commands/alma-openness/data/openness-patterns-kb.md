# Openness Patterns Knowledge Base

> Design patterns, specifications, and best practices for the ALMA Openness Squad.

---

## 1. MCP Protocol Best Practices for Multi-Client Scenarios

### Architecture Models

**Model A: Shared Process (Current)**
- One `ALMAMCPServer` process handles all connections
- Works for stdio mode where a single MCP client connects
- Limitation: stdio is inherently single-client (one stdin/stdout pair)

**Model B: HTTP Mode Multi-Client (Recommended)**
- One `ALMAMCPServer` running in HTTP mode at a known port
- Multiple MCP clients connect via HTTP to the same server
- Naturally supports concurrent connections
- Each request is independent; server handles serialization

**Model C: Hybrid (Best of Both)**
- Primary tool (e.g., Claude Code) connects via stdio for low latency
- Secondary tools connect via HTTP for flexibility
- Single ALMA instance shared across both modes

### Client Identification Protocol

Every MCP client should identify itself so ALMA can track provenance:

```
Client-ID Header: X-ALMA-Client: claude-code/1.2.3
Tool Parameter:   source_tool="claude-code" (fallback for stdio mode)
```

Priority order for identification:
1. HTTP header `X-ALMA-Client` (HTTP mode)
2. MCP initialization `clientInfo` field (MCP protocol standard)
3. Explicit `source_tool` parameter on tool calls (universal fallback)

### Session Management

- Each client connection gets a unique `session_id`
- Sessions are independent but share the same ALMA instance
- A session maps to one client tool instance (e.g., one Claude Code window)
- Session metadata: `tool_name`, `tool_version`, `connected_at`, `last_activity`

### Error Handling in Multi-Client

- Client disconnections should not affect other clients
- Failed writes should return clear error codes, not crash the server
- Rate limiting per client to prevent one tool from monopolizing the backend

---

## 2. Personal Brain Entity Type Definitions

### Thought

A raw, unprocessed idea, observation, or reflection captured in the moment. Thoughts are the most basic unit of the personal brain -- they represent what the user is thinking before any synthesis or analysis.

**Examples:**
- "I wonder if we should use PostgreSQL instead of SQLite for production"
- "The meeting with Sarah went well -- she seems supportive of the new direction"
- "Noticed that our API response times spike every Monday morning"

**Attributes:**
| Attribute | Type | Required | Description |
|-----------|------|----------|-------------|
| `content` | str | Yes | The thought itself |
| `context` | str | No | Where/when the thought occurred |
| `mood` | str | No | Emotional context (curious, frustrated, excited, neutral) |
| `source_tool` | str | No | Which AI tool captured this thought |
| `tags` | list[str] | No | User-defined tags for categorization |

### Insight

A refined understanding derived from one or more thoughts. Insights emerge when patterns are recognized, connections are made, or conclusions are drawn from accumulated thoughts.

**Examples:**
- "Our Monday API spikes correlate with the batch job that runs at 6 AM -- we should stagger it"
- "Sarah's team uses event-driven architecture consistently -- this could work for our notification system"
- "After reviewing 3 projects, it is clear that type safety catches more bugs than unit tests alone"

**Attributes:**
| Attribute | Type | Required | Description |
|-----------|------|----------|-------------|
| `summary` | str | Yes | The insight statement |
| `confidence` | float | No | How confident the user is (0.0-1.0) |
| `evidence` | list[str] | No | Supporting thoughts or observations |
| `domain` | str | No | Knowledge domain this insight belongs to |
| `source_thoughts` | list[str] | No | IDs of thoughts that led to this insight |

### Decision

A choice made with explicit rationale and considered alternatives. Decisions are high-signal memories that capture not just what was chosen, but why.

**Examples:**
- "Chose PostgreSQL+pgvector over Pinecone for storage: lower cost, self-hosted, pgvector is good enough for our scale"
- "Decided to delay the launch by 2 weeks to add proper error handling -- the risk of a bad first impression is too high"

**Attributes:**
| Attribute | Type | Required | Description |
|-----------|------|----------|-------------|
| `choice` | str | Yes | What was decided |
| `alternatives` | list[str] | No | Options that were considered |
| `rationale` | str | Yes | Why this choice was made |
| `outcome` | str | No | What happened as a result (filled later) |
| `status` | str | No | active, reversed, superseded |

### Person

A person referenced in the user's knowledge graph. People entities allow the brain to track relationships, contexts, and interaction history.

**Examples:**
- "Sarah Chen -- Engineering Lead at Acme, strong advocate for microservices"
- "Dr. Marcus Lee -- My thesis advisor, expert in distributed systems"

**Attributes:**
| Attribute | Type | Required | Description |
|-----------|------|----------|-------------|
| `name` | str | Yes | Person's name |
| `role` | str | No | Professional role or title |
| `relationship` | str | No | How the user knows them (colleague, mentor, friend) |
| `context` | str | No | Primary context of interaction |
| `last_interaction` | str | No | When last interacted with |

### Project

A project, initiative, or area of ongoing work. Projects serve as containers that group related thoughts, insights, decisions, and lessons.

**Examples:**
- "ALMA v1.0 Launch -- Making the memory library production-ready by Q2"
- "Kitchen Renovation -- Budget $50k, timeline 3 months"

**Attributes:**
| Attribute | Type | Required | Description |
|-----------|------|----------|-------------|
| `name` | str | Yes | Project name |
| `status` | str | No | active, paused, completed, abandoned |
| `goals` | list[str] | No | What this project aims to achieve |
| `domain` | str | No | Domain (work, personal, health, etc.) |
| `priority` | str | No | high, medium, low |

### Lesson

A lesson learned from experience, reusable across contexts. Lessons are the highest-signal entity type -- they represent distilled wisdom that should be recalled when relevant situations arise.

**Examples:**
- "Never deploy on Friday afternoon -- the one time we did, we spent the entire weekend debugging"
- "When estimating project timelines, multiply the initial estimate by 2.5 -- this has been accurate for the last 4 projects"

**Attributes:**
| Attribute | Type | Required | Description |
|-----------|------|----------|-------------|
| `summary` | str | Yes | The lesson statement |
| `context` | str | No | Situation where this was learned |
| `applicability` | str | No | When this lesson applies |
| `cost_of_ignoring` | str | No | What happens if this lesson is ignored |
| `source_experience` | str | No | The experience that taught this lesson |

---

## 3. Relationship Type Catalog

### inspires (thought -> thought, insight -> thought)

One thought or insight was inspired by another. Tracks the provenance of ideas.

**Use cases:**
- Linking a new thought to the observation that triggered it
- Connecting an insight back to the thoughts that formed it
- Building "inspiration chains" to understand how ideas evolved

**Implementation note:** Since ALMA's `RelationshipType` requires a single source_type/target_type pair, implement as multiple relationship entries:
- `thought_inspires_thought` (thought -> thought)
- `insight_inspired_by_thought` (insight -> thought)

### contradicts (insight -> insight, decision -> decision)

Two memories are in tension or conflict. This is high-value metadata because it helps the brain surface conflicting information rather than silently ignoring it.

**Use cases:**
- "Always deploy with feature flags" vs. "Feature flags add unnecessary complexity for small changes"
- Tracking when a new decision reverses a previous one
- Surfacing unresolved tensions in the user's thinking

### builds_on (insight -> thought, insight -> insight)

An insight deepens, refines, or extends a prior thought or insight. This is the primary "knowledge synthesis" relationship.

**Use cases:**
- Insight A refines raw Thought B into a more actionable understanding
- Insight C extends Insight D with additional evidence or nuance
- Building knowledge depth over time

### involves (thought -> person, decision -> person, lesson -> person)

A memory references a specific person. This enables person-centric retrieval -- "What do I know about Sarah?"

**Use cases:**
- Link a decision to the people who influenced or were affected by it
- Track which lessons came from interactions with specific people
- Enable "tell me everything related to [person]" queries

### belongs_to (thought -> project, insight -> project, decision -> project)

A memory is part of a project context. This enables project-scoped retrieval -- "Show me everything about the ALMA launch."

**Use cases:**
- Group all thoughts, insights, and decisions under a project
- Track knowledge accumulation per project over time
- Enable project handoff by exporting all related memories

### leads_to (decision -> lesson, thought -> lesson)

An experience or decision leads to a lesson. This captures the causal chain from experience to wisdom.

**Use cases:**
- Decision D led to bad outcome, which produced Lesson L
- A series of thoughts about a topic crystallized into a reusable lesson
- Tracing why a lesson exists (what experience taught it)

---

## 4. Memory Migration Format Specifications

### Claude Memory Export

**Format:** JSON file from Claude's settings export.

**Structure:**
```json
{
  "memories": [
    {
      "id": "mem_abc123",
      "content": "User prefers TypeScript over JavaScript",
      "created_at": "2025-11-15T10:30:00Z",
      "source": "conversation",
      "context": "Discussion about project tooling"
    }
  ]
}
```

**Mapping to Personal Brain:**
- Short factual statements -> `lesson` or `insight`
- Preferences -> `lesson` (they represent learned preferences)
- Observations -> `thought`
- Decisions mentioned -> `decision`

**Metadata preservation:** source, context, created_at, original_id

### ChatGPT Memory Export

**Format:** JSON file from ChatGPT Settings > Data Controls > Export Data.

**Structure:**
```json
{
  "memories": [
    {
      "id": "...",
      "text": "Prefers dark mode in all editors",
      "created_at": "2025-10-20",
      "updated_at": "2025-11-01"
    }
  ]
}
```

**Mapping to Personal Brain:**
- Similar heuristic-based classification as Claude exports
- `text` field is typically shorter and more declarative than Claude's

**Metadata preservation:** created_at, updated_at, original_id

### Obsidian Vault Export

**Format:** Directory of Markdown files with YAML frontmatter and wiki links.

**Structure:**
```
vault/
  daily/          -> thought entities (daily notes)
  projects/       -> project entities
  people/         -> person entities
  insights/       -> insight entities
  decisions/      -> decision entities (if structured)
  *.md            -> general notes (classify by content)
```

**Frontmatter mapping:**
```yaml
---
title: "Note Title"    -> entity name/summary
tags: [tag1, tag2]     -> ALMA tags metadata
created: 2025-11-01    -> ALMA timestamp
status: active         -> entity status attribute
type: project          -> entity type (if explicit)
---
```

**Wiki link mapping:**
- `[[Person Name]]` -> `involves` relationship to person entity
- `[[Project Name]]` -> `belongs_to` relationship to project entity
- `[[Other Note]]` -> `builds_on` or `inspires` relationship

**Metadata preservation target:** >90%
- Content: 100% (markdown preserved or converted)
- Tags: 100% (direct mapping)
- Frontmatter fields: 90%+ (most map directly)
- Wiki links: 80%+ (converted to relationships where target entity exists)

---

## 5. Concurrent Write Handling Patterns

### Optimistic Concurrency Control

Used by Azure Cosmos DB and recommended as the default strategy for database-backed storage.

**Pattern:**
1. Read the current version (ETag/version number)
2. Perform the write with the version as a precondition
3. If version mismatch (someone else wrote first), retry with the new version
4. Max retries: 3 (then fail with ConflictError)

```python
async def write_with_optimistic_concurrency(storage, key, value, max_retries=3):
    for attempt in range(max_retries):
        current = await storage.read(key)
        etag = current.get("_etag") if current else None
        try:
            await storage.write(key, value, etag=etag)
            return  # Success
        except ConcurrencyConflict:
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(0.1 * (attempt + 1))  # Backoff
```

### Write Queue Serialization

Used by SQLite and file-based backends where concurrent writes are unsafe.

**Pattern:**
1. All write operations go through an asyncio.Queue
2. A single consumer processes writes sequentially
3. Reads are not queued (concurrent reads are safe)

```python
class WriteQueue:
    def __init__(self):
        self._queue = asyncio.Queue()
        self._consumer_task = asyncio.create_task(self._consume())

    async def enqueue_write(self, operation):
        future = asyncio.get_event_loop().create_future()
        await self._queue.put((operation, future))
        return await future

    async def _consume(self):
        while True:
            operation, future = await self._queue.get()
            try:
                result = await operation()
                future.set_result(result)
            except Exception as e:
                future.set_exception(e)
```

### Cross-Tool Deduplication Strategies

**Strategy 1: Content Hash (Fast, Exact)**
- SHA-256 hash of normalized content
- Check against recent writes within time window (60s default)
- O(1) lookup, but misses semantic duplicates

**Strategy 2: Semantic Similarity (Slower, Fuzzy)**
- Use ALMA's `DeduplicationEngine` from `alma/consolidation/deduplication.py`
- Compare embedding similarity with threshold (0.85 default)
- Catches paraphrased duplicates but requires embedding computation

**Strategy 3: Two-Phase (Recommended)**
1. First pass: content hash dedup (instant, catches exact dupes)
2. Second pass: semantic dedup on recent writes (background, catches near-dupes)
3. The second pass can run asynchronously after the write succeeds

### Last-Write-Wins with Audit Trail

The simplest conflict resolution strategy. When two tools write the same key:
1. The later write wins
2. The earlier write is preserved in an audit log
3. Both `source_tool` values are recorded

This is acceptable for the Personal Brain because:
- Most writes are *new* memories, not updates to existing ones
- When updates happen, the user's most recent AI tool interaction is usually the one that matters
- The audit trail allows recovery if needed

---

## 6. Quickstart UX Best Practices

### Progressive Disclosure

Only ask what is needed at each step. Start with the minimum viable configuration and offer advanced options later.

**Good:**
```
Step 1: Choose storage backend
  [1] SQLite (simplest, local)
  [2] PostgreSQL (recommended for multi-tool)
```

**Bad:**
```
Step 1: Configure storage
  Backend type: ___
  Connection string: ___
  Pool size: ___
  Max retries: ___
  Timeout (ms): ___
```

### Smart Defaults

Every option should have a sensible default so the user can press Enter to continue:

| Option | Default | Rationale |
|--------|---------|-----------|
| Storage backend | SQLite+FAISS | Zero setup, works immediately |
| Embedding provider | OpenAI | Best quality, most users have API key |
| Domain schema | personal_brain | The whole point of open-brain setup |
| MCP config | Generate .mcp.json | Most users want Claude Code integration |
| Project ID | "open-brain" | Descriptive, can be changed later |

### Error Messages

When something fails, tell the user exactly what went wrong and how to fix it:

**Good:**
```
ERROR: Could not connect to PostgreSQL at localhost:5432
  - Is PostgreSQL running? Try: pg_isready
  - Is the database created? Try: createdb alma
  - Check your connection string: postgresql://user:pass@localhost:5432/alma
```

**Bad:**
```
ERROR: Connection refused
```

### Validation at Each Step

Validate each configuration step before moving to the next:
1. Storage backend selected -> verify connection (for PostgreSQL)
2. Embedding provider selected -> verify API key works (for OpenAI)
3. All config written -> parse it back with ConfigLoader
4. MCP config written -> verify the server can start

### Time Budget

Target: 45 minutes total from `pip install` to first successful retrieve.

| Step | Target Time |
|------|------------|
| `pip install alma-memory` | 2 min |
| `alma init --open-brain` | 5 min |
| Run SQL migration (if PostgreSQL) | 3 min |
| Configure MCP in Claude Code | 5 min |
| First thought capture | 1 min |
| First thought retrieval | 1 min |
| Buffer for troubleshooting | 28 min |

---

## 7. ALMA Codebase Reference

### Key Files for Openness Squad

| File | Purpose | Relevance |
|------|---------|-----------|
| `alma/domains/types.py` | DomainSchema, EntityType, RelationshipType | Schema design |
| `alma/domains/schemas.py` | 6 pre-built schemas | Pattern to follow |
| `alma/domains/factory.py` | DomainMemoryFactory | Schema registration |
| `alma/mcp/server.py` | ALMAMCPServer | Multi-client integration |
| `alma/mcp/tools.py` | 22 MCP tools | Tool modification for source_tool |
| `alma/consolidation/deduplication.py` | DeduplicationEngine | Cross-tool dedup |
| `alma/storage/base.py` | StorageBackend ABC | Concurrency characteristics |
| `alma/config/loader.py` | ConfigLoader | Config generation target |
| `alma/events/emitter.py` | Event emitter | Cross-tool notifications |
| `alma/core.py` | ALMA class | Core API for all operations |
| `alma/types.py` | Heuristic, Outcome | Memory data structures |

### Testing Utilities

```python
from alma.testing import MockStorage, MockEmbedder
from alma.testing.factories import (
    create_test_heuristic,
    create_test_outcome,
)
```

---

*Openness Patterns KB v1.0.0 -- ALMA Openness Squad*
*Source: Derived from open-brain-kb.md, ALMA codebase analysis, and MCP protocol research*
