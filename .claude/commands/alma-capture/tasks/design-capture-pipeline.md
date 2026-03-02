# Task: Design Capture Pipeline

## Goal

Design the end-to-end capture pipeline architecture for ALMA's Open Brain thought capture system. This is the foundational design that all other tasks in the alma-capture squad depend on.

## Agent

**capture-chief** ([agents/capture-chief.md](../agents/capture-chief.md))

## Requires

None -- this is the first task in the dependency chain.

## Inputs

- **ALMA codebase:** `alma/` -- study existing architecture, especially:
  - `alma/mcp/tools.py` -- 22 existing MCP tools (patterns to follow)
  - `alma/storage/base.py` -- StorageBackend ABC (storage abstraction)
  - `alma/retrieval/` -- retrieval engine (how memories are searched)
  - `alma/extraction/` -- existing auto-learning from conversations
  - `alma/types.py` -- memory data structures
  - `alma/exceptions.py` -- exception hierarchy
- **Open Brain KB:** [data/open-brain-kb.md](../data/open-brain-kb.md) -- target architecture and patterns

## Steps

### Step 1: Analyze Existing ALMA MCP Tools

Study at least 5 existing MCP tools in `alma/mcp/tools.py` to understand:
- How tools are registered with the MCP server
- Parameter patterns (required, optional, validation)
- Return type patterns (success responses, error responses)
- How tools interact with storage backends
- How tools handle async execution
- Error handling patterns (which exceptions, how they propagate)

Document the patterns found.

### Step 2: Map the Capture Flow

Design the end-to-end pipeline:

```
1. VALIDATE  -- content not empty, importance in range, source_tool valid
2. EMBED     -- generate vector embedding via configured embedder
3. EXTRACT   -- extract metadata via LLM (type, topics, people, action_items, importance)
4. STORE     -- persist thought + embedding + metadata via storage backend
5. CONFIRM   -- return success response with stored ID, metadata preview, timestamp
```

For each step, define:
- Input/output data shapes
- Error handling (what happens if this step fails?)
- Whether the step is blocking or can be async
- Fallback behavior (e.g., store without metadata if extraction fails)

### Step 3: Design MCP Tool API Surface

Define the API for three new MCP tools:

**alma_capture_thought:**
- Parameters: content (str, required), source_tool (str, optional), importance_override (int, optional)
- Returns: { id, content_preview, metadata, stored_at }
- Errors: ValidationError, StorageError

**alma_list_memories:**
- Parameters: date_from, date_to, type_filter, topic_filter, limit, offset
- Returns: { memories[], total_count, has_more }

**alma_browse_timeline:**
- Parameters: date, range_days, direction
- Returns: { timeline[{date, memories[]}], date_range, total_count }

### Step 4: Design Metadata Extraction Prompt

Draft the LLM prompt for metadata extraction. The prompt must:
- Accept raw thought text as input
- Return structured JSON with: type, topics, people, action_items, importance
- Handle all 8 thought types (decision, person_note, insight, meeting, task, idea, reference, lesson)
- Be token-efficient (minimize prompt tokens for cost)
- Include examples for ambiguous cases
- Define the importance scale clearly (1=trivial, 5=critical)

Reference the metadata schema in [templates/metadata-schema-tmpl.md](../templates/metadata-schema-tmpl.md).

### Step 5: Document Data Flow Diagram

Create a data flow document covering:
- Where the new code lives in ALMA's module structure
- How new modules interact with existing modules
- Configuration requirements (LLM provider, embedder, storage backend)
- Testing strategy (unit tests, integration tests, test corpus)

## Output

Architecture document covering:
1. Data flow diagram (capture -> embed -> extract -> store -> confirm)
2. MCP tool API surface (parameters, return types, error handling)
3. Metadata extraction prompt design
4. Module placement within ALMA codebase
5. Error handling and fallback strategy
6. Configuration requirements

## Gate

- [ ] Architecture covers the full pipeline: capture -> embed -> extract metadata -> store -> confirm
- [ ] MCP tool API surface is defined for all 3 tools (parameters, return types, errors)
- [ ] Metadata extraction prompt is designed and covers all 8 thought types
- [ ] Error handling strategy is documented for each pipeline step
- [ ] Module placement is decided and does not conflict with existing ALMA structure
- [ ] Dependencies on existing ALMA modules are identified
