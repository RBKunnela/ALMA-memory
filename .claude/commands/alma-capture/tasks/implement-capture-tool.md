# Task: Implement Capture Tool

## Goal

Implement the `alma_capture_thought` MCP tool -- the primary tool for capturing thoughts into ALMA with automatic embedding and metadata extraction.

## Agent

**mcp-capture-dev** ([agents/mcp-capture-dev.md](../agents/mcp-capture-dev.md))

## Requires

- **build-metadata-extractor** (completed) -- the metadata extraction module must be working and tested before this tool can use it

## Steps

### Step 1: Study Existing MCP Tool Patterns

Read `alma/mcp/tools.py` and study at least 3 existing tools. Document:

- How tools are decorated/registered with the MCP server
- How parameters are defined (types, required vs optional, descriptions)
- How tools access the ALMA core instance and storage backend
- How tools format success responses
- How tools handle and format errors
- How tools handle async execution

**Do NOT skip this step.** The new tool must match existing patterns exactly.

### Step 2: Implement alma_capture_thought

Create the tool with the following specification:

```
Tool: alma_capture_thought
Description: Capture a thought into memory with automatic embedding and metadata extraction.

Parameters:
  content: str (required)
    The thought text to capture. Must not be empty.

  source_tool: str (optional, default: "manual")
    Which AI tool or interface sent this thought.
    Valid values: claude, chatgpt, cursor, slack, manual

  importance_override: int (optional, 1-5)
    Override the LLM-assigned importance score.
    If not provided, importance is determined by metadata extraction.

Returns:
  {
    "id": "uuid",
    "content_preview": "first 100 chars...",
    "metadata": {
      "type": "insight",
      "topics": ["memory", "architecture"],
      "people": [],
      "action_items": [],
      "importance": 3,
      "source_tool": "claude"
    },
    "stored_at": "2026-03-03T10:30:00Z",
    "embedding_status": "success" | "failed",
    "extraction_status": "success" | "partial" | "failed"
  }
```

### Step 3: Implement the Capture Flow

The tool must execute this pipeline:

```
1. VALIDATE
   - content is not empty (raise ValidationError if empty)
   - importance_override is 1-5 if provided (raise ValidationError if out of range)
   - source_tool is from valid set if provided

2. EMBED
   - Generate vector embedding via configured embedder
   - On failure: log warning, continue without embedding
   - Set embedding_status accordingly

3. EXTRACT
   - Call MetadataExtractor.extract(content)
   - Apply importance_override if provided (overrides LLM result)
   - Add source_tool to metadata
   - Add captured_at timestamp
   - On failure: log warning, use minimal metadata (type="unknown", importance=3)
   - Set extraction_status accordingly

4. STORE
   - Store thought + embedding + metadata via storage backend
   - On failure: raise StorageError (this is a hard failure)

5. CONFIRM
   - Return success response with stored ID, metadata, timestamps
```

### Step 4: Register with MCP Server

Register the new tool with ALMA's MCP server infrastructure. This involves:

- Adding the tool function to `alma/mcp/tools.py`
- Registering it with the MCP tool registry
- Ensuring it appears in tool listings
- Testing it is callable via MCP protocol

### Step 5: Write Tests

Write comprehensive tests in `tests/unit/test_mcp_capture_tools.py`:

**Happy path tests:**
- Capture a thought with all fields
- Capture a thought with only required fields
- Capture a thought with importance_override
- Capture a thought with source_tool specified

**Error handling tests:**
- Empty content raises ValidationError
- Invalid importance_override raises ValidationError
- Embedding failure stores thought without embedding
- Metadata extraction failure stores thought with minimal metadata
- Storage failure propagates StorageError

**Integration tests:**
- Captured thought is retrievable via existing `alma_retrieve`
- Metadata is correctly stored and queryable
- Embedding enables semantic search

Use `alma.testing.MockStorage` and `alma.testing.MockEmbedder` for unit tests.

## Output

- **Tool function:** New function `alma_capture_thought` in `alma/mcp/tools.py`
- **Tests:** `tests/unit/test_mcp_capture_tools.py`

## Gate

- [ ] MCP tool is registered and callable via MCP protocol
- [ ] Thought is stored with embedding + metadata in configured storage backend
- [ ] Captured thought is retrievable via existing `alma_retrieve`
- [ ] Empty content raises ValidationError
- [ ] Embedding failure does not block storage (graceful degradation)
- [ ] Metadata extraction failure does not block storage (graceful degradation)
- [ ] importance_override overrides LLM-assigned importance
- [ ] source_tool is recorded in metadata
- [ ] All tests pass
- [ ] Code follows ALMA patterns: same registration, parameter, response patterns as existing tools
