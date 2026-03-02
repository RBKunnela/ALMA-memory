# Capture Quality Gate Checklist

**Squad:** alma-capture
**Phase:** 1 (Foundation -- Thought Capture)
**Purpose:** Verify all Phase 1 deliverables meet acceptance criteria before proceeding to Phase 2.

---

## MCP Tools

- [ ] `alma_capture_thought` MCP tool is registered and callable via MCP protocol
- [ ] `alma_list_memories` MCP tool is registered and callable via MCP protocol
- [ ] `alma_browse_timeline` MCP tool is registered and callable via MCP protocol
- [ ] All three tools appear in MCP tool listings

## Capture Pipeline

- [ ] Capture flow executes: validate -> embed -> extract metadata -> store -> confirm
- [ ] Empty content input raises ValidationError
- [ ] Embedding failure does not block storage (graceful degradation)
- [ ] Metadata extraction failure does not block storage (graceful degradation)
- [ ] `importance_override` parameter overrides LLM-assigned importance
- [ ] `source_tool` parameter is recorded in stored metadata
- [ ] Captured thoughts are retrievable via existing `alma_retrieve`

## Metadata Extraction

- [ ] Metadata extraction returns valid JSON with all 5 fields (type, topics, people, action_items, importance)
- [ ] Extraction handles all 8 thought types (decision, person_note, insight, meeting, task, idea, reference, lesson)
- [ ] Extraction accuracy >80% on test corpus (weighted average across all fields)
- [ ] Extraction works with configurable LLM provider (not hardcoded)
- [ ] Malformed LLM output is handled gracefully (no crashes, sensible defaults)
- [ ] JSON schema validation catches invalid extraction results

## List and Browse

- [ ] `alma_list_memories` supports date filtering (date_from, date_to)
- [ ] `alma_list_memories` supports type filtering
- [ ] `alma_list_memories` supports topic filtering (case-insensitive)
- [ ] Pagination works correctly (limit, offset, has_more flag)
- [ ] `alma_browse_timeline` returns memories grouped by day in chronological order
- [ ] `alma_browse_timeline` supports before, after, and around directions
- [ ] Both tools work with all 7 ALMA storage backends

## Code Quality

- [ ] All new code follows ALMA coding standards (ruff format, ruff check passes)
- [ ] Type hints on all public API functions
- [ ] Google-style docstrings on all public API functions
- [ ] Error handling uses ALMA's exception hierarchy (no bare except)
- [ ] Both sync and async variants provided where applicable

## Testing

- [ ] Unit tests written for all new modules
- [ ] Metadata extraction test corpus has 20+ diverse examples
- [ ] All new tests pass
- [ ] No regressions in existing 1,210 tests
- [ ] New code coverage >80%

## Migration (Design Only)

- [ ] Migration architecture document is complete
- [ ] Common intermediate format is defined
- [ ] Metadata mapping documented for Claude, ChatGPT, and Obsidian
- [ ] Importer plugin architecture is designed
- [ ] `alma/migration/` module skeleton is created

---

## Sign-off

| Reviewer | Status | Date |
|----------|--------|------|
| capture-chief | Pending | |
| metadata-extractor | Pending | |
| mcp-capture-dev | Pending | |
| migration-engineer | Pending | |

**Phase 1 is complete when all checkboxes above are checked and all agents have signed off.**

On completion, proceed to Phase 2 (Intelligence -- Synthesis & Patterns) per [workflows/alma-open-brain-evolution.yaml](../workflows/alma-open-brain-evolution.yaml).
