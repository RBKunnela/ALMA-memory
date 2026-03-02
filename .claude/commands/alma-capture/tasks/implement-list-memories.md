# Task: Implement List Memories

## Goal

Implement two MCP tools for browsing and filtering stored memories: `alma_list_memories` and `alma_browse_timeline`.

## Agent

**mcp-capture-dev** ([agents/mcp-capture-dev.md](../agents/mcp-capture-dev.md))

## Requires

- **implement-capture-tool** (completed) -- the capture tool must be working so there are memories to list and browse. The list/browse tools also need to follow the same registration and response patterns.

## Steps

### Step 1: Implement alma_list_memories

Create the tool for listing recent memories with filtering and pagination:

```
Tool: alma_list_memories
Description: List stored memories with optional filtering by date, type, and topic.

Parameters:
  date_from: str (optional, ISO-8601)
    Only include memories created on or after this date.

  date_to: str (optional, ISO-8601)
    Only include memories created on or before this date.

  type_filter: str (optional)
    Filter by thought type (decision, person_note, insight, meeting, task, idea, reference, lesson).

  topic_filter: str (optional)
    Filter by topic. Matches memories that include this topic in their topics list.

  limit: int (optional, default 20, max 100)
    Maximum number of memories to return.

  offset: int (optional, default 0)
    Number of memories to skip for pagination.

Returns:
  {
    "memories": [
      {
        "id": "uuid",
        "content_preview": "first 100 chars...",
        "type": "insight",
        "topics": ["memory", "architecture"],
        "people": ["Sarah"],
        "importance": 3,
        "source_tool": "claude",
        "created_at": "2026-03-03T10:30:00Z"
      }
    ],
    "total_count": 42,
    "has_more": true,
    "filters_applied": {
      "date_from": "2026-03-01",
      "type_filter": "insight"
    }
  }
```

Implementation notes:
- Results are ordered by `created_at` descending (newest first)
- `total_count` reflects the count with filters applied, not total memories
- `has_more` is true if there are more results beyond the current page
- Date parsing should accept ISO-8601 format and handle invalid dates gracefully
- Topic filter should be case-insensitive partial match
- Must work with all 7 ALMA storage backends via the storage abstraction

### Step 2: Implement alma_browse_timeline

Create the tool for chronological timeline browsing:

```
Tool: alma_browse_timeline
Description: Browse memories chronologically, grouped by day.

Parameters:
  date: str (optional, ISO-8601, default: today)
    The center date for the timeline view.

  range_days: int (optional, default 7, max 30)
    Number of days to include in the timeline.

  direction: str (optional, default "before")
    Which direction to look from the center date.
    "before" -- show range_days before the date
    "after" -- show range_days after the date
    "around" -- show range_days/2 before and after the date

Returns:
  {
    "timeline": [
      {
        "date": "2026-03-03",
        "count": 5,
        "memories": [
          {
            "id": "uuid",
            "content_preview": "first 100 chars...",
            "type": "insight",
            "topics": ["memory"],
            "importance": 3,
            "created_at": "2026-03-03T10:30:00Z"
          }
        ]
      },
      {
        "date": "2026-03-02",
        "count": 3,
        "memories": [...]
      }
    ],
    "date_range": {
      "from": "2026-02-25",
      "to": "2026-03-03"
    },
    "total_count": 28
  }
```

Implementation notes:
- Timeline entries are ordered chronologically (newest date first)
- Within each day, memories are ordered by time (newest first)
- Days with no memories are omitted from the timeline
- The `date_range` shows the actual range queried, not just dates with results
- Must work with all 7 storage backends

### Step 3: Implement Storage Backend Queries

Both tools need to query stored memories with filtering. This likely requires:
- Adding date-range query support to the storage backend abstraction if not already present
- Adding metadata-based filtering (by type, by topic) to the storage layer
- Ensuring pagination is handled at the storage level (not in-memory filtering)

Check `alma/storage/base.py` to see what query capabilities already exist. If date/metadata filtering is not supported, extend the storage abstraction with backward-compatible methods.

### Step 4: Write Tests

Write comprehensive tests for both tools:

**alma_list_memories tests:**
- List with no filters returns recent memories
- Date filtering works (date_from, date_to, both)
- Type filtering works
- Topic filtering works (case-insensitive)
- Multiple filters applied simultaneously
- Pagination: limit and offset work correctly
- has_more flag is correct
- Empty results return gracefully
- Invalid date format handling

**alma_browse_timeline tests:**
- Default parameters (today, 7 days, before)
- Custom date and range
- Direction: before, after, around
- Empty days are omitted
- date_range is correct
- Memories are grouped by day correctly
- Memories within a day are time-ordered

Use `alma.testing.MockStorage` for unit tests. Pre-populate with test memories.

## Output

- **Tools:** Two new functions in `alma/mcp/tools.py`:
  - `alma_list_memories`
  - `alma_browse_timeline`
- **Tests:** Additional tests in `tests/unit/test_mcp_capture_tools.py`

## Gate

- [ ] `alma_list_memories` supports date filtering (date_from, date_to)
- [ ] `alma_list_memories` supports type filtering
- [ ] `alma_list_memories` supports topic filtering (case-insensitive)
- [ ] Pagination works correctly (limit, offset, has_more)
- [ ] `alma_browse_timeline` returns memories grouped by day in chronological order
- [ ] `alma_browse_timeline` supports before, after, and around directions
- [ ] Both tools work with all 7 storage backends (via storage abstraction)
- [ ] Both tools are registered with MCP server
- [ ] All tests pass
- [ ] Code follows ALMA patterns: same registration, parameter, response patterns as existing tools
