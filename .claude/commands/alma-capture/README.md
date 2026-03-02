# alma-capture

**Squad:** alma-capture
**Phase:** 1 (Foundation -- Thought Capture)
**Mission:** Build the thought capture pipeline -- the foundation layer. Without capture, nothing else works.

---

## Overview

The alma-capture squad is responsible for Phase 1 of the ALMA Open Brain evolution. We build the core ingestion pipeline that transforms raw thoughts into structured, embedded, retrievable memories. Every other squad depends on our output -- if capture does not work, nothing else works.

The capture pipeline implements the flow: **capture -> embed -> extract metadata -> store -> confirm**.

See [data/open-brain-kb.md](data/open-brain-kb.md) for the full Open Brain architecture and vision.

---

## Agents

| Agent | Role | File |
|-------|------|------|
| **capture-chief** | Squad orchestrator -- designs pipeline, coordinates agents, quality gates | [agents/capture-chief.md](agents/capture-chief.md) |
| **metadata-extractor** | LLM-powered metadata extraction specialist | [agents/metadata-extractor.md](agents/metadata-extractor.md) |
| **mcp-capture-dev** | MCP tool developer -- implements capture/list/browse tools | [agents/mcp-capture-dev.md](agents/mcp-capture-dev.md) |
| **migration-engineer** | Memory migration specialist -- imports from external memory stores | [agents/migration-engineer.md](agents/migration-engineer.md) |

## Tasks

| Task | Agent | Depends On | File |
|------|-------|------------|------|
| Design Capture Pipeline | capture-chief | -- | [tasks/design-capture-pipeline.md](tasks/design-capture-pipeline.md) |
| Build Metadata Extractor | metadata-extractor | design-capture-pipeline | [tasks/build-metadata-extractor.md](tasks/build-metadata-extractor.md) |
| Implement Capture Tool | mcp-capture-dev | build-metadata-extractor | [tasks/implement-capture-tool.md](tasks/implement-capture-tool.md) |
| Implement List Memories | mcp-capture-dev | implement-capture-tool | [tasks/implement-list-memories.md](tasks/implement-list-memories.md) |
| Design Migration Format | migration-engineer | -- | [tasks/design-migration-format.md](tasks/design-migration-format.md) |

## Task Dependency Graph

```
design-capture-pipeline
        |
        v
build-metadata-extractor        design-migration-format
        |                              (parallel)
        v
implement-capture-tool
        |
        v
implement-list-memories
```

---

## Squad Structure

```
alma-capture/
  README.md                              # This file
  agents/
    capture-chief.md                     # Squad orchestrator
    metadata-extractor.md                # LLM metadata extraction specialist
    mcp-capture-dev.md                   # MCP tool developer
    migration-engineer.md                # Memory migration specialist
  tasks/
    design-capture-pipeline.md           # Architecture design task
    build-metadata-extractor.md          # Metadata extraction module
    implement-capture-tool.md            # alma_capture_thought MCP tool
    implement-list-memories.md           # alma_list_memories + alma_browse_timeline
    design-migration-format.md           # Migration format + importer architecture
  checklists/
    capture-quality-gate.md              # Phase 1 completion quality gate
  data/
    open-brain-kb.md                     # Shared knowledge base (Open Brain architecture)
  templates/
    metadata-schema-tmpl.md              # Metadata extraction schema template
  workflows/
    alma-open-brain-evolution.yaml       # Master workflow (all 3 squads)
```

---

## Key References

- **Open Brain KB:** [data/open-brain-kb.md](data/open-brain-kb.md) -- extracted architecture and patterns from Nate B Jones' article
- **Master Workflow:** [workflows/alma-open-brain-evolution.yaml](workflows/alma-open-brain-evolution.yaml) -- orchestration across all 3 squads
- **ALMA Codebase:** `alma/` -- core library (107 source files, 18 subpackages)
- **Existing MCP Tools:** `alma/mcp/tools.py` -- 22 existing tools to follow as patterns
- **Existing Extraction:** `alma/extraction/` -- auto-learning from conversations (extend, not replace)

## Phase 1 Success Criteria

1. Can capture a thought via MCP and retrieve it
2. Metadata is automatically extracted (type, topics, people, action_items, importance)
3. Can list recent memories with date/type/topic filtering
4. Can browse memories chronologically
5. All existing 1,210 tests still pass
6. New test coverage >80% for new modules

## How to Start

1. Read [data/open-brain-kb.md](data/open-brain-kb.md) for full context
2. Start with [tasks/design-capture-pipeline.md](tasks/design-capture-pipeline.md) -- everything depends on it
3. Follow the dependency graph above
4. Use [checklists/capture-quality-gate.md](checklists/capture-quality-gate.md) to verify completion
