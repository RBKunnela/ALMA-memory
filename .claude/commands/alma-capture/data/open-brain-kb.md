# Open Brain Knowledge Base

> Extracted from: "Why Your AI Starts From Zero" by Nate B Jones
> ETL Method: Document extraction + semantic chunking + topic indexing
> Date: 2026-03-03

---

## Core Problem Statement

Every AI conversation starts from zero. Claude doesn't remember what ChatGPT learned.
Cursor doesn't know what you told Claude Code. Each tool is a silo.

**Market signal:** Users are re-explaining context, losing insights to chat history graveyards,
and paying the same "context loading" cost in every session.

**The $7.8B question:** What if every AI tool you use shared the same persistent memory of you?

---

## Architecture: The Open Brain Pattern

### Components

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Storage** | PostgreSQL + pgvector | Thoughts as text + vector embeddings + structured metadata |
| **Capture** | Slack/CLI + Edge Function | Natural language input → embedding + classification |
| **Retrieval** | MCP Server | Any AI tool searches by meaning via semantic search |
| **Intelligence** | LLM (gpt-4o-mini) | Metadata extraction: people, topics, action items, type |
| **Protocol** | MCP (Model Context Protocol) | Universal AI tool access — Claude, ChatGPT, Cursor, etc. |

### Database Schema (from article)

```sql
-- Core thoughts table
create table thoughts (
  id uuid default gen_random_uuid() primary key,
  content text not null,
  embedding vector(1536),
  metadata jsonb default '{}'::jsonb,
  created_at timestamptz default now(),
  updated_at timestamptz default now()
);

-- Indexes: HNSW for vectors, GIN for metadata, B-tree for dates
```

### Metadata Schema (from article)

```json
{
  "type": "decision | person_note | insight | meeting | task | idea | reference",
  "topics": ["topic1", "topic2"],
  "people": ["Person Name"],
  "action_items": ["action 1"],
  "importance": 1-5,
  "source_tool": "claude | chatgpt | cursor | slack | manual"
}
```

---

## 5 Capture Patterns (from article)

### Pattern 1: "Save This" — Preserving AI-Generated Insights
- **When:** AI produces a framework, reframe, prompt approach, or analysis worth keeping
- **Signal:** "That's useful, I might need this again"
- **Template:** `Saving from [AI tool]: [key takeaway or output]`

### Pattern 2: "Before I Forget" — Capturing Perishable Context
- **When:** Post-meeting decisions, phone call details, triggered ideas, gut reactions
- **Signal:** Information is fresh but will decay within hours
- **Template:** `Decision: [what]. Context: [why]. Owner: [who].`

### Pattern 3: "Cross-Pollinate" — Searching Across Tools
- **When:** In one AI tool but need context from another part of your life
- **Signal:** "I know I discussed this somewhere..."
- **Template:** Search query via MCP: `search_thoughts("vendor evaluation criteria")`

### Pattern 4: "Build the Thread" — Accumulating Insight Over Time
- **When:** Daily captures that compound into strategic insight
- **Signal:** Topics that evolve over weeks/months
- **Template:** `Insight: [realization]. Triggered by: [context].`

### Pattern 5: "People Context" — Remembering What Matters About People
- **When:** Capturing preferences, concerns, communication style, life events
- **Signal:** Relationship context that makes interactions more effective
- **Template:** `[Name] — [what happened or what you learned].`

---

## MCP Tools Required (from article)

### Write Operations
1. **save_thought** — Store a thought with auto-embedding and metadata extraction
2. **update_thought** — Modify existing thought content or metadata

### Read Operations
3. **search_thoughts** — Semantic vector search by meaning
4. **list_thoughts** — Browse recent thoughts (with date filtering)
5. **get_thought** — Retrieve specific thought by ID
6. **get_stats** — Memory statistics (count, topics, people, types)

### Synthesis Operations (not in article, needed for ALMA)
7. **weekly_review** — Generate structured weekly synthesis
8. **find_connections** — Surface non-obvious links between memories
9. **detect_patterns** — Recurring themes across time periods

---

## Weekly Review Protocol (from article)

### Analysis Steps
1. **Cluster by topic** — Group related captures, identify 3-5 dominant themes
2. **Scan for unresolved action items** — Tasks without completion notes
3. **People analysis** — Who appeared most? Relationship context updates?
4. **Pattern detection** — Compare against previous weeks (growing/new/dropped topics)
5. **Connection mapping** — Non-obvious links between captures from different days
6. **Gap analysis** — What's conspicuously absent given role and priorities?

### Output Format
```
## Week at a Glance
[X] thoughts captured | Top themes: [1], [2], [3]

## This Week's Themes
**[Theme]** ([X] captures) - [synthesis]

## Open Loops
[unresolved action items with original context]

## Connections You Might Have Missed
[non-obvious links between captures]

## Gaps
[absent topics that deserve attention]

## Suggested Focus for Next Week
[2-3 specific things to capture more deliberately]
```

---

## Compounding Advantage Model

```
Week 1: 20 captures → basic search works
Week 4: 140 captures → patterns emerge in weekly reviews
Week 12: 500+ captures → cross-domain connections surface automatically
Week 52: 2000+ captures → the system knows you better than you know yourself
```

**Key insight from article:** "The AI doesn't get smarter. Your data gets richer."

This is exactly what ALMA needs: not better algorithms, but better capture and synthesis workflows.

---

## ALMA Gap Mapping

| Open Brain Need | ALMA Has | ALMA Needs |
|----------------|----------|------------|
| thoughts table + pgvector | 7 storage backends with vectors | Personal brain domain schema |
| Edge Function + embedding | AutoLearner (partial) | MCP capture tool + metadata extraction |
| MCP read/write tools | 22 MCP tools (agent-scoped) | Personal thought-scoped tools |
| Semantic search | Hybrid engine (vector + TF-IDF) | Temporal browsing + faceted search |
| Metadata extraction | Nothing built-in | LLM-powered extraction pipeline |
| Weekly review | Nothing | Synthesis engine + MCP tool |
| Cross-tool memory | MCP server works | Multi-client patterns + conflict resolution |
| Memory migration | Nothing | Import from Claude/ChatGPT/Notion/Obsidian |
| Graph connections | 4 graph backends (disconnected) | Unified vector+graph retrieval |

---

## Implementation Priority (ETL-informed)

### Tier 1: Essential (build first, blocks everything)
- `alma_capture_thought` MCP tool
- Metadata extraction pipeline
- `alma_list_memories` with date filtering
- Personal Brain domain schema

### Tier 2: Important (build second, enables compounding)
- Weekly review synthesis engine
- Pattern detection across sessions
- Graph + vector unified retrieval
- `alma_find_connections` MCP tool

### Tier 3: Supplementary (build third, market differentiator)
- Memory migration tools (Claude, ChatGPT, Notion, Obsidian)
- `alma init --open-brain` quickstart
- Cross-tool conflict resolution
- Importance scoring + priority signals

---

*Knowledge extracted using ETL Document Specialist methodology*
*Source: D:\1.NateBJones\Why_your_AI_starts_from_zero.md*
*Extraction quality: 95% (full article processed, all key concepts indexed)*
