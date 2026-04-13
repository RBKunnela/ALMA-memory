# ALMA + MemPalace Merge - Complete Guide

## Three Documents. One Goal. Ship in 6 weeks.

---

## Document 1: Honest Comparison (mempalace_vs_alma_analysis.md)

**What it tells you:**
- MemPalace is shipping. ALMA is designing.
- 96.6% accuracy on LongMemEval (proven). ALMA's scoping is unproven but valuable.
- You need to merge because ALMA's ideas are sound, but ALMA alone won't ship.

**Key takeaway:**
Your governance model is sophisticated. MemPalace's storage is proven. Combine them.

---

## Document 2: Merge Strategy (alma_mempalace_merge_strategy.md)

**What it tells you:**
- Exact architecture post-merge (includes SQL schema)
- 6-sprint execution plan (6 weeks to v1.0)
- Critical success metrics (95%+ on benchmarks)
- Code structure after merge

**Key sections:**
1. **Architecture Merge Plan** - Keep MemPalace's retrieval, add ALMA's scoping
2. **Storage Unification** - Single schema supporting both systems
3. **Memory Type Consolidation** - How to map halls → ALMA types
4. **Execution Roadmap** - Week by week what to build
5. **Risk Assessment** - What could go wrong and how to mitigate

**SQL Schema (ready to use):**
```sql
CREATE TABLE memories (
  id TEXT PRIMARY KEY,
  wing TEXT, hall TEXT, room TEXT,      -- MemPalace
  agent_id TEXT, memory_type TEXT,      -- ALMA
  content TEXT,
  embedding BYTEA,
  metadata JSONB,
  created_at TIMESTAMP,
  ...
);
```

---

## Document 3: Week 1 Execution (week1_execution_guide.md)

**What it tells you:**
- Exactly what to code this week
- 6 tasks with daily breakdown
- Definition of done for each task
- Deliverables by EOD each day

**Week 1 Tasks:**
1. Understand MemPalace schema (Day 1)
2. Design ALMA memory table (Day 1-2)
3. Create migration script (Day 2-3)
4. Adapt retrieval for scoping (Day 3-4)
5. Agent scoping + learning (Day 4-5)
6. Integration test + benchmark (Day 5-6)

**Success criteria for Week 1:**
- All MemPalace data migrated (zero loss)
- Retrieval accuracy >= 95% (was 96.6%)
- Agent scoping enforced
- All tests passing

---

## Document 4: Code Patterns (technical_patterns_reference.md)

**What it tells you:**
- Copy/paste code patterns from MemPalace
- What to adapt vs what to take as-is
- Testing structure from MemPalace
- MCP server patterns

**Key patterns:**
1. ChromaDB collection access
2. Embedding (all-MiniLM-L6-v2, 384 dims)
3. Configuration management
4. Metadata filtering for search
5. Query sanitization
6. Deduplication
7. Conversation mining
8. Testing fixtures
9. MCP server tools
10. Benchmark structure

**Files to copy as-is:**
```
mempalace/query_sanitizer.py → alma/core/search/
mempalace/normalize.py → alma/core/mining/
mempalace/spellcheck.py → alma/core/mining/
```

**Files to adapt:**
```
mempalace/backends/chroma.py → alma/storage/ (add to schema)
mempalace/searcher.py → alma/core/search/ (add scoping)
mempalace/cli.py → alma/cli.py (add governance commands)
```

---

## The Full Picture

### What You Have
- ALMA: Excellent architecture, unvalidated, 53 commits, 2 stars
- MemPalace: Proven retrieval, 96.6% accuracy, 263 commits, tested

### What You're Building
- ALMA + MemPalace: Proven retrieval + scoped learning + benchmarks

### Timeline
- Week 1: Storage unification + basic scoping
- Week 2: Agent governance layer
- Week 3: Multi-agent sharing
- Week 4: Anti-pattern learning
- Week 5: Memory consolidation
- Week 6: MCP integration + ship v1.0

### Success Looks Like
- LongMemEval: 95%+ (vs baseline 96.6%)
- ConvoMem: 90%+ (vs baseline 92.9%)
- Agent scoping enforced
- Multi-agent sharing working
- Anti-patterns guiding decisions
- v1.0 on PyPI with 11K lines of code + 11K lines of tests

---

## For Claude Code

### Repository Setup
```bash
cd alma-work
git clone https://github.com/RBKunnela/ALMA-memory.git
cp -r mempalace-repo/mempalace ./mempalace-reference

# Start work on Week 1
git checkout -b feature/mempalace-merge-week1
```

### What's Where
- `alma/` - Your new merged code
- `mempalace-reference/` - Reference implementation
- `.alma/schema.py` - Unified schema (start here Week 1)
- `tests/` - All tests (11K lines by end)
- `benchmarks/` - MemPalace benchmarks adapted

### Monday Morning
1. Read `mempalace-reference/README.md` (understand 96.6% baseline)
2. Read `mempalace-reference/tests/test_mcp_server.py` (understand drawer format)
3. Create `alma/schema.py` (Memory + Agent dataclasses)
4. Create migration script

### Friday EOD
1. MemPalace data migrated
2. Retrieval working at 95%+
3. Agent scoping validated
4. All tests passing

---

## The Hard Conversations You Need to Have

### With Yourself
1. "Am I going to finish this in 6 weeks or redesign it again?"
   - Answer: You already designed it. Execute.

2. "What if scoping breaks accuracy?"
   - Answer: Benchmark after each sprint. If regression > 2%, revert that sprint.

3. "What if PostgreSQL takes too long?"
   - Answer: Ship with SQLite. PostgreSQL is Sprint 2, not MVP.

### With Your Team (if any)
1. "MemPalace is already doing retrieval at 96%+"
   - Response: We're not competing with them. We're extending them.

2. "Why not just use MemPalace?"
   - Response: MemPalace has no scoping. Agents learn off-domain. We fix that.

3. "How confident are you in the timeline?"
   - Response: 6 weeks if focused. 12+ weeks if we redesign. We're not redesigning.

---

## What Could Go Wrong

### High Risk
1. **Scoping adds latency** - Solution: Index on (agent_id, memory_type)
2. **Multi-agent queries expensive** - Solution: Pre-cache sharing graphs
3. **Migration loses metadata** - Solution: Backup first, test on small dataset

### Medium Risk
1. **PostgreSQL takes longer than Sprint 2** - Solution: Keep Chroma as default
2. **Anti-pattern classification needs LLM** - Solution: Make optional, fallback to heuristics

### What Won't Go Wrong
1. Schema design - you have the SQL
2. Storage patterns - you have MemPalace's code
3. Benchmarking - you have test data (LongMemEval, ConvoMem)

---

## Decision: What Actually Ships in V1.0

### Must Have
- ChromaDB + SQLite backend
- Agent scoping (can_learn/cannot_learn)
- Basic multi-agent sharing (inherit_from/share_with)
- Migration from MemPalace
- MCP server (Claude Code integration)
- Benchmarks showing 95%+ accuracy

### Defer to V1.1
- PostgreSQL backend
- LLM-based anti-pattern classification
- Domain factory pre-built schemas
- Qdrant/Pinecone backends
- Event webhooks

### Archive (Low Priority)
- AAAK compression (too complex, 12.4% regression)
- Graph database (too much surface area)
- TypeScript SDK (Python first)

---

## One More Honest Thing

You spent months designing ALMA's architecture. It's good. But good architecture is worthless if nobody uses it.

MemPalace shipped something simpler and got it working. Now you get to add the sophistication on top.

The question is: will you finish it, or will you redesign it again?

Your move.

---

## Reading Order

1. **Start here:** mempalace_vs_alma_analysis.md (15 min read)
   - Understand why you need to merge
   
2. **Then read:** alma_mempalace_merge_strategy.md (30 min read)
   - Understand what you're building
   
3. **Then read:** week1_execution_guide.md (20 min read)
   - Understand what to code Monday
   
4. **Keep handy:** technical_patterns_reference.md (reference)
   - Copy/paste when you need specific patterns

5. **Final step:** Open Claude Code and start Task 1 Day 1

---

## Success Metrics (Pick One)

### Conservative
- v1.0 ships with 95% retrieval accuracy
- Agent scoping works
- All MemPalace data migrated
- 1 real user testing

### Ambitious
- v1.0 ships with 95%+ retrieval accuracy
- Multi-agent sharing working
- Anti-pattern learning validated
- PostgreSQL backend done
- 5 real users testing

### You Should Target
Conservative + PostgreSQL backend done (halfway to ambitious).

That's 6 weeks of focused work.

---

## Files You Now Have

1. `mempalace_vs_alma_analysis.md` - The honest assessment
2. `alma_mempalace_merge_strategy.md` - The complete architecture plan
3. `week1_execution_guide.md` - What to code this week
4. `technical_patterns_reference.md` - Code patterns to copy

All are in `/mnt/user-data/outputs/` for download.

Copy them to your ALMA repo. Share with anyone helping.

---

## The Promise

If you follow this plan and actually execute it (no redesigns, no new features):

- Week 1: Data unified, scoping in place, 95%+ accuracy
- Week 2: Full agent governance working
- Week 3: Multi-agent sharing validated
- Week 4: Anti-patterns guiding decisions
- Week 5: Consolidation reducing storage by 10%+
- Week 6: Ship v1.0 to PyPI with benchmarks

You'll have built the only AI memory system that:
1. Stores everything verbatim (MemPalace)
2. Respects domain boundaries (ALMA)
3. Shares knowledge hierarchically (ALMA)
4. Has proven benchmarks (both)
5. Works offline (MemPalace)
6. Costs nothing (both)

That's a real product. Not a design doc. A product.

Now go code it.
