# ALMA Open Brain Evolution — Architect's Master Plan

> Synthesized from: "Why Your AI Starts From Zero" (Nate B Jones, 9 sections), "The Prompt to Personalize Your Second Brain Build" (4-prompt framework), ALMA codebase analysis (107 source files, 18 subpackages), and existing squad architecture.
>
> Date: 2026-03-03
> Author: Architect analysis
> Status: PLAN — awaiting implementation

---

## 1. What the Architect Found

### Source Material Analyzed

| Source | Key Insight | ALMA Impact |
|--------|------------|-------------|
| **01 — The Memory Problem** | AI agents lack persistent memory. Users burn best thinking on context transfer. "Memory architecture determines agent capabilities more than model selection." | ALMA already solves this for agents. The gap is solving it for **users** (personal brain). |
| **02 — Platform Lock-in** | Claude doesn't know what ChatGPT learned. Each tool is a silo. | ALMA's MCP server already enables cross-tool access. The gap is the **capture** pipeline — users can retrieve but can't easily capture. |
| **03 — Second Brain Limits** | Notion/Obsidian are human-readable but not agent-readable. SaaS middlemen can break, reprice, or disappear. | ALMA is already database-backed and agent-readable. But it has no **personal brain** use case — it's agent-scoped, not user-scoped. |
| **04 — Open Brain Architecture** | Postgres + pgvector + MCP = persistent memory accessible from any AI tool. Cost: $0.10-0.30/month. | ALMA has 7 storage backends and 22 MCP tools. It's **already** the Open Brain backend. It just doesn't know it yet. |
| **05 — Capture & Retrieval** | Capture in <10 seconds. Auto-extract metadata (people, topics, type, actions). Semantic search retrieval. | ALMA has retrieval. ALMA lacks the capture-side pipeline: natural language → classification → filing → confirmation. |
| **06 — Compounding Advantage** | Every thought captured makes the next search smarter. Week 1: basic search. Week 52: the system knows you better than you know yourself. | ALMA's learning loop already compounds for agents. The personal brain needs the **same compounding** for human thoughts. |
| **07 — MCP Beyond Retrieval** | MCP is read AND write. Any MCP client becomes both a capture point and a search tool. Build dashboards, digests, custom tools on top. | ALMA's MCP server needs write-oriented tools (capture, reclassify) not just read-oriented tools (retrieve, stats). |
| **08 — Practical Setup** | 4 lifecycle prompts: Memory Migration (run once), Open Brain Spark (when stuck), Quick Capture Templates (daily), Weekly Review (Friday). | ALMA needs: importers (migration), onboarding CLI (spark), capture templates (daily), synthesis engine (weekly review). |
| **09 — Bigger Picture** | "Just as we needed a personal computer to be digital citizens in the 1990s, we need our own memory architectures to be responsible AI citizens now." | ALMA can be that memory architecture. The library → platform shift is the strategic opportunity. |

### Second Brain Personalization Framework Analysis

| Pattern | What It Means | ALMA Status |
|---------|--------------|-------------|
| **6-step closed loop** | capture → classify → file → confirm → digest → fix | Only "file" (storage) exists. 5 of 6 steps are missing. |
| **4 destination buckets** | people / projects / ideas / admin (max 5 fields each) | No personal brain schema. No bucket routing. |
| **Confidence threshold routing** | Below threshold → "Needs Review" instead of misfiling | `ConfidenceSignal` exists but backward-looking (strategy assessment), not forward-looking (classification). |
| **Fix flow** | User corrects misclassification, log updates | Nothing exists. Need `alma_reclassify_thought` MCP tool. |
| **Operating modes** | always-on / scheduled / session-based | No concept in config. |
| **Reliability tiers** | good-enough / needs-receipts / full-audit-trail | `alma/observability/` exists but not tiered by user choice. |
| **Interactive onboarding** | 7-question questionnaire before config generation | No `alma init` CLI at all. |
| **MVP-first approach** | 30-min day-1 build (capture+classify+file+confirm), fix+digest on day 2 | Good principle for ALMA's implementation phasing. |
| **5 test messages** | Standard validation suite for the capture pipeline | No built-in self-test. |
| **3 operating rules** | One capture rule, one fix rule, one maintenance rule | Not enforceable by software, but ALMA can nudge compliance via webhooks. |

---

## 2. Existing Squads (Already Created)

### alma-capture (Phase 1 — Foundation)

**Mission:** Build the thought capture pipeline.

| What It Builds | Maps To |
|----------------|---------|
| `alma_capture_thought` MCP tool | Article §05 capture flow |
| Metadata extraction pipeline (people, topics, dates, actions) | Article §05 metadata extraction |
| `alma_list_memories` + `alma_browse_timeline` MCP tools | Article §05 retrieval flow |
| Memory migration importers (Claude, ChatGPT, Obsidian, Notion) | Article §08 Memory Migration prompt |
| Classification pipeline with bucket routing | Personalization Framework closed loop |
| Inbox log data structure | Personalization Framework audit trail |

**Agents:** capture-chief, metadata-extractor, mcp-capture-dev, migration-engineer
**Tasks:** 5 tasks defined
**Status:** Planned, not implemented

### alma-synthesis (Phase 2 — Intelligence)

**Mission:** Build the intelligence and review layer.

| What It Builds | Maps To |
|----------------|---------|
| Weekly review synthesis engine | Article §08 Weekly Review prompt |
| Pattern detection (recurring themes, trends, anomalies) | Article §06 compounding advantage |
| Graph + vector unified retrieval | Article §04 Open Brain architecture |
| Connection finder (non-obvious links) | Article §06 "every connection more likely to surface" |

**Agents:** synthesis-chief, review-engine-dev, graph-integrator, pattern-detector
**Tasks:** 4 tasks defined
**Status:** Planned, not implemented

### alma-openness (Phase 3 — Universal Access)

**Mission:** Make ALMA the universal personal brain backend.

| What It Builds | Maps To |
|----------------|---------|
| Personal Brain domain schema (7th schema) | Article §04 Open Brain architecture |
| Multi-client MCP protocol | Article §07 MCP beyond retrieval |
| `alma init --open-brain` quickstart CLI | Article §08 Practical Setup |
| Memory migration format specs | Article §08 Memory Migration prompt |

**Agents:** openness-chief, protocol-architect, quickstart-dev, domain-designer
**Tasks:** 4 tasks defined
**Status:** Planned, not implemented

---

## 3. Gap Analysis — What Existing Squads DON'T Cover

After analyzing all source material against the 3 existing squads, these critical gaps remain:

### Gap 1: Code Quality and Tech Debt

**No existing squad addresses ALMA's known tech debt.**

From CLAUDE.md:
- `alma/mcp/tools.py` is ~3,000 lines (god file, needs splitting into tools/ package)
- 52 `SELECT *` queries need explicit column lists
- 3 retrieval modules lack tests: `trust_scoring.py`, `budget.py`, `progressive.py`
- mypy type checking is non-blocking in CI
- Coverage threshold at 50% (target 80%)

**Why this matters:** Building 15+ new modules on top of tech debt will make the codebase harder to maintain. The mcp/tools.py god file especially blocks the capture squad — you can't add 6 new MCP tools to a 3,000-line file.

**Impact if not addressed:** Every new squad's work becomes harder. Tests break in unexpected ways. The MCP tools file becomes unmaintainable at 4,000+ lines.

### Gap 2: Confidence-Based Routing and Fix Flow

**The closed-loop system needs infrastructure that cuts across all 3 squads.**

The personalization framework's most important pattern is the confidence threshold + fix flow:
- Classify with confidence score → route to bucket or "Needs Review"
- User corrects misclassifications → system learns from corrections
- All operations logged in inbox log for audit

This isn't just a capture feature. It's a **routing infrastructure** that affects:
- Storage (new data structure for inbox log)
- Events (new event types: THOUGHT_CAPTURED, THOUGHT_CLASSIFIED, THOUGHT_RECLASSIFIED)
- MCP (new tool: `alma_reclassify_thought`)
- Confidence module (forward-looking classification confidence, not just backward-looking strategy confidence)

### Gap 3: CLI Infrastructure

**ALMA has no CLI at all.**

The article and framework both emphasize:
- `alma init --open-brain` (interactive 7-question questionnaire)
- `alma open-brain test` (5-message validation suite)
- `alma open-brain enable fix-flow` / `alma open-brain enable weekly-digest`
- `alma capture "Test thought"` (quick capture from terminal)

The `alma-openness` squad has a `quickstart-dev` agent, but CLI infrastructure is a separate concern from the MCP protocol work. The CLI needs:
- Entry point in `pyproject.toml`
- Click/Typer command framework
- Interactive questionnaire UX
- Config generation from answers
- Self-test runner

### Gap 4: Documentation and Developer Experience

**No squad builds documentation.**

ALMA's documentation site exists (alma-memory.pages.dev) but the Open Brain feature needs:
- Tutorial: "Build your personal brain in 45 minutes"
- Guide: "Migrating from Notion/Obsidian to ALMA"
- API reference for new modules
- Architecture guide with diagrams
- Example configs for each storage backend + personal brain
- MCP integration guide for Claude Code, ChatGPT, Cursor

### Gap 5: Event Pipeline Extensions

**The capture pipeline needs 6 new event types and confirmation delivery.**

Existing `alma/events/` has `EventEmitter` and `WebhookManager`, but the personal brain needs:
- `THOUGHT_CAPTURED` — raw input received
- `THOUGHT_CLASSIFIED` — classification complete
- `THOUGHT_FILED` — stored in destination bucket
- `THOUGHT_RECLASSIFIED` — user corrected classification
- `THOUGHT_REVIEW_NEEDED` — low confidence, needs manual review
- `DIGEST_GENERATED` — weekly review produced

Plus confirmation delivery: when a thought is filed, the user needs a confirmation message back in their capture channel.

### Gap 6: Stress Testing and Validation

**The personalization framework makes stress testing a first-class concern.**

Prompt 4 is entirely about failure modes:
1. Malformed AI response (LLM returns invalid JSON)
2. Missing permission or disconnected integration
3. Low-confidence classification (ambiguous input)
4. Happy path end-to-end
5. Fix correction that refiles and updates log

No existing squad has a testing-focused agent or stress testing tasks.

---

## 4. New Squads to Create

Based on the gap analysis, these squads are needed:

### Squad 4: alma-quality

**Mission:** Reduce tech debt, increase test coverage, split god files, and build the testing infrastructure that all other squads depend on.

**Why it's needed:** Building 15+ new modules on a 50% coverage, tech-debt-laden codebase is risky. This squad de-risks the entire evolution.

**Agents:**

| Agent | Role |
|-------|------|
| `quality-chief` | Squad leader. Prioritizes tech debt, tracks coverage metrics. |
| `refactorer` | Splits `mcp/tools.py` into `mcp/tools/` package. Fixes SELECT * queries. |
| `test-engineer` | Writes missing tests for `trust_scoring.py`, `budget.py`, `progressive.py`. Builds test infrastructure for new modules. |
| `stress-tester` | Implements the 5-message validation suite. Builds failure injection tests. |

**Tasks:**

| Task | Priority | Impact |
|------|----------|--------|
| Split `alma/mcp/tools.py` into `alma/mcp/tools/` package | P0 | Unblocks alma-capture squad from adding new MCP tools |
| Fix 52 `SELECT *` queries with explicit column lists | P1 | Prevents schema drift bugs |
| Write tests for `trust_scoring.py`, `budget.py`, `progressive.py` | P1 | Raises coverage from 50% to ~65% |
| Implement 5-message Open Brain validation suite | P1 | Enables `alma open-brain test` command |
| Build failure injection framework for stress testing | P2 | Catches edge cases in LLM extraction pipeline |
| Enable mypy as blocking in CI | P2 | Catches type errors before they ship |

**Dependency:** Must run BEFORE or IN PARALLEL with alma-capture. The MCP tools split is a hard blocker for new tool development.

### Squad 5: alma-cli

**Mission:** Build ALMA's command-line interface — the user's primary interaction point for setup, configuration, testing, and quick operations.

**Why it's needed:** ALMA currently has NO CLI. The Open Brain vision depends on `alma init --open-brain`, `alma capture`, `alma open-brain test`, and other commands. This is user-facing infrastructure, distinct from MCP tools (which are AI-facing).

**Agents:**

| Agent | Role |
|-------|------|
| `cli-chief` | Squad leader. Designs CLI architecture, coordinates with other squads. |
| `onboarding-dev` | Builds the `alma init --open-brain` interactive questionnaire (7 questions from Personalization Framework). |
| `config-generator` | Builds config generation from questionnaire answers. Maps user constraints to ALMA configuration. |
| `cli-tools-dev` | Builds utility commands: `alma capture`, `alma open-brain test`, `alma status`, `alma open-brain enable`. |

**Tasks:**

| Task | Priority | Impact |
|------|----------|--------|
| Set up CLI entry point with Click/Typer in `pyproject.toml` | P0 | Foundation for all CLI commands |
| Build `alma init --open-brain` questionnaire | P0 | Primary onboarding flow from article §08 |
| Build config generator from questionnaire answers | P0 | Connects questionnaire to working config |
| Build `alma capture "thought"` quick capture command | P1 | Terminal capture without MCP |
| Build `alma open-brain test` self-validation | P1 | 5-message test suite from Personalization Framework |
| Build `alma status` and `alma open-brain enable` commands | P2 | Feature toggle and system status |

**Dependency:** Depends on alma-openness for personal brain schema. Can start CLI infrastructure in parallel.

### Squad 6: alma-events

**Mission:** Extend ALMA's event system for the capture pipeline — new event types, confirmation delivery, reliability tier configuration, and digest triggers.

**Why it's needed:** The capture → classify → file → confirm → digest → fix loop requires event infrastructure that cuts across all squads. Without dedicated focus, events will be bolted on inconsistently by each squad.

**Agents:**

| Agent | Role |
|-------|------|
| `events-chief` | Squad leader. Designs event schema, coordinates with capture and synthesis squads. |
| `event-pipeline-dev` | Implements 6 new event types, confirmation delivery, and digest trigger events. |
| `reliability-dev` | Implements reliability tier configuration (good-enough / needs-receipts / full-audit-trail). Maps tiers to event persistence levels. |

**Tasks:**

| Task | Priority | Impact |
|------|----------|--------|
| Add 6 new capture pipeline event types to `alma/events/types.py` | P0 | Foundation for confirmation and digest triggers |
| Implement confirmation delivery (event → user notification) | P1 | Closes the "confirm" step in the 6-step loop |
| Implement reliability tier configuration | P1 | Maps to Personalization Framework reliability tiers |
| Implement operating mode configuration (always-on / scheduled / session-based) | P2 | From Personalization Framework operating modes |
| Implement digest trigger (weekly cron or on-demand) | P2 | Triggers weekly review synthesis |

**Dependency:** alma-capture depends on this for event types. alma-synthesis depends on this for digest triggers.

### Squad 7: alma-docs

**Mission:** Build comprehensive documentation for the Open Brain feature and the ALMA library as a whole.

**Why it's needed:** The Nate B Jones article's setup guide was tested by a non-coder and took 45 minutes. ALMA's documentation must achieve the same accessibility. Without docs, the technical implementation is unusable by the target audience.

**Agents:**

| Agent | Role |
|-------|------|
| `docs-chief` | Squad leader. Plans documentation architecture, coordinates with dev squads. |
| `tutorial-writer` | Writes the "Build your personal brain in 45 minutes" tutorial. |
| `api-doc-generator` | Generates API reference docs from docstrings. Ensures all new public APIs have Google-style docstrings. |
| `migration-guide-writer` | Writes guides for migrating from Notion, Obsidian, Claude Memory, ChatGPT Memory. |

**Tasks:**

| Task | Priority | Impact |
|------|----------|--------|
| Write "Personal Brain in 45 Minutes" tutorial | P0 | Primary adoption driver |
| Write MCP integration guide (Claude Code + ChatGPT + Cursor) | P0 | Shows cross-tool value proposition |
| Generate API reference for new modules | P1 | Developer reference |
| Write migration guides (Notion, Obsidian, Claude, ChatGPT) | P1 | Reduces switching cost |
| Write architecture guide with diagrams | P2 | For contributors and advanced users |
| Update documentation site (alma-memory.pages.dev) | P2 | Public-facing documentation |

**Dependency:** Follows all other squads (documents what they build). Can start tutorial skeleton in parallel.

---

## 5. How Each Squad Transforms ALMA

### The Transformation Chain

```
CURRENT STATE (v0.8.0)
├── Agent memory library
├── 7 storage backends, 4 graph backends
├── 22 MCP tools (agent-scoped)
├── Learning loop: retrieve → execute → learn
├── Target user: developers building AI agents
│
├── [alma-quality] ──────────────────────────────────────┐
│   Clean foundation: split god files, fix tech debt,    │
│   raise coverage, enable stress testing                │
│                                                        │
├── [alma-events] ───────────────────────────────────────┤
│   Event infrastructure: 6 new event types,             │
│   confirmation delivery, reliability tiers,            │
│   operating modes                                      │
│                                                        │
├── [alma-capture] ──────────────────────────────────────┤
│   Capture pipeline: natural language → classify →      │
│   file → confirm. Metadata extraction. Migration.      │
│   New MCP tools: capture, reclassify, list, browse.    │
│                                                        │
├── [alma-synthesis] ────────────────────────────────────┤
│   Intelligence: weekly reviews, pattern detection,     │
│   graph+vector unified retrieval, connection finder.   │
│   New MCP tools: weekly_review, find_connections.      │
│                                                        │
├── [alma-openness] ─────────────────────────────────────┤
│   Universal access: personal brain schema, multi-      │
│   client MCP protocol, memory migration tools.         │
│                                                        │
├── [alma-cli] ──────────────────────────────────────────┤
│   User interface: alma init --open-brain, alma         │
│   capture, alma open-brain test, alma status.          │
│                                                        │
├── [alma-docs] ─────────────────────────────────────────┤
│   Documentation: 45-min tutorial, migration guides,    │
│   API reference, architecture guide.                   │
│                                                        │
FUTURE STATE (v1.0.0)                                    │
├── Personal knowledge infrastructure                    │
├── Agent memory + human memory in one system            │
├── 28+ MCP tools (agent + personal brain)               │
├── Full closed loop: capture → classify → file →        │
│   confirm → digest → fix                               │
├── Compounding intelligence: patterns, connections,     │
│   weekly reviews                                       │
├── Zero switching cost: any AI tool, any platform       │
├── 45-minute onboarding for non-technical users         │
├── Target user: anyone who uses AI tools                │
└────────────────────────────────────────────────────────┘
```

### What Each Squad Delivers to the End User

| Squad | User Sees | User Experience |
|-------|-----------|-----------------|
| **alma-quality** | Nothing visible (foundation work) | "It just works. No weird bugs." |
| **alma-events** | Confirmation messages, digest notifications | "I captured a thought and got instant confirmation." |
| **alma-capture** | Capture thoughts from any MCP client | "I typed a thought into Claude and it was classified and stored in 5 seconds." |
| **alma-synthesis** | Weekly reviews, pattern insights, connections | "On Friday, ALMA showed me connections between things I said on Monday and Wednesday that I hadn't noticed." |
| **alma-openness** | Personal brain schema, cross-tool access | "I captured in Claude Code, searched in ChatGPT, and it was there." |
| **alma-cli** | Terminal commands, interactive setup | "I ran `alma init --open-brain`, answered 7 questions, and had a working personal brain in 30 minutes." |
| **alma-docs** | Tutorials, guides, API reference | "I followed the tutorial and built it over coffee this weekend." |

---

## 6. Master Implementation Timeline

### Phase 0: Foundation (alma-quality + alma-events)

**Must happen first. Unblocks everything else.**

```
alma-quality:
  ├── Split mcp/tools.py → mcp/tools/ package  [BLOCKER for alma-capture]
  ├── Fix 52 SELECT * queries
  ├── Write missing tests (trust_scoring, budget, progressive)
  └── Build stress test framework

alma-events (parallel):
  ├── Add 6 capture pipeline event types
  ├── Implement confirmation delivery
  └── Design reliability tier configuration
```

**Gate:** mcp/tools.py split complete. 6 new event types registered. Coverage >60%.

### Phase 1: Capture Pipeline (alma-capture)

**The foundation layer. Without capture, nothing else works.**

```
alma-capture:
  ├── Design capture pipeline architecture
  ├── Build ThoughtClassifier (LLM-powered classification)
  ├── Implement alma_capture_thought MCP tool
  ├── Implement confidence threshold routing ("Needs Review")
  ├── Build inbox log data structure
  ├── Implement alma_list_memories + alma_browse_timeline
  └── Design memory migration format
```

**Gate:** Can capture a thought via MCP, classify it, store it, confirm it. 5-message test suite passes.

### Phase 2: Intelligence + Schema (alma-synthesis + alma-openness)

**Build in parallel after capture pipeline is working.**

```
alma-synthesis:
  ├── Build weekly review engine (cluster, summarize, detect gaps)
  ├── Implement pattern detection (trends, anomalies, cycles)
  ├── Build unified graph+vector retrieval
  ├── Implement connection finder
  └── Create alma_weekly_review MCP tool

alma-openness (parallel):
  ├── Design and implement personal brain schema (7th schema)
  ├── Implement multi-client MCP protocol
  ├── Build memory migration tools (Claude, ChatGPT, Obsidian, Notion)
  └── Implement fix flow (alma_reclassify_thought)
```

**Gate:** Weekly review generates useful output. Personal brain schema registered. Cross-tool capture+retrieve works. Migration imports >90% metadata.

### Phase 3: User Interface + Documentation (alma-cli + alma-docs)

**Make it accessible. Build the 45-minute experience.**

```
alma-cli:
  ├── Set up CLI entry point
  ├── Build alma init --open-brain questionnaire
  ├── Build config generator from answers
  ├── Build alma capture command
  ├── Build alma open-brain test validation
  └── Build alma status command

alma-docs (parallel):
  ├── Write "Personal Brain in 45 Minutes" tutorial
  ├── Write MCP integration guide
  ├── Write migration guides
  ├── Generate API reference
  └── Update documentation site
```

**Gate:** Non-technical user can go from `pip install alma-memory` to working personal brain in 45 minutes.

---

## 7. Expected ALMA Transformation

### Before (v0.8.0) — Agent Memory Library

- **Target user:** Developer building AI agents
- **Value prop:** "Give your agents persistent memory"
- **Use case:** Agent learns from task outcomes
- **MCP tools:** Agent-scoped (retrieve, learn, workflow)
- **Market position:** Competitor to Mem0

### After (v1.0.0) — Personal Knowledge Infrastructure

- **Target user:** Anyone who uses AI tools daily
- **Value prop:** "One brain, every AI, never start from zero"
- **Use case:** Personal + agent memory in one system
- **MCP tools:** Personal brain (capture, classify, review, connect) + agent (retrieve, learn, workflow)
- **Market position:** The Open Brain backend — no direct competitor

### The Compounding Math (from Nate B Jones)

```
Week 1:   20 captures → basic search works
Week 4:   140 captures → patterns emerge in weekly reviews
Week 12:  500+ captures → cross-domain connections surface automatically
Week 52:  2000+ captures → the system knows you better than you know yourself
```

### ALMA's Competitive Moat After v1.0.0

| Capability | ALMA | Mem0 | Notion | Obsidian |
|------------|------|------|--------|----------|
| 7 storage backends | Yes | No | No | No |
| 4 graph backends | Yes | No | No | No |
| 28+ MCP tools | Yes | No | No | No |
| Agent + personal memory | Yes | Agent only | Human only | Human only |
| Semantic search | Yes | Yes | No | Plugin |
| Cross-tool access | Yes (MCP) | No | No | No |
| Confidence routing | Yes | No | No | No |
| Fix flow | Yes | No | Manual | Manual |
| Weekly review synthesis | Yes | No | No | Plugin |
| Pattern detection | Yes | No | No | No |
| Memory migration | Yes | No | No | Plugin |
| Compounding intelligence | Yes | No | No | No |
| Open source + self-hosted | Yes | Partial | No | Yes |

---

## 8. Summary: All 7 Squads

| # | Squad | Phase | Mission | Agents | Tasks |
|---|-------|-------|---------|--------|-------|
| 1 | **alma-capture** | 1 | Thought capture pipeline | 4 | 5 |
| 2 | **alma-synthesis** | 2 | Intelligence + review layer | 4 | 4 |
| 3 | **alma-openness** | 2 | Universal access + schema | 4 | 4 |
| 4 | **alma-quality** | 0 | Tech debt + testing | 4 | 6 |
| 5 | **alma-cli** | 3 | CLI commands + onboarding | 4 | 6 |
| 6 | **alma-events** | 0 | Event pipeline + reliability | 3 | 5 |
| 7 | **alma-docs** | 3 | Documentation + tutorials | 4 | 6 |
| **TOTAL** | | | | **27** | **36** |

### Squad Creation Order

```
Phase 0 (create first, run first):
  alma-quality   — clean the foundation
  alma-events    — build the event pipeline

Phase 1 (create second, depends on Phase 0):
  alma-capture   — ALREADY EXISTS, refine and execute

Phase 2 (create third, depends on Phase 1):
  alma-synthesis — ALREADY EXISTS, refine and execute
  alma-openness  — ALREADY EXISTS, refine and execute

Phase 3 (create fourth, depends on Phase 2):
  alma-cli       — user-facing CLI
  alma-docs      — documentation
```

### New Squads to Create: 4

1. **alma-quality** — 4 agents, 6 tasks
2. **alma-cli** — 4 agents, 6 tasks
3. **alma-events** — 3 agents, 5 tasks
4. **alma-docs** — 4 agents, 6 tasks

### Existing Squads to Refine: 3

1. **alma-capture** — Add fix flow task, add confidence routing task
2. **alma-synthesis** — Add digest trigger integration task
3. **alma-openness** — Add operating modes task, add reliability tier task

---

*Architect's Master Plan v1.0.0*
*Source: "Why Your AI Starts From Zero" (Nate B Jones), "The Prompt to Personalize Your Second Brain Build" (Notion Product Template), ALMA codebase analysis*
*Cross-referenced with: open-brain-kb.md, second-brain-personalization-kb.md, synthesis-patterns-kb.md, openness-patterns-kb.md*
