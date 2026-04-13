# ALMA-memory vs MemPalace: Definitive Comparative Analysis

**Date:** 2026-04-13
**Research Method:** 5 parallel deep-research agents reading actual source code, domain-decoding both architectures, reviewing prior research, and scanning the competitive landscape.
**Prior research (zip files) verdict:** Fundamentally flawed -- evaluated ALMA from README surface metrics, got the codebase state massively wrong. This document replaces all prior analysis.

---

## Executive Summary

ALMA-memory is not the "unvalidated design document" the prior research claimed. It is a **52,058-line production library** with 1,682 passing tests, 7 storage backends, and the most sophisticated learning architecture in the AI agent memory space. MemPalace is a **simpler, more focused tool** (11,263 lines) that went viral (42.5k stars) on benchmark claims that were subsequently **debunked** by the community.

**The merge question is wrong.** ALMA should not absorb MemPalace's codebase. ALMA should absorb **4 specific capabilities** through an adapter layer, then prove itself with published benchmarks.

---

## 1. The Numbers (Code-Verified, Not README-Estimated)

| Metric | ALMA-memory | MemPalace |
|--------|------------|-----------|
| Source lines | 52,058 (107 files, 18 packages) | 11,263 (27 files, flat) |
| Test lines | 38,825 (86 test files) | 8,583 (32 test files) |
| Tests passing | 1,682 passing, 181 skipped, 0 failing | Not published (85% coverage target) |
| Storage backends | 7 (SQLite+FAISS, PostgreSQL+pgvector, Azure Cosmos, Qdrant, Chroma, Pinecone, File) | 1 (ChromaDB only, tightly coupled) |
| Graph backends | 4 (Neo4j, Memgraph, Kuzu, In-Memory) | 1 (SQLite triples) |
| MCP tools | 22 tools + 2 resources (5 modules) | 19+ tools (1 monolith file) |
| Memory types | 5 typed dataclasses (Heuristic, Outcome, AntiPattern, DomainKnowledge, UserPreference) | Untyped metadata dicts |
| Learning system | Auto-heuristic generation, decay, forgetting, anti-patterns | None (passive storage only) |
| Async support | Yes (asyncio.to_thread wrappers) | None |
| Observability | OpenTelemetry + Prometheus + structured logging | Basic logging |
| Dependencies | Many (per backend) | 2 (chromadb, pyyaml) |
| CI/CD | Full (lint, typecheck, security, tests on 3.10/3.11/3.12) | GitHub Actions |
| PyPI version | 0.8.0 alpha | 3.2.0 beta |
| PyPI downloads/mo | 259 | ~76,000 |
| GitHub stars | 22 | 42,500 |
| Published benchmarks | **None** | **Debunked** (96.6% measures ChromaDB, not palace logic) |

---

## 2. What Each System Actually Does (Source Code Reading)

### ALMA: Stores Lessons, Not Just Memories

ALMA's core innovation is the **learning loop**: outcomes feed into heuristics, repeated failures become anti-patterns, confidence decays over time, and memories are forgotten when they stop being useful.

```
Task execution -> Outcome (success/failure record)
                    |
                    v
            After 3+ similar outcomes -> Heuristic (condition -> strategy, with confidence)
            After 2+ similar failures -> AntiPattern (pattern + why_bad + better_alternative)
                    |
                    v
            Retrieval: 4-factor scoring (similarity 0.4 + recency 0.3 + success_rate 0.2 + confidence 0.1)
                    |
                    v
            Decay: strength = e^(-0.693 * days/half_life) * importance_factor
            When strength < 0.1 -> forget
```

**Retrieval sophistication:** 5 cognitive modes (BROAD/PRECISE/DIAGNOSTIC/LEARNING/RECALL), hybrid search (vector + BM25 via RRF), trust scoring per agent, budget-aware token allocation, progressive disclosure (4 levels), verified retrieval with ground-truth checking, TTL cache.

**Scoped multi-agent:** MemoryScope with can_learn/cannot_learn lists, share_with/inherit_from hierarchies. Write isolation, read sharing. No competitor has this.

### MemPalace: Stores Verbatim Text, Organizes by Metaphor

MemPalace's philosophy is "No summaries. Ever." It stores raw text in 800-char chunks and relies on ChromaDB's default embeddings for retrieval.

```
File/Chat -> Normalize (6 formats) -> Chunk (800 chars) -> Route to wing/room -> ChromaDB
                                                                                    |
                                                                                    v
                                                            Search: cosine similarity via ChromaDB
                                                                                    |
                                                                                    v
                                                            4-layer context loading:
                                                              L0: identity (~100 tok)
                                                              L1: essential story (~800 tok)
                                                              L2: on-demand (~500 tok)
                                                              L3: deep search (unlimited)
```

**What it does well:** Chat format normalization (6 formats), query sanitization (prevents system prompt contamination), the palace metaphor (wings/rooms/halls) adds navigational structure that improves retrieval by ~34% via metadata filtering, zero-API local operation.

**What it lacks:** No learning, no confidence, no decay, no forgetting, no consolidation, no async, no multi-agent, no observability, no type safety, single storage backend.

---

## 3. The Benchmark Truth

### MemPalace Benchmarks: Debunked

Independent developers (Issue #214) proved:
- The 96.6% LongMemEval R@5 measures **raw ChromaDB retrieval**, not any palace-specific functionality
- Modes that use palace logic score **LOWER**: room-based boosting = 89.4%, AAAK compression = 84.2%
- The benchmark searched only ~50 pre-filtered sessions, not the full corpus
- Full-corpus retrieval (19,195 sessions) drops to **30% R@5**
- BM25 keyword search alone achieves 93.8% on the artificial haystack
- "30x lossless compression" actually has 12.4% quality degradation
- "Contradiction detection" feature doesn't exist in the codebase
- Maintainers acknowledged all of this and retired the headline metric

### ALMA Benchmarks: Non-Existent

ALMA has zero published benchmark numbers. This is the single biggest gap. The architecture is sophisticated but unproven.

### Actual Market Leaders (Verified Scores)

| System | LongMemEval | LoCoMo | Notes |
|--------|------------|--------|-------|
| Hindsight | **91.4%** | - | Multi-strategy hybrid, real score |
| MemMachine | - | **91.7%** | 80% fewer tokens than Mem0 |
| Letta (MemGPT) | - | ~83.2% | OS-inspired 3-tier |
| Zep/Graphiti | 63.8% | - | Temporal knowledge graph |
| Mem0 | 49.0% | 66.9% | Market leader by adoption |
| MemPalace (real) | ~30% full-corpus | - | 96.6% was ChromaDB baseline |
| ALMA | **Unknown** | **Unknown** | Never benchmarked |

---

## 4. Domain Model Comparison (DDD Analysis)

### Ubiquitous Language Mapping

| MemPalace Term | ALMA Term | Shared Concept |
|---------------|-----------|----------------|
| Palace | ALMA (the system) | The memory store |
| Wing | project_id | Top-level namespace |
| Room | domain / task_type | Sub-categorization |
| Drawer | Outcome + DomainKnowledge | A unit of stored content |
| Hall/Tunnel | Graph Relationship | Cross-domain connections |
| Layer 0 (identity) | (none) | Agent identity |
| Layer 1-3 (memory stack) | MemorySlice.to_prompt() | Context injection |
| (none) | Heuristic | Learned rule with confidence |
| (none) | AntiPattern | Learned "what NOT to do" |
| (none) | MemoryScope | Multi-agent access control |
| (none) | MemoryStrength | Decay-based forgetting |
| importance (metadata) | confidence (field) | Reliability score |
| emotional_weight | (none) | Emotional significance |
| Entity (KG) | Entity (graph) | Same concept, same name |
| Triple (KG) | Relationship (graph) | Same concept, different name |

### Type System Mismatch (Core Friction Point)

MemPalace stores everything as untyped ChromaDB documents with flat metadata dictionaries. ALMA stores everything as typed Python dataclasses with explicit fields, enums, and validation.

This is the fundamental reason a naive code merge would fail. You cannot pour untyped drawers into ALMA's type system without an adapter.

---

## 5. Competitive Position

### What's Table-Stakes in 2026

ALMA has ALL of these:
- Multi-scope memory (user/session/agent)
- Semantic vector search
- Multiple storage backends
- MCP integration
- Async writes
- Temporal awareness

### ALMA's Unique Differentiators (No Competitor Has These)

1. **Anti-pattern learning** -- Tracking what NOT to do. Zero competitors offer this.
2. **Auto-heuristic generation** -- Promoting repeated outcomes to rules. No competitor has a learning feedback loop.
3. **5 typed memory types** -- Structured learning vs. raw storage. Everyone else stores text.
4. **Trust scoring per agent** -- 5-dimensional trust profiles with decay and streak bonuses.
5. **7 storage + 4 graph backends** -- Enterprise flexibility matched only by Mem0.

### ALMA's Critical Gaps

1. **Zero published benchmarks** -- Invisible to anyone doing comparison research
2. **Zero market presence** -- 22 stars, 259 downloads/mo, no blog posts, no articles
3. **Zero community** -- Single contributor, no issues, no PRs
4. **retrieve_with_scope bug** -- scope_filter is computed but never passed to storage (confirmed in code)
5. **50% test coverage threshold** -- Should be 80% for a production library

---

## 6. The Verdict: What To Do

### DO NOT do a code merge.

The prior research proposed replacing ALMA's code with MemPalace's simpler implementations. This would be a **downgrade**:
- Replace 7 storage backends with 1
- Replace typed dataclasses with untyped dicts
- Replace multi-factor scoring with raw cosine similarity
- Replace learning system with passive storage
- Replace modular MCP (5 files) with monolith (1 file)
- Lose observability, decay, forgetting, trust scoring, budget retrieval, progressive disclosure

### DO absorb 4 specific capabilities through adapters:

| Capability | Source | Integration Path | Effort |
|-----------|--------|-----------------|--------|
| **File/conversation ingestion** | miner.py, convo_miner.py, normalize.py, general_extractor.py (~2,100 lines) | Adapt miners to output ALMA types (Outcome, DomainKnowledge, UserPreference) instead of ChromaDB drawers | 2-3 days |
| **4-layer MemoryStack** | layers.py (493 lines) | Wrap ALMA's RetrievalEngine with layer concept. L0=identity file, L1=top heuristics, L2=filtered retrieval, L3=full search | 1-2 days |
| **Query sanitization** | query_sanitizer.py (188 lines) | Port directly into ALMA's retrieval pipeline as a pre-processing step | 0.5 day |
| **Temporal graph edges** | knowledge_graph.py valid_from/valid_to pattern | Add valid_from/valid_to to ALMA's Relationship dataclass, update GraphBackend ABC | 1-2 days |

### DO fix critical ALMA issues:

| Issue | Fix | Effort |
|-------|-----|--------|
| `retrieve_with_scope` doesn't filter | Pass scope_filter to storage.get_* methods | 1 day |
| No published benchmarks | Run LongMemEval + LoCoMo with existing code, publish results | 2-3 days |
| 50% coverage threshold | Raise to 80%, add missing tests | 1-2 weeks |
| Zero market presence | Write blog post, submit to "awesome" lists, post comparison | 1 week |
| MockStorage ignores scope_filter | Update mock to implement filtering | 0.5 day |

### DO leverage MemPalace's insights (not code):

1. **Verbatim storage option** -- Add a "raw" memory type that stores unprocessed text. MemPalace proved this outperforms LLM extraction for retrieval.
2. **Palace metaphor as UI** -- The wing/room/hall navigation is intuitive. Consider it for ALMA's CLI (which doesn't exist yet).
3. **AAAK as a compression level** -- MemPalace's lossy dialect could be an AGGRESSIVE+ compression mode, with the caveat that it degrades retrieval by ~12%.
4. **Hook-based auto-save** -- MemPalace's session hooks (save on stop, save before compaction) are a pattern ALMA should adopt for Claude Code integration.

---

## 7. Strategic Roadmap: ALMA to v1.0

### Phase 1: Prove It (Weeks 1-2)
- [ ] Fix `retrieve_with_scope` bug
- [ ] Fix MockStorage scope_filter
- [ ] Run LongMemEval benchmark, publish score
- [ ] Run LoCoMo benchmark, publish score
- [ ] Write "ALMA vs The Field" blog post with real numbers

### Phase 2: Absorb (Weeks 3-4)
- [ ] Port MemPalace query sanitizer into retrieval pipeline
- [ ] Port MemPalace file/conversation ingestion as `alma.ingestion` package
- [ ] Add temporal validity to graph edges
- [ ] Implement 4-layer MemoryStack wrapper

### Phase 3: Ship (Weeks 5-6)
- [ ] Build CLI (`alma mine`, `alma search`, `alma status`)
- [ ] Raise coverage to 80%
- [ ] Write documentation site
- [ ] Publish v1.0.0 to PyPI
- [ ] Submit to awesome-ai-agents, awesome-llm lists

### Phase 4: Differentiate (Weeks 7-8)
- [ ] Publish anti-pattern learning as headline feature
- [ ] Publish multi-agent sharing demo
- [ ] Create Claude Code plugin (like MemPalace's)
- [ ] Write comparison benchmark: ALMA vs Mem0 vs Hindsight vs MemPalace

---

## 8. Prior Research Corrections

The 8 documents in docs/research/files.zip contained these errors:

| Claim | Reality |
|-------|---------|
| "53 commits" | 119 commits |
| "Fragments of incomplete implementation" | 52,058 lines of production code |
| "No CI/CD" | Full CI: lint, typecheck, security, tests on 3 Python versions |
| "6 backend interfaces, none hardened" | 7 backends, 1,000-3,000 lines each |
| "MCP server skeleton" | 22 tools, 4,395 lines, 5 modules |
| "No tests" | 38,825 lines of test code, 1,682 passing |
| "Needs 6 weeks to build what exists" | Most proposed features already implemented |
| "MemPalace 96.6% LongMemEval" | Debunked -- measures ChromaDB, not palace logic |
| "MemPalace is a finished product" | Has fundamental gaps: no learning, no async, no multi-agent |

**Root cause of errors:** The prior research evaluated ALMA from README surface metrics (star count, commit count) without reading a single source file. It evaluated MemPalace from its (since-debunked) benchmark claims.

---

## 9. Competitive Landscape Summary (April 2026)

| Project | Stars | Funding | Architecture | Verified Score | ALMA Comparison |
|---------|-------|---------|-------------|---------------|-----------------|
| Mem0 | ~48k | $24M | Vector+Graph+KV | 49.0% LongMemEval | ALMA has better learning, fewer backends |
| MemPalace | 42.5k | None | ChromaDB+SQLite | ~30% full-corpus | ALMA is architecturally superior |
| Zep/Graphiti | ~24k | VC | Temporal KG | 63.8% LongMemEval | ALMA has more backends, Zep has better graph |
| Letta | ~21k | VC | OS-inspired 3-tier | ~83.2% LoCoMo | Different philosophy, Letta more mature |
| Cognee | ~12k | Unknown | KG+Vector pipeline | Not published | Similar scope, ALMA has learning |
| Hindsight | ~4k | Unknown | Multi-strategy hybrid | **91.4% LongMemEval** | The benchmark target to beat |
| ALMA | 22 | None | 5-type learning architecture | **Unknown** | Needs benchmarks urgently |

---

## Appendix: Research Methodology

5 parallel agents, total runtime ~20 minutes:

1. **MemPalace Analyst** -- Read all 27 .py source files (11,263 lines), benchmarks, tests, docs, integrations
2. **ALMA Analyst** -- Read all 107 source files (52,058 lines) across 18 subpackages, test infrastructure
3. **Research Reviewer** -- Critically assessed all 8 prior research documents against actual codebases
4. **Domain Decoder** -- DDD bounded context mapping, ubiquitous language glossary, merge compatibility matrix
5. **Web Researcher** -- PyPI stats, GitHub activity, benchmark controversy, competitive landscape, market positioning

All findings are evidence-based with line numbers and file references from actual source code.
