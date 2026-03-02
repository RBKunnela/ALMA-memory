# Synthesis Quality Gate

> Quality gate checklist for the ALMA Synthesis Squad. All items must pass before Phase 2 (Intelligence) is considered complete.

## Weekly Review Engine

- [ ] Weekly review generates structured synthesis from 20+ memories
- [ ] Themes identified match actual content topics (>80% relevance)
- [ ] Open loops correctly identify unresolved action items (heuristics without outcomes, pending outcomes)
- [ ] Gap analysis identifies domains with insufficient recent coverage
- [ ] People analysis extracts and ranks entity mentions
- [ ] Review output follows the Weekly Review Protocol format from `open-brain-kb.md`
- [ ] `alma_weekly_review` MCP tool registered, callable, returns formatted review
- [ ] Edge cases handled: no memories, single memory, all memories in one cluster, all singletons

## Pattern Detection

- [ ] Pattern detection finds genuine recurring themes (not noise) across 2+ time windows
- [ ] Topic frequency tracking shows accurate evolution over time (emerging, stable, declining)
- [ ] Anomaly detection correctly flags statistical outliers against baseline
- [ ] Confidence scoring produces meaningful differentiation (scores in [0.0, 1.0])
- [ ] False positive rate for patterns < 20%
- [ ] Handles edge cases: too few memories (<5), single-topic weeks, burst capture events
- [ ] Reuses `SemanticClusterer` from review engine (no duplicate clustering logic)

## Unified Retrieval (Graph + Vector)

- [ ] Unified retrieval combines vector + graph results in a single query
- [ ] Graph edges measurably improve retrieval relevance (connected > isolated at equal vector similarity)
- [ ] Works with `InMemoryGraphBackend` in unit tests
- [ ] Uses only `GraphBackend` ABC methods (no backend-specific code, compatible with all 4 backends)
- [ ] Graceful degradation when graph backend is `None` (falls back to pure vector search)
- [ ] Graceful degradation on graph traversal timeout (returns vector-only results with warning)
- [ ] Deduplication merges results found via both vector and graph paths

## Connection Finder

- [ ] Connection finder surfaces non-obvious links between memories
- [ ] Connections found via both vector + graph paths ranked highest
- [ ] Cross-domain connections discovered (different topics linked through shared entities)
- [ ] Connection explanations are meaningful and accurate (match connection type)
- [ ] Template-based explanations work without LLM provider
- [ ] `alma_find_connections` MCP tool registered, callable, returns structured results
- [ ] Accepts both `memory_id` and `query` as input modes
- [ ] Empty result returned (not error) when no connections found

## Code Quality

- [ ] All new code follows ALMA standards: ruff check + ruff format pass
- [ ] Type hints on all public APIs
- [ ] Google-style docstrings on all public classes and methods
- [ ] Unit tests cover > 80% of new modules
- [ ] All tests use `alma.testing.MockStorage`, `alma.testing.MockEmbedder`, `alma.testing.factories`
- [ ] No regressions in existing 1,210+ tests
- [ ] New modules are under `alma/synthesis/` package (except `alma/retrieval/unified.py`)
- [ ] Async variants (`async_*`) provided for all core methods using `asyncio.to_thread()`

## Integration

- [ ] `alma/synthesis/__init__.py` exports public API cleanly
- [ ] MCP tools registered in MCP server with proper parameter schemas
- [ ] Events emitted: `synthesis.review.completed`, `synthesis.pattern.detected` (via `alma/events/emitter.py`)
- [ ] Observability: logging via `alma/observability/logging` for all major operations
- [ ] Config: `SynthesisConfig` with sensible defaults, overridable via YAML config

## Phase 2 Sign-Off

- [ ] All 4 tasks completed and individual gates passed
- [ ] Synthesis squad chief (`@synthesis-chief`) has reviewed all modules
- [ ] Demo: Weekly review generates useful synthesis from sample data
- [ ] Demo: Pattern detection finds genuine patterns in multi-week data
- [ ] Demo: Connection finder surfaces non-obvious links
- [ ] Ready for Phase 3 (Openness) or Phase 4 (Compound) integration

---

*Synthesis Quality Gate — alma-synthesis squad v1.0.0*
