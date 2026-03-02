---
task: Build Weekly Review Synthesis Engine
agent: "@review-engine-dev"
persona: Recap
squad: alma-synthesis
phase: "2 — Intelligence"
version: 1.0.0
---

# Build Weekly Review Synthesis Engine

## Goal

Build the weekly review synthesis engine and `alma_weekly_review` MCP tool that transforms raw stored memories into structured, actionable weekly intelligence. The engine clusters recent memories by topic, detects open loops, performs people analysis, maps connections, identifies knowledge gaps, and produces a formatted synthesis output.

## Agent

`@review-engine-dev` (Recap) — Weekly review engine builder, clustering and summarization specialist.

## Requires

- Phase 1 (Foundation) capture pipeline functional — memories can be stored and retrieved
- Existing ALMA storage backends operational (`alma/storage/base.py` ABC)
- Existing retrieval engine available (`alma/retrieval/engine.py`)

## Steps

### Step 1: Design Review Algorithm

Define the review pipeline that processes a configurable time window of memories:

1. **Cluster by topic** — Group related memories using semantic similarity on embeddings. Use agglomerative clustering with cosine distance (scipy). Configurable similarity threshold (default 0.75), minimum cluster size of 2, singletons reported separately.
2. **Scan action items** — Detect open loops: heuristics without recent outcome validation, outcomes marked `pending`/`in_progress` without resolution, domain knowledge referencing untaken actions, strategies lacking outcome feedback.
3. **People analysis** — Extract and rank entity mentions (people, tools, organizations) from memory fields. Cross-reference with graph entities if `GraphBackend` is available.
4. **Pattern detection** — Compare current week's topic distribution against previous weeks. Flag growing, new, and dropped topics. (Lightweight version; full pattern detection is a separate task.)
5. **Connection mapping** — Find non-obvious links between captures from different days or clusters using cross-cluster embedding similarity.
6. **Gap analysis** — Compare domains/tags in recent memories against: the full set of domains the agent has ever stored, domains in `MemoryScope.can_learn`, and expected domains from domain schemas.

Reference: Weekly Review Protocol in `alma-capture/data/open-brain-kb.md` (lines 112-141).

### Step 2: Build `alma/synthesis/review.py` Module

Implement the `WeeklyReviewEngine` class:

```python
# alma/synthesis/review.py
from alma.storage.base import StorageBackend
from alma.synthesis.clustering import SemanticClusterer
from alma.synthesis.types import (
    SynthesisResult, Cluster, OpenLoop, KnowledgeGap, PersonMention
)

class WeeklyReviewEngine:
    """Generates periodic synthesis reviews from stored memories."""

    def __init__(
        self,
        storage: StorageBackend,
        embedder: Optional[Any] = None,
        llm_provider: Optional[Any] = None,
        config: Optional[SynthesisConfig] = None,
    ): ...

    def generate_review(
        self,
        project_id: str,
        agent: str,
        days: int = 7,
    ) -> SynthesisResult: ...

    async def async_generate_review(
        self,
        project_id: str,
        agent: str,
        days: int = 7,
    ) -> SynthesisResult: ...
```

Also implement supporting types in `alma/synthesis/types.py`:
- `SynthesisResult` — top-level dataclass with themes, open loops, connections, gaps, focus suggestions
- `Cluster` — group of related memories with label and summary
- `OpenLoop` — unresolved action item with original context
- `KnowledgeGap` — domain or topic area with insufficient coverage
- `PersonMention` — entity mention frequency and context

Also implement `alma/synthesis/clustering.py`:
- `SemanticClusterer` — agglomerative clustering using cosine distance
- Configurable threshold, minimum cluster size
- No external dependencies beyond numpy/scipy

### Step 3: Implement `alma_weekly_review` MCP Tool

Add the MCP tool that exposes the review engine:

```python
# In alma/mcp/tools/ (synthesis tools)
def alma_weekly_review(
    project_id: str,
    agent: str,
    days: int = 7,
    include_suggestions: bool = True,
) -> dict:
    """Generate a structured weekly review synthesis."""
```

Register the tool in the MCP server with proper parameter schemas and descriptions.

### Step 4: Format Output as Structured Weekly Synthesis

The `SynthesisResult` output must follow this structure (matching the Weekly Review Protocol):

```
## Week at a Glance
[X] memories captured | Top themes: [1], [2], [3]

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

The MCP tool returns both:
- A structured dict (for programmatic use)
- A formatted markdown string (for human consumption)

### Step 5: Test with Sample Data (20+ Memories over 1 Week)

Write comprehensive tests using ALMA's testing infrastructure:

- Use `alma.testing.MockStorage` with pre-loaded test memories
- Use `alma.testing.MockEmbedder` for deterministic embeddings
- Use `alma.testing.factories.create_test_heuristic()` and `create_test_outcome()` for test data
- Create a corpus of 20+ test memories spanning 7 days with known themes, open loops, and gaps
- Test edge cases: no memories, single memory, all memories in one cluster, all singletons, zero open loops

## Output

| Artifact | Path | Description |
|----------|------|-------------|
| Review engine | `alma/synthesis/review.py` | `WeeklyReviewEngine` class |
| Clustering | `alma/synthesis/clustering.py` | `SemanticClusterer` class |
| Types | `alma/synthesis/types.py` | `SynthesisResult`, `Cluster`, `OpenLoop`, `KnowledgeGap` dataclasses |
| Config | `alma/synthesis/config.py` | `SynthesisConfig` with sensible defaults |
| Package init | `alma/synthesis/__init__.py` | Public API exports |
| MCP tool | `alma/mcp/tools.py` or `alma/mcp/tools/synthesis.py` | `alma_weekly_review` function |
| Unit tests | `tests/unit/test_synthesis_review.py` | Tests for review engine |

## Gate

- [ ] `WeeklyReviewEngine.generate_review()` returns a valid `SynthesisResult` from 20+ test memories
- [ ] Themes identified match actual content topics (>80% relevance)
- [ ] Open loops correctly identify unresolved action items (heuristics without outcomes, pending outcomes)
- [ ] Gap analysis identifies domains with insufficient recent coverage
- [ ] People analysis extracts and ranks entity mentions
- [ ] `alma_weekly_review` MCP tool is registered, callable, and returns formatted review
- [ ] Output matches the Weekly Review Protocol format from `open-brain-kb.md`
- [ ] Edge cases handled: no memories, single memory, all singletons
- [ ] All unit tests pass with >80% coverage on new modules
- [ ] Code passes `ruff check` and `ruff format --check`
- [ ] Type hints on all public APIs, Google-style docstrings

---

*Task: Build Weekly Review — alma-synthesis squad v1.0.0*
