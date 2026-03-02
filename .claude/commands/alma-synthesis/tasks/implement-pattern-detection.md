---
task: Implement Cross-Session Pattern Detection
agent: "@pattern-detector"
persona: Pulse
squad: alma-synthesis
phase: "2 — Intelligence"
version: 1.0.0
---

# Implement Cross-Session Pattern Detection

## Goal

Build the cross-session pattern detection engine that identifies recurring themes, tracks topic frequency evolution over time, detects emerging and declining topics, finds cyclical patterns, and scores pattern confidence. This module transforms raw memory frequency data into actionable intelligence about what the agent is learning, what themes persist, and what topics are shifting.

## Agent

`@pattern-detector` (Pulse) — Pattern detection specialist, temporal analysis, noise filtering.

## Requires

- **build-weekly-review** (completed) — The review engine provides clustering infrastructure (`SemanticClusterer`) and types (`Cluster`, `SynthesisResult`) that pattern detection builds upon.
- `alma/synthesis/types.py` exists with base dataclasses.
- `alma/synthesis/clustering.py` exists with `SemanticClusterer`.

## Steps

### Step 1: Design Pattern Detection Algorithm

Define three core detection strategies:

**1. Recurring Theme Detection (Sliding Window)**
- Retrieve memories across multiple time windows (e.g., 4 weeks)
- Extract key terms from `condition`, `strategy`, `content` fields per window
- Cluster memories within each window using `SemanticClusterer`
- Track cluster labels across windows — a theme is "recurring" if it appears in 2+ non-adjacent windows
- Filter by `min_occurrences` threshold (default: 3 total occurrences across all windows)

**2. Topic Frequency Evolution (Time Series)**
- Divide the analysis period into equal sub-periods (e.g., 4 weeks of 7 days)
- Count memories per domain/tag/cluster per sub-period
- Calculate trend direction using simple linear regression on period counts:
  - **Emerging**: positive slope, frequency increasing
  - **Stable**: slope near zero, frequency roughly constant
  - **Declining**: negative slope, frequency decreasing
  - **Cyclical**: alternating high/low pattern (detect via autocorrelation)
- Use TF-IDF evolution to track which terms are becoming more/less distinctive over time

**3. Anomaly Detection (Statistical Baseline)**
- Calculate baseline statistics from a longer period (default: 90 days) — mean and standard deviation per domain
- Compare recent activity (default: 7 days) against baseline
- Flag domains where recent count exceeds baseline mean + 2 standard deviations
- Flag new domains that never appeared in baseline
- Flag domains that disappeared (present in baseline, absent recently)

### Step 2: Build `alma/synthesis/patterns.py`

Implement the `PatternDetector` class:

```python
# alma/synthesis/patterns.py
from alma.storage.base import StorageBackend
from alma.synthesis.clustering import SemanticClusterer
from alma.synthesis.types import (
    Pattern, PatternType, TrendDirection,
    PatternReport, Anomaly
)

class PatternDetector:
    """Detects recurring patterns, trends, and anomalies in stored memories."""

    def __init__(
        self,
        storage: StorageBackend,
        clusterer: Optional[SemanticClusterer] = None,
        config: Optional[SynthesisConfig] = None,
    ): ...

    def detect_patterns(
        self,
        project_id: str,
        agent: str,
        days: int = 30,
        min_occurrences: int = 3,
    ) -> PatternReport: ...

    def track_evolution(
        self,
        project_id: str,
        agent: str,
        periods: int = 4,
        period_days: int = 7,
    ) -> List[Pattern]: ...

    def find_anomalies(
        self,
        project_id: str,
        agent: str,
        days: int = 7,
        baseline_days: int = 90,
    ) -> List[Anomaly]: ...
```

### Step 3: Track Topic Frequency Over Time with Sliding Windows

Implement the sliding window infrastructure:

- `TopicFrequencyTracker` — maintains per-period frequency counts
- Configurable window size (default: 7 days) and window count (default: 4)
- Overlap option for smoother trends (default: no overlap)
- Store frequency snapshots in a format compatible with the review engine
- Reuse `SemanticClusterer` from the review engine for consistent topic grouping

### Step 4: Detect Pattern Types

Implement detection for each pattern type:

| Pattern Type | Detection Method | Confidence Score |
|-------------|-----------------|------------------|
| Recurring theme | Appears in 2+ non-adjacent windows | `occurrences / total_windows` |
| Emerging topic | Positive linear regression slope | `r_squared * slope_magnitude` |
| Declining topic | Negative linear regression slope | `r_squared * abs(slope_magnitude)` |
| Cyclical pattern | Autocorrelation peak at lag > 1 | `autocorrelation_coefficient` |
| Anomaly spike | Count > mean + 2*std_dev | `(count - mean) / std_dev` (z-score) |
| New domain | Not present in baseline | `1.0` (binary) |
| Disappeared domain | Present in baseline, absent recently | `baseline_frequency / max_frequency` |

### Step 5: Build Scoring System for Pattern Confidence

Each detected pattern gets a confidence score (0.0 to 1.0):

- **Frequency weight** (0.4): How often does the pattern appear relative to total memories?
- **Consistency weight** (0.3): Does it appear across multiple distinct time windows?
- **Statistical weight** (0.3): How strong is the regression R-squared or z-score?

Configurable thresholds:
- `min_confidence`: Minimum score to report a pattern (default: 0.3)
- `min_occurrences`: Minimum raw count (default: 3)
- `max_patterns`: Maximum patterns to return (default: 20, sorted by confidence)

### Step 6: Test with Multi-Week Sample Data

Build comprehensive tests:

- **Known patterns**: Create 4 weeks of test data with planted patterns (3 memories about "API testing" per week, 5 about "caching" in weeks 1-2 then none)
- **No patterns**: All unique topics, no recurrence
- **Clear evolution**: Topic A emerging (0, 2, 5, 10), Topic B declining (10, 5, 2, 0)
- **Cyclical**: Topic C alternates high/low (5, 1, 5, 1)
- **Anomaly injection**: Sudden spike in a domain that normally has 1-2 memories
- **Edge cases**: Too few memories (<5 total), single-topic weeks, burst capture events (10+ memories in one day)

Use `alma.testing.MockStorage` and `alma.testing.factories` for all test data. Use `alma.testing.MockEmbedder` for deterministic embeddings.

## Output

| Artifact | Path | Description |
|----------|------|-------------|
| Pattern detector | `alma/synthesis/patterns.py` | `PatternDetector` class |
| Types (additions) | `alma/synthesis/types.py` | `Pattern`, `PatternType`, `TrendDirection`, `PatternReport`, `Anomaly` dataclasses |
| Unit tests | `tests/unit/test_synthesis_patterns.py` | Tests for pattern detection |

## Gate

- [ ] `PatternDetector.detect_patterns()` finds recurring themes planted in test data across multiple weeks
- [ ] `PatternDetector.track_evolution()` correctly classifies topics as emerging, stable, or declining
- [ ] `PatternDetector.find_anomalies()` detects injected spikes against baseline
- [ ] Confidence scoring produces scores in [0.0, 1.0] range with meaningful differentiation
- [ ] False positive rate < 20% (measured as: patterns reported that do not match planted patterns)
- [ ] Handles edge cases: too few memories, single-topic weeks, burst capture events
- [ ] Reuses `SemanticClusterer` from review engine (no duplicate clustering logic)
- [ ] All unit tests pass with >80% coverage on new code
- [ ] Code passes `ruff check` and `ruff format --check`
- [ ] Type hints on all public APIs, Google-style docstrings

---

*Task: Implement Pattern Detection — alma-synthesis squad v1.0.0*
