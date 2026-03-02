# Synthesis Patterns Knowledge Base

> Reference material for the ALMA Synthesis Squad. Contains algorithms, best practices, scoring formulas, sample patterns, and edge case documentation for the intelligence layer.

---

## 1. Pattern Detection Algorithms

### 1.1 Sliding Window Analysis

The sliding window approach divides a time range into overlapping or non-overlapping windows and analyzes memory frequency within each.

**Algorithm:**

```
Input: memories[], window_size=7 days, window_count=4, overlap=0
Output: frequency_matrix[window][topic]

1. Sort memories by created_at timestamp
2. Calculate total_range = window_count * window_size (adjusted for overlap)
3. For each window i in [0, window_count):
     start = now - total_range + (i * (window_size - overlap))
     end = start + window_size
     window_memories = filter(memories, start <= created_at < end)
     For each memory in window_memories:
       topics = extract_topics(memory)  # from tags, domains, cluster labels
       frequency_matrix[i][topic] += 1
4. Return frequency_matrix
```

**Choosing window parameters:**
- `window_size=7` (weekly) is the default for most analysis
- `window_size=1` (daily) for granular burst detection
- `window_count=4` provides 1 month of trend data
- `overlap=0` for clean separation; `overlap=window_size/2` for smoother trends

### 1.2 Frequency Analysis

Simple frequency counting with normalization for fair cross-topic comparison.

**Raw frequency:**
```
freq(topic, window) = count of memories with topic in window
```

**Normalized frequency (TF-like):**
```
norm_freq(topic, window) = freq(topic, window) / total_memories_in_window
```

**Cross-window frequency:**
```
presence(topic) = count of windows where freq(topic, window) > 0
persistence(topic) = presence(topic) / total_windows
```

A topic is considered "recurring" when `persistence >= 0.5` (appears in at least half the windows).

### 1.3 TF-IDF Evolution

Track how distinctive a topic is over time by computing TF-IDF per window and comparing across windows.

**Per-window TF-IDF:**
```
tf(topic, window) = freq(topic, window) / total_memories_in_window
idf(topic) = log(total_windows / windows_containing_topic)
tfidf(topic, window) = tf * idf
```

**Evolution signal:**
- If `tfidf(topic, window_N) >> tfidf(topic, window_1)`: topic is becoming more distinctive (emerging)
- If `tfidf(topic, window_N) << tfidf(topic, window_1)`: topic is becoming less distinctive (fading)
- Stable topics have roughly constant TF-IDF across windows

### 1.4 Trend Detection via Linear Regression

For topic frequency over time periods, fit a simple linear regression:

```
y = mx + b
where:
  x = window index [0, 1, 2, ..., N-1]
  y = frequency count in that window
  m = slope (positive = emerging, negative = declining)
  b = intercept

R-squared indicates how well the trend fits:
  R^2 > 0.7: strong trend
  R^2 0.3-0.7: moderate trend
  R^2 < 0.3: no clear trend (noise or cyclical)
```

**Trend classification:**
| Slope | R-squared | Classification |
|-------|-----------|---------------|
| m > 0 | R^2 > 0.5 | Emerging |
| m < 0 | R^2 > 0.5 | Declining |
| abs(m) < threshold | any | Stable |
| any | R^2 < 0.3 | No trend (check for cyclical) |

### 1.5 Cyclical Pattern Detection

Use autocorrelation to detect periodic patterns:

```
autocorr(lag) = correlation(freq_series, shifted_freq_series_by_lag)

If autocorr(lag=2) or autocorr(lag=3) is significant (> 0.5):
  Pattern is cyclical with period = lag * window_size
```

Common cyclical patterns in personal knowledge:
- Weekly rhythms (work topics Mon-Fri, personal topics weekends)
- Biweekly sprint cycles (planning topics spike every 2 weeks)
- Monthly patterns (reporting, reviews)

---

## 2. Weekly Review Best Practices

Derived from the Weekly Review Protocol in `alma-capture/data/open-brain-kb.md`.

### 2.1 The Six-Step Review Pipeline

1. **Cluster by topic** -- Group related captures using semantic similarity. Target 3-5 dominant themes per week. Singletons (unclustered memories) should be reported separately as "one-off captures."

2. **Scan for unresolved action items** -- An open loop is:
   - A heuristic with no recent outcome validating or invalidating it
   - An outcome marked as `pending` or `in_progress` without resolution
   - A domain knowledge entry referencing an action not yet taken
   - Strategy mentions in heuristics that lack outcome feedback

3. **People analysis** -- Who appeared most? Extract entity mentions from memory fields. Cross-reference with graph entities. Track relationship context updates.

4. **Pattern detection (lightweight)** -- Compare against previous weeks:
   - Growing topics (more captures this week than last)
   - New topics (first appearance this week)
   - Dropped topics (present last week, absent this week)

5. **Connection mapping** -- Find non-obvious links between captures from different days or clusters. Cross-cluster embedding similarity reveals connections the user may not have noticed.

6. **Gap analysis** -- What is conspicuously absent?
   - Domains the agent has historically captured for but not this week
   - Domains in `MemoryScope.can_learn` with zero captures
   - Expected domains based on project type and domain schemas

### 2.2 Review Output Format

```markdown
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

### 2.3 Quality Signals for Good Reviews

- **Theme relevance > 80%**: At least 4 out of 5 identified themes should genuinely represent the week's content
- **Actionable suggestions**: "Focus on X" is better than "Think about X"
- **Specificity**: Reference actual memory content, not generic observations
- **Brevity**: A review should be scannable in 2 minutes

---

## 3. Graph Traversal Strategies for Connection Finding

### 3.1 Breadth-First Expansion

Start from a source entity and expand outward by hop count:

```
Hop 0: Source memory entity
Hop 1: Direct relationships (strongest signal)
Hop 2: Two-hop connections (moderate signal)
Hop 3+: Diminishing returns (use sparingly)
```

**Recommendation**: Default `max_hops=2`. Hop 3 should only be used for sparse graphs where hop 2 yields fewer than 3 results.

### 3.2 Relation Type Filtering

Not all relationship types are equally valuable for connection finding:

| Relation Type | Signal Strength | Notes |
|--------------|-----------------|-------|
| `relates_to` | High | Direct topical connection |
| `causes` / `caused_by` | High | Causal chains are very valuable |
| `mentions` | Medium | Entity co-occurrence |
| `referenced_in` | Medium | Shared references |
| `similar_to` | Low | Already captured by vector similarity |
| `temporal_proximity` | Low | Same-day captures, often obvious |

### 3.3 Path Scoring

Score a graph path based on:

```
path_score = product(edge.weight for edge in path) / (hop_count ** decay_factor)

where:
  decay_factor = 1.5 (default, makes distant connections worth less)
  edge.weight = 0.0 to 1.0 (relationship strength)
```

For multiple paths between the same source and target, use the highest-scoring path.

### 3.4 Non-Obvious Connection Heuristics

A connection is "non-obvious" if:
- Source and target are in different semantic clusters
- Source and target were captured more than 3 days apart
- Source and target have different memory types (e.g., heuristic + domain knowledge)
- Source and target share no explicit tags in common
- The connection is only discoverable via 2+ hop graph traversal

---

## 4. Relevance Scoring Formulas

### 4.1 Vector Similarity Score

ALMA's existing retrieval engine uses cosine similarity:

```
vector_score = cosine_similarity(query_embedding, memory_embedding)
Range: [-1.0, 1.0], typically [0.0, 1.0] for normalized embeddings
```

### 4.2 Graph Relationship Score

```
graph_score = normalize(
    sum(edge.weight / (hop ** 1.5) for edge, hop in paths),
    min=0.0, max=1.0
)
```

With diversity bonus:
```
diversity = len(unique_relation_types) / len(total_edges)
graph_score_final = graph_score * (1.0 + 0.2 * diversity)
```

### 4.3 Recency Decay

ALMA's existing decay formula from `alma/learning/decay.py`:

```
recency_score = exp(-lambda * days_since_access)
where lambda is configurable (default: 0.1)
```

### 4.4 Unified Scoring Formula

For the `UnifiedRetriever`:

```
unified_score = (vector_score * w_vector) + (graph_score * w_graph) + (recency_score * w_recency)

Default weights:
  w_vector = 0.60  # Embedding similarity is primary signal
  w_graph  = 0.25  # Graph relationships enhance ranking
  w_recency = 0.15 # Recent memories get a small boost
```

For connection finding (where novelty matters more):

```
connection_strength = (similarity * 0.4) + (graph_weight * 0.3) + (novelty * 0.3)

novelty = 1.0 - overlap_score(source, target)
overlap_score considers: same day (0.3), same tags (0.3), same type (0.2), same cluster (0.2)
```

---

## 5. Sample Patterns and What They Look Like in Data

### 5.1 Recurring Theme: "API Design"

```
Week 1: 3 memories about REST API patterns, endpoint naming
Week 2: 2 memories about API versioning, deprecation
Week 3: 4 memories about GraphQL vs REST, schema design
Week 4: 2 memories about API testing, contract testing

Pattern: "API Design" recurring theme
  persistence = 4/4 = 1.0 (present every week)
  total_occurrences = 11
  trend = stable (slope ~0, R^2 low)
  confidence = 0.85
```

### 5.2 Emerging Topic: "Observability"

```
Week 1: 0 memories
Week 2: 1 memory about logging
Week 3: 3 memories about tracing, metrics
Week 4: 6 memories about OpenTelemetry, dashboards

Pattern: "Observability" emerging topic
  slope = +2.0 per week
  R^2 = 0.95
  trend = emerging
  confidence = 0.90
```

### 5.3 Declining Topic: "Legacy Migration"

```
Week 1: 5 memories about migration planning
Week 2: 3 memories about data migration
Week 3: 1 memory about migration testing
Week 4: 0 memories

Pattern: "Legacy Migration" declining topic
  slope = -1.67 per week
  R^2 = 0.98
  trend = declining
  confidence = 0.92
```

### 5.4 Cyclical Pattern: "Sprint Planning"

```
Week 1: 4 memories (sprint start)
Week 2: 0 memories
Week 3: 5 memories (sprint start)
Week 4: 0 memories

Pattern: "Sprint Planning" cyclical
  autocorrelation(lag=2) = 0.95
  cycle_period = 2 weeks
  confidence = 0.88
```

### 5.5 Anomaly: Sudden Spike in "Security"

```
Baseline (90 days): avg 1.2 memories/week about security, std_dev 0.8
This week: 8 memories about security vulnerabilities, patching

Anomaly: "Security" spike
  z_score = (8 - 1.2) / 0.8 = 8.5
  classification = anomaly (z > 2.0)
  confidence = 1.0
```

---

## 6. Edge Cases

### 6.1 Too Few Memories (< 5 Total)

**Problem**: Not enough data for meaningful clustering or pattern detection.

**Handling**:
- Weekly review: Skip clustering, list all memories individually. No gap analysis (insufficient baseline).
- Pattern detection: Return empty `PatternReport` with a `status: "insufficient_data"` flag.
- Connection finder: Fall back to pure vector similarity (skip graph traversal if <3 entities).

**Threshold**: Minimum 5 memories for clustering, minimum 10 memories across 2+ windows for pattern detection.

### 6.2 Single-Topic Weeks

**Problem**: All memories are about the same topic. Clustering produces one giant cluster and zero insights.

**Handling**:
- Weekly review: Report the single dominant theme. Focus gap analysis on what is missing.
- Pattern detection: Flag as "single-focus period" rather than forcing pattern extraction.
- Suggested focus: Recommend diversifying capture to other domains.

### 6.3 Burst Capture Events (10+ Memories in One Day)

**Problem**: A single productive day skews frequency analysis for the entire week.

**Handling**:
- Normalize by unique days with captures, not raw count.
- Weight formula: `adjusted_freq = raw_freq * (unique_capture_days / window_size)`
- Pattern detection: Use daily deduplication — multiple memories about the same topic on the same day count as 1 for frequency analysis.
- Flag burst events in the review: "Note: 12 memories captured on Tuesday alone, which may skew topic weights."

### 6.4 Graph Backend Not Configured

**Problem**: User has no graph backend set up. Graph-dependent features fail.

**Handling**:
- `UnifiedRetriever`: Fall back to pure vector search. Log degradation.
- `ConnectionFinder`: Return vector-only connections. Explanation mentions that graph connections are unavailable.
- `WeeklyReviewEngine`: Skip people/entity analysis that depends on graph. Note the limitation in the review output.
- Never raise an error for missing graph backend — it is an optional enhancement.

### 6.5 All Singletons (No Clusters Form)

**Problem**: All memories are unique, none are similar enough to cluster (similarity < threshold).

**Handling**:
- Weekly review: List all memories as individual items. Report "no dominant themes detected."
- Pattern detection: No recurring themes to report. Focus on anomaly detection instead.
- Connection finder: May still find graph-based connections even without semantic clusters.
- Consider suggesting a lower similarity threshold if the user's memories are naturally diverse.

### 6.6 Empty Time Window

**Problem**: No memories captured in the requested time period.

**Handling**:
- Return an empty `SynthesisResult` with `status: "no_data"`.
- Weekly review: Report "No memories captured this week" and suggest capture patterns from `open-brain-kb.md`.
- Pattern detection: Skip the empty window in trend calculation (do not count as zero).
- Never raise an error — an empty week is a valid state.

---

*Synthesis Patterns Knowledge Base — alma-synthesis squad v1.0.0*
