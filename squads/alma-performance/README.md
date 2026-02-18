# alma-performance Squad

**Performance optimization and bottleneck resolution for ALMA-memory**

| Component | Value |
|-----------|-------|
| **Squad Type** | Performance Optimization |
| **Total Agents** | 9 (1 orchestrator + 3 masters + 5 specialists) |
| **Elite Minds** | Raymond Hettinger, Brandon Rhodes, David Beazley |
| **Quality Mode** | ✅ Full research + synthesis |
| **Version** | 1.0.0 |

---

## Squad Purpose

alma-performance is a specialized agent squad designed to identify, analyze, and resolve performance bottlenecks in ALMA-memory. It synthesizes three elite performance frameworks:

1. **Raymond Hettinger**: Pythonic code patterns & built-in optimization
2. **Brandon Rhodes**: Profiling methodology & bottleneck identification
3. **David Beazley**: Systems-level optimization & deep analysis

The squad uses a **sequential 3-phase approach**:
1. Make code Pythonic first (free, quick wins)
2. Profile to identify real bottleneck (data-driven)
3. Optimize at system level (algorithmic, vectorization, compilation)

---

## Agent Structure

### Tier 0: Orchestrator

**⚡ performance-master-chief** - Master Performance Orchestrator
- Receives performance requests
- Routes to appropriate specialists
- Synthesizes recommendations from all 3 frameworks
- Commands: `*audit-performance`, `*quick-diagnose`, `*optimization-roadmap`, `*benchmark-comparison`
- Framework: Hettinger → Rhodes → Beazley (sequential)

---

### Tier 1: Masters (Framework Synthesizers)

**🐍 hettinger-idioms** - Pythonic Code Performance Specialist
- Makes code simultaneously readable and fast
- Detects non-Pythonic patterns (loops vs comprehensions, etc.)
- Recommends built-in optimizations
- Typical improvement: 2-5x via idioms
- Commands: `*check-idioms`, `*suggest-idiom-improvements`, `*validate-pythonic-style`

**📊 rhodes-profiler** - Profiling & Bottleneck Identification Specialist
- Profiles entire application (cProfile, py-spy)
- Identifies hotspot (80% rule)
- Classifies bottleneck type (CPU/I/O/Memory)
- Routes to appropriate specialist
- Commands: `*profile-application`, `*analyze-hotspot`, `*identify-bottleneck`, `*classify-bottleneck`

**⚙️ beayley-systems** - Systems-Level Performance Optimization Specialist
- Analyzes algorithmic complexity (Big O)
- Optimizes data structures for cache efficiency
- Recommends vectorization or compilation
- Typical improvement: 10-100x via deep optimization
- Commands: `*analyze-algorithm`, `*optimize-data-structure`, `*recommend-vectorization`, `*estimate-compilation-gain`

---

### Tier 2: Specialists (Domain-Specific Optimizers)

**💾 cache-analyzer** - Memory & CPU Cache Optimization Expert
- Memory profiling (tracemalloc, memory_profiler)
- Cache efficiency analysis (perf, data layout)
- Memory leak detection (gc, objgraph)
- GC tuning
- Commands: `*analyze-memory-usage`, `*detect-memory-leaks`, `*optimize-cache-layout`, `*tune-gc`

**🔗 i-o-optimizer** - I/O & Database Query Optimization Expert
- N+1 query problem detection & resolution
- Async pattern design
- Connection pooling recommendations
- Query batching & caching strategies
- Commands: `*analyze-database-queries`, `*detect-n-plus-one`, `*batch-operation-strategy`, `*recommend-async-patterns`

**📈 algorithm-auditor** - Algorithmic Complexity & Optimization Auditor
- Big O complexity analysis
- Algorithm selection (linear search vs binary, etc.)
- Sorting algorithm optimization
- Memoization & dynamic programming opportunities
- Commands: `*analyze-complexity`, `*find-algorithmic-bottlenecks`, `*recommend-better-algorithm`, `*estimate-speedup`

**🧠 memory-profiler** - Advanced Memory & Leak Detection Expert
- Line-by-line memory profiling
- Reference cycle detection
- Memory footprint optimization
- Allocation pattern analysis
- Commands: `*profile-memory-by-line`, `*diagnose-memory-leak`, `*find-large-objects`, `*recommend-memory-reduction`

**⚡ vectorization-expert** - NumPy & SIMD Vectorization Specialist
- NumPy vectorization (broadcasting, array operations)
- Numba JIT compilation recommendation
- SIMD optimization
- Array shape optimization
- Commands: `*vectorize-loop`, `*analyze-vectorization-opportunity`, `*recommend-numba-compilation`, `*profile-vectorized-vs-loop`

---

## Typical Performance Audit Workflow

```
User Request: "Performance audit for alma.retrieval.engine"
       ↓
⚡ performance-master-chief receives request
       ↓
   PHASE 1: HETTINGER (Quick Wins)
   └─ 🐍 hettinger-idioms analyzes code
      └─ Finds 5 idiom improvements (2-3x)
      └─ Reports back to master-chief
       ↓
   PHASE 2: RHODES (Profiling)
   └─ 📊 rhodes-profiler profiles application
      └─ Identifies bottleneck: semantic_search_pgvector (45%)
      └─ Classifies: I/O-bound (database queries)
      └─ Routes to: i-o-optimizer
       ↓
   PHASE 3: BEAYLEY (Deep Optimization)
   └─ 🔗 i-o-optimizer handles I/O bottleneck
      └─ Detects N+1 query pattern
      └─ Recommends batching (5-10x)
       ↓
   └─ ⚡ vectorization-expert handles secondary bottleneck
      └─ Vectorizes similarity calculations (2-3x)
       ↓
   └─ All findings synthesized by master-chief
       ↓
COMPREHENSIVE REPORT
  - Phase 1 idiom wins: 2-3x (apply immediately)
  - Phase 2-3 optimization roadmap: 5-10x (prioritized)
  - Total potential: 12-30x improvement
  - Effort estimates + implementation sequence
```

---

## Key Concepts

### Sequential Approach (Not Parallel)

Unlike alma-architecture which presents alternatives, alma-performance applies optimizations **in sequence**:

1. **Hettinger** (always first): Free 2-3x improvements via better code style
2. **Rhodes** (if improvements needed): Profile to confirm Phase 1 solved it
3. **Beayley** (if profiling shows bottleneck): Deep system-level optimization

This prevents:
- Wasted effort (80% optimization focus on non-bottleneck)
- Wrong optimizations (guessing at bottleneck type)
- Premature compilation (Phase 3 without profiling)

### 80/20 Rule

Performance optimization is about finding where 80% of time is spent and focusing **only** on that hotspot. Optimizing the other 80% of code wastes effort.

### Bottleneck Classification

Every bottleneck is classified as:
- **CPU-bound**: High CPU utilization → Algorithm/vectorization/compilation
- **I/O-bound**: Low CPU, high wait time → Async/batching/caching
- **Memory-bound**: High memory usage/leaks → Memory optimization

Different bottleneck types require different optimization strategies.

### Quality Metrics

- **Phase 1 idioms**: Usually 2-5x improvement, 30 min effort
- **Phase 2 profiling**: Eliminates guesswork, 1 hour overhead
- **Phase 3 optimization**: 10-100x potential, 4-8 hours effort

Total typical audit: 6-10 hours for 12-30x improvement.

---

## ALMA-Specific Knowledge

The squad understands ALMA's architecture:

- **Storage backends**: PostgreSQL+pgvector, Azure Cosmos, Qdrant, etc.
- **Retrieval patterns**: Semantic search, BM25, hybrid, verification
- **Consolidation**: LLM-powered memory deduplication
- **Compression**: Token-based memory compression
- **Common bottlenecks**:
  - N+1 queries in retrieval
  - Large embedding vector operations (vectorization opportunity)
  - Memory-heavy LLM context in consolidation
  - Sequential batch processing (async opportunity)

---

## Commands

### Master Chief (Orchestrator)

- `*audit-performance {module}` - Full 3-phase audit
- `*quick-diagnose {function}` - Fast bottleneck ID
- `*optimization-roadmap {module}` - Implementation sequence
- `*benchmark-comparison {before|after}` - Verify improvements
- `*estimate-roi {optimization}` - Cost vs benefit analysis

### Specialist Routing

Commands generally follow pattern: `*{action}-{target}`

Examples:
- `*check-idioms alma/retrieval/engine.py`
- `*profile-application alma.consolidation`
- `*analyze-algorithm score_reranker`
- `*vectorize-loop embedding_similarity`
- `*detect-n-plus-one memory_queries`
- `*detect-memory-leaks consolidation_module`

---

## Success Indicators

✅ A performance optimization is successful when:
- Profiling data confirms improvement
- Actual wall-clock time reduced (not just theory)
- Code remains correct (tests still pass)
- Improvement is proportional to effort
- Bottleneck doesn't shift elsewhere (measure end-to-end)

---

## Anti-Patterns (What Doesn't Work)

❌ **Never**:
- Optimize without profiling first (80% wasted effort)
- Skip Phase 1 idioms (free improvements)
- Guess at bottleneck type (measure instead)
- Optimize non-bottleneck code (focus on hotspot)
- Micro-optimize when algorithmic change possible (10x vs 2x)
- Compile code before profiling (high complexity cost)
- Assume obvious bottleneck is real one (profile confirms)

---

## Handoff Patterns

| From | To | Trigger |
|------|----|----|
| master-chief | hettinger-idioms | Always (Phase 1) |
| hettinger-idioms | rhodes-profiler | After idiom wins applied |
| rhodes-profiler | beayley-systems | If CPU-bound |
| rhodes-profiler | i-o-optimizer | If I/O-bound |
| beayley-systems | algorithm-auditor | If algorithmic issue |
| beayley-systems | vectorization-expert | If vectorizable |
| beayley-systems | cache-analyzer | If memory-layout issue |
| All specialists | master-chief | Final synthesis |

---

## Integration with alma-architecture

- **alma-architecture** (module structure, SOLID principles, dependencies)
- **alma-quality** (code standards, test coverage, security)
- **alma-performance** (speed, latency, throughput)
- **alma-integration** (cross-module compatibility)

These squads work together:
- Architecture defines structure
- Quality ensures correctness
- **Performance optimizes that correct structure**
- Integration validates cross-squad consistency

---

## Example Audit Scenarios

### Scenario 1: Query Slowness
```
Issue: Retrieval queries taking 2.4 seconds
Phase 1: Idioms (2-3x) → 0.8-1.2 seconds
Phase 2: Profile → I/O-bound bottleneck (N+1 queries)
Phase 3: Batch queries (5-10x) → 0.1-0.2 seconds
Total: 12-24x improvement
```

### Scenario 2: Memory Leak
```
Issue: Memory grows 2MB per consolidation
Phase 1: Idioms (minimal impact) → Still growing
Phase 2: Profile → Memory-bound bottleneck
Phase 3: Detect leak, fix with lru_cache
Total: Memory stabilized, no crashes
```

### Scenario 3: Vector Operations Slow
```
Issue: Similarity calculations taking 430ms
Phase 1: Idioms (minimal) → Still 430ms
Phase 2: Profile → CPU-bound (vectorizable)
Phase 3: NumPy vectorization (2-10x) → 20-50ms
Total: 8-20x improvement
```

---

## Getting Started

To use the alma-performance squad:

1. **Start with master chief**: `*audit-performance {module}`
   - Receives your ALMA component
   - Performs full 3-phase analysis
   - Provides comprehensive report

2. **Follow recommendations**:
   - Apply Phase 1 idiom wins (always quick)
   - If needed, let Phase 2 profiling guide deeper work
   - Route to specialists based on bottleneck type

3. **Measure improvements**:
   - Before optimization (baseline)
   - After each phase
   - Confirm improvements match predictions

---

## FAQ

**Q: How long does a typical audit take?**
A: 1-2 hours for initial analysis, 4-8 hours for full optimization.

**Q: Which phase is most important?**
A: Phase 2 (profiling) — eliminates 80% of wasted optimization effort.

**Q: Should I apply all optimizations?**
A: No. Apply Phase 1 (always worth it), then prioritize Phase 3 by ROI.

**Q: What if profiling shows no clear bottleneck?**
A: Phase 1 idioms likely solved it. Measure end-to-end and confirm.

---

## Contact

Route specialized questions to appropriate agent:
- Pythonic code: `@hettinger-idioms`
- Profiling/measurement: `@rhodes-profiler`
- System-level optimization: `@beayley-systems`
- I/O optimization: `@i-o-optimizer`
- Memory issues: `@memory-profiler`, `@cache-analyzer`
- Algorithms: `@algorithm-auditor`
- Vectorization: `@vectorization-expert`

---

*alma-performance - ALMA's performance optimization squad*
*Created: 2026-02-18 | Framework synthesis: Hettinger + Rhodes + Beazley*
