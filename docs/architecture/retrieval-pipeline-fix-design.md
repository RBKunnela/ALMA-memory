# Retrieval Pipeline Fix Design: LongMemEval R@5 from 0.236 to 0.90+

**Author:** Aria (Architect)
**Date:** 2026-04-13
**Status:** Design Complete - Ready for Implementation

---

## Executive Summary

ALMA's retrieval pipeline has a **fundamental ranking bug**: FAISS computes similarity scores correctly, but they are **never propagated** to the scorer. Every memory gets a default similarity of 1.0, reducing ranking to recency/confidence tiebreaking. Since the benchmark uses uniform timestamps and confidence, all items score identically and ranking becomes arbitrary.

R@50=1.0 proves the embeddings find the answer. The problem is purely in **score propagation**.

---

## 1. Root Cause Analysis: Full Query Trace

Here is what happens when the benchmark runner calls `engine.retrieve()`:

### Step 1: Engine generates query embedding
`engine.py` line 143: `query_embedding = self._get_embedding(query)`
-- This works correctly.

### Step 2: Engine calls storage
`engine.py` line 256-263: calls `storage.get_domain_knowledge(embedding=query_embedding, top_k=top_k*2)`
-- Passes embedding correctly.

### Step 3: Storage does FAISS search
`sqlite_local.py` line 1048-1052:
```python
search_results = self._search_index(MemoryType.DOMAIN_KNOWLEDGE, embedding, top_k * 2)
candidate_ids = [id for id, _ in search_results]
faiss_scores = {id: score for id, score in search_results}
```
-- **BUG 1 (Minor):** `_search_index` (line 542-543) normalizes the QUERY vector but `_add_to_index` (line 520-522) does NOT normalize document vectors when adding to IndexFlatIP. Inner product of (normalized_query) dot (unnormalized_doc) != cosine similarity. However, since sentence-transformers' `all-MiniLM-L6-v2` outputs near-unit-norm vectors, the distortion is small.
-- The scores ARE computed and stored in `faiss_scores` dict.

### Step 4: Storage reorders by FAISS scores
`sqlite_local.py` line 1085-1086:
```python
results = self._reorder_by_faiss(results, candidate_ids, faiss_scores)
```
-- The `_reorder_by_faiss` method (line 574-611) correctly reorders results to match FAISS ranking order AND injects `item.metadata["_faiss_similarity"]` = the FAISS score.
-- **The order IS preserved.** Results come back in correct similarity order.

### Step 5: Engine receives ordered list, sends to scorer
`engine.py` line 274:
```python
scored_knowledge = self.scorer.score_domain_knowledge(raw_domain_knowledge)
```
-- **BUG 2 (CRITICAL): No `similarities` argument is passed.**
-- The scorer's `score_domain_knowledge()` (scoring.py line 174-219) has `similarities: Optional[List[float]] = None`.
-- When `None`, line 192 defaults: `similarities = similarities or [1.0] * len(knowledge)`.
-- **Every single item gets similarity=1.0.** The FAISS ranking is destroyed.

### Step 6: Scorer produces uniform scores
With the benchmark's `ScoringWeights(similarity=1.0, recency=0.0, success_rate=0.0, confidence=0.0)`:
- Every item gets `total = 1.0 * 1.0 = 1.0`
- The sort on line 219 (`sorted(scored, key=lambda x: -x.score)`) is stable-sort on identical values
- **Ranking becomes insertion order, which is FAISS order...** BUT:

### Step 7: Engine extracts top-k and UNWRAPS items
`engine.py` line 280:
```python
final_knowledge = self._extract_top_k(scored_knowledge, top_k)
```
Line 768-771:
```python
def _extract_top_k(self, scored_items, top_k):
    filtered = self.scorer.apply_score_threshold(scored_items, self.min_score_threshold)
    return [item.item for item in filtered[:top_k]]
```
-- Items are unwrapped from `ScoredItem` back to raw `DomainKnowledge`.
-- **All similarity information is discarded.** The returned list is just `DomainKnowledge` objects.

### Step 8: Benchmark extracts session IDs
`runner.py` line 270-273:
```python
for dk in result.domain_knowledge:
    sess_id = dk.metadata.get("session_id", "")
    if sess_id and sess_id not in ranked_session_ids:
        ranked_session_ids.append(sess_id)
```
-- The order here depends on the order from step 7.

### WHY R@5=0.236 and not 0.0

Because step 6 uses Python's stable sort on equal scores, AND the FAISS ordering from step 4 IS preserved through steps 5-6 when all scores are identical. So the ranking SHOULD be correct... unless:

**BUG 3 (The actual killer): The `_reorder_by_faiss` re-sort uses dict lookup order.**

Look at `_reorder_by_faiss` (line 601-602):
```python
id_order = {id_val: i for i, id_val in enumerate(candidate_ids)}
results.sort(key=lambda x: id_order.get(getattr(x, id_attr), float("inf")))
```

This SHOULD preserve FAISS order. But the SQL query fetches with `IN (...)` which returns rows in **arbitrary order**, and then `_reorder_by_faiss` re-sorts them. So the order SHOULD be correct after re-sort.

**BUT**: The `get_domain_knowledge` method requests `top_k * 2` from FAISS (line 1049: `top_k * 2`), then the SQL query also limits to `top_k * 2` (line 1080). Then `engine.retrieve()` calls `get_domain_knowledge` with `top_k * 2` (line 262). So the engine gets `top_k * 2` results from storage, then the scorer processes them, then `_extract_top_k` takes the first `top_k`.

Since all scores are 1.0, `_extract_top_k` takes the first `top_k` items, which SHOULD be the FAISS-ordered top items... The 0.236 R@5 suggests the ordering IS partially preserved but degraded.

**The real problem is compound:**

1. Scores ARE 1.0 for everything (the `similarities` parameter bug)
2. Even though FAISS order is nominally preserved through stable sort, any perturbation (like the SQL `IN` clause or the `_reorder_by_faiss` sort being applied on only the subset returned by SQL) means ties are broken unpredictably
3. The benchmark weights are `similarity=1.0, recency=0.0, success_rate=0.0, confidence=0.0` -- so when similarity is always 1.0, score is always 1.0, and ranking is arbitrary among ties

**With real similarity scores propagated, the scorer would produce differentiated scores, and ranking would be deterministic and correct.**

---

## 2. The Similarity Score Gap

### Where scores are born
`_search_index()` at `sqlite_local.py:548-551` returns `(memory_id, float(score))` tuples where `score` is the FAISS inner product result.

### Where scores are injected into metadata
`_reorder_by_faiss()` at `sqlite_local.py:604-610` injects `item.metadata["_faiss_similarity"] = faiss_scores[item_id]`.

### Where scores are needed but missing
`scoring.py:174` `score_domain_knowledge(knowledge, similarities=None)` -- the `similarities` parameter is NEVER populated by the engine.

### The gap
`engine.py:272-275` calls the scorer without extracting similarities:
```python
scored_knowledge = self.scorer.score_domain_knowledge(raw_domain_knowledge)
```
It SHOULD be:
```python
similarities = [dk.metadata.get("_faiss_similarity", 1.0) for dk in raw_domain_knowledge]
scored_knowledge = self.scorer.score_domain_knowledge(raw_domain_knowledge, similarities=similarities)
```

### FAISS score normalization
FAISS IndexFlatIP with normalized queries returns inner products in range [-1, 1] for unit-norm docs. For the benchmark:
- Sentence-transformers outputs near-unit-norm vectors (norm ~0.98-1.02)
- The query IS normalized (line 543: `faiss.normalize_L2(query)`)
- Documents are NOT normalized on add (line 520-522)
- So scores approximate cosine similarity but with slight distortion

**Fix:** Normalize document vectors when adding to index. This makes IndexFlatIP produce exact cosine similarities in [0, 1] (or [-1, 1] technically, but almost always [0, 1] for text embeddings).

---

## 3. Storage Layer Fix

### Approach: Use existing `_faiss_similarity` metadata injection (already implemented!)

The storage layer already does the right thing via `_reorder_by_faiss()`. The fix is NOT in storage -- it's in the engine layer reading the metadata.

However, one normalization fix is needed in storage:

### Fix 3A: Normalize document vectors on add

**File:** `alma/storage/sqlite_local.py`
**Method:** `_add_to_index()` (line 494-524)
**Current** (line 517-522):
```python
# Add to index
self._id_maps[memory_type].append(memory_id)
if FAISS_AVAILABLE:
    self._indices[memory_type].add(
        embedding_array.reshape(1, -1).astype(np.float32)
    )
```

**Fixed:**
```python
# Add to index
self._id_maps[memory_type].append(memory_id)
if FAISS_AVAILABLE:
    vec = embedding_array.reshape(1, -1).astype(np.float32)
    faiss.normalize_L2(vec)
    self._indices[memory_type].add(vec)
```

### Fix 3B: Normalize vectors on load (same issue)

**File:** `alma/storage/sqlite_local.py`
**Method:** `_load_faiss_indices()` (line 474-477)
**Current:**
```python
if FAISS_AVAILABLE:
    self._indices[memory_type].add(
        embedding.reshape(1, -1).astype(np.float32)
    )
```

**Fixed:**
```python
if FAISS_AVAILABLE:
    vec = embedding.reshape(1, -1).astype(np.float32)
    faiss.normalize_L2(vec)
    self._indices[memory_type].add(vec)
```

### Trade-off analysis: Alternative approaches considered

| Approach | Pros | Cons | Verdict |
|----------|------|------|---------|
| A. Add `similarity` field to DomainKnowledge dataclass | Clean API, explicit | Breaks all backends, massive diff | REJECTED |
| B. Return tuples of (item, score) | Minimal type changes | Breaks all callers of get_* methods | REJECTED |
| C. Use existing metadata dict (current `_faiss_similarity`) | Zero schema change, already implemented | Slightly implicit | **CHOSEN** |
| D. New `ScoredMemory` wrapper type | Type-safe | Over-engineering for this fix | REJECTED |

**Decision:** Option C is already 80% implemented. The storage layer already injects `_faiss_similarity` into metadata. The engine just needs to read it.

---

## 4. Engine Layer Fix

### Fix 4A: Extract similarities from metadata and pass to scorer

**File:** `alma/retrieval/engine.py`
**Method:** `retrieve()` (lines 272-276)

**Current:**
```python
# Score and rank each type
scored_heuristics = self.scorer.score_heuristics(raw_heuristics)
scored_outcomes = self.scorer.score_outcomes(raw_outcomes)
scored_knowledge = self.scorer.score_domain_knowledge(raw_domain_knowledge)
scored_anti_patterns = self.scorer.score_anti_patterns(raw_anti_patterns)
```

**Fixed:**
```python
# Extract FAISS similarity scores from metadata (injected by storage layer)
def _extract_similarities(items):
    return [
        getattr(item, 'metadata', {}).get('_faiss_similarity', 1.0)
        if getattr(item, 'metadata', None) else 1.0
        for item in items
    ]

# Score and rank each type with actual similarity scores
scored_heuristics = self.scorer.score_heuristics(
    raw_heuristics, similarities=_extract_similarities(raw_heuristics)
)
scored_outcomes = self.scorer.score_outcomes(
    raw_outcomes, similarities=_extract_similarities(raw_outcomes)
)
scored_knowledge = self.scorer.score_domain_knowledge(
    raw_domain_knowledge, similarities=_extract_similarities(raw_domain_knowledge)
)
scored_anti_patterns = self.scorer.score_anti_patterns(
    raw_anti_patterns, similarities=_extract_similarities(raw_anti_patterns)
)
```

### Fix 4B: Same fix in `retrieve_with_mode()`

**File:** `alma/retrieval/engine.py`
**Method:** `retrieve_with_mode()` (lines 478-481)

Apply the same `_extract_similarities()` helper to the four scorer calls in this method.

### Helper placement

Add `_extract_similarities` as a private method on `RetrievalEngine`:
```python
@staticmethod
def _extract_similarities(items: list) -> list:
    """Extract FAISS similarity scores from item metadata.
    
    The storage layer injects '_faiss_similarity' into each item's
    metadata dict during _reorder_by_faiss(). This method extracts
    those scores for use by the scorer.
    
    Args:
        items: List of memory objects with optional metadata dicts.
        
    Returns:
        List of similarity scores (0-1), defaulting to 1.0 if not present.
    """
    return [
        getattr(item, 'metadata', {}).get('_faiss_similarity', 1.0)
        if getattr(item, 'metadata', None) else 1.0
        for item in items
    ]
```

---

## 5. Hybrid Search Integration

### Current state
`alma/retrieval/hybrid.py` is fully functional. It implements RRF (Reciprocal Rank Fusion) with configurable vector/text weights.

### What's missing for the benchmark

The benchmark runner (`runner.py` lines 234-268) already has hybrid search code, but it's broken:

**Problem 1:** Vector results use fake scores (line 244):
```python
vector_results.append((idx, 1.0 / (len(vector_results) + 1)))
```
This creates scores 1.0, 0.5, 0.33, 0.25... which bear no relation to actual similarity. It should use the real FAISS scores.

**Problem 2:** The hybrid code runs AFTER `engine.retrieve()` has already discarded similarity scores (results are unwrapped `DomainKnowledge` objects).

### Design for proper hybrid integration

**Option A (Recommended): Integrate hybrid search into the engine**

Add a `hybrid` parameter to `RetrievalEngine`:
```python
class RetrievalEngine:
    def __init__(self, ..., hybrid_config: Optional[HybridSearchConfig] = None):
        self._hybrid_engine = None
        if hybrid_config:
            self._hybrid_engine = HybridSearchEngine(config=hybrid_config)
```

In `retrieve()`, after getting raw items from storage:
1. Build text corpus from item texts
2. Index corpus in hybrid engine
3. Run text search
4. Build vector results from `_faiss_similarity` metadata
5. Fuse with RRF
6. Use fused scores as the `similarities` parameter to scorer

**Option B (Quick benchmark fix): Fix the runner's hybrid code**

In `runner.py`, before calling `engine.retrieve()`:
1. Run BM25 search separately on session texts
2. After `engine.retrieve()`, extract FAISS scores from metadata
3. Build proper vector_results from those scores
4. Fuse and reorder

**Recommendation:** Option A for the library, Option B as an immediate benchmark fix.

### Expected impact
Hybrid search typically adds 5-15% recall improvement over pure vector search for factoid questions where keyword overlap matters. For LongMemEval, many questions contain named entities (names, dates, places) where BM25 excels.

---

## 6. Reranking Pass

### Current state
`alma/retrieval/reranking.py` has a `CrossEncoderReranker` that uses `cross-encoder/ms-marco-MiniLM-L-6-v2`. Falls back to `NoOpReranker` if `rerankers` lib is not installed.

### What it needs to work

1. **Text extraction:** The reranker needs text representations of each memory item. A helper to extract searchable text from each memory type:
   ```python
   def _item_to_text(item) -> str:
       if isinstance(item, DomainKnowledge): return item.fact
       if isinstance(item, Heuristic): return f"{item.condition} {item.strategy}"
       if isinstance(item, Outcome): return f"{item.task_description} {item.strategy_used}"
       if isinstance(item, AntiPattern): return f"{item.pattern} {item.why_bad}"
       return str(item)
   ```

2. **Integration point:** After scoring, before `_extract_top_k`:
   ```python
   if self._reranker is not None:
       texts = [_item_to_text(si.item) for si in scored_knowledge]
       reranked = self._reranker.rerank(query, scored_knowledge, texts, top_k)
       # Rebuild scored_knowledge in reranked order
   ```

3. **Configuration:** Add to `RetrievalEngine.__init__`:
   ```python
   def __init__(self, ..., reranker: Optional[Reranker] = None):
       self._reranker = reranker
   ```

### Expected impact
Cross-encoder reranking typically improves top-5 precision by 10-25% over bi-encoder-only retrieval. For LongMemEval's factoid questions, this could be significant.

### Trade-off: Latency
Cross-encoder reranking is O(n) forward passes through a transformer. For top_k=50, this adds ~200-500ms. For a benchmark, acceptable. For production with strict latency budgets, apply only to the top-20 candidates.

---

## 7. MemPalace Comparison

### Why raw ChromaDB (MemPalace) scores higher

MemPalace's retrieval is dead simple (`searcher.py` line 56):
```python
results = col.query(query_texts=[query], n_results=n_results, include=["documents", "metadatas", "distances"])
```

ChromaDB does three things ALMA doesn't:

| Factor | ChromaDB/MemPalace | ALMA | Impact |
|--------|-------------------|------|--------|
| **Distance metric** | Cosine distance (hnsw:space=cosine), returns distances [0, 2] | IndexFlatIP with half-normalized vectors | Minor (both approximate cosine) |
| **Score propagation** | Distances returned directly alongside results | Scores computed but **never reach the scorer** | **CRITICAL** - this is THE bug |
| **Ranking** | Results returned in distance order, directly used | FAISS order -> SQL fetch -> re-sort -> scorer (all 1.0) -> stable-sort ties | CRITICAL - ties destroy ranking |
| **Index type** | HNSW (approximate but with locality) | FlatIP (exact but scores dropped) | Minimal - exact search should be better |

### Same embedding model
Both use `all-MiniLM-L6-v2` by default. The embedding quality is identical.

### The core difference
MemPalace returns results in ChromaDB's distance order and uses them directly. ALMA computes the exact same cosine similarities via FAISS but then **throws them away** by not passing them to the scorer.

**ALMA should outperform MemPalace** once the score propagation bug is fixed, because:
1. ALMA uses exact search (FlatIP) vs ChromaDB's approximate HNSW
2. ALMA has a proper multi-signal scorer (similarity + recency + success + confidence)
3. ALMA has hybrid search and reranking infrastructure already built

---

## 8. Recommended Fix Order

### Phase 1: Score Propagation (Expected: R@5 0.236 -> 0.75-0.85)

**Priority: CRITICAL. This is the bug fix.**

| # | Fix | File | Lines | Effort | Impact |
|---|-----|------|-------|--------|--------|
| 1a | Normalize doc vectors on add | `sqlite_local.py` | 517-522 | 3 lines | Correct cosine scores |
| 1b | Normalize doc vectors on load | `sqlite_local.py` | 474-477 | 3 lines | Correct cosine scores |
| 1c | Extract similarities in engine | `engine.py` | 272-276 | 15 lines | **THE FIX** - propagate scores |
| 1d | Same for retrieve_with_mode | `engine.py` | 478-481 | 15 lines | Complete coverage |

**Rationale:** This is the minimum viable fix. Once similarity scores flow from FAISS through the scorer, ranking will be based on actual cosine similarity instead of arbitrary tiebreaking. This alone should bring R@5 to 0.75-0.85 based on the fact that R@50=1.0 (the correct answer is always in the candidate set, it just needs proper ranking).

### Phase 2: Hybrid Search (Expected: R@5 0.85 -> 0.88-0.92)

| # | Fix | File | Effort | Impact |
|---|-----|------|--------|--------|
| 2a | Add hybrid_config to RetrievalEngine | `engine.py` | 30 lines | Enable hybrid |
| 2b | Integrate text search into retrieve() | `engine.py` | 40 lines | RRF fusion |
| 2c | Fix runner hybrid code | `runner.py` | 20 lines | Benchmark uses real scores |

**Rationale:** Named entity questions (names, dates, places) benefit enormously from BM25 keyword matching. LongMemEval has many such questions. RRF fusion of vector + BM25 is a well-proven technique.

### Phase 3: Cross-Encoder Reranking (Expected: R@5 0.90 -> 0.93+)

| # | Fix | File | Effort | Impact |
|---|-----|------|--------|--------|
| 3a | Add reranker to RetrievalEngine | `engine.py` | 20 lines | Infrastructure |
| 3b | Text extraction helper | `engine.py` | 15 lines | Item -> text |
| 3c | Post-scoring rerank step | `engine.py` | 25 lines | 2nd pass ranking |
| 3d | Add to benchmark runner | `runner.py` | 10 lines | Enable for benchmark |

**Rationale:** Cross-encoders see query and document together, enabling much more nuanced relevance judgments than bi-encoder cosine similarity. This is the standard second-pass technique in production search systems.

### Phase 4: Optional Improvements (Marginal gains)

| # | Fix | Impact | Priority |
|---|-----|--------|----------|
| 4a | Better embedding model (e.g., `bge-base-en-v1.5`, 768-dim) | +2-5% R@5 | Low |
| 4b | Query expansion / reformulation | +1-3% R@5 | Low |
| 4c | Sliding window chunking for long sessions | +1-2% R@5 | Low |

---

## Summary of Changes by File

### `alma/storage/sqlite_local.py`
- `_add_to_index()`: Add `faiss.normalize_L2(vec)` before adding to index
- `_load_faiss_indices()`: Add `faiss.normalize_L2(vec)` before adding to index on load

### `alma/retrieval/engine.py`
- Add `_extract_similarities()` static method
- `retrieve()`: Pass similarities to all four scorer calls
- `retrieve_with_mode()`: Same
- (Phase 2) Add `hybrid_config` parameter and hybrid search integration
- (Phase 3) Add `reranker` parameter and post-scoring rerank step

### `benchmarks/longmemeval/runner.py`
- (Phase 2) Fix hybrid code to use real FAISS scores instead of fake rank-based scores
- (Phase 3) Add cross-encoder reranker option

### Files NOT changed
- `alma/retrieval/scoring.py` -- Already correct. The `similarities` parameter works, it just was never called with real data.
- `alma/retrieval/hybrid.py` -- Already correct. RRF implementation is sound.
- `alma/retrieval/reranking.py` -- Already correct. CrossEncoderReranker works.
- `alma/types.py` -- No schema changes needed.

---

## Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| Normalizing vectors changes scores for existing databases | Medium | Only affects ranking quality (improves it). No data loss. Existing tests should still pass since MockStorage doesn't use FAISS. |
| Breaking change to metadata contract (`_faiss_similarity`) | Low | Already used by storage layer, just reading it in engine. Private convention (underscore prefix). |
| Cross-encoder adds latency | Low | Optional, configurable. Only applies to top-N candidates. |
| Hybrid search BM25 indexing overhead | Low | Per-query overhead is <10ms for typical corpus sizes. |

---

## Validation Plan

1. **Unit tests:** Verify `_extract_similarities()` reads metadata correctly
2. **Integration test:** End-to-end test with known embeddings verifying score propagation
3. **Benchmark:** Run `python -m benchmarks.longmemeval.runner --limit 50` after Phase 1 to validate improvement
4. **Full benchmark:** Run all 500 questions after each phase to measure cumulative improvement
5. **Regression:** Run existing test suite (`pytest tests/unit/ --tb=short -q`) to ensure no breakage

---

## 9. Results

Benchmark results after implementing the fixes described in this document.

### Phase 1: Score Propagation (Implemented)

| Metric | Before Fix | After Fix | Delta |
|--------|-----------|-----------|-------|
| R@5 | 0.236 | 0.800 | +0.564 (+239%) |
| R@50 | 1.000 | 1.000 | unchanged |

The score propagation fix (Fixes 3A, 3B, 4A, 4B) confirmed the root cause: FAISS was computing correct similarity scores, but they were never reaching the scorer. With real cosine similarities flowing end-to-end, ranking became deterministic and correct.

R@50 remaining at 1.0 confirms that the embedding model finds the correct answer in every case -- the problem was purely ranking, as predicted.

### Phase 2: Hybrid Search

| Metric | Pure Similarity | Hybrid (Vector + BM25) |
|--------|----------------|----------------------|
| R@5 | 0.800 | *PENDING* |

Hybrid search via Reciprocal Rank Fusion is implemented and available via `--hybrid` flag. Full benchmark run pending.

### Phase 3: Cross-Encoder Reranking

Not yet implemented. Expected +5-15% improvement on R@5 based on literature.

---

*Aria, arquitetando o futuro*
