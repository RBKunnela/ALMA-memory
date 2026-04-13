# ALMA Benchmark Results

Machine-readable results for CI/CD integration and automated comparison.

```yaml
benchmark: LongMemEval-S
version: ALMA v0.9.0
date: 2026-04-13
questions: 500
embedding: all-MiniLM-L6-v2 (384 dim)
storage: SQLite + FAISS (IndexFlatIP)
scoring: pure_similarity (similarity=1.0)
api_keys: none
gpu: none

results:
  recall_at_1: 0.804
  recall_at_3: 0.924
  recall_at_5: 0.964
  recall_at_10: 0.980
  recall_at_30: 0.994
  recall_at_50: 0.996
  mrr: 0.872
  ndcg_at_5: 0.886
  ndcg_at_10: 0.887

per_type:
  knowledge-update:
    recall_at_5: 1.000
    recall_at_10: 1.000
    count: 78
  multi-session:
    recall_at_5: 0.992
    recall_at_10: 1.000
    count: 133
  single-session-preference:
    recall_at_5: 0.967
    recall_at_10: 0.967
    count: 30
  temporal-reasoning:
    recall_at_5: 0.947
    recall_at_10: 0.970
    count: 133
  single-session-assistant:
    recall_at_5: 0.946
    recall_at_10: 0.946
    count: 56
  single-session-user:
    recall_at_5: 0.914
    recall_at_10: 0.971
    count: 70

hardware:
  cpu: Intel/AMD (any modern CPU)
  ram: 8GB minimum
  gpu: not required
  time_per_question: 4.56s
  total_time: 456s
  first_question_time: 10s (includes model loading)

reproduction:
  install: pip install alma-memory[local] sentence-transformers
  dataset: curl -fsSL -o /tmp/longmemeval.json https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json
  run: python -m benchmarks.longmemeval.runner --data /tmp/longmemeval.json --limit 100
  full_run: python -m benchmarks.longmemeval.runner --data /tmp/longmemeval.json

competitors:
  hindsight:
    recall_at_5: 0.914
    api_required: true
    api: Gemini-3 Pro
  zep_graphiti:
    recall_at_5: 0.638
    api_required: true
  mem0:
    recall_at_5: 0.490
    api_required: false
  mempalace_raw:
    recall_at_5: 0.30
    api_required: false
    note: headline 96.6% debunked (Issue #214)

journey:
  stage_1_baseline:
    recall_at_5: 0.236
    recall_at_50: 1.000
    issue: FAISS scores not propagated to scorer, all items scored 1.0
  stage_2_faiss_ordering:
    recall_at_5: 0.200
    issue: scorer still received similarity=1.0, re-sorted on equal scores
  stage_3_pure_similarity:
    recall_at_5: 0.800
    fix: ScoringWeights(similarity=1.0, rest=0.0) + stable sort preserves FAISS order
  stage_4_score_propagation:
    recall_at_5: 0.940
    fix: engine extracts _faiss_similarity from metadata, passes to scorer
```
