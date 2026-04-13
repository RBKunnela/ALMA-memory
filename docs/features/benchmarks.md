# Benchmark Infrastructure

**Module:** `benchmarks/`
**Since:** v0.9.0

## Overview

ALMA includes a benchmark infrastructure for reproducible evaluation of retrieval quality. The primary benchmark is LongMemEval (ICLR 2025), which tests retrieval across 500 questions with ~53 conversation sessions per question as the search haystack.

No LLM API keys required. Uses local embeddings (sentence-transformers) and SQLite+FAISS storage.

## Prerequisites

```bash
# Install ALMA with dev dependencies
pip install -e ".[dev]"

# Local embedding model (included via sentence-transformers)
# FAISS is optional but recommended
pip install faiss-cpu
```

## Running LongMemEval

### Quick Sanity Check (20 questions)

```bash
.venv/bin/python -m benchmarks.longmemeval.runner --limit 20
```

### Full Benchmark (500 questions)

```bash
.venv/bin/python -m benchmarks.longmemeval.runner
```

Takes approximately 15-30 minutes depending on hardware.

### With Pre-Downloaded Dataset

```bash
# Download manually
mkdir -p /tmp/alma-benchmark-data
curl -fsSL -o /tmp/alma-benchmark-data/longmemeval_s_cleaned.json \
  https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json

# Run with local data
.venv/bin/python -m benchmarks.longmemeval.runner --data /tmp/alma-benchmark-data/longmemeval_s_cleaned.json
```

### Hybrid Search (Vector + BM25)

```bash
.venv/bin/python -m benchmarks.longmemeval.runner --hybrid --limit 50
```

Hybrid search uses Reciprocal Rank Fusion (RRF) to combine vector similarity with BM25 keyword matching. Particularly effective for questions containing named entities (names, dates, places).

### All Options

```bash
.venv/bin/python -m benchmarks.longmemeval.runner \
  --data path/to/dataset.json \   # Custom dataset path
  --limit 50 \                    # Number of questions to evaluate
  --skip 100 \                    # Skip first N questions (for resume)
  --mode session \                # session (default) or full
  --output results.json \         # Save per-question results
  --embedding mock \              # Use mock embeddings (fast pipeline test)
  --hybrid                        # Enable hybrid search
```

### Indexing Modes

| Mode | Description |
|------|-------------|
| `session` | Index user turns per session (default). Each session becomes one DomainKnowledge memory. |
| `full` | Index all turns (user + assistant). More text per memory, potentially higher recall. |

## Interpreting Results

### Output Format

```
================================================================
  ALMA x LongMemEval Benchmark Results
================================================================
  Questions: 500    Time: 900.0s

  SESSION-LEVEL METRICS:
     K    Recall@K      NDCG@K      Prec@K
   ---    --------      ------      ------
     1       0.xxx       0.xxx       0.xxx
     5       0.xxx       0.xxx       0.xxx
    10       0.xxx       0.xxx       0.xxx
    50       0.xxx       0.xxx       0.xxx

  MRR: 0.xxx
```

### Key Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| R@5 (Recall@5) | Fraction of questions where the correct session appears in top 5 results | >= 0.80 |
| R@50 | Same at top 50. If R@50 is high but R@5 is low, ranking is the problem. | >= 0.95 |
| NDCG@K | Normalized Discounted Cumulative Gain. Rewards correct answers ranked higher. | >= 0.70 |
| MRR | Mean Reciprocal Rank. Average of 1/rank for the first correct result. | >= 0.60 |

### Baseline Results (v0.9.0)

| Configuration | R@5 | R@50 | MRR |
|--------------|-----|------|-----|
| Before fixes (v0.8.0) | 0.236 | 1.000 | -- |
| Pure similarity scoring (v0.9.0) | 0.800 | 1.000 | -- |
| Hybrid search (v0.9.0) | *pending* | -- | -- |

The jump from 0.236 to 0.800 came from fixing the critical score propagation bug where FAISS similarity scores were computed but never passed to the scorer.

## How It Works

1. **Dataset Loading**: Downloads LongMemEval from HuggingFace (500 questions, each with ~53 conversation sessions as a haystack)
2. **Per-Question Evaluation**: For each question:
   - Creates a fresh ALMA SQLite+FAISS database (avoids cross-contamination)
   - Ingests all haystack sessions as `DomainKnowledge` memories with embeddings
   - Queries ALMA's `RetrievalEngine` with the question text
   - Checks if the ground-truth session appears in the top-K results
3. **Metrics**: Calculates R@K, NDCG@K, Precision@K, and MRR across all questions

### Design Decisions

- **DomainKnowledge as memory type**: LongMemEval sessions are factual conversation records, mapping naturally to ALMA's domain knowledge type.
- **Fresh DB per question**: Each question has its own haystack, preventing cross-contamination.
- **Shared embedder**: The sentence-transformers model (`all-MiniLM-L6-v2`, 384-dim) loads once and is shared across all questions.
- **No LLM required**: Pure local embeddings. No API keys, no GPU required.

## Comparison with Other Systems

The LongMemEval benchmark provides a standardized way to compare ALMA's retrieval against other memory systems. For context on the retrieval pipeline architecture and the fixes that improved R@5, see `docs/architecture/retrieval-pipeline-fix-design.md`.

## Requirements

- Python 3.10+
- `sentence-transformers` (for local embeddings)
- `faiss-cpu` (optional, falls back to numpy cosine similarity)
- ~300 MB disk for LongMemEval data
- ~15-30 minutes for full 500-question benchmark
