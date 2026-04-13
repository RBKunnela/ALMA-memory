# ALMA Benchmarks

Standard benchmark suites for evaluating ALMA's memory retrieval performance.

## Setup

```bash
# Ensure ALMA is installed with dev dependencies
pip install -e ".[dev]"

# The local embedding model (all-MiniLM-L6-v2) is included via sentence-transformers
# FAISS is optional but recommended for speed
pip install faiss-cpu
```

## Benchmark 1: LongMemEval (500 questions)

Tests retrieval across ~53 conversation sessions per question. The standard
benchmark for evaluating AI memory systems (ICLR 2025).

### Quick Start

```bash
# Run with 20 questions first (quick sanity check)
.venv/bin/python -m benchmarks.longmemeval.runner --limit 20

# Full benchmark (500 questions, ~15-30 min depending on hardware)
.venv/bin/python -m benchmarks.longmemeval.runner

# With a pre-downloaded dataset file
.venv/bin/python -m benchmarks.longmemeval.runner --data /tmp/longmemeval_s_cleaned.json
```

### Download Data Manually

```bash
mkdir -p /tmp/alma-benchmark-data
curl -fsSL -o /tmp/alma-benchmark-data/longmemeval_s_cleaned.json \
  https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json
```

### Modes

| Mode | Description | Command |
|------|-------------|---------|
| `session` | Index user turns per session (default) | `--mode session` |
| `full` | Index all turns (user + assistant) | `--mode full` |

### Options

```bash
# Save per-question results for analysis
.venv/bin/python -m benchmarks.longmemeval.runner --output results.json

# Skip first N questions (for resume)
.venv/bin/python -m benchmarks.longmemeval.runner --skip 100 --limit 100

# Use mock embeddings (fast, for testing the pipeline)
.venv/bin/python -m benchmarks.longmemeval.runner --embedding mock --limit 5
```

### Expected Output

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

  MRR: 0.xxx
```

### How It Works

1. **Dataset Loading**: Downloads LongMemEval from HuggingFace (500 questions, each with ~53 conversation sessions as a haystack)
2. **Per-Question Evaluation**: For each question:
   - Creates a fresh ALMA SQLite+FAISS database
   - Ingests all haystack sessions as `DomainKnowledge` memories with embeddings
   - Queries ALMA's `RetrievalEngine` with the question text
   - Checks if the ground-truth session appears in the top-K results
3. **Metrics**: Calculates Recall@K, NDCG@K, Precision@K, and MRR across all questions

### Key Design Decisions

- **DomainKnowledge as memory type**: LongMemEval sessions map naturally to ALMA's
  domain knowledge -- they are factual conversation records, not heuristics or outcomes.
- **Fresh DB per question**: Each question has its own haystack, so we create a fresh
  storage instance per question to avoid cross-contamination.
- **Shared embedder**: The sentence-transformers model is loaded once and shared across
  all questions to avoid the ~2s model load overhead per question.
- **No LLM required**: Uses local embeddings (all-MiniLM-L6-v2, 384-dim) -- no API keys needed.

## Requirements

- Python 3.10+
- `sentence-transformers` (for local embeddings)
- `faiss-cpu` (optional, falls back to numpy cosine similarity)
- ~300MB disk for LongMemEval data
- ~15-30 minutes for full 500-question benchmark
- No API key. No GPU required.
