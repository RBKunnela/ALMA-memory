# LoCoMo Benchmark for ALMA

Evaluates ALMA's turn-level retrieval on the LoCoMo (Long Conversational
Memory) benchmark — Maharana et al. 2024, [arXiv:2402.17753](https://arxiv.org/abs/2402.17753).
Dataset: [snap-research/locomo](https://github.com/snap-research/locomo).

## Modes

| Mode | Status | LLM keys | What it measures |
|------|--------|----------|------------------|
| `retrieval` (default) | v1.0 — shipped | None | Turn-level recall@k / NDCG@k / MRR / adversarial refusal rate |
| `end-to-end` | v1.1 — scaffolded, not implemented | Yes | Answer quality (BERTScore, F1, LLM-as-judge) |

v1.0 intentionally preserves ALMA's "no API keys required" positioning — the
retrieval pipeline runs entirely on local embeddings (sentence-transformers
`all-MiniLM-L6-v2`) and SQLite+FAISS storage.

## Quick start

```bash
# Smoke test on 10 conversations -- downloads locomo10.json on first run
.venv/Scripts/python -m benchmarks.locomo.runner \
    --mode retrieval --limit 10 --output /tmp/locomo_smoke.json
```

## Categories

LoCoMo QA pairs cover five categories:

| ID | Label | Scoring |
|----|---------------|---------|
| 1 | `single_hop` | recall@k / NDCG@k / MRR |
| 2 | `multi_hop` | recall@k / NDCG@k / MRR |
| 3 | `temporal` | recall@k / NDCG@k / MRR |
| 4 | `open_domain` | recall@k / NDCG@k / MRR |
| 5 | `adversarial` | **refusal rate** — empty retrieval = success |

## Data flow

1. `LoCoMoDataset.load()` — load `locomo10.json` (auto-download if missing)
2. `ingest_conversation()` — each turn → `DomainKnowledge` memory, tagged
   with the LoCoMo evidence turn ID (`"D{session}:{turn}"`) in
   `metadata["turn_id"]`
3. For every QA pair, `LoCoMoRunner` queries ALMA and extracts the retrieved
   memories' turn IDs
4. `LoCoMoMetrics.aggregate_by_category()` emits overall and per-category
   numbers; adversarial items are scored separately (refusal rate)

## End-to-end mode (v1.1 preview)

The `llm_adapters.py` module scaffolds five providers. When end-to-end
lands in v1.1, set the matching environment variable and pass
`--mode end-to-end --llm-provider <name>`:

| Provider | Env var |
|----------|---------|
| `openrouter` | `OPENROUTER_API_KEY` |
| `zai` | `ZAI_API_KEY` |
| `ollama_cloud` | `OLLAMA_CLOUD_API_KEY` |
| `openai` | `OPENAI_API_KEY` |
| `anthropic` | `ANTHROPIC_API_KEY` |

Calling `--mode end-to-end` in v1.0 raises a clear `NotImplementedError`
pointing to v1.1.

## Files

| File | Purpose |
|------|---------|
| `dataset.py` | JSON loader, auto-download, turn-ID parsing |
| `evidence_mapping.py` | Ingest turns into ALMA as `DomainKnowledge` |
| `metrics.py` | Retrieval metrics + category aggregation |
| `llm_adapters.py` | Provider stubs (v1.1) |
| `runner.py` | CLI entry point — `python -m benchmarks.locomo.runner` |
