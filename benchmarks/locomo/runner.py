#!/usr/bin/env python3
"""
ALMA x LoCoMo Benchmark Runner (retrieval-only, v1.0)

Evaluates ALMA's turn-level retrieval against the LoCoMo benchmark:

1. Load conversations + QA pairs (download locomo10.json if missing)
2. For each conversation: stand up a fresh SQLite+FAISS ALMA, ingest every
   turn as a DomainKnowledge memory tagged with its LoCoMo turn ID
3. For each QA pair: query ALMA, map retrieved memories back to turn IDs,
   score against ground-truth evidence turn IDs
4. Aggregate into per-category metrics (single-hop, multi-hop, temporal,
   open-domain, adversarial) and dump JSON

End-to-end mode (``--mode end-to-end``) is scaffolded via
:mod:`benchmarks.locomo.llm_adapters` but raises :class:`NotImplementedError`
until v1.1. No LLM API keys are required in v1.0.

Usage::

    python -m benchmarks.locomo.runner --mode retrieval --limit 2 \\
        --output /tmp/locomo_smoke.json
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root on sys.path for editable-style invocation
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from benchmarks.locomo.dataset import (  # noqa: E402
    LoCoMoConversation,
    LoCoMoDataset,
)
from benchmarks.locomo.evidence_mapping import ingest_conversation  # noqa: E402
from benchmarks.locomo.llm_adapters import resolve_adapter  # noqa: E402
from benchmarks.locomo.metrics import (  # noqa: E402
    LoCoMoMetrics,
    LoCoMoQAResult,
)


class LoCoMoRunner:
    """
    Orchestrates: download -> ingest -> query -> score for LoCoMo.

    One SQLite DB is created per conversation; each QA pair issues a retrieval
    against that conversation's memories and we score on the returned turn IDs.
    """

    def __init__(
        self,
        embedding_provider: str = "local",
        tmp_dir: Optional[str] = None,
        top_k: int = 10,
        min_score: float = 0.0,
        embedder_model: Optional[str] = None,
    ):
        """
        Args:
            embedding_provider: "local" or "mock" ALMA embedding backend.
            tmp_dir: Scratch dir for per-conversation SQLite DBs.
            top_k: Retrieval depth per QA pair.
            min_score: Minimum similarity threshold passed to
                ``RetrievalEngine.min_score_threshold``. Memories scoring
                below this are filtered out; if every candidate falls
                below the threshold the engine returns an empty set,
                which registers as a refusal on adversarial QA pairs.
            embedder_model: Optional sentence-transformers model name to
                override ALMA's default ``all-MiniLM-L6-v2``. Only honored
                when ``embedding_provider == "local"``. Examples:
                ``"BAAI/bge-large-en-v1.5"`` (1024-dim),
                ``"BAAI/bge-base-en-v1.5"`` (768-dim).
        """
        self.embedding_provider = embedding_provider
        self.top_k = top_k
        self.min_score = min_score
        self.embedder_model = embedder_model
        self._tmp_dir = tmp_dir or tempfile.mkdtemp(prefix="alma_locomo_")
        self._conv_count = 0
        self._shared_embedder = None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_embedder(self):
        if self._shared_embedder is None:
            if self.embedding_provider == "mock":
                from alma.retrieval.embeddings import MockEmbedder

                self._shared_embedder = MockEmbedder()
            else:
                from alma.retrieval.embeddings import LocalEmbedder

                if self.embedder_model:
                    self._shared_embedder = LocalEmbedder(
                        model_name=self.embedder_model
                    )
                else:
                    self._shared_embedder = LocalEmbedder()
        return self._shared_embedder

    def _run_conversation(self, conv: LoCoMoConversation) -> List[LoCoMoQAResult]:
        """Ingest ``conv`` and retrieve an answer list for every QA pair."""
        from alma.retrieval.engine import RetrievalEngine
        from alma.retrieval.scoring import ScoringWeights
        from alma.storage.sqlite_local import SQLiteStorage
        from alma.types import MemoryScope

        self._conv_count += 1
        db_path = Path(self._tmp_dir) / f"conv_{self._conv_count}.db"

        embedder = self._get_embedder()
        storage = SQLiteStorage(
            db_path=db_path,
            embedding_dim=embedder.dimension,
        )

        project_id = f"locomo_{conv.conv_id}"
        agent = "benchmark"

        # Ingest every turn with its LoCoMo turn_id in metadata
        turn_to_memory = ingest_conversation(
            alma=storage,
            conv=conv,
            project_id=project_id,
            agent=agent,
            embedder=embedder,
        )
        # Reverse for lookup memory_id -> turn_id. Not strictly needed -- we
        # read turn_id from metadata -- but kept for debugging.
        _ = {v: k for k, v in turn_to_memory.items()}

        # Pure-similarity scoring; recency/success/confidence are noise here
        weights = ScoringWeights(
            similarity=1.0,
            recency=0.0,
            success_rate=0.0,
            confidence=0.0,
        )
        engine = RetrievalEngine(
            storage=storage,
            embedding_provider=self.embedding_provider,
            enable_cache=False,
            scoring_weights=weights,
            min_score_threshold=self.min_score,
        )
        engine._embedder = embedder

        scope = MemoryScope(
            agent_name=agent,
            can_learn=["conversation"],
            cannot_learn=[],
        )

        qa_results: List[LoCoMoQAResult] = []
        for qa in conv.qa_pairs:
            t0 = time.time()
            result = engine.retrieve(
                query=qa.question,
                agent=agent,
                project_id=project_id,
                top_k=self.top_k,
                scope=scope,
            )
            elapsed_ms = (time.time() - t0) * 1000

            retrieved_turn_ids: List[str] = []
            for dk in result.domain_knowledge:
                tid = dk.metadata.get("turn_id", "")
                if tid and tid not in retrieved_turn_ids:
                    retrieved_turn_ids.append(tid)

            qa_results.append(
                LoCoMoQAResult(
                    qa_id=qa.qa_id,
                    category=qa.category,
                    question=qa.question,
                    evidence_turn_ids=list(qa.evidence_turn_ids),
                    retrieved_turn_ids=retrieved_turn_ids,
                    retrieval_time_ms=elapsed_ms,
                )
            )

        try:
            db_path.unlink(missing_ok=True)
        except Exception:
            pass

        return qa_results

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        dataset: LoCoMoDataset,
        k_values: List[int],
    ) -> tuple[LoCoMoMetrics, List[LoCoMoQAResult]]:
        """Run retrieval over every conversation in ``dataset``."""
        all_qa_results: List[LoCoMoQAResult] = []
        start = time.time()

        for i, conv in enumerate(dataset.conversations, start=1):
            c_start = time.time()
            qa_results = self._run_conversation(conv)
            all_qa_results.extend(qa_results)
            elapsed = time.time() - c_start
            print(
                f"  [{i:3}/{len(dataset)}] conv={conv.conv_id} "
                f"turns={conv.num_turns} qa={len(qa_results)} "
                f"({elapsed:.1f}s)"
            )

        total_time = time.time() - start
        metrics = LoCoMoMetrics()
        metrics.aggregate_by_category(all_qa_results, k_values=k_values)
        metrics.total_time_s = total_time
        return metrics, all_qa_results

    def cleanup(self):
        try:
            shutil.rmtree(self._tmp_dir, ignore_errors=True)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def _format_report(metrics: LoCoMoMetrics) -> str:
    sep = "=" * 64
    thin = "-" * 64
    lines = [
        "",
        sep,
        "  ALMA x LoCoMo Benchmark (retrieval-only)",
        sep,
        f"  QA pairs: {metrics.total_qa}    Time: {metrics.total_time_s:.1f}s",
        thin,
        "",
        "  OVERALL (non-adversarial):",
        f"  {'K':>4}  {'Recall@K':>10}  {'NDCG@K':>10}",
        f"  {'---':>4}  {'--------':>10}  {'------':>10}",
    ]
    for k in sorted(metrics.recall_at_k):
        r = metrics.recall_at_k.get(k, 0.0)
        n = metrics.ndcg_at_k.get(k, 0.0)
        lines.append(f"  {k:>4}  {r:>10.3f}  {n:>10.3f}")
    lines.append(f"\n  MRR: {metrics.mrr:.3f}")
    lines.append(f"  Adversarial refusal rate: {metrics.adversarial_refusal_rate:.3f}")
    lines.append(f"\n{thin}")
    lines.append("  PER-CATEGORY:")
    for cat, block in metrics.per_category.items():
        parts = [f"count={int(block.get('count', 0))}"]
        if cat == "adversarial":
            parts.append(f"refusal_rate={block.get('refusal_rate', 0.0):.3f}")
        else:
            for key in sorted(block):
                if key == "count":
                    continue
                parts.append(f"{key}={block[key]:.3f}")
        lines.append(f"  {cat:<14} " + "  ".join(parts))
    lines.append(sep)
    return "\n".join(lines)


def _write_output(
    path: str,
    metrics: LoCoMoMetrics,
    qa_results: List[LoCoMoQAResult],
    mode: str,
    embedding_provider: str,
    top_k: int,
    min_score: float = 0.0,
):
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "benchmark": "locomo",
        "system": "ALMA",
        "mode": mode,
        "embedding_provider": embedding_provider,
        "top_k": top_k,
        "min_score": min_score,
        "metrics": metrics.to_dict(),
        "qa_results": [
            {
                "qa_id": qa.qa_id,
                "category": qa.category,
                "question": qa.question,
                "evidence_turn_ids": qa.evidence_turn_ids,
                "retrieved_turn_ids": qa.retrieved_turn_ids[:top_k],
                "retrieval_time_ms": qa.retrieval_time_ms,
            }
            for qa in qa_results
        ],
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"  Results saved to: {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_k_values(raw: str) -> List[int]:
    return [int(x) for x in raw.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(
        description="ALMA x LoCoMo Benchmark Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m benchmarks.locomo.runner --mode retrieval --limit 2\n"
            "  python -m benchmarks.locomo.runner --mode retrieval "
            "--output /tmp/locomo_results.json\n"
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["retrieval", "end-to-end"],
        default="retrieval",
        help="retrieval (default, v1.0) or end-to-end (v1.1, not implemented)",
    )
    parser.add_argument("--data", type=str, default=None, help="Path to locomo10.json")
    parser.add_argument(
        "--output", type=str, default=None, help="Path for JSON results"
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="Max conversations (0 = all)"
    )
    parser.add_argument("--top-k", type=int, default=10, help="Retrieval depth")
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        help=(
            "Minimum similarity threshold for retrieval. "
            "Scores below this are filtered; empty results register as "
            "refusal for adversarial QA pairs (default: 0.0)"
        ),
    )
    parser.add_argument(
        "--k",
        type=str,
        default="1,5,10",
        help="Comma-separated K values for recall@k/ndcg@k (default: 1,5,10)",
    )
    parser.add_argument(
        "--embedding",
        choices=["local", "mock"],
        default="local",
        help="ALMA embedding provider",
    )
    parser.add_argument(
        "--embedder-model",
        type=str,
        default=None,
        help=(
            "Optional sentence-transformers model name for the local "
            "embedder. Overrides ALMA's default (all-MiniLM-L6-v2). "
            "Example: BAAI/bge-large-en-v1.5"
        ),
    )

    # End-to-end mode plumbing (not active in v1.0)
    parser.add_argument("--llm-provider", type=str, default=None)
    parser.add_argument("--llm-model", type=str, default=None)
    parser.add_argument("--judge-provider", type=str, default=None)
    parser.add_argument("--judge-model", type=str, default=None)

    args = parser.parse_args()

    if args.mode == "end-to-end":
        # Eagerly resolve adapters so we fail fast with a clear message
        if args.llm_provider:
            resolve_adapter(args.llm_provider, args.llm_model)
        if args.judge_provider:
            resolve_adapter(args.judge_provider, args.judge_model)
        raise NotImplementedError(
            "End-to-end LoCoMo mode is coming in v1.1. "
            "Use --mode retrieval for the current build. "
            "See benchmarks/locomo/README.md for the mode matrix."
        )

    k_values = _parse_k_values(args.k)

    print("\n" + "=" * 64)
    print("  ALMA x LoCoMo Benchmark")
    print("=" * 64)
    print(f"  Mode:        {args.mode}")
    print(f"  Top-K:       {args.top_k}")
    print(f"  Min-Score:   {args.min_score}")
    print(f"  Embeddings:  {args.embedding}")
    print(f"  K values:    {k_values}")
    print("-" * 64 + "\n")

    dataset = LoCoMoDataset.load(path=args.data, limit=args.limit)
    print("  " + dataset.summary().replace("\n", "\n  ") + "\n")

    runner = LoCoMoRunner(
        embedding_provider=args.embedding,
        top_k=args.top_k,
        min_score=args.min_score,
        embedder_model=args.embedder_model,
    )
    try:
        metrics, qa_results = runner.run(dataset, k_values=k_values)
    finally:
        runner.cleanup()

    print(_format_report(metrics))

    if args.output:
        _write_output(
            path=args.output,
            metrics=metrics,
            qa_results=qa_results,
            mode=args.mode,
            embedding_provider=args.embedding,
            top_k=args.top_k,
            min_score=args.min_score,
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n  Interrupted by user.")
        sys.exit(1)
    except NotImplementedError as e:
        print(f"\n  {e}")
        sys.exit(2)
    except FileNotFoundError as e:
        print(f"\n  ERROR: {e}")
        sys.exit(1)
