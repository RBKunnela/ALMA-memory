#!/usr/bin/env python3
"""
ALMA x LongMemEval Benchmark Runner

Evaluates ALMA's retrieval performance against the LongMemEval benchmark.
No LLM API keys required -- uses local embeddings (sentence-transformers)
and SQLite+FAISS storage.

For each of the 500 questions:
1. Ingest all haystack sessions as ALMA DomainKnowledge memories
2. Query ALMA with the question
3. Score retrieval against ground-truth answer sessions
4. Calculate R@K, NDCG@K, MRR

Modes:
    session  -- one memory per session (user turns joined)
    full     -- one memory per session (all turns: user + assistant)

Usage:
    python -m benchmarks.longmemeval.runner
    python -m benchmarks.longmemeval.runner --data path/to/longmemeval_s_cleaned.json
    python -m benchmarks.longmemeval.runner --limit 20
    python -m benchmarks.longmemeval.runner --mode full
"""

import argparse
import json
import shutil
import sys
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure the project root is on sys.path so alma and benchmarks can be imported
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from benchmarks.longmemeval.dataset import (  # noqa: E402
    BenchmarkQuestion,
    load_dataset,
)
from benchmarks.longmemeval.metrics import (  # noqa: E402
    BenchmarkMetrics,
    QuestionResult,
    compute_all_metrics,
    format_results,
)

# ---------------------------------------------------------------------------
# ALMA Backend Adapter
# ---------------------------------------------------------------------------


class ALMABenchmarkBackend:
    """
    Wraps ALMA's SQLite+FAISS storage and RetrievalEngine for benchmarking.

    Each benchmark question gets a fresh database to avoid cross-contamination
    between questions (each question has its own haystack).

    The adapter stores conversation sessions as DomainKnowledge entries --
    ALMA's typed fact store with embedding-based similarity search.
    """

    def __init__(
        self,
        embedding_provider: str = "local",
        tmp_dir: Optional[str] = None,
    ):
        """
        Initialize the benchmark backend.

        Args:
            embedding_provider: Embedding provider for ALMA's retrieval engine.
                                "local" uses sentence-transformers (all-MiniLM-L6-v2).
            tmp_dir: Base directory for temporary databases. If None, uses tempfile.
        """
        self.embedding_provider = embedding_provider
        self._tmp_dir = tmp_dir or tempfile.mkdtemp(prefix="alma_bench_")
        self._question_count = 0

        # Lazily initialized -- shared across questions to avoid reloading
        # the embedding model for every question
        self._shared_embedder = None

    def _get_embedder(self):
        """
        Get or create the shared embedding model.

        Lazy initialization avoids loading the model until first use,
        and sharing it avoids the ~2s load time per question.
        """
        if self._shared_embedder is None:
            if self.embedding_provider == "local":
                from alma.retrieval.embeddings import LocalEmbedder

                self._shared_embedder = LocalEmbedder()
            elif self.embedding_provider == "mock":
                from alma.retrieval.embeddings import MockEmbedder

                self._shared_embedder = MockEmbedder()
            else:
                from alma.retrieval.embeddings import LocalEmbedder

                self._shared_embedder = LocalEmbedder()
        return self._shared_embedder

    def ingest_and_retrieve(
        self,
        question: BenchmarkQuestion,
        mode: str = "session",
        top_k: int = 50,
    ) -> Tuple[List[str], float]:
        """
        Ingest haystack sessions and retrieve for one question.

        Creates a fresh ALMA storage instance per question, ingests all
        sessions as DomainKnowledge, then queries with the question text.

        Args:
            question: A BenchmarkQuestion with haystack sessions
            mode: "session" (user turns only) or "full" (user + assistant)
            top_k: Number of results to retrieve

        Returns:
            Tuple of (ranked_session_ids, retrieval_time_ms)
        """
        from alma.retrieval.engine import RetrievalEngine
        from alma.storage.sqlite_local import SQLiteStorage
        from alma.types import DomainKnowledge, MemoryScope

        self._question_count += 1
        db_path = Path(self._tmp_dir) / f"q_{self._question_count}.db"

        embedder = self._get_embedder()
        embedding_dim = embedder.dimension

        # Create fresh storage for this question
        storage = SQLiteStorage(
            db_path=db_path,
            embedding_dim=embedding_dim,
        )

        # Create retrieval engine sharing the embedder
        engine = RetrievalEngine(
            storage=storage,
            embedding_provider=self.embedding_provider,
            enable_cache=False,  # No caching needed for benchmark
        )
        # Inject shared embedder to avoid reloading the model
        engine._embedder = embedder

        # Ingest sessions as DomainKnowledge
        project_id = "longmemeval"
        agent = "benchmark"

        for session in question.haystack:
            if mode == "full":
                text = session.full_text()
            else:
                text = session.user_text()

            if not text.strip():
                continue

            # Generate embedding for this session
            embedding = embedder.encode(text)

            knowledge = DomainKnowledge(
                id=str(uuid.uuid4()),
                agent=agent,
                project_id=project_id,
                domain="conversation",
                fact=text,
                source="longmemeval_haystack",
                confidence=1.0,
                last_verified=datetime.now(timezone.utc),
                embedding=embedding,
                metadata={
                    "session_id": session.session_id,
                    "date": session.date,
                },
            )
            storage.save_domain_knowledge(knowledge)

        # Retrieve
        start_time = time.time()

        scope = MemoryScope(
            agent_name=agent,
            can_learn=["conversation"],
            cannot_learn=[],
        )

        result = engine.retrieve(
            query=question.question,
            agent=agent,
            project_id=project_id,
            top_k=top_k,
            scope=scope,
        )

        retrieval_time_ms = (time.time() - start_time) * 1000

        # Extract session IDs from retrieved domain knowledge
        ranked_session_ids = []
        for dk in result.domain_knowledge:
            sess_id = dk.metadata.get("session_id", "")
            if sess_id and sess_id not in ranked_session_ids:
                ranked_session_ids.append(sess_id)

        # Clean up DB file to save disk space
        try:
            db_path.unlink(missing_ok=True)
        except Exception:
            pass

        return ranked_session_ids, retrieval_time_ms

    def cleanup(self):
        """Remove temporary directory and all benchmark databases."""
        try:
            shutil.rmtree(self._tmp_dir, ignore_errors=True)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Benchmark Runner
# ---------------------------------------------------------------------------


def run_benchmark(
    data_path: Optional[str] = None,
    mode: str = "session",
    limit: int = 0,
    skip: int = 0,
    top_k: int = 50,
    output_file: Optional[str] = None,
    embedding_provider: str = "local",
) -> BenchmarkMetrics:
    """
    Run the LongMemEval benchmark against ALMA.

    Args:
        data_path: Path to dataset JSON. If None, downloads from HuggingFace.
        mode: "session" (user turns) or "full" (all turns)
        limit: Max questions to evaluate (0 = all)
        skip: Questions to skip from start
        top_k: Retrieval depth (default 50)
        output_file: Optional path to save per-question JSONL results
        embedding_provider: ALMA embedding provider ("local" or "mock")

    Returns:
        BenchmarkMetrics with R@K, NDCG@K, MRR scores
    """
    # Load dataset
    dataset = load_dataset(path=data_path, limit=limit, skip=skip)

    print(f"\n{'=' * 64}")
    print("  ALMA x LongMemEval Benchmark")
    print(f"{'=' * 64}")
    print(f"  Questions:   {len(dataset)}")
    print(f"  Mode:        {mode}")
    print(f"  Top-K:       {top_k}")
    print(f"  Embeddings:  {embedding_provider}")
    print(f"{'─' * 64}\n")

    # Initialize ALMA backend
    backend = ALMABenchmarkBackend(
        embedding_provider=embedding_provider,
    )

    # Run benchmark
    results: List[QuestionResult] = []
    results_log: List[Dict[str, Any]] = []
    start_time = time.time()

    for i, question in enumerate(dataset.questions):
        q_start = time.time()

        ranked_ids, retrieval_ms = backend.ingest_and_retrieve(
            question=question,
            mode=mode,
            top_k=top_k,
        )

        q_elapsed = (time.time() - q_start) * 1000

        result = QuestionResult(
            question_id=question.question_id,
            question_type=question.question_type,
            correct_ids=question.correct_session_ids,
            ranked_ids=ranked_ids,
            retrieval_time_ms=retrieval_ms,
        )
        results.append(result)

        # Check if hit at K=5
        top5 = set(ranked_ids[:5])
        hit = bool(top5 & question.correct_session_ids)
        status = "HIT" if hit else "miss"

        # Check R@10
        top10 = set(ranked_ids[:10])
        hit10 = bool(top10 & question.correct_session_ids)

        print(
            f"  [{i + 1:4}/{len(dataset)}] "
            f"{question.question_id[:30]:<30} "
            f"R@5={'1' if hit else '0'} "
            f"R@10={'1' if hit10 else '0'}  "
            f"{status}  "
            f"({q_elapsed:.0f}ms)"
        )

        # Log for output file
        if output_file:
            results_log.append(
                {
                    "question_id": question.question_id,
                    "question_type": question.question_type,
                    "question": question.question,
                    "answer": question.answer,
                    "correct_session_ids": list(question.correct_session_ids),
                    "ranked_session_ids": ranked_ids[:50],
                    "hit_at_5": hit,
                    "hit_at_10": hit10,
                    "retrieval_time_ms": retrieval_ms,
                    "total_time_ms": q_elapsed,
                }
            )

    total_time = time.time() - start_time

    # Compute metrics
    metrics = compute_all_metrics(
        results=results,
        ks=[1, 3, 5, 10, 30, 50],
        total_time_s=total_time,
    )

    # Print results
    print(format_results(metrics, title="ALMA"))

    # Save detailed results
    if output_file:
        out_path = Path(output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(
                {
                    "benchmark": "longmemeval",
                    "system": "ALMA",
                    "mode": mode,
                    "embedding_provider": embedding_provider,
                    "top_k": top_k,
                    "total_questions": len(results),
                    "total_time_s": total_time,
                    "metrics": {
                        "recall_at_k": metrics.recall_at_k,
                        "ndcg_at_k": {str(k): v for k, v in metrics.ndcg_at_k.items()},
                        "mrr": metrics.mrr,
                    },
                    "per_type": {
                        qtype: {
                            "count": tm.total_questions,
                            "recall_at_5": tm.recall_at_k.get(5, 0.0),
                            "recall_at_10": tm.recall_at_k.get(10, 0.0),
                        }
                        for qtype, tm in metrics.per_type.items()
                    },
                    "questions": results_log,
                },
                f,
                indent=2,
            )
        print(f"  Results saved to: {out_path}")

    # Cleanup
    backend.cleanup()

    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    """CLI entry point for the LongMemEval benchmark."""
    parser = argparse.ArgumentParser(
        description="ALMA x LongMemEval Benchmark Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full benchmark (downloads dataset automatically)
  python -m benchmarks.longmemeval.runner

  # Run with local dataset file
  python -m benchmarks.longmemeval.runner --data /tmp/longmemeval_s_cleaned.json

  # Quick test with 20 questions
  python -m benchmarks.longmemeval.runner --limit 20

  # Full-turn mode (index user + assistant text)
  python -m benchmarks.longmemeval.runner --mode full

  # Save per-question results
  python -m benchmarks.longmemeval.runner --output results.json
        """,
    )

    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to longmemeval_s_cleaned.json (downloads if not provided)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["session", "full"],
        default="session",
        help="Ingestion mode: session (user turns) or full (all turns)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max questions to evaluate (0 = all 500)",
    )
    parser.add_argument(
        "--skip",
        type=int,
        default=0,
        help="Skip first N questions",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Retrieval depth (default: 50)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save per-question results as JSON",
    )
    parser.add_argument(
        "--embedding",
        type=str,
        default="local",
        choices=["local", "mock"],
        help="Embedding provider (local = sentence-transformers, mock = random)",
    )

    args = parser.parse_args()

    try:
        run_benchmark(
            data_path=args.data,
            mode=args.mode,
            limit=args.limit,
            skip=args.skip,
            top_k=args.top_k,
            output_file=args.output,
            embedding_provider=args.embedding,
        )
    except KeyboardInterrupt:
        print("\n  Benchmark interrupted by user.")
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"\n  ERROR: {e}")
        sys.exit(1)
    except ImportError as e:
        print(f"\n  ERROR: Missing dependency: {e}")
        print("  Install with: pip install sentence-transformers faiss-cpu")
        sys.exit(1)


if __name__ == "__main__":
    main()
