#!/usr/bin/env python3
"""
ALMA Feedback Learning Benchmark (FLB) Runner.

Measures how much ALMA's retrieval improves when feedback signals
(USED / IGNORED) are recorded and used to re-rank future queries.

Unlike the LongMemEval runner which creates a fresh DB per question,
FLB uses ONE persistent SQLite+FAISS database across ALL questions
and runs multiple rounds over the same dataset to measure improvement.

Flow:
    1. Ingest all sessions from LongMemEval into ONE persistent DB
    2. Round 1: query all questions, record baseline R@5 / MRR / NDCG
    3. For each result, simulate feedback via FeedbackSimulator
    4. Record feedback via FeedbackTracker.record_usage()
    5. Round 2: re-query with FeedbackAwareScorer active, record metrics
    6. Optional Round 3+ for convergence measurement
    7. Output: JSON with per-round metrics + deltas

Usage:
    python -m benchmarks.feedback_learning.runner --data path/to/longmemeval.json
    python -m benchmarks.feedback_learning.runner --data data.json --rounds 3
    python -m benchmarks.feedback_learning.runner --data data.json --simulator oracle
    python -m benchmarks.feedback_learning.runner --data data.json --weights 0.10,0.15,0.25
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
from typing import Any, Dict, List, Optional, Set, Tuple

# Ensure the project root is on sys.path
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from benchmarks.feedback_learning.simulator import FeedbackSimulator  # noqa: E402
from benchmarks.longmemeval.dataset import (  # noqa: E402
    BenchmarkQuestion,
    load_dataset,
)
from benchmarks.longmemeval.metrics import (  # noqa: E402
    BenchmarkMetrics,
    QuestionResult,
    compute_all_metrics,
)

# ---------------------------------------------------------------------------
# Persistent ALMA Backend (single DB across all questions)
# ---------------------------------------------------------------------------


class PersistentALMABackend:
    """ALMA backend with a single persistent database for all questions.

    Unlike ALMABenchmarkBackend (one DB per question), this backend
    ingests all sessions once and keeps the database alive across
    multiple retrieval rounds to observe feedback-driven improvement.

    Args:
        embedding_provider: Embedding provider name (``"local"`` or ``"mock"``).
        tmp_dir: Base directory for the database. Auto-created if None.
        feedback_weight: Weight for FeedbackAwareScorer blending (0.0 to 1.0).
    """

    def __init__(
        self,
        embedding_provider: str = "local",
        tmp_dir: Optional[str] = None,
        feedback_weight: float = 0.15,
    ):
        self.embedding_provider = embedding_provider
        self.feedback_weight = feedback_weight
        self._tmp_dir = tmp_dir or tempfile.mkdtemp(prefix="alma_flb_")
        self._db_path = Path(self._tmp_dir) / "flb_persistent.db"

        # Lazily initialized
        self._shared_embedder = None
        self._storage = None
        self._engine = None
        self._feedback_tracker = None
        self._feedback_scorer = None

        # Maps session_id -> memory_id for feedback recording
        self._session_to_memory_id: Dict[str, str] = {}

    def _get_embedder(self):
        """Get or create the shared embedding model."""
        if self._shared_embedder is None:
            if self.embedding_provider == "mock":
                from alma.retrieval.embeddings import MockEmbedder

                self._shared_embedder = MockEmbedder()
            else:
                from alma.retrieval.embeddings import LocalEmbedder

                self._shared_embedder = LocalEmbedder()
        return self._shared_embedder

    def _init_storage(self):
        """Initialize storage, retrieval engine, and feedback components."""
        from alma.retrieval.engine import RetrievalEngine
        from alma.retrieval.feedback import FeedbackAwareScorer, FeedbackTracker
        from alma.retrieval.scoring import ScoringWeights
        from alma.storage.sqlite_local import SQLiteStorage

        embedder = self._get_embedder()

        self._storage = SQLiteStorage(
            db_path=self._db_path,
            embedding_dim=embedder.dimension,
        )

        self._feedback_tracker = FeedbackTracker(self._storage)
        self._feedback_scorer = FeedbackAwareScorer(
            feedback_tracker=self._feedback_tracker,
            feedback_weight=self.feedback_weight,
        )

        # Scoring weights: pure similarity for benchmarks (recency/success are noise)
        benchmark_weights = ScoringWeights(
            similarity=1.0,
            recency=0.0,
            success_rate=0.0,
            confidence=0.0,
        )

        self._engine = RetrievalEngine(
            storage=self._storage,
            embedding_provider=self.embedding_provider,
            enable_cache=False,
            scoring_weights=benchmark_weights,
            min_score_threshold=0.0,
            feedback_scorer=self._feedback_scorer,
        )
        # Inject shared embedder
        self._engine._embedder = embedder

    def ingest_sessions(
        self,
        questions: List[BenchmarkQuestion],
        mode: str = "session",
    ) -> int:
        """Ingest all unique sessions from all questions into the persistent DB.

        Each session is stored once, even if it appears in multiple questions'
        haystacks. The mapping from session_id to ALMA memory_id is tracked
        for feedback recording.

        Args:
            questions: List of benchmark questions with haystack sessions.
            mode: ``"session"`` (user turns only) or ``"full"`` (all turns).

        Returns:
            Number of unique sessions ingested.
        """
        from alma.types import DomainKnowledge

        if self._storage is None:
            self._init_storage()

        embedder = self._get_embedder()
        ingested_sessions: Set[str] = set()
        project_id = "feedback_learning"
        agent = "benchmark"

        for question in questions:
            for session in question.haystack:
                if session.session_id in ingested_sessions:
                    continue

                if mode == "full":
                    text = session.full_text()
                else:
                    text = session.user_text()

                if not text.strip():
                    continue

                memory_id = str(uuid.uuid4())
                embedding = embedder.encode(text)

                knowledge = DomainKnowledge(
                    id=memory_id,
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
                self._storage.save_domain_knowledge(knowledge)

                self._session_to_memory_id[session.session_id] = memory_id
                ingested_sessions.add(session.session_id)

        return len(ingested_sessions)

    def retrieve(
        self,
        question: BenchmarkQuestion,
        top_k: int = 50,
    ) -> Tuple[List[str], float]:
        """Retrieve sessions for a question from the persistent DB.

        Args:
            question: Benchmark question to query.
            top_k: Number of results to return.

        Returns:
            Tuple of (ranked_session_ids, retrieval_time_ms).
        """
        from alma.types import MemoryScope

        agent = "benchmark"
        project_id = "feedback_learning"

        scope = MemoryScope(
            agent_name=agent,
            can_learn=["conversation"],
            cannot_learn=[],
        )

        start_time = time.time()
        result = self._engine.retrieve(
            query=question.question,
            agent=agent,
            project_id=project_id,
            top_k=top_k,
            scope=scope,
        )
        retrieval_time_ms = (time.time() - start_time) * 1000

        ranked_session_ids: List[str] = []
        for dk in result.domain_knowledge:
            sess_id = dk.metadata.get("session_id", "")
            if sess_id and sess_id not in ranked_session_ids:
                ranked_session_ids.append(sess_id)

        return ranked_session_ids, retrieval_time_ms

    def record_feedback(
        self,
        retrieved_session_ids: List[str],
        feedback_signals: Dict[str, Any],
    ) -> int:
        """Record simulated feedback for retrieved sessions.

        Translates session-level feedback into memory-level feedback
        and persists it via FeedbackTracker.

        Args:
            retrieved_session_ids: Session IDs that were retrieved.
            feedback_signals: Dict mapping session_id to FeedbackSignal.

        Returns:
            Number of feedback records created.
        """
        from alma.types import FeedbackSignal, MemoryType

        # Translate session IDs to memory IDs and split into used/ignored
        retrieved_memory_ids: List[str] = []
        used_memory_ids: List[str] = []

        for sess_id in retrieved_session_ids:
            memory_id = self._session_to_memory_id.get(sess_id)
            if memory_id is None:
                continue
            retrieved_memory_ids.append(memory_id)
            signal = feedback_signals.get(sess_id)
            if signal == FeedbackSignal.USED:
                used_memory_ids.append(memory_id)

        if not retrieved_memory_ids:
            return 0

        feedback_ids = self._feedback_tracker.record_usage(
            retrieved_ids=retrieved_memory_ids,
            used_ids=used_memory_ids,
            memory_type=MemoryType.DOMAIN_KNOWLEDGE,
            agent="benchmark",
            project_id="feedback_learning",
            query="",
        )
        return len(feedback_ids)

    def update_feedback_weight(self, weight: float) -> None:
        """Update the feedback weight on the scorer.

        Args:
            weight: New feedback weight (0.0 to 1.0).
        """
        if self._feedback_scorer is not None:
            self._feedback_scorer.feedback_weight = weight
            self.feedback_weight = weight

    def cleanup(self) -> None:
        """Remove temporary directory and database."""
        try:
            shutil.rmtree(self._tmp_dir, ignore_errors=True)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Round Result
# ---------------------------------------------------------------------------


def _run_round(
    backend: PersistentALMABackend,
    questions: List[BenchmarkQuestion],
    simulator: Optional[FeedbackSimulator],
    round_num: int,
    top_k: int = 50,
    record_feedback: bool = True,
) -> Tuple[BenchmarkMetrics, List[QuestionResult]]:
    """Run one retrieval round over all questions.

    Args:
        backend: Persistent ALMA backend.
        questions: Benchmark questions to evaluate.
        simulator: Feedback simulator (None to skip feedback recording).
        round_num: Current round number (for display).
        top_k: Retrieval depth.
        record_feedback: Whether to record feedback after this round.

    Returns:
        Tuple of (aggregated metrics, per-question results).
    """
    results: List[QuestionResult] = []
    feedback_count = 0
    start_time = time.time()

    for i, question in enumerate(questions):
        ranked_ids, retrieval_ms = backend.retrieve(question, top_k=top_k)

        result = QuestionResult(
            question_id=question.question_id,
            question_type=question.question_type,
            correct_ids=question.correct_session_ids,
            ranked_ids=ranked_ids,
            retrieval_time_ms=retrieval_ms,
        )
        results.append(result)

        # Record simulated feedback
        if record_feedback and simulator is not None:
            feedback_signals = simulator.generate_feedback(
                retrieved_ids=ranked_ids,
                correct_ids=question.correct_session_ids,
            )
            n = backend.record_feedback(ranked_ids, feedback_signals)
            feedback_count += n

        # Progress
        hit = bool(set(ranked_ids[:5]) & question.correct_session_ids)
        status = "HIT" if hit else "miss"
        if (i + 1) % 50 == 0 or i == len(questions) - 1:
            print(
                f"    Round {round_num} [{i + 1:4}/{len(questions)}] "
                f"R@5={'1' if hit else '0'}  {status}"
            )

    total_time = time.time() - start_time

    metrics = compute_all_metrics(
        results=results,
        ks=[1, 3, 5, 10, 30, 50],
        total_time_s=total_time,
    )

    if record_feedback:
        print(f"    Feedback signals recorded: {feedback_count}")

    return metrics, results


# ---------------------------------------------------------------------------
# Benchmark Runner
# ---------------------------------------------------------------------------


def run_feedback_benchmark(
    data_path: Optional[str] = None,
    rounds: int = 2,
    weights: Optional[List[float]] = None,
    simulator_mode: str = "realistic",
    limit: int = 0,
    top_k: int = 50,
    output_file: Optional[str] = None,
    embedding_provider: str = "local",
    seed: int = 42,
) -> Dict[str, Any]:
    """Run the Feedback Learning Benchmark.

    Ingests all LongMemEval sessions into a single persistent database,
    then runs multiple retrieval rounds with simulated feedback to measure
    how much retrieval quality improves.

    Args:
        data_path: Path to LongMemEval dataset JSON. Downloads if None.
        rounds: Number of retrieval rounds (minimum 2: baseline + feedback).
        weights: List of feedback_weight values to sweep. If provided,
            runs the full round sequence for each weight and compares.
            Default: [0.15].
        simulator_mode: Feedback simulation mode (oracle, realistic, noisy).
        limit: Max questions to evaluate (0 = all).
        top_k: Retrieval depth.
        output_file: Path to save JSON results.
        embedding_provider: ALMA embedding provider (local or mock).
        seed: Random seed for reproducibility.

    Returns:
        Dict with per-round metrics, deltas, and configuration.
    """
    if weights is None:
        weights = [0.15]

    dataset = load_dataset(path=data_path, limit=limit)
    questions = dataset.questions

    print(f"\n{'=' * 64}")
    print("  ALMA Feedback Learning Benchmark (FLB)")
    print(f"{'=' * 64}")
    print(f"  Questions:     {len(questions)}")
    print(f"  Rounds:        {rounds}")
    print(f"  Simulator:     {simulator_mode}")
    print(f"  Weights:       {weights}")
    print(f"  Embeddings:    {embedding_provider}")
    print(f"  Seed:          {seed}")
    print(f"{'-' * 64}")

    all_results: Dict[str, Any] = {
        "benchmark": "feedback_learning",
        "system": "ALMA",
        "config": {
            "rounds": rounds,
            "simulator_mode": simulator_mode,
            "weights": weights,
            "top_k": top_k,
            "embedding_provider": embedding_provider,
            "seed": seed,
            "total_questions": len(questions),
        },
        "weight_runs": {},
    }

    for weight in weights:
        print(f"\n{'=' * 64}")
        print(f"  Weight: {weight}")
        print(f"{'=' * 64}")

        backend = PersistentALMABackend(
            embedding_provider=embedding_provider,
            feedback_weight=weight,
        )

        # Ingest all sessions once
        print("\n  Ingesting sessions...")
        ingest_start = time.time()
        n_sessions = backend.ingest_sessions(questions, mode="session")
        ingest_time = time.time() - ingest_start
        print(f"  Ingested {n_sessions} unique sessions in {ingest_time:.1f}s")

        simulator = FeedbackSimulator(mode=simulator_mode, seed=seed)

        round_data: List[Dict[str, Any]] = []

        for round_num in range(1, rounds + 1):
            print(f"\n  --- Round {round_num} ---")

            # Round 1 is baseline: no feedback exists yet, but we record it
            # Subsequent rounds benefit from accumulated feedback
            is_last_round = round_num == rounds
            record = not is_last_round  # Don't record on the last round

            metrics, results = _run_round(
                backend=backend,
                questions=questions,
                simulator=simulator,
                round_num=round_num,
                top_k=top_k,
                record_feedback=record,
            )

            r5 = metrics.recall_at_k.get(5, 0.0)
            mrr = metrics.mrr
            ndcg5 = metrics.ndcg_at_k.get(5, 0.0)

            round_info = {
                "round": round_num,
                "recall_at_5": r5,
                "recall_at_10": metrics.recall_at_k.get(10, 0.0),
                "mrr": mrr,
                "ndcg_at_5": ndcg5,
                "total_time_s": metrics.total_time_s,
                "recall_at_k": metrics.recall_at_k,
                "ndcg_at_k": {str(k): v for k, v in metrics.ndcg_at_k.items()},
            }
            round_data.append(round_info)

            print(f"  R@5={r5:.4f}  MRR={mrr:.4f}  NDCG@5={ndcg5:.4f}")

        # Calculate deltas between rounds
        deltas = []
        for i in range(1, len(round_data)):
            prev = round_data[i - 1]
            curr = round_data[i]
            delta = {
                "from_round": prev["round"],
                "to_round": curr["round"],
                "recall_at_5_delta": curr["recall_at_5"] - prev["recall_at_5"],
                "mrr_delta": curr["mrr"] - prev["mrr"],
                "ndcg_at_5_delta": curr["ndcg_at_5"] - prev["ndcg_at_5"],
            }
            deltas.append(delta)
            print(
                f"\n  Delta R{prev['round']}->R{curr['round']}: "
                f"R@5={delta['recall_at_5_delta']:+.4f}  "
                f"MRR={delta['mrr_delta']:+.4f}  "
                f"NDCG@5={delta['ndcg_at_5_delta']:+.4f}"
            )

        all_results["weight_runs"][str(weight)] = {
            "weight": weight,
            "rounds": round_data,
            "deltas": deltas,
        }

        backend.cleanup()

    # Save results
    if output_file:
        out_path = Path(output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n  Results saved to: {out_path}")

    print(f"\n{'=' * 64}")
    print("  FLB Complete")
    print(f"{'=' * 64}\n")

    return all_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    """CLI entry point for the Feedback Learning Benchmark."""
    parser = argparse.ArgumentParser(
        description="ALMA Feedback Learning Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with defaults (2 rounds, realistic simulator, weight=0.15)
  python -m benchmarks.feedback_learning.runner --data /path/to/data.json

  # Run 3 rounds with oracle simulator
  python -m benchmarks.feedback_learning.runner --data data.json --rounds 3 --simulator oracle

  # Sweep multiple feedback weights
  python -m benchmarks.feedback_learning.runner --data data.json --weights 0.10,0.15,0.25

  # Quick test with 20 questions
  python -m benchmarks.feedback_learning.runner --data data.json --limit 20

  # Save results to JSON
  python -m benchmarks.feedback_learning.runner --data data.json --output results.json
        """,
    )

    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to LongMemEval data JSON (downloads if not provided)",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=2,
        help="Number of retrieval rounds (default: 2)",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="0.15",
        help="Comma-separated feedback_weight values to sweep (default: 0.15)",
    )
    parser.add_argument(
        "--simulator",
        choices=["oracle", "realistic", "noisy"],
        default="realistic",
        help="Feedback simulation mode (default: realistic)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of questions (0 = all)",
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
        help="Output JSON path for results",
    )
    parser.add_argument(
        "--embedding",
        type=str,
        default="local",
        choices=["local", "mock"],
        help="Embedding provider (default: local)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()

    # Parse weights
    weight_list = [float(w.strip()) for w in args.weights.split(",")]

    try:
        run_feedback_benchmark(
            data_path=args.data,
            rounds=args.rounds,
            weights=weight_list,
            simulator_mode=args.simulator,
            limit=args.limit,
            top_k=args.top_k,
            output_file=args.output,
            embedding_provider=args.embedding,
            seed=args.seed,
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
