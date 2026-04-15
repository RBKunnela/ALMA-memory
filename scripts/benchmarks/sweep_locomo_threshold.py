#!/usr/bin/env python3
"""
LoCoMo similarity-threshold sweep.

Runs the ALMA x LoCoMo retrieval benchmark once per threshold value and
prints a tradeoff table: as ``min_score`` climbs, adversarial refusal
rate should rise (good) while non-adversarial Recall@5 should only
degrade mildly (cost). We look for the knee.

Usage::

    .venv/Scripts/python scripts/benchmarks/sweep_locomo_threshold.py
    .venv/Scripts/python scripts/benchmarks/sweep_locomo_threshold.py --fast
    .venv/Scripts/python scripts/benchmarks/sweep_locomo_threshold.py \\
        --thresholds 0.0,0.3,0.35,0.4 --limit 10

Output is printed as a markdown table and saved to
``benchmarks/locomo/results_sweep.json`` (array of per-threshold
result blocks). Does NOT overwrite ``results_smoke.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Ensure project root on sys.path
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from benchmarks.locomo.dataset import LoCoMoDataset  # noqa: E402
from benchmarks.locomo.metrics import LoCoMoMetrics  # noqa: E402
from benchmarks.locomo.runner import LoCoMoRunner  # noqa: E402

# Default full and fast sweeps. The fast preset is what we run when the
# full 8-point sweep would blow past ~35 minutes on local CPU.
FULL_THRESHOLDS: List[float] = [0.0, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
FAST_THRESHOLDS: List[float] = [0.0, 0.30, 0.35, 0.40]


def _summarize(metrics: LoCoMoMetrics, threshold: float) -> Dict[str, Any]:
    """Flatten a :class:`LoCoMoMetrics` into a sweep row.

    Keeps only the columns we need for the comparison table plus a raw
    ``per_category`` block for downstream analysis.
    """
    per_cat_r5: Dict[str, float] = {}
    for cat, block in metrics.per_category.items():
        if cat == "adversarial":
            continue
        r5 = block.get("recall@5")
        if r5 is not None:
            per_cat_r5[cat] = float(r5)

    return {
        "min_score": threshold,
        "recall@5": float(metrics.recall_at_k.get(5, 0.0)),
        "mrr": float(metrics.mrr),
        "adversarial_refusal_rate": float(metrics.adversarial_refusal_rate),
        "per_category_recall@5": per_cat_r5,
        "per_category": metrics.per_category,
        "total_qa": int(metrics.total_qa),
        "total_time_s": float(metrics.total_time_s),
    }


def _format_table(rows: List[Dict[str, Any]]) -> str:
    """Render a markdown table with the key tradeoff columns."""
    # Collect all category keys we saw (non-adversarial only).
    cats: List[str] = []
    for row in rows:
        for c in row["per_category_recall@5"]:
            if c not in cats:
                cats.append(c)
    cats.sort()

    header = ["min_score", "R@5", "MRR", "adv_refusal"] + [f"{c}_R@5" for c in cats]
    lines = ["| " + " | ".join(header) + " |"]
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for row in rows:
        parts = [
            f"{row['min_score']:.2f}",
            f"{row['recall@5']:.3f}",
            f"{row['mrr']:.3f}",
            f"{row['adversarial_refusal_rate']:.3f}",
        ]
        for c in cats:
            v = row["per_category_recall@5"].get(c)
            parts.append("-" if v is None else f"{v:.3f}")
        lines.append("| " + " | ".join(parts) + " |")
    return "\n".join(lines)


def _recommend(rows: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    """Pick the knee: refusal >=0.70 and non-adv R@5 drop <=10% vs baseline.

    The baseline is the threshold=0.0 row. Returns ``None`` if no
    threshold in the sweep satisfies both constraints.
    """
    if not rows:
        return None
    baseline = rows[0]
    base_r5 = baseline["recall@5"] or 1e-9
    candidates = []
    for row in rows:
        if row["min_score"] == 0.0:
            continue
        refusal = row["adversarial_refusal_rate"]
        r5_drop = (base_r5 - row["recall@5"]) / base_r5
        if refusal >= 0.70 and r5_drop <= 0.10:
            candidates.append((row, refusal, r5_drop))
    if not candidates:
        return None
    # Prefer the highest refusal, then the smallest R@5 drop.
    candidates.sort(key=lambda t: (-t[1], t[2]))
    row, refusal, r5_drop = candidates[0]
    return {
        "min_score": row["min_score"],
        "recall@5": row["recall@5"],
        "adversarial_refusal_rate": refusal,
        "r5_drop_vs_baseline": r5_drop,
    }


def run_sweep(
    thresholds: List[float],
    limit: int,
    embedding: str,
    output_path: Path,
    embedder_model: str | None = None,
) -> List[Dict[str, Any]]:
    """Run the benchmark once per threshold, return the flattened rows.

    Args:
        thresholds: Min-score values to sweep.
        limit: Max conversations (0 = all).
        embedding: "local" or "mock".
        output_path: Where incremental results are persisted.
        embedder_model: Optional override for the local embedder model name
            (e.g. ``"BAAI/bge-large-en-v1.5"``). ``None`` uses the default.
    """
    # Load dataset ONCE -- downloads locomo10.json on first call.
    dataset = LoCoMoDataset.load(path=None, limit=limit)
    print(f"Loaded dataset: {dataset.summary()}")
    print(f"Sweeping {len(thresholds)} thresholds: {thresholds}")
    if embedder_model:
        print(f"Embedder model override: {embedder_model}")
    print()

    rows: List[Dict[str, Any]] = []
    for i, threshold in enumerate(thresholds, start=1):
        print("=" * 64)
        print(f"  [{i}/{len(thresholds)}] min_score = {threshold}")
        print("=" * 64)
        t0 = time.time()
        runner = LoCoMoRunner(
            embedding_provider=embedding,
            top_k=10,
            min_score=threshold,
            embedder_model=embedder_model,
        )
        try:
            metrics, _ = runner.run(dataset, k_values=[1, 5, 10])
        finally:
            runner.cleanup()
        elapsed = time.time() - t0
        row = _summarize(metrics, threshold)
        row["elapsed_s"] = elapsed
        rows.append(row)
        print(
            f"  --> R@5={row['recall@5']:.3f} "
            f"MRR={row['mrr']:.3f} "
            f"refusal={row['adversarial_refusal_rate']:.3f} "
            f"({elapsed:.1f}s)\n"
        )
        # Persist incrementally so a crash mid-sweep is not fatal.
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2)

    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sweep LoCoMo min-score threshold and report the knee."
    )
    parser.add_argument(
        "--thresholds",
        type=str,
        default=None,
        help=(
            "Comma-separated thresholds. Overrides --fast/--full. "
            "Example: 0.0,0.3,0.35,0.4"
        ),
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help=f"Use the 4-point fast preset: {FAST_THRESHOLDS}",
    )
    parser.add_argument("--limit", type=int, default=10, help="Conv limit")
    parser.add_argument("--embedding", type=str, default="local")
    parser.add_argument(
        "--embedder-model",
        type=str,
        default=None,
        help=(
            "Optional sentence-transformers model override for the local "
            "embedder. Example: BAAI/bge-large-en-v1.5"
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(_project_root / "benchmarks/locomo/results_sweep.json"),
    )
    args = parser.parse_args()

    if args.thresholds:
        thresholds = [float(x) for x in args.thresholds.split(",") if x.strip()]
    elif args.fast:
        thresholds = FAST_THRESHOLDS
    else:
        thresholds = FULL_THRESHOLDS

    output_path = Path(args.output)
    rows = run_sweep(
        thresholds=thresholds,
        limit=args.limit,
        embedding=args.embedding,
        output_path=output_path,
        embedder_model=args.embedder_model,
    )

    print("\n" + "=" * 64)
    print("  SWEEP RESULTS")
    print("=" * 64 + "\n")
    print(_format_table(rows))

    rec = _recommend(rows)
    print("\n" + "-" * 64)
    if rec:
        print(
            f"  RECOMMENDED min_score = {rec['min_score']:.2f}  "
            f"(refusal={rec['adversarial_refusal_rate']:.3f}, "
            f"R@5 drop={rec['r5_drop_vs_baseline'] * 100:.1f}% vs baseline)"
        )
    else:
        print(
            "  NO RECOMMENDATION: no threshold hit the knee "
            "(refusal>=0.70 and R@5 drop<=10%). "
            "Consider sweeping finer around the best-refusal row."
        )
    print(f"  Results saved to: {output_path}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n  Interrupted by user.")
        sys.exit(1)
