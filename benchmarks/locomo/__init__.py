"""
LoCoMo Benchmark for ALMA

Evaluates ALMA's retrieval performance against the LoCoMo benchmark
(Long Conversational Memory, Maharana et al. 2024). Unlike LongMemEval,
LoCoMo evaluates retrieval at the turn level (not session level) and
includes five QA categories: single-hop, multi-hop, temporal, open-domain,
and adversarial.

This v1.0 build ships the retrieval-only mode. End-to-end mode with LLM
answer generation + judge scoring is scaffolded (`llm_adapters.py`) but
not implemented -- coming in v1.1.

Dataset: https://github.com/snap-research/locomo
Paper:   https://arxiv.org/abs/2402.17753
"""

from benchmarks.locomo.dataset import (
    LoCoMoConversation,
    LoCoMoDataset,
    LoCoMoQA,
)
from benchmarks.locomo.metrics import LoCoMoMetrics
from benchmarks.locomo.runner import LoCoMoRunner

__all__ = [
    "LoCoMoConversation",
    "LoCoMoDataset",
    "LoCoMoMetrics",
    "LoCoMoQA",
    "LoCoMoRunner",
]
