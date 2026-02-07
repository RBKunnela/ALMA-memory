"""
ALMA Memory Enhancer.

Core scoring logic that applies memory signals to RAG results.
Computes memory signals per chunk by cross-referencing ALMA's stored
memories with the chunk content.
"""

import logging
from typing import List, Optional

from alma.rag.types import EnhancedChunk, MemorySignals, RAGChunk
from alma.types import AntiPattern, Heuristic, MemorySlice, Outcome

logger = logging.getLogger(__name__)


class MemoryEnhancer:
    """Enhances RAG chunks with ALMA memory signals.

    Given a MemorySlice (ALMA's retrieval results) and a list of RAG chunks,
    computes per-chunk memory signals and an overall boost/demotion factor.

    The enhancement logic:
    1. For each chunk, find related heuristics by keyword overlap.
    2. Compute historical success rate from related outcomes.
    3. Check for anti-pattern warnings.
    4. Compute a boost factor that adjusts the chunk's score.
    """

    def __init__(
        self,
        success_boost: float = 1.3,
        anti_pattern_penalty: float = 0.7,
        no_signal_neutral: float = 1.0,
    ) -> None:
        """Initialize the enhancer.

        Args:
            success_boost: Score multiplier for chunks matching successful strategies.
            anti_pattern_penalty: Score multiplier for chunks matching anti-patterns.
            no_signal_neutral: Score multiplier when no memory signals found.
        """
        self.success_boost = success_boost
        self.anti_pattern_penalty = anti_pattern_penalty
        self.no_signal_neutral = no_signal_neutral

    def enhance_chunks(
        self,
        chunks: List[RAGChunk],
        memory_slice: MemorySlice,
    ) -> List[EnhancedChunk]:
        """Enhance RAG chunks with memory signals.

        Args:
            chunks: Raw chunks from external RAG system.
            memory_slice: ALMA retrieval results for the query.

        Returns:
            List of EnhancedChunks sorted by enhanced_score descending.
        """
        enhanced = []
        for chunk in chunks:
            signals = self._compute_signals(chunk, memory_slice)
            enhanced_score = chunk.score * signals.boost_factor
            enhanced.append(
                EnhancedChunk(
                    chunk=chunk,
                    signals=signals,
                    enhanced_score=enhanced_score,
                )
            )

        # Sort by enhanced score descending
        enhanced.sort(key=lambda e: -e.enhanced_score)

        # Assign ranks
        for i, item in enumerate(enhanced):
            item.rank = i + 1

        return enhanced

    def generate_augmentation(
        self,
        memory_slice: MemorySlice,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate memory augmentation text for prompt injection.

        Creates a formatted text block with relevant strategies,
        anti-patterns, and domain knowledge from ALMA's memory.

        Args:
            memory_slice: ALMA retrieval results.
            max_tokens: Optional token budget limit.

        Returns:
            Formatted augmentation text.
        """
        sections = []

        if memory_slice.heuristics:
            lines = ["## Learned Strategies"]
            for h in sorted(memory_slice.heuristics, key=lambda x: -x.confidence)[:5]:
                lines.append(
                    f"- When: {h.condition} -> {h.strategy} "
                    f"(confidence: {h.confidence:.0%}, "
                    f"success: {h.success_rate:.0%})"
                )
            sections.append("\n".join(lines))

        if memory_slice.anti_patterns:
            lines = ["## Anti-Patterns to Avoid"]
            for ap in memory_slice.anti_patterns[:3]:
                lines.append(
                    f"- Avoid: {ap.pattern}\n"
                    f"  Reason: {ap.why_bad}\n"
                    f"  Instead: {ap.better_alternative}"
                )
            sections.append("\n".join(lines))

        if memory_slice.domain_knowledge:
            lines = ["## Domain Context"]
            for dk in memory_slice.domain_knowledge[:5]:
                lines.append(f"- [{dk.domain}] {dk.fact}")
            sections.append("\n".join(lines))

        if memory_slice.preferences:
            lines = ["## User Preferences"]
            for p in memory_slice.preferences[:3]:
                lines.append(f"- {p.preference}")
            sections.append("\n".join(lines))

        text = "\n\n".join(sections)

        if max_tokens is not None:
            # Rough token estimation: ~4 chars per token
            max_chars = max_tokens * 4
            if len(text) > max_chars:
                text = text[:max_chars] + "\n[truncated]"

        return text

    def _compute_signals(
        self,
        chunk: RAGChunk,
        memory_slice: MemorySlice,
    ) -> MemorySignals:
        """Compute memory signals for a single chunk."""
        chunk_words = set(chunk.text.lower().split())

        related_heuristics = self._find_related_heuristics(
            chunk_words, memory_slice.heuristics
        )
        related_outcomes = self._find_related_outcomes(
            chunk_words, memory_slice.outcomes
        )
        anti_warnings = self._find_anti_pattern_warnings(
            chunk_words, memory_slice.anti_patterns
        )

        # Compute success rate from related outcomes
        success_rate = 0.5  # neutral default
        if related_outcomes:
            successes = sum(
                1
                for oid in related_outcomes
                for o in memory_slice.outcomes
                if o.id == oid and o.success
            )
            success_rate = (
                successes / len(related_outcomes) if related_outcomes else 0.5
            )

        # Compute confidence based on signal strength
        signal_count = (
            len(related_heuristics) + len(related_outcomes) + len(anti_warnings)
        )
        confidence = min(signal_count / 5.0, 1.0)  # Saturates at 5 signals

        # Compute boost factor
        boost = self.no_signal_neutral
        if related_heuristics and success_rate > 0.6:
            boost = self.success_boost * (0.5 + 0.5 * success_rate)
        if anti_warnings:
            boost *= self.anti_pattern_penalty

        # Trust score derived from heuristic confidence
        trust = 0.5
        if related_heuristics:
            h_confidences = [
                h.confidence
                for h in memory_slice.heuristics
                if h.id in related_heuristics
            ]
            if h_confidences:
                trust = sum(h_confidences) / len(h_confidences)

        return MemorySignals(
            related_heuristics=related_heuristics,
            related_outcomes=related_outcomes,
            trust_score=trust,
            historical_success_rate=success_rate,
            confidence=confidence,
            anti_pattern_warnings=anti_warnings,
            boost_factor=boost,
        )

    def _find_related_heuristics(
        self,
        chunk_words: set,
        heuristics: List[Heuristic],
    ) -> List[str]:
        """Find heuristics related to chunk content by keyword overlap."""
        related = []
        for h in heuristics:
            h_words = set(h.strategy.lower().split()) | set(h.condition.lower().split())
            overlap = len(chunk_words & h_words)
            if overlap >= 3:
                related.append(h.id)
        return related

    def _find_related_outcomes(
        self,
        chunk_words: set,
        outcomes: List[Outcome],
    ) -> List[str]:
        """Find outcomes related to chunk content by keyword overlap."""
        related = []
        for o in outcomes:
            o_words = set(o.task_description.lower().split()) | set(
                o.strategy_used.lower().split()
            )
            overlap = len(chunk_words & o_words)
            if overlap >= 3:
                related.append(o.id)
        return related

    def _find_anti_pattern_warnings(
        self,
        chunk_words: set,
        anti_patterns: List[AntiPattern],
    ) -> List[str]:
        """Find anti-patterns related to chunk content."""
        warnings = []
        for ap in anti_patterns:
            ap_words = set(ap.pattern.lower().split()) | set(ap.why_bad.lower().split())
            overlap = len(chunk_words & ap_words)
            if overlap >= 2:
                warnings.append(f"Avoid: {ap.pattern} - {ap.why_bad}")
        return warnings
