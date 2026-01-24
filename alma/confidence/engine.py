"""
Confidence Engine.

Computes forward-looking confidence for strategies.
Not just "this worked before" but "this will likely work now."
"""

import logging
from typing import Any, List, Optional, Tuple

from alma.confidence.types import (
    ConfidenceSignal,
    OpportunitySignal,
    RiskSignal,
)

logger = logging.getLogger(__name__)


class ConfidenceEngine:
    """
    Compute forward-looking confidence for strategies.

    The engine combines:
    1. Historical data (past outcomes, heuristics)
    2. Context analysis (similarity to past successes/failures)
    3. Risk detection (anti-patterns, untested contexts)
    4. Opportunity detection (proven patterns, recent successes)

    Usage:
        engine = ConfidenceEngine(alma)

        # Assess a single strategy
        signal = engine.assess_strategy(
            strategy="Use incremental validation for forms",
            context="Testing a 5-field registration form",
            agent="Helena",
        )
        print(f"Confidence: {signal.confidence_score:.0%}")
        print(f"Recommendation: {signal.recommendation}")

        # Rank multiple strategies
        rankings = engine.rank_strategies(
            strategies=["Strategy A", "Strategy B", "Strategy C"],
            context="Current task context",
            agent="Helena",
        )
        for strategy, signal in rankings:
            print(f"{strategy}: {signal.recommendation}")
    """

    def __init__(
        self,
        alma: Optional[Any] = None,
        similarity_threshold: float = 0.7,
        min_occurrences_for_confidence: int = 3,
    ):
        """
        Initialize the ConfidenceEngine.

        Args:
            alma: ALMA instance for memory access
            similarity_threshold: Minimum similarity for "high similarity" signals
            min_occurrences_for_confidence: Minimum uses before trusting historical rate
        """
        self.alma = alma
        self.similarity_threshold = similarity_threshold
        self.min_occurrences_for_confidence = min_occurrences_for_confidence

    def assess_strategy(
        self,
        strategy: str,
        context: str,
        agent: str,
        heuristic: Optional[Any] = None,
    ) -> ConfidenceSignal:
        """
        Assess confidence for a strategy in the current context.

        Args:
            strategy: The strategy to assess
            context: Current task/context description
            agent: Agent name
            heuristic: Optional existing heuristic for this strategy

        Returns:
            ConfidenceSignal with full assessment
        """
        signal = ConfidenceSignal.create(
            strategy=strategy,
            context=context,
            agent=agent,
            heuristic_id=heuristic.id if heuristic else None,
        )

        # 1. Load historical data from heuristic
        if heuristic:
            signal.occurrence_count = getattr(heuristic, 'occurrence_count', 0)
            success_count = getattr(heuristic, 'success_count', 0)
            if signal.occurrence_count > 0:
                signal.historical_success_rate = success_count / signal.occurrence_count
            signal.metadata["heuristic_confidence"] = getattr(heuristic, 'confidence', 0.5)

        # 2. Analyze context similarity
        signal.context_similarity = self._compute_context_similarity(
            strategy=strategy,
            context=context,
            agent=agent,
        )

        # 3. Compute predicted success
        signal.predicted_success = self._predict_success(
            strategy=strategy,
            context=context,
            agent=agent,
            historical_rate=signal.historical_success_rate,
            context_similarity=signal.context_similarity,
        )

        # 4. Compute uncertainty
        signal.uncertainty = self._compute_uncertainty(
            occurrence_count=signal.occurrence_count,
            context_similarity=signal.context_similarity,
        )

        # 5. Detect risks
        risks = self.detect_risks(strategy, context, agent)
        for risk in risks:
            signal.risk_signals.append(risk)

        # 6. Detect opportunities
        opportunities = self.detect_opportunities(strategy, context, agent)
        for opp in opportunities:
            signal.opportunity_signals.append(opp)

        # 7. Recalculate scores (triggers recommendation update)
        signal._recalculate_scores()

        # 8. Generate reasoning
        signal.reasoning = self._generate_reasoning(signal)

        logger.debug(
            f"Assessed strategy '{strategy[:30]}...': "
            f"confidence={signal.confidence_score:.2f}, "
            f"recommendation={signal.recommendation}"
        )

        return signal

    def rank_strategies(
        self,
        strategies: List[str],
        context: str,
        agent: str,
    ) -> List[Tuple[str, ConfidenceSignal]]:
        """
        Rank multiple strategies by confidence.

        Args:
            strategies: List of strategies to rank
            context: Current context
            agent: Agent name

        Returns:
            List of (strategy, signal) tuples, sorted by confidence (highest first)
        """
        results = []

        for strategy in strategies:
            signal = self.assess_strategy(
                strategy=strategy,
                context=context,
                agent=agent,
            )
            results.append((strategy, signal))

        # Sort by confidence score (highest first)
        results.sort(key=lambda x: x[1].confidence_score, reverse=True)

        return results

    def detect_risks(
        self,
        strategy: str,
        context: str,
        agent: str,
    ) -> List[RiskSignal]:
        """
        Detect risk signals for a strategy.

        Checks for:
        - Similar past failures (anti-patterns)
        - Untested contexts
        - High complexity indicators
        - Missing prerequisites

        Args:
            strategy: Strategy to assess
            context: Current context
            agent: Agent name

        Returns:
            List of detected risk signals
        """
        risks = []

        # Check for anti-patterns in ALMA
        if self.alma:
            try:
                # Search for similar anti-patterns
                memories = self.alma.retrieve(
                    task=f"{strategy} {context}",
                    agent=agent,
                    top_k=10,
                )

                if memories and hasattr(memories, 'anti_patterns'):
                    for ap in memories.anti_patterns[:3]:
                        # Check if this anti-pattern relates to our strategy
                        if self._is_similar(strategy, ap.strategy):
                            risks.append(RiskSignal(
                                signal_type="similar_to_failure",
                                description=f"Similar to known anti-pattern: {ap.reason[:100]}",
                                severity=0.7,
                                source=f"anti_pattern:{ap.id}",
                                related_memories=[ap.id],
                            ))
            except Exception as e:
                logger.warning(f"Failed to check anti-patterns: {e}")

        # Check for complexity indicators
        complexity_keywords = ["complex", "multiple", "all", "every", "entire", "complete"]
        complexity_score = sum(1 for kw in complexity_keywords if kw in strategy.lower())
        if complexity_score >= 2:
            risks.append(RiskSignal(
                signal_type="high_complexity",
                description="Strategy appears complex - consider breaking into smaller steps",
                severity=0.4,
                source="context_analysis",
            ))

        # Check for risky patterns
        risky_patterns = [
            ("sleep", "Time-based waits can cause flaky behavior", 0.6),
            ("force", "Force operations can have unintended side effects", 0.5),
            ("delete all", "Bulk deletions are high-risk", 0.8),
            ("production", "Production operations require extra caution", 0.7),
        ]
        for pattern, description, severity in risky_patterns:
            if pattern in strategy.lower():
                risks.append(RiskSignal(
                    signal_type="risky_pattern",
                    description=description,
                    severity=severity,
                    source="pattern_match",
                ))

        return risks

    def detect_opportunities(
        self,
        strategy: str,
        context: str,
        agent: str,
    ) -> List[OpportunitySignal]:
        """
        Detect opportunity signals for a strategy.

        Checks for:
        - Proven patterns with high success rate
        - High similarity to past successes
        - Recent successful uses
        - Strong prerequisite matches

        Args:
            strategy: Strategy to assess
            context: Current context
            agent: Agent name

        Returns:
            List of detected opportunity signals
        """
        opportunities = []

        # Check for matching heuristics in ALMA
        if self.alma:
            try:
                memories = self.alma.retrieve(
                    task=f"{strategy} {context}",
                    agent=agent,
                    top_k=10,
                )

                if memories and hasattr(memories, 'heuristics'):
                    for h in memories.heuristics[:3]:
                        # Check success rate
                        if h.occurrence_count >= self.min_occurrences_for_confidence:
                            success_rate = h.success_count / h.occurrence_count if h.occurrence_count > 0 else 0
                            if success_rate >= 0.8:
                                opportunities.append(OpportunitySignal(
                                    signal_type="proven_pattern",
                                    description=f"Proven strategy with {success_rate:.0%} success rate over {h.occurrence_count} uses",
                                    strength=min(0.9, success_rate),
                                    source=f"heuristic:{h.id}",
                                    related_memories=[h.id],
                                ))

                # Check for recent successes in outcomes
                if hasattr(memories, 'outcomes'):
                    recent_successes = [
                        o for o in memories.outcomes
                        if getattr(o, 'outcome', '') == 'success'
                    ][:3]
                    if recent_successes:
                        opportunities.append(OpportunitySignal(
                            signal_type="recent_success",
                            description=f"Similar approach succeeded recently ({len(recent_successes)} recent successes)",
                            strength=0.6,
                            source="outcome_analysis",
                            related_memories=[o.id for o in recent_successes],
                        ))

            except Exception as e:
                logger.warning(f"Failed to check opportunities: {e}")

        # Check for best practice patterns
        best_practices = [
            ("incremental", "Incremental approaches reduce risk", 0.5),
            ("test first", "Test-first approaches catch issues early", 0.6),
            ("validate", "Validation prevents downstream errors", 0.5),
            ("small steps", "Small steps are easier to debug", 0.4),
        ]
        for pattern, description, strength in best_practices:
            if pattern in strategy.lower():
                opportunities.append(OpportunitySignal(
                    signal_type="best_practice",
                    description=description,
                    strength=strength,
                    source="pattern_match",
                ))

        return opportunities

    def _compute_context_similarity(
        self,
        strategy: str,
        context: str,
        agent: str,
    ) -> float:
        """Compute similarity between current context and past successful contexts."""
        if not self.alma:
            return 0.5  # Default when no memory available

        try:
            # Retrieve relevant memories
            memories = self.alma.retrieve(
                task=context,
                agent=agent,
                top_k=5,
            )

            if not memories:
                return 0.3  # Low similarity for novel contexts

            # Check if any outcomes match our strategy
            if hasattr(memories, 'outcomes'):
                matching_outcomes = [
                    o for o in memories.outcomes
                    if self._is_similar(strategy, getattr(o, 'strategy_used', ''))
                ]
                if matching_outcomes:
                    return 0.8  # High similarity

            # Check heuristics
            if hasattr(memories, 'heuristics'):
                matching_heuristics = [
                    h for h in memories.heuristics
                    if self._is_similar(strategy, getattr(h, 'strategy', ''))
                ]
                if matching_heuristics:
                    return 0.7

            return 0.5  # Moderate similarity

        except Exception as e:
            logger.warning(f"Failed to compute context similarity: {e}")
            return 0.5

    def _predict_success(
        self,
        strategy: str,
        context: str,
        agent: str,
        historical_rate: float,
        context_similarity: float,
    ) -> float:
        """
        Predict success probability for the strategy in current context.

        Combines historical rate with context similarity adjustment.
        """
        # Base prediction from historical rate
        if historical_rate > 0:
            base = historical_rate
        else:
            base = 0.5  # Unknown strategies start at 50%

        # Adjust for context similarity
        # High similarity → trust historical rate more
        # Low similarity → regress toward 50%
        similarity_weight = context_similarity
        predicted = (similarity_weight * base) + ((1 - similarity_weight) * 0.5)

        return predicted

    def _compute_uncertainty(
        self,
        occurrence_count: int,
        context_similarity: float,
    ) -> float:
        """
        Compute uncertainty in the prediction.

        Higher uncertainty when:
        - Few occurrences (limited data)
        - Low context similarity (novel situation)
        """
        # Uncertainty decreases with more data
        if occurrence_count >= 10:
            data_uncertainty = 0.1
        elif occurrence_count >= 5:
            data_uncertainty = 0.3
        elif occurrence_count >= 2:
            data_uncertainty = 0.5
        else:
            data_uncertainty = 0.8

        # Uncertainty increases with novel contexts
        context_uncertainty = 1.0 - context_similarity

        # Combined uncertainty
        return min(1.0, (data_uncertainty + context_uncertainty) / 2)

    def _is_similar(self, text1: str, text2: str) -> bool:
        """Simple similarity check between two texts."""
        if not text1 or not text2:
            return False

        # Normalize
        t1 = text1.lower().strip()
        t2 = text2.lower().strip()

        # Exact match
        if t1 == t2:
            return True

        # Substring match
        if t1 in t2 or t2 in t1:
            return True

        # Word overlap
        words1 = set(t1.split())
        words2 = set(t2.split())
        overlap = len(words1 & words2)
        total = len(words1 | words2)

        if total > 0 and overlap / total >= 0.5:
            return True

        return False

    def _generate_reasoning(self, signal: ConfidenceSignal) -> str:
        """Generate human-readable reasoning for the assessment."""
        parts = []

        # Historical data
        if signal.occurrence_count >= self.min_occurrences_for_confidence:
            parts.append(
                f"Historical data shows {signal.historical_success_rate:.0%} success rate "
                f"over {signal.occurrence_count} uses."
            )
        elif signal.occurrence_count > 0:
            parts.append(
                f"Limited historical data ({signal.occurrence_count} uses) - "
                f"prediction has higher uncertainty."
            )
        else:
            parts.append("No historical data for this strategy - treating as novel.")

        # Context similarity
        if signal.context_similarity >= 0.7:
            parts.append("Current context is highly similar to past successful applications.")
        elif signal.context_similarity <= 0.3:
            parts.append("Current context is quite different from past applications.")

        # Key risks
        high_risks = [r for r in signal.risk_signals if r.severity >= 0.6]
        if high_risks:
            parts.append(f"WARNING: {len(high_risks)} significant risk(s) detected.")

        # Key opportunities
        strong_opps = [o for o in signal.opportunity_signals if o.strength >= 0.6]
        if strong_opps:
            parts.append(f"POSITIVE: {len(strong_opps)} strong opportunity signal(s) detected.")

        return " ".join(parts)
