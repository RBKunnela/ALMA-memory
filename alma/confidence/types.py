"""
Confidence Types.

Forward-looking confidence signals for strategies.
Inspired by Ilya Sutskever's insight: emotions are forward-looking value functions.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional


@dataclass
class RiskSignal:
    """
    A risk indicator for a strategy.

    Risks are signals that a strategy may not work in the current context.
    They can come from:
    - Similar past failures
    - Untested contexts
    - High complexity
    - Missing prerequisites
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Type of risk
    signal_type: str = (
        ""  # "similar_to_failure", "untested_context", "high_complexity", etc.
    )

    # Human-readable description
    description: str = ""

    # Severity: 0.0 = low risk, 1.0 = critical risk
    severity: float = 0.0

    # What triggered this signal
    source: str = ""  # "heuristic:h123", "anti_pattern:ap456", "context_analysis"

    # Related memory IDs
    related_memories: List[str] = field(default_factory=list)

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    detected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "signal_type": self.signal_type,
            "description": self.description,
            "severity": self.severity,
            "source": self.source,
            "related_memories": self.related_memories,
            "metadata": self.metadata,
            "detected_at": self.detected_at.isoformat(),
        }


@dataclass
class OpportunitySignal:
    """
    An opportunity indicator for a strategy.

    Opportunities are signals that a strategy is likely to succeed.
    They can come from:
    - Proven patterns with high success rate
    - High similarity to past successes
    - Recent successful uses
    - Strong prerequisites met
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Type of opportunity
    signal_type: str = ""  # "proven_pattern", "high_similarity", "recent_success", etc.

    # Human-readable description
    description: str = ""

    # Strength: 0.0 = weak signal, 1.0 = strong signal
    strength: float = 0.0

    # What triggered this signal
    source: str = ""  # "heuristic:h123", "outcome:o456", "pattern_match"

    # Related memory IDs
    related_memories: List[str] = field(default_factory=list)

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    detected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "signal_type": self.signal_type,
            "description": self.description,
            "strength": self.strength,
            "source": self.source,
            "related_memories": self.related_memories,
            "metadata": self.metadata,
            "detected_at": self.detected_at.isoformat(),
        }


# Recommendation levels
Recommendation = Literal["strong_yes", "yes", "neutral", "caution", "avoid"]


@dataclass
class ConfidenceSignal:
    """
    Forward-looking confidence assessment for a strategy.

    Combines backward-looking metrics (historical success) with
    forward-looking predictions (expected success in current context).

    This is the "gut feeling" that tells an agent whether a strategy
    is likely to work before trying it.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # What we're assessing
    strategy: str = ""
    context: str = ""
    agent: str = ""

    # Optional link to existing heuristic
    heuristic_id: Optional[str] = None

    # Backward-looking metrics (from historical data)
    historical_success_rate: float = 0.0  # 0-1, based on past outcomes
    occurrence_count: int = 0  # How many times this strategy was tried

    # Forward-looking predictions (computed for current context)
    predicted_success: float = 0.5  # Expected success in THIS context
    uncertainty: float = (
        0.5  # How uncertain is the prediction (0=certain, 1=very uncertain)
    )
    context_similarity: float = 0.0  # How similar is current context to past successes

    # Risk signals
    risk_signals: List[RiskSignal] = field(default_factory=list)
    total_risk_score: float = 0.0  # Aggregated risk (0=no risk, 1=high risk)

    # Opportunity signals
    opportunity_signals: List[OpportunitySignal] = field(default_factory=list)
    total_opportunity_score: float = 0.0  # Aggregated opportunity (0=none, 1=high)

    # Combined assessment
    confidence_score: float = 0.5  # Final weighted score (0-1)
    recommendation: Recommendation = "neutral"

    # Explanation
    reasoning: str = ""

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    assessed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @classmethod
    def create(
        cls,
        strategy: str,
        context: str,
        agent: str,
        heuristic_id: Optional[str] = None,
    ) -> "ConfidenceSignal":
        """Create a new confidence signal."""
        return cls(
            strategy=strategy,
            context=context,
            agent=agent,
            heuristic_id=heuristic_id,
        )

    def add_risk(
        self,
        signal_type: str,
        description: str,
        severity: float,
        source: str = "",
    ) -> RiskSignal:
        """Add a risk signal."""
        risk = RiskSignal(
            signal_type=signal_type,
            description=description,
            severity=severity,
            source=source,
        )
        self.risk_signals.append(risk)
        self._recalculate_scores()
        return risk

    def add_opportunity(
        self,
        signal_type: str,
        description: str,
        strength: float,
        source: str = "",
    ) -> OpportunitySignal:
        """Add an opportunity signal."""
        opportunity = OpportunitySignal(
            signal_type=signal_type,
            description=description,
            strength=strength,
            source=source,
        )
        self.opportunity_signals.append(opportunity)
        self._recalculate_scores()
        return opportunity

    def _recalculate_scores(self) -> None:
        """Recalculate total scores and recommendation."""
        # Aggregate risk
        if self.risk_signals:
            # Use max risk as the dominant signal
            self.total_risk_score = max(r.severity for r in self.risk_signals)
        else:
            self.total_risk_score = 0.0

        # Aggregate opportunity
        if self.opportunity_signals:
            # Use max opportunity as the dominant signal
            self.total_opportunity_score = max(
                o.strength for o in self.opportunity_signals
            )
        else:
            self.total_opportunity_score = 0.0

        # Combined confidence score
        # Weighs historical success, predicted success, and risk/opportunity balance
        base_confidence = (
            0.3 * self.historical_success_rate
            + 0.4 * self.predicted_success
            + 0.15 * self.context_similarity
            + 0.15 * (1.0 - self.uncertainty)
        )

        # Adjust for risk/opportunity
        risk_adjustment = -0.2 * self.total_risk_score
        opportunity_adjustment = 0.2 * self.total_opportunity_score

        self.confidence_score = max(
            0.0, min(1.0, base_confidence + risk_adjustment + opportunity_adjustment)
        )

        # Determine recommendation
        self._update_recommendation()

    def _update_recommendation(self) -> None:
        """Update recommendation based on confidence score and signals."""
        # High risk signals can override confidence
        if self.total_risk_score >= 0.8:
            self.recommendation = "avoid"
        elif self.total_risk_score >= 0.6:
            self.recommendation = "caution"
        elif self.confidence_score >= 0.8:
            self.recommendation = "strong_yes"
        elif self.confidence_score >= 0.6:
            self.recommendation = "yes"
        elif self.confidence_score >= 0.4:
            self.recommendation = "neutral"
        elif self.confidence_score >= 0.2:
            self.recommendation = "caution"
        else:
            self.recommendation = "avoid"

    def to_prompt(self) -> str:
        """Format confidence signal for prompt injection."""
        lines = [
            f"## Confidence Assessment: {self.strategy[:50]}...",
            f"**Recommendation: {self.recommendation.upper()}** (score: {self.confidence_score:.2f})",
            "",
        ]

        # Metrics
        lines.append("### Metrics")
        lines.append(
            f"- Historical success: {self.historical_success_rate:.0%} ({self.occurrence_count} uses)"
        )
        lines.append(f"- Predicted success: {self.predicted_success:.0%}")
        lines.append(f"- Context similarity: {self.context_similarity:.0%}")
        lines.append(f"- Uncertainty: {self.uncertainty:.0%}")
        lines.append("")

        # Risks
        if self.risk_signals:
            lines.append("### Risks")
            for risk in self.risk_signals:
                severity_label = (
                    "HIGH"
                    if risk.severity >= 0.7
                    else "MEDIUM"
                    if risk.severity >= 0.4
                    else "LOW"
                )
                lines.append(f"- [{severity_label}] {risk.description}")
            lines.append("")

        # Opportunities
        if self.opportunity_signals:
            lines.append("### Opportunities")
            for opp in self.opportunity_signals:
                strength_label = (
                    "STRONG"
                    if opp.strength >= 0.7
                    else "MODERATE"
                    if opp.strength >= 0.4
                    else "WEAK"
                )
                lines.append(f"- [{strength_label}] {opp.description}")
            lines.append("")

        # Reasoning
        if self.reasoning:
            lines.append("### Analysis")
            lines.append(self.reasoning)

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "strategy": self.strategy,
            "context": self.context,
            "agent": self.agent,
            "heuristic_id": self.heuristic_id,
            "historical_success_rate": self.historical_success_rate,
            "occurrence_count": self.occurrence_count,
            "predicted_success": self.predicted_success,
            "uncertainty": self.uncertainty,
            "context_similarity": self.context_similarity,
            "risk_signals": [r.to_dict() for r in self.risk_signals],
            "total_risk_score": self.total_risk_score,
            "opportunity_signals": [o.to_dict() for o in self.opportunity_signals],
            "total_opportunity_score": self.total_opportunity_score,
            "confidence_score": self.confidence_score,
            "recommendation": self.recommendation,
            "reasoning": self.reasoning,
            "metadata": self.metadata,
            "assessed_at": self.assessed_at.isoformat(),
        }
