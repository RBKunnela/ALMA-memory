"""
ALMA Auto-Learning Module.

Bridges LLM-powered fact extraction with ALMA's learning protocols.
Enables Mem0-style automatic learning from conversations.
"""

import logging
from typing import Any, Dict, List, Optional

from alma.extraction.extractor import (
    ExtractedFact,
    FactExtractor,
    FactType,
    create_extractor,
)

logger = logging.getLogger(__name__)


class AutoLearner:
    """
    Automatic learning from conversations.

    This class bridges the gap between Mem0's automatic extraction
    and ALMA's explicit learning protocols. It:

    1. Extracts facts from conversations using LLM or rules
    2. Validates facts against agent scopes
    3. Deduplicates against existing memories
    4. Commits valid facts to ALMA storage

    Usage:
        alma = ALMA.from_config(".alma/config.yaml")
        auto_learner = AutoLearner(alma)

        # After a conversation
        results = auto_learner.learn_from_conversation(
            messages=[
                {"role": "user", "content": "Test the login form"},
                {"role": "assistant", "content": "I tested using incremental validation..."},
            ],
            agent="helena",
        )
    """

    def __init__(
        self,
        alma,  # ALMA instance - avoid circular import
        extractor: Optional[FactExtractor] = None,
        auto_commit: bool = True,
        min_confidence: float = 0.5,
    ):
        """
        Initialize AutoLearner.

        Args:
            alma: ALMA instance for storage and retrieval
            extractor: Custom extractor, or None for auto-detection
            auto_commit: Whether to automatically commit extracted facts
            min_confidence: Minimum confidence threshold for facts
        """
        self.alma = alma
        self.extractor = extractor or create_extractor()
        self.auto_commit = auto_commit
        self.min_confidence = min_confidence

    def learn_from_conversation(
        self,
        messages: List[Dict[str, str]],
        agent: str,
        user_id: Optional[str] = None,
        commit: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Extract and learn from a conversation.

        Args:
            messages: Conversation messages
            agent: Agent that had the conversation
            user_id: Optional user ID for preferences
            commit: Override auto_commit setting

        Returns:
            Dict with extraction results and commit status
        """
        should_commit = commit if commit is not None else self.auto_commit

        # Get agent scope for context
        scope = self.alma.scopes.get(agent)
        agent_context = None
        if scope:
            agent_context = f"Agent '{agent}' can learn: {scope.can_learn}. Cannot learn: {scope.cannot_learn}"

        # Get existing facts to avoid duplicates
        existing_memories = self.alma.retrieve(
            task=" ".join(m["content"] for m in messages[-3:]),  # Recent context
            agent=agent,
            top_k=20,
        )
        existing_facts = []
        for h in existing_memories.heuristics:
            existing_facts.append(f"{h.condition}: {h.strategy}")
        for ap in existing_memories.anti_patterns:
            existing_facts.append(f"AVOID: {ap.pattern}")
        for dk in existing_memories.domain_knowledge:
            existing_facts.append(dk.fact)

        # Extract facts
        extraction_result = self.extractor.extract(
            messages=messages,
            agent_context=agent_context,
            existing_facts=existing_facts if existing_facts else None,
        )

        # Filter by confidence and scope
        valid_facts = []
        rejected_facts = []

        for fact in extraction_result.facts:
            # Check confidence
            if fact.confidence < self.min_confidence:
                rejected_facts.append(
                    {
                        "fact": fact,
                        "reason": f"Low confidence: {fact.confidence} < {self.min_confidence}",
                    }
                )
                continue

            # Check scope for heuristics and anti-patterns
            if scope and fact.fact_type in (FactType.HEURISTIC, FactType.ANTI_PATTERN):
                # Infer domain from content
                inferred_domain = self._infer_domain(fact.content)
                if inferred_domain and not scope.is_allowed(inferred_domain):
                    rejected_facts.append(
                        {
                            "fact": fact,
                            "reason": f"Outside agent scope: {inferred_domain}",
                        }
                    )
                    continue

            valid_facts.append(fact)

        # Commit if enabled
        committed = []
        if should_commit:
            for fact in valid_facts:
                try:
                    result = self._commit_fact(fact, agent, user_id)
                    if result:
                        committed.append({"fact": fact, "id": result})
                except Exception as e:
                    logger.error(f"Failed to commit fact: {e}")
                    rejected_facts.append(
                        {
                            "fact": fact,
                            "reason": f"Commit failed: {str(e)}",
                        }
                    )

        return {
            "extracted_count": len(extraction_result.facts),
            "valid_count": len(valid_facts),
            "committed_count": len(committed),
            "rejected_count": len(rejected_facts),
            "extraction_time_ms": extraction_result.extraction_time_ms,
            "tokens_used": extraction_result.tokens_used,
            "committed": committed,
            "rejected": rejected_facts,
            "valid_facts": valid_facts,
        }

    def _commit_fact(
        self,
        fact: ExtractedFact,
        agent: str,
        user_id: Optional[str],
    ) -> Optional[str]:
        """Commit a single fact to ALMA storage."""

        if fact.fact_type == FactType.HEURISTIC:
            # Use learning protocol for heuristics
            return self.alma.learning.add_heuristic_direct(
                agent=agent,
                project_id=self.alma.project_id,
                condition=fact.condition or fact.content,
                strategy=fact.strategy or fact.content,
                confidence=fact.confidence,
                metadata={"source": "auto_extraction"},
            )

        elif fact.fact_type == FactType.ANTI_PATTERN:
            return self.alma.learning.add_anti_pattern(
                agent=agent,
                project_id=self.alma.project_id,
                pattern=fact.content,
                why_bad=fact.condition,
                better_alternative=fact.strategy,
            )

        elif fact.fact_type == FactType.PREFERENCE:
            if user_id:
                pref = self.alma.add_user_preference(
                    user_id=user_id,
                    category=fact.category or "general",
                    preference=fact.content,
                    source="auto_extraction",
                )
                return pref.id

        elif fact.fact_type == FactType.DOMAIN_KNOWLEDGE:
            # add_domain_knowledge now raises ScopeViolationError instead of returning None
            knowledge = self.alma.add_domain_knowledge(
                agent=agent,
                domain=fact.domain or "general",
                fact=fact.content,
                source="auto_extraction",
            )
            return knowledge.id

        elif fact.fact_type == FactType.OUTCOME:
            # Outcomes need success/failure info we don't have
            # Store as domain knowledge instead
            knowledge = self.alma.add_domain_knowledge(
                agent=agent,
                domain="outcomes",
                fact=fact.content,
                source="auto_extraction",
            )
            return knowledge.id

        return None

    def _infer_domain(self, content: str) -> Optional[str]:
        """Infer domain from fact content using keywords."""
        content_lower = content.lower()

        domain_keywords = {
            "testing": ["test", "assert", "selenium", "playwright", "cypress"],
            "frontend": ["css", "html", "react", "vue", "ui", "button", "form"],
            "backend": ["api", "database", "sql", "server", "endpoint"],
            "security": ["auth", "token", "password", "encrypt", "csrf"],
            "performance": ["latency", "cache", "optimize", "slow", "fast"],
        }

        for domain, keywords in domain_keywords.items():
            if any(kw in content_lower for kw in keywords):
                return domain

        return None


def add_auto_learning_to_alma(alma) -> AutoLearner:
    """
    Convenience function to add auto-learning to an ALMA instance.

    Usage:
        alma = ALMA.from_config(".alma/config.yaml")
        auto_learner = add_auto_learning_to_alma(alma)

        # Now use auto_learner.learn_from_conversation()
    """
    return AutoLearner(alma)
