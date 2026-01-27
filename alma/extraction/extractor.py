"""
ALMA Fact Extraction Module.

LLM-powered extraction of facts, preferences, and learnings from conversations.
This bridges the gap between Mem0's automatic extraction and ALMA's explicit learning.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class FactType(Enum):
    """Types of facts that can be extracted from conversations."""

    HEURISTIC = "heuristic"  # Strategy that worked
    ANTI_PATTERN = "anti_pattern"  # What NOT to do
    PREFERENCE = "preference"  # User preference
    DOMAIN_KNOWLEDGE = "domain_knowledge"  # Factual information
    OUTCOME = "outcome"  # Task result


@dataclass
class ExtractedFact:
    """A fact extracted from conversation."""

    fact_type: FactType
    content: str
    confidence: float  # 0.0 to 1.0
    source_text: str  # Original text this was extracted from
    metadata: Dict[str, Any] = None

    # For heuristics/anti-patterns
    condition: Optional[str] = None  # When does this apply?
    strategy: Optional[str] = None  # What to do?

    # For preferences
    category: Optional[str] = None

    # For domain knowledge
    domain: Optional[str] = None


@dataclass
class ExtractionResult:
    """Result of fact extraction from a conversation."""

    facts: List[ExtractedFact]
    raw_response: str  # LLM's raw response for debugging
    tokens_used: int
    extraction_time_ms: int


class FactExtractor(ABC):
    """Abstract base class for fact extraction."""

    @abstractmethod
    def extract(
        self,
        messages: List[Dict[str, str]],
        agent_context: Optional[str] = None,
        existing_facts: Optional[List[str]] = None,
    ) -> ExtractionResult:
        """
        Extract facts from a conversation.

        Args:
            messages: List of {"role": "user"|"assistant", "content": "..."}
            agent_context: Optional context about the agent's domain
            existing_facts: Optional list of already-known facts to avoid duplicates

        Returns:
            ExtractionResult with extracted facts
        """
        pass


class LLMFactExtractor(FactExtractor):
    """
    LLM-powered fact extraction.

    Uses structured prompting to extract facts, preferences, and learnings
    from conversations. Supports OpenAI, Anthropic, and local models.
    """

    EXTRACTION_PROMPT = """You are a fact extraction system for an AI agent memory architecture.

Analyze the following conversation and extract facts worth remembering.

IMPORTANT: Only extract facts that are:
1. Specific and actionable (not vague observations)
2. Likely to be useful in future similar situations
3. Not already in the existing facts list

Categorize each fact as one of:
- HEURISTIC: A strategy or approach that worked well
- ANTI_PATTERN: Something that failed or should be avoided
- PREFERENCE: A user preference or constraint
- DOMAIN_KNOWLEDGE: A factual piece of information about the domain
- OUTCOME: The result of a specific task

For HEURISTIC and ANTI_PATTERN, also extract:
- condition: When does this apply?
- strategy: What to do (or not do)?

For PREFERENCE, extract:
- category: What type of preference (communication, code_style, workflow, etc.)

For DOMAIN_KNOWLEDGE, extract:
- domain: What knowledge domain this belongs to

{agent_context}

{existing_facts_section}

CONVERSATION:
{conversation}

Respond in JSON format:
```json
{{
  "facts": [
    {{
      "fact_type": "HEURISTIC|ANTI_PATTERN|PREFERENCE|DOMAIN_KNOWLEDGE|OUTCOME",
      "content": "The main fact statement",
      "confidence": 0.0-1.0,
      "condition": "optional - when this applies",
      "strategy": "optional - what to do",
      "category": "optional - preference category",
      "domain": "optional - knowledge domain"
    }}
  ]
}}
```

If no facts worth extracting, return: {{"facts": []}}
"""

    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        temperature: float = 0.1,
    ):
        """
        Initialize LLM fact extractor.

        Args:
            provider: "openai", "anthropic", or "local"
            model: Model name/identifier
            api_key: API key (or use environment variable)
            temperature: LLM temperature for extraction
        """
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self._client = None

    def _get_client(self):
        """Lazy initialization of LLM client."""
        if self._client is None:
            if self.provider == "openai":
                from openai import OpenAI

                self._client = OpenAI(api_key=self.api_key)
            elif self.provider == "anthropic":
                from anthropic import Anthropic

                self._client = Anthropic(api_key=self.api_key)
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
        return self._client

    def extract(
        self,
        messages: List[Dict[str, str]],
        agent_context: Optional[str] = None,
        existing_facts: Optional[List[str]] = None,
    ) -> ExtractionResult:
        """Extract facts from conversation using LLM."""
        import time

        start_time = time.time()

        # Format conversation
        conversation = "\n".join(
            f"{msg['role'].upper()}: {msg['content']}" for msg in messages
        )

        # Build prompt
        agent_context_section = ""
        if agent_context:
            agent_context_section = f"\nAGENT CONTEXT:\n{agent_context}\n"

        existing_facts_section = ""
        if existing_facts:
            facts_list = "\n".join(f"- {f}" for f in existing_facts)
            existing_facts_section = (
                f"\nEXISTING FACTS (do not duplicate):\n{facts_list}\n"
            )

        prompt = self.EXTRACTION_PROMPT.format(
            agent_context=agent_context_section,
            existing_facts_section=existing_facts_section,
            conversation=conversation,
        )

        # Call LLM
        client = self._get_client()
        tokens_used = 0

        if self.provider == "openai":
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
            )
            raw_response = response.choices[0].message.content
            tokens_used = response.usage.total_tokens if response.usage else 0

        elif self.provider == "anthropic":
            response = client.messages.create(
                model=self.model,
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}],
            )
            raw_response = response.content[0].text
            tokens_used = response.usage.input_tokens + response.usage.output_tokens

        # Parse response
        facts = self._parse_response(raw_response, conversation)

        extraction_time_ms = int((time.time() - start_time) * 1000)

        return ExtractionResult(
            facts=facts,
            raw_response=raw_response,
            tokens_used=tokens_used,
            extraction_time_ms=extraction_time_ms,
        )

    def _parse_response(
        self,
        raw_response: str,
        source_text: str,
    ) -> List[ExtractedFact]:
        """Parse LLM response into ExtractedFact objects."""
        import json
        import re

        # Extract JSON from response (handle markdown code blocks)
        json_match = re.search(r"```json\s*(.*?)\s*```", raw_response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find raw JSON
            json_match = re.search(r"\{.*\}", raw_response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                logger.warning(
                    f"Could not parse JSON from response: {raw_response[:200]}"
                )
                return []

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error: {e}")
            return []

        facts = []
        for item in data.get("facts", []):
            try:
                fact_type = FactType[item["fact_type"].upper()]
                facts.append(
                    ExtractedFact(
                        fact_type=fact_type,
                        content=item["content"],
                        confidence=float(item.get("confidence", 0.7)),
                        source_text=source_text[:500],  # Truncate for storage
                        condition=item.get("condition"),
                        strategy=item.get("strategy"),
                        category=item.get("category"),
                        domain=item.get("domain"),
                    )
                )
            except (KeyError, ValueError) as e:
                logger.warning(f"Could not parse fact: {item}, error: {e}")
                continue

        return facts


class RuleBasedExtractor(FactExtractor):
    """
    Rule-based fact extraction for offline/free usage.

    Uses pattern matching and heuristics instead of LLM calls.
    Less accurate but free and fast.
    """

    # Patterns that indicate different fact types
    HEURISTIC_PATTERNS = [
        r"(?:worked|succeeded|fixed|solved|helped).*(?:by|using|with)",
        r"(?:better|best|good)\s+(?:to|approach|way|strategy)",
        r"(?:should|always|recommend).*(?:use|try|do)",
    ]

    ANTI_PATTERN_PATTERNS = [
        r"(?:don't|do not|never|avoid).*(?:use|do|try)",
        r"(?:failed|broke|caused|error).*(?:because|when|due)",
        r"(?:bad|wrong|incorrect)\s+(?:to|approach|way)",
    ]

    PREFERENCE_PATTERNS = [
        r"(?:i|user)\s+(?:prefer|like|want|need)",
        r"(?:always|never).*(?:for me|i want)",
    ]

    def extract(
        self,
        messages: List[Dict[str, str]],
        agent_context: Optional[str] = None,
        existing_facts: Optional[List[str]] = None,
    ) -> ExtractionResult:
        """Extract facts using pattern matching."""
        import re
        import time

        start_time = time.time()
        facts = []

        for msg in messages:
            content = msg["content"].lower()

            # Check for heuristics
            for pattern in self.HEURISTIC_PATTERNS:
                if re.search(pattern, content, re.IGNORECASE):
                    facts.append(
                        ExtractedFact(
                            fact_type=FactType.HEURISTIC,
                            content=msg["content"][:200],
                            confidence=0.5,  # Lower confidence for rule-based
                            source_text=msg["content"],
                        )
                    )
                    break

            # Check for anti-patterns
            for pattern in self.ANTI_PATTERN_PATTERNS:
                if re.search(pattern, content, re.IGNORECASE):
                    facts.append(
                        ExtractedFact(
                            fact_type=FactType.ANTI_PATTERN,
                            content=msg["content"][:200],
                            confidence=0.5,
                            source_text=msg["content"],
                        )
                    )
                    break

            # Check for preferences
            for pattern in self.PREFERENCE_PATTERNS:
                if re.search(pattern, content, re.IGNORECASE):
                    facts.append(
                        ExtractedFact(
                            fact_type=FactType.PREFERENCE,
                            content=msg["content"][:200],
                            confidence=0.5,
                            source_text=msg["content"],
                        )
                    )
                    break

        extraction_time_ms = int((time.time() - start_time) * 1000)

        return ExtractionResult(
            facts=facts,
            raw_response="rule-based extraction",
            tokens_used=0,
            extraction_time_ms=extraction_time_ms,
        )


def create_extractor(
    provider: str = "auto",
    **kwargs,
) -> FactExtractor:
    """
    Factory function to create appropriate extractor.

    Args:
        provider: "openai", "anthropic", "local", "rule-based", or "auto"
        **kwargs: Additional arguments for the extractor

    Returns:
        Configured FactExtractor instance
    """
    if provider == "auto":
        # Try to use LLM if API key is available
        import os

        if os.environ.get("OPENAI_API_KEY"):
            provider = "openai"
        elif os.environ.get("ANTHROPIC_API_KEY"):
            provider = "anthropic"
        else:
            provider = "rule-based"

    if provider == "rule-based":
        return RuleBasedExtractor()
    else:
        return LLMFactExtractor(provider=provider, **kwargs)
