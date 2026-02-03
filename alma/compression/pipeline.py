"""
ALMA Memory Compression Pipeline.

Provides intelligent compression of verbose content into structured, efficient memories.
Supports both LLM-based intelligent extraction and rule-based fallback.

Target compression ratios:
- LIGHT: 1.5x (remove redundancy)
- MEDIUM: 3x (extract key points)
- AGGRESSIVE: 5x+ (maximum compression)
"""

import logging
import re
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol

logger = logging.getLogger(__name__)


class CompressionLevel(Enum):
    """Compression level for memory content."""

    NONE = "none"  # No compression, preserve original
    LIGHT = "light"  # Remove redundancy only (~1.5x)
    MEDIUM = "medium"  # Extract key points (~3x)
    AGGRESSIVE = "aggressive"  # Maximum compression (~5x+)


@dataclass
class CompressedMemory:
    """
    Result of compressing verbose content.

    Attributes:
        original_length: Length of original content in characters
        compressed_length: Length of compressed summary
        compression_ratio: Ratio of original to compressed length
        key_facts: Essential facts extracted from content
        constraints: Limitations or requirements discovered
        patterns: Reusable patterns identified
        summary: Compressed content for storage
        full_content: Original content preserved for verification
    """

    original_length: int
    compressed_length: int
    compression_ratio: float
    key_facts: List[str]
    constraints: List[str]
    patterns: List[str]
    summary: str
    full_content: str

    def to_metadata(self) -> Dict[str, Any]:
        """Generate metadata to store with memory."""
        return {
            "compressed": True,
            "compression_ratio": round(self.compression_ratio, 2),
            "original_length": self.original_length,
            "compressed_length": self.compressed_length,
            "key_facts": self.key_facts,
            "constraints": self.constraints,
            "patterns": self.patterns,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "original_length": self.original_length,
            "compressed_length": self.compressed_length,
            "compression_ratio": self.compression_ratio,
            "key_facts": self.key_facts,
            "constraints": self.constraints,
            "patterns": self.patterns,
            "summary": self.summary,
            "full_content": self.full_content,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CompressedMemory":
        """Create from dictionary."""
        return cls(
            original_length=data["original_length"],
            compressed_length=data["compressed_length"],
            compression_ratio=data["compression_ratio"],
            key_facts=data.get("key_facts", []),
            constraints=data.get("constraints", []),
            patterns=data.get("patterns", []),
            summary=data["summary"],
            full_content=data["full_content"],
        )


@dataclass
class CompressionResult:
    """
    Full result of a compression operation.

    Attributes:
        compressed: The compressed memory
        level: Compression level used
        method: Method used (llm or rule_based)
        compression_time_ms: Time taken for compression
        success: Whether compression succeeded
        error: Error message if failed
    """

    compressed: Optional[CompressedMemory]
    level: CompressionLevel
    method: str  # "llm" or "rule_based"
    compression_time_ms: int = 0
    success: bool = True
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "compressed": self.compressed.to_dict() if self.compressed else None,
            "level": self.level.value,
            "method": self.method,
            "compression_time_ms": self.compression_time_ms,
            "success": self.success,
            "error": self.error,
        }


@dataclass
class CompressionConfig:
    """
    Configuration for compression behavior.

    Attributes:
        default_level: Default compression level
        min_length_for_compression: Minimum content length to compress
        max_key_facts: Maximum number of key facts to extract
        max_constraints: Maximum number of constraints to extract
        max_patterns: Maximum number of patterns to extract
        preserve_full_content: Whether to preserve original content
        llm_timeout_seconds: Timeout for LLM-based compression
    """

    default_level: CompressionLevel = CompressionLevel.MEDIUM
    min_length_for_compression: int = 200
    max_key_facts: int = 5
    max_constraints: int = 3
    max_patterns: int = 3
    preserve_full_content: bool = True
    llm_timeout_seconds: float = 10.0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CompressionConfig":
        """Create from dictionary."""
        level_str = data.get("default_level", "medium")
        return cls(
            default_level=CompressionLevel(level_str),
            min_length_for_compression=data.get("min_length_for_compression", 200),
            max_key_facts=data.get("max_key_facts", 5),
            max_constraints=data.get("max_constraints", 3),
            max_patterns=data.get("max_patterns", 3),
            preserve_full_content=data.get("preserve_full_content", True),
            llm_timeout_seconds=data.get("llm_timeout_seconds", 10.0),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "default_level": self.default_level.value,
            "min_length_for_compression": self.min_length_for_compression,
            "max_key_facts": self.max_key_facts,
            "max_constraints": self.max_constraints,
            "max_patterns": self.max_patterns,
            "preserve_full_content": self.preserve_full_content,
            "llm_timeout_seconds": self.llm_timeout_seconds,
        }


class LLMClient(Protocol):
    """Protocol for LLM clients used in compression."""

    def complete(self, prompt: str, timeout: Optional[float] = None) -> str:
        """Complete a prompt and return the response."""
        ...


class MemoryCompressor:
    """
    Intelligent compression of verbose content into structured memories.

    Supports both LLM-based intelligent extraction and rule-based fallback.
    Achieves 3-5x compression ratio while preserving essential information.

    Example:
        compressor = MemoryCompressor(llm_client=llm)
        result = compressor.compress_outcome(
            "Long verbose task outcome description...",
            level=CompressionLevel.MEDIUM
        )
        print(f"Compressed {result.compression_ratio:.1f}x")
        print(f"Key facts: {result.key_facts}")
    """

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        config: Optional[CompressionConfig] = None,
    ):
        """
        Initialize compressor.

        Args:
            llm_client: Optional LLM client for intelligent compression
            config: Compression configuration
        """
        self.llm = llm_client
        self.config = config or CompressionConfig()

    def compress(
        self,
        content: str,
        level: Optional[CompressionLevel] = None,
        content_type: str = "general",
    ) -> CompressionResult:
        """
        Compress content using the appropriate method.

        Args:
            content: Content to compress
            level: Compression level (default from config)
            content_type: Type of content (outcome, conversation, general)

        Returns:
            CompressionResult with compressed memory and metadata
        """
        level = level or self.config.default_level
        start_time = time.time()

        try:
            # Skip compression for short content
            if len(content) < self.config.min_length_for_compression:
                compressed = self._no_compression(content)
                return CompressionResult(
                    compressed=compressed,
                    level=CompressionLevel.NONE,
                    method="skip",
                    compression_time_ms=int((time.time() - start_time) * 1000),
                )

            # No compression requested
            if level == CompressionLevel.NONE:
                compressed = self._no_compression(content)
                return CompressionResult(
                    compressed=compressed,
                    level=level,
                    method="none",
                    compression_time_ms=int((time.time() - start_time) * 1000),
                )

            # Try LLM-based compression
            if self.llm:
                try:
                    if content_type == "outcome":
                        compressed = self._llm_compress_outcome(content, level)
                    elif content_type == "conversation":
                        compressed = self._llm_compress_conversation(content, level)
                    else:
                        compressed = self._llm_compress_general(content, level)

                    return CompressionResult(
                        compressed=compressed,
                        level=level,
                        method="llm",
                        compression_time_ms=int((time.time() - start_time) * 1000),
                    )
                except Exception as e:
                    logger.warning(f"LLM compression failed, falling back: {e}")
                    # Fall through to rule-based

            # Rule-based fallback
            compressed = self._rule_based_compression(content, level)
            return CompressionResult(
                compressed=compressed,
                level=level,
                method="rule_based",
                compression_time_ms=int((time.time() - start_time) * 1000),
            )

        except Exception as e:
            logger.error(f"Compression failed: {e}")
            return CompressionResult(
                compressed=None,
                level=level,
                method="error",
                compression_time_ms=int((time.time() - start_time) * 1000),
                success=False,
                error=str(e),
            )

    def compress_outcome(
        self,
        verbose_outcome: str,
        level: Optional[CompressionLevel] = None,
    ) -> CompressedMemory:
        """
        Compress a task outcome into structured memory.

        Args:
            verbose_outcome: Verbose task outcome description
            level: Compression level

        Returns:
            CompressedMemory with extracted key information
        """
        result = self.compress(verbose_outcome, level, content_type="outcome")
        if result.compressed:
            return result.compressed
        # Fallback to no compression on error
        return self._no_compression(verbose_outcome)

    def compress_conversation(
        self,
        conversation: str,
        focus: Optional[str] = None,
        level: Optional[CompressionLevel] = None,
    ) -> CompressedMemory:
        """
        Extract learnable knowledge from a conversation.

        Args:
            conversation: Conversation content
            focus: Optional focus area for extraction
            level: Compression level

        Returns:
            CompressedMemory with extracted knowledge
        """
        level = level or self.config.default_level

        if level == CompressionLevel.NONE:
            return self._no_compression(conversation)

        if self.llm:
            try:
                return self._llm_compress_conversation(conversation, level, focus)
            except Exception as e:
                logger.warning(f"LLM conversation compression failed: {e}")

        return self._rule_based_compression(conversation, level)

    def extract_heuristic(
        self,
        experiences: List[str],
        min_experiences: int = 3,
    ) -> Optional[str]:
        """
        Extract a general rule from multiple similar experiences.

        Args:
            experiences: List of similar experience descriptions
            min_experiences: Minimum number of experiences required

        Returns:
            Extracted heuristic rule, or None if no pattern found
        """
        if len(experiences) < min_experiences:
            logger.debug(
                f"Not enough experiences ({len(experiences)}) for heuristic extraction"
            )
            return None

        if not self.llm:
            # Try simple rule-based extraction
            return self._rule_based_heuristic(experiences)

        prompt = f"""Analyze these {len(experiences)} similar experiences and extract a general rule.

EXPERIENCES:
{chr(10).join(f"{i + 1}. {e}" for i, e in enumerate(experiences[:10]))}

If a clear, actionable pattern exists, state it as:
"When [specific situation], then [specific action] because [brief reason]."

Requirements:
- The pattern must apply to ALL experiences
- Be specific and actionable, not vague
- Keep it to ONE sentence

If no clear pattern exists or experiences are too different, respond exactly: NO_PATTERN"""

        try:
            response = self.llm.complete(
                prompt, timeout=self.config.llm_timeout_seconds
            ).strip()

            if "NO_PATTERN" in response.upper():
                return None

            # Clean up response
            response = response.strip('"').strip()
            if response and len(response) > 10:
                return response
            return None

        except Exception as e:
            logger.warning(f"Heuristic extraction failed: {e}")
            return self._rule_based_heuristic(experiences)

    def deduplicate_knowledge(
        self,
        new_knowledge: str,
        existing_knowledge: List[str],
    ) -> Optional[str]:
        """
        Merge new knowledge with existing, removing redundancy.

        Args:
            new_knowledge: New knowledge to potentially add
            existing_knowledge: List of existing knowledge items

        Returns:
            - new_knowledge if unique
            - Merged statement if overlapping
            - None if duplicate
        """
        if not existing_knowledge:
            return new_knowledge

        # Find similar existing knowledge
        similar = self._find_similar(new_knowledge, existing_knowledge)
        if not similar:
            return new_knowledge

        if not self.llm:
            # Simple dedup: check for high overlap
            for existing in similar:
                if self._is_duplicate(new_knowledge, existing):
                    return None
            return new_knowledge

        prompt = f"""Compare new knowledge with existing similar items and decide how to handle.

NEW KNOWLEDGE:
{new_knowledge}

EXISTING SIMILAR KNOWLEDGE:
{chr(10).join(f"- {k}" for k in similar[:5])}

Decide:
1. If new knowledge is completely redundant (says nothing new): respond "DUPLICATE"
2. If new adds information: respond with a merged statement that combines both
3. If new contradicts existing: respond "CONTRADICTION: [brief explanation]"

Keep merged statements concise (1-2 sentences max)."""

        try:
            response = self.llm.complete(
                prompt, timeout=self.config.llm_timeout_seconds
            ).strip()

            if "DUPLICATE" in response.upper():
                return None
            if response.upper().startswith("CONTRADICTION"):
                # Log but return new knowledge (let caller decide)
                logger.warning(f"Knowledge contradiction detected: {response}")
                return new_knowledge

            # Return merged statement
            return response if response else new_knowledge

        except Exception as e:
            logger.warning(f"Deduplication failed: {e}")
            return new_knowledge

    def batch_compress(
        self,
        contents: List[str],
        level: Optional[CompressionLevel] = None,
    ) -> List[CompressionResult]:
        """
        Compress multiple content items.

        Args:
            contents: List of content strings to compress
            level: Compression level for all items

        Returns:
            List of CompressionResult objects
        """
        return [self.compress(content, level) for content in contents]

    # ==================== LLM-BASED COMPRESSION ====================

    def _llm_compress_outcome(
        self,
        content: str,
        level: CompressionLevel,
    ) -> CompressedMemory:
        """LLM-based outcome compression."""
        level_instruction = self._get_level_instruction(level)

        prompt = f"""Compress this task outcome into essential information only.
{level_instruction}

TASK OUTCOME:
{content}

Extract in this exact format:
SUMMARY: [1-2 sentence compressed summary]
KEY_FACTS:
- [fact 1]
- [fact 2]
(max 5 facts)
CONSTRAINTS:
- [constraint 1]
(max 3 constraints, or "None" if none)
PATTERNS:
- [reusable pattern]
(max 3 patterns, or "None" if none)

Every word must earn its place. Be extremely concise."""

        response = self.llm.complete(prompt, timeout=self.config.llm_timeout_seconds)
        return self._parse_llm_response(response, content)

    def _llm_compress_general(
        self,
        content: str,
        level: CompressionLevel,
    ) -> CompressedMemory:
        """LLM-based general content compression."""
        level_instruction = self._get_level_instruction(level)

        prompt = f"""Compress this content, extracting only essential information.
{level_instruction}

CONTENT:
{content}

Extract in this exact format:
SUMMARY: [Compressed content - be concise]
KEY_FACTS:
- [fact 1]
- [fact 2]
(max 5 most important facts)
CONSTRAINTS:
- [limitation or requirement]
(max 3, or "None")
PATTERNS:
- [reusable insight]
(max 3, or "None")"""

        response = self.llm.complete(prompt, timeout=self.config.llm_timeout_seconds)
        return self._parse_llm_response(response, content)

    def _llm_compress_conversation(
        self,
        content: str,
        level: CompressionLevel,
        focus: Optional[str] = None,
    ) -> CompressedMemory:
        """LLM-based conversation compression."""
        level_instruction = self._get_level_instruction(level)
        focus_clause = f"\nFocus specifically on: {focus}" if focus else ""

        prompt = f"""Extract learnable knowledge from this conversation.
{level_instruction}{focus_clause}

Skip pleasantries, tangents, and filler. Extract only actionable knowledge.

CONVERSATION:
{content}

Extract in this exact format:
SUMMARY: [Key takeaways in 1-2 sentences]
KEY_FACTS:
- [Confirmed fact or decision]
(max 5 most important)
CONSTRAINTS:
- [Limitation or requirement discovered]
(max 3, or "None")
PATTERNS:
- [Reusable pattern or rule identified]
(max 3, or "None")"""

        response = self.llm.complete(prompt, timeout=self.config.llm_timeout_seconds)
        return self._parse_llm_response(response, content)

    def _get_level_instruction(self, level: CompressionLevel) -> str:
        """Get compression instruction based on level."""
        if level == CompressionLevel.LIGHT:
            return "COMPRESSION: Light - Remove redundancy but preserve detail."
        elif level == CompressionLevel.MEDIUM:
            return (
                "COMPRESSION: Medium - Extract key points only. Target 3x compression."
            )
        elif level == CompressionLevel.AGGRESSIVE:
            return "COMPRESSION: Aggressive - Maximum compression. Only absolute essentials. Target 5x+ compression."
        return ""

    def _parse_llm_response(
        self,
        response: str,
        original_content: str,
    ) -> CompressedMemory:
        """Parse LLM response into CompressedMemory."""
        lines = response.strip().split("\n")

        summary = ""
        key_facts: List[str] = []
        constraints: List[str] = []
        patterns: List[str] = []

        current_section = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check for section headers
            upper_line = line.upper()
            if upper_line.startswith("SUMMARY:"):
                summary = line.split(":", 1)[1].strip() if ":" in line else ""
                current_section = "summary"
            elif upper_line.startswith("KEY_FACTS:") or upper_line.startswith(
                "KEY FACTS:"
            ):
                current_section = "facts"
            elif upper_line.startswith("CONSTRAINTS:"):
                current_section = "constraints"
            elif upper_line.startswith("PATTERNS:"):
                current_section = "patterns"
            elif line.startswith("-") or line.startswith("•"):
                # Bullet point
                item = line.lstrip("-•").strip()
                if item.lower() == "none" or not item:
                    continue
                if (
                    current_section == "facts"
                    and len(key_facts) < self.config.max_key_facts
                ):
                    key_facts.append(item)
                elif (
                    current_section == "constraints"
                    and len(constraints) < self.config.max_constraints
                ):
                    constraints.append(item)
                elif (
                    current_section == "patterns"
                    and len(patterns) < self.config.max_patterns
                ):
                    patterns.append(item)
            elif current_section == "summary" and not summary:
                # Continuation of summary
                summary = line

        # Fallback if no summary extracted
        if not summary:
            summary = (
                original_content[:500] + "..."
                if len(original_content) > 500
                else original_content
            )

        compressed_length = len(summary)
        original_length = len(original_content)

        return CompressedMemory(
            original_length=original_length,
            compressed_length=compressed_length,
            compression_ratio=original_length / compressed_length
            if compressed_length > 0
            else 1.0,
            key_facts=key_facts,
            constraints=constraints,
            patterns=patterns,
            summary=summary,
            full_content=original_content if self.config.preserve_full_content else "",
        )

    # ==================== RULE-BASED COMPRESSION ====================

    def _no_compression(self, content: str) -> CompressedMemory:
        """Return content without compression."""
        return CompressedMemory(
            original_length=len(content),
            compressed_length=len(content),
            compression_ratio=1.0,
            key_facts=[],
            constraints=[],
            patterns=[],
            summary=content,
            full_content=content,
        )

    def _rule_based_compression(
        self,
        text: str,
        level: CompressionLevel,
    ) -> CompressedMemory:
        """
        Rule-based compression fallback when LLM unavailable.

        Uses sentence-level heuristics to extract key information.
        """
        # Split into sentences
        sentences = self._split_sentences(text)

        if level == CompressionLevel.LIGHT:
            summary = self._light_compression(sentences)
        elif level == CompressionLevel.MEDIUM:
            summary = self._medium_compression(sentences)
        else:  # AGGRESSIVE
            summary = self._aggressive_compression(sentences, text)

        # Extract key facts using indicators
        key_facts = self._extract_key_facts(sentences)

        # Extract constraints
        constraints = self._extract_constraints(sentences)

        # No pattern extraction without LLM
        patterns: List[str] = []

        compressed_length = len(summary)
        original_length = len(text)

        return CompressedMemory(
            original_length=original_length,
            compressed_length=compressed_length,
            compression_ratio=original_length / compressed_length
            if compressed_length > 0
            else 1.0,
            key_facts=key_facts[: self.config.max_key_facts],
            constraints=constraints[: self.config.max_constraints],
            patterns=patterns,
            summary=summary,
            full_content=text if self.config.preserve_full_content else "",
        )

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if s.strip()]

    def _light_compression(self, sentences: List[str]) -> str:
        """Light compression: remove duplicate/similar sentences."""
        if not sentences:
            return ""

        unique_sentences: List[str] = []
        seen_normalized: set = set()

        for sentence in sentences:
            normalized = self._normalize_sentence(sentence)
            if normalized not in seen_normalized:
                seen_normalized.add(normalized)
                unique_sentences.append(sentence)

        return " ".join(unique_sentences)

    def _medium_compression(self, sentences: List[str]) -> str:
        """Medium compression: extract key sentences."""
        if not sentences:
            return ""

        # Indicators of important sentences
        importance_indicators = [
            " is ",
            " are ",
            " was ",
            " were ",
            " should ",
            " must ",
            " need ",
            " because ",
            " therefore ",
            " however ",
            " key ",
            " important ",
            " essential ",
            " result ",
            " conclusion ",
            " found ",
        ]

        # Score sentences by importance
        scored: List[tuple] = []
        for i, sentence in enumerate(sentences):
            lower = sentence.lower()
            score = sum(1 for ind in importance_indicators if ind in lower)
            # Boost first and last sentences
            if i == 0:
                score += 2
            if i == len(sentences) - 1:
                score += 1
            scored.append((score, i, sentence))

        # Sort by score descending, then by position
        scored.sort(key=lambda x: (-x[0], x[1]))

        # Take top sentences, maintaining original order
        target_count = max(2, len(sentences) // 3)
        selected_indices = sorted([s[1] for s in scored[:target_count]])
        selected_sentences = [sentences[i] for i in selected_indices]

        return " ".join(selected_sentences)

    def _aggressive_compression(self, sentences: List[str], original: str) -> str:
        """Aggressive compression: minimal content."""
        if not sentences:
            return original[:200] + "..." if len(original) > 200 else original

        if len(sentences) == 1:
            # Single sentence - truncate if needed
            return (
                sentences[0][:300] + "..." if len(sentences[0]) > 300 else sentences[0]
            )

        # First sentence (context) + most important + last (conclusion)
        first = sentences[0]
        last = sentences[-1] if len(sentences) > 1 else ""

        # Find most important middle sentence
        middle_sentences = sentences[1:-1] if len(sentences) > 2 else []
        best_middle = ""
        if middle_sentences:
            importance_words = [
                "must",
                "should",
                "key",
                "important",
                "because",
                "therefore",
            ]
            for sentence in middle_sentences:
                if any(w in sentence.lower() for w in importance_words):
                    best_middle = sentence
                    break

        parts = [first]
        if best_middle and best_middle != last:
            parts.append(best_middle)
        if last and last != first:
            parts.append(last)

        return " ".join(parts)

    def _extract_key_facts(self, sentences: List[str]) -> List[str]:
        """Extract sentences that appear to be stating facts."""
        fact_indicators = [" is ", " are ", " has ", " have ", " was ", " were "]
        facts = []

        for sentence in sentences:
            lower = sentence.lower()
            if any(ind in lower for ind in fact_indicators):
                # Clean up the sentence
                clean = sentence.strip()
                if clean and len(clean) > 10:
                    facts.append(clean)

        return facts[: self.config.max_key_facts]

    def _extract_constraints(self, sentences: List[str]) -> List[str]:
        """Extract sentences that appear to describe constraints."""
        constraint_indicators = [
            " must ",
            " cannot ",
            " should not ",
            " shouldn't ",
            " limit ",
            " require ",
            " only ",
            " never ",
            " avoid ",
            " prevent ",
            " restrict ",
        ]
        constraints = []

        for sentence in sentences:
            lower = sentence.lower()
            if any(ind in lower for ind in constraint_indicators):
                clean = sentence.strip()
                if clean and len(clean) > 10:
                    constraints.append(clean)

        return constraints[: self.config.max_constraints]

    def _normalize_sentence(self, sentence: str) -> str:
        """Normalize sentence for deduplication."""
        # Lowercase, remove extra whitespace, remove punctuation
        normalized = sentence.lower()
        normalized = re.sub(r"[^\w\s]", "", normalized)
        normalized = " ".join(normalized.split())
        return normalized

    def _rule_based_heuristic(self, experiences: List[str]) -> Optional[str]:
        """Try to extract heuristic using simple pattern matching."""
        # Look for common words across experiences
        word_counts: Dict[str, int] = {}
        for exp in experiences:
            words = set(exp.lower().split())
            for word in words:
                if len(word) > 3:  # Skip short words
                    word_counts[word] = word_counts.get(word, 0) + 1

        # Find words common to most experiences
        threshold = len(experiences) * 0.6
        common_words = [w for w, c in word_counts.items() if c >= threshold]

        if not common_words:
            return None

        # Very basic pattern: if common action words found
        action_words = ["use", "apply", "check", "verify", "add", "remove", "update"]
        found_actions = [w for w in common_words if w in action_words]

        if found_actions:
            return f"Consider using '{found_actions[0]}' approach based on {len(experiences)} similar experiences."

        return None

    def _find_similar(
        self,
        query: str,
        candidates: List[str],
        threshold: float = 0.3,
    ) -> List[str]:
        """Find candidates similar to query using word overlap."""
        query_words = set(query.lower().split())
        similar = []

        for candidate in candidates:
            cand_words = set(candidate.lower().split())
            if not query_words or not cand_words:
                continue

            # Jaccard similarity
            intersection = len(query_words & cand_words)
            union = len(query_words | cand_words)
            similarity = intersection / union if union > 0 else 0

            if similarity >= threshold:
                similar.append(candidate)

        return similar[:5]

    def _is_duplicate(self, new: str, existing: str, threshold: float = 0.8) -> bool:
        """Check if new content is duplicate of existing."""
        new_words = set(new.lower().split())
        existing_words = set(existing.lower().split())

        if not new_words or not existing_words:
            return False

        intersection = len(new_words & existing_words)
        smaller = min(len(new_words), len(existing_words))

        return intersection / smaller >= threshold if smaller > 0 else False


def create_compressor(
    llm_client: Optional[LLMClient] = None,
    config: Optional[CompressionConfig] = None,
) -> MemoryCompressor:
    """
    Factory function to create a MemoryCompressor.

    Args:
        llm_client: Optional LLM client for intelligent compression
        config: Compression configuration

    Returns:
        Configured MemoryCompressor
    """
    return MemoryCompressor(llm_client=llm_client, config=config)
