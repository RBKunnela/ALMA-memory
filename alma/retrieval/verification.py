"""
ALMA Two-Stage Verified Retrieval.

Provides verification of retrieved memories for high-stakes applications.
Based on Memory Wall principles: "Pair fuzzy retrieval with exact verification."

Two-stage process:
1. Fuzzy Recall: Semantic search with expanded candidate set
2. Verification: Validate against ground truth, cross-verify, or confidence fallback

Verification statuses:
- VERIFIED: Safe to use, confirmed accurate
- UNCERTAIN: Use with caution, unconfirmed
- CONTRADICTED: Needs review, conflicts detected
- UNVERIFIABLE: No verification method available
"""

import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, Union

logger = logging.getLogger(__name__)


class VerificationStatus(Enum):
    """Status of memory verification."""

    VERIFIED = "verified"  # Confirmed accurate, safe to use
    UNCERTAIN = "uncertain"  # Unconfirmed, use with caution
    CONTRADICTED = "contradicted"  # Conflicts detected, needs review
    UNVERIFIABLE = "unverifiable"  # No verification method available


class VerificationMethod(Enum):
    """Method used for verification."""

    GROUND_TRUTH = "ground_truth"  # Verified against authoritative sources
    CROSS_VERIFY = "cross_verify"  # Verified against other memories
    CONFIDENCE = "confidence"  # Confidence-based fallback (no LLM)
    NONE = "none"  # No verification performed


@dataclass
class Verification:
    """
    Result of verifying a single memory.

    Attributes:
        status: Verification status (VERIFIED, UNCERTAIN, etc.)
        confidence: Confidence in verification result (0.0 - 1.0)
        reason: Human-readable explanation
        method: Method used for verification
        contradicting_source: Source of contradiction if status is CONTRADICTED
        verification_time_ms: Time taken for verification in milliseconds
    """

    status: VerificationStatus
    confidence: float
    reason: str
    method: VerificationMethod = VerificationMethod.NONE
    contradicting_source: Optional[str] = None
    verification_time_ms: int = 0

    def __post_init__(self):
        """Validate confidence is in range."""
        self.confidence = max(0.0, min(1.0, self.confidence))

    def is_usable(self) -> bool:
        """Check if memory is safe to use."""
        return self.status in (VerificationStatus.VERIFIED, VerificationStatus.UNCERTAIN)

    def needs_review(self) -> bool:
        """Check if memory needs human review."""
        return self.status == VerificationStatus.CONTRADICTED

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "status": self.status.value,
            "confidence": self.confidence,
            "reason": self.reason,
            "method": self.method.value,
            "contradicting_source": self.contradicting_source,
            "verification_time_ms": self.verification_time_ms,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Verification":
        """Create from dictionary."""
        return cls(
            status=VerificationStatus(data["status"]),
            confidence=data["confidence"],
            reason=data["reason"],
            method=VerificationMethod(data.get("method", "none")),
            contradicting_source=data.get("contradicting_source"),
            verification_time_ms=data.get("verification_time_ms", 0),
        )


@dataclass
class VerifiedMemory:
    """
    Memory with verification result attached.

    Attributes:
        memory: The original memory object
        verification: Verification result
        retrieval_score: Original similarity/relevance score from retrieval
    """

    memory: Any
    verification: Verification
    retrieval_score: float = 0.0

    @property
    def status(self) -> VerificationStatus:
        """Shortcut to verification status."""
        return self.verification.status

    @property
    def is_verified(self) -> bool:
        """Check if memory is verified."""
        return self.status == VerificationStatus.VERIFIED

    @property
    def is_usable(self) -> bool:
        """Check if memory is usable (verified or uncertain)."""
        return self.verification.is_usable()

    def combined_score(self, verification_weight: float = 0.5) -> float:
        """
        Compute combined score from retrieval and verification.

        Args:
            verification_weight: Weight for verification confidence (0-1)

        Returns:
            Combined score between 0 and 1
        """
        retrieval_weight = 1.0 - verification_weight
        return (
            self.retrieval_score * retrieval_weight
            + self.verification.confidence * verification_weight
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        memory_dict = (
            self.memory.to_dict()
            if hasattr(self.memory, "to_dict")
            else {"content": str(self.memory)}
        )
        return {
            "memory": memory_dict,
            "verification": self.verification.to_dict(),
            "retrieval_score": self.retrieval_score,
        }


@dataclass
class VerifiedResults:
    """
    Container for categorized verification results.

    Organizes verified memories by status for easy access.

    Attributes:
        verified: Memories confirmed accurate
        uncertain: Memories with uncertain status
        contradicted: Memories with conflicts detected
        unverifiable: Memories that couldn't be verified
        metadata: Additional information about the verification process
    """

    verified: List[VerifiedMemory] = field(default_factory=list)
    uncertain: List[VerifiedMemory] = field(default_factory=list)
    contradicted: List[VerifiedMemory] = field(default_factory=list)
    unverifiable: List[VerifiedMemory] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def all_usable(self) -> List[VerifiedMemory]:
        """Get all memories safe to use (verified + uncertain)."""
        return self.verified + self.uncertain

    @property
    def high_confidence(self) -> List[VerifiedMemory]:
        """Get only verified memories."""
        return self.verified

    @property
    def needs_review(self) -> List[VerifiedMemory]:
        """Get memories that need human review."""
        return self.contradicted

    @property
    def total_count(self) -> int:
        """Total number of memories processed."""
        return (
            len(self.verified)
            + len(self.uncertain)
            + len(self.contradicted)
            + len(self.unverifiable)
        )

    def summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        total = self.total_count
        return {
            "verified": len(self.verified),
            "uncertain": len(self.uncertain),
            "contradicted": len(self.contradicted),
            "unverifiable": len(self.unverifiable),
            "total": total,
            "usable_count": len(self.all_usable),
            "usable_ratio": len(self.all_usable) / total if total > 0 else 0.0,
            "verification_time_ms": self.metadata.get("total_verification_time_ms", 0),
        }

    def get_by_status(self, status: VerificationStatus) -> List[VerifiedMemory]:
        """Get memories by verification status."""
        if status == VerificationStatus.VERIFIED:
            return self.verified
        elif status == VerificationStatus.UNCERTAIN:
            return self.uncertain
        elif status == VerificationStatus.CONTRADICTED:
            return self.contradicted
        else:
            return self.unverifiable

    def sort_by_confidence(self, descending: bool = True) -> None:
        """Sort all categories by verification confidence."""

        def get_confidence(vm: VerifiedMemory) -> float:
            return vm.verification.confidence

        self.verified.sort(key=get_confidence, reverse=descending)
        self.uncertain.sort(key=get_confidence, reverse=descending)
        self.contradicted.sort(key=get_confidence, reverse=descending)
        self.unverifiable.sort(key=get_confidence, reverse=descending)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "verified": [vm.to_dict() for vm in self.verified],
            "uncertain": [vm.to_dict() for vm in self.uncertain],
            "contradicted": [vm.to_dict() for vm in self.contradicted],
            "unverifiable": [vm.to_dict() for vm in self.unverifiable],
            "metadata": self.metadata,
            "summary": self.summary(),
        }


@dataclass
class VerificationConfig:
    """
    Configuration for verification behavior.

    Attributes:
        enabled: Whether verification is enabled
        default_method: Default verification method to use
        confidence_threshold: Threshold for confidence-based verification
        llm_timeout_seconds: Timeout for LLM-based verification
        expand_candidates_factor: Factor to expand candidate set (default 4x)
        max_sources_for_verification: Max sources to use for ground truth
        max_memories_for_cross_verify: Max other memories to cross-verify against
    """

    enabled: bool = False  # Opt-in by default
    default_method: str = "confidence"  # confidence | cross_verify | ground_truth
    confidence_threshold: float = 0.7
    llm_timeout_seconds: float = 5.0
    expand_candidates_factor: int = 4
    max_sources_for_verification: int = 5
    max_memories_for_cross_verify: int = 5

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VerificationConfig":
        """Create from dictionary."""
        return cls(
            enabled=data.get("enabled", False),
            default_method=data.get("default_method", "confidence"),
            confidence_threshold=data.get("confidence_threshold", 0.7),
            llm_timeout_seconds=data.get("llm_timeout_seconds", 5.0),
            expand_candidates_factor=data.get("expand_candidates_factor", 4),
            max_sources_for_verification=data.get("max_sources_for_verification", 5),
            max_memories_for_cross_verify=data.get("max_memories_for_cross_verify", 5),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "enabled": self.enabled,
            "default_method": self.default_method,
            "confidence_threshold": self.confidence_threshold,
            "llm_timeout_seconds": self.llm_timeout_seconds,
            "expand_candidates_factor": self.expand_candidates_factor,
            "max_sources_for_verification": self.max_sources_for_verification,
            "max_memories_for_cross_verify": self.max_memories_for_cross_verify,
        }


class LLMClient(Protocol):
    """Protocol for LLM clients used in verification."""

    def complete(self, prompt: str, timeout: Optional[float] = None) -> str:
        """Complete a prompt and return the response."""
        ...


class VerifiedRetriever:
    """
    Two-stage retrieval with verification.

    Stage 1: Fuzzy recall with expanded candidate set
    Stage 2: Verify candidates using one of:
        - Ground truth sources (with LLM)
        - Cross-verification against other memories (with LLM)
        - Confidence-based fallback (no LLM required)

    Example:
        retriever = VerifiedRetriever(
            retrieval_engine=engine,
            llm_client=llm,
            config=VerificationConfig(enabled=True)
        )
        results = retriever.retrieve_verified(
            query="How to handle API rate limits?",
            agent="dev-agent",
            project_id="my-project"
        )
        for memory in results.high_confidence:
            print(f"Verified: {memory.memory}")
    """

    def __init__(
        self,
        retrieval_engine: Any,
        llm_client: Optional[LLMClient] = None,
        config: Optional[VerificationConfig] = None,
    ):
        """
        Initialize verified retriever.

        Args:
            retrieval_engine: RetrievalEngine or compatible retriever
            llm_client: Optional LLM client for verification
            config: Verification configuration
        """
        self.retrieval_engine = retrieval_engine
        self.llm = llm_client
        self.config = config or VerificationConfig()

    def retrieve_verified(
        self,
        query: str,
        agent: str,
        project_id: str,
        ground_truth_sources: Optional[List[str]] = None,
        cross_verify: Optional[bool] = None,
        top_k: int = 5,
        **retrieval_kwargs: Any,
    ) -> VerifiedResults:
        """
        Two-stage retrieval with verification.

        Args:
            query: Query string for retrieval
            agent: Agent requesting memories
            project_id: Project context
            ground_truth_sources: Optional authoritative sources for verification
            cross_verify: Whether to cross-verify (None = use config default)
            top_k: Number of final results to return
            **retrieval_kwargs: Additional arguments for retrieval engine

        Returns:
            VerifiedResults with categorized memories
        """
        start_time = time.time()

        # Stage 1: Fuzzy recall with expanded candidates
        recall_k = top_k * self.config.expand_candidates_factor
        retrieval_kwargs["top_k"] = recall_k
        retrieval_kwargs["agent"] = agent
        retrieval_kwargs["project_id"] = project_id

        # Call retrieval engine
        memory_slice = self.retrieval_engine.retrieve(query, **retrieval_kwargs)

        # Extract all memories from slice
        candidates = self._extract_candidates(memory_slice)

        # Stage 2: Verification
        results = self._verify_candidates(
            candidates=candidates,
            ground_truth_sources=ground_truth_sources,
            cross_verify=cross_verify,
        )

        # Limit results to top_k per category
        results.verified = results.verified[:top_k]
        results.uncertain = results.uncertain[:top_k]
        results.contradicted = results.contradicted[:top_k]
        results.unverifiable = results.unverifiable[:top_k]

        # Record metadata
        total_time_ms = int((time.time() - start_time) * 1000)
        results.metadata["total_verification_time_ms"] = total_time_ms
        results.metadata["total_candidates"] = len(candidates)
        results.metadata["query"] = query
        results.metadata["top_k"] = top_k

        return results

    def _extract_candidates(self, memory_slice: Any) -> List[Any]:
        """Extract all memory objects from a MemorySlice or similar container."""
        candidates = []

        # Handle MemorySlice structure
        if hasattr(memory_slice, "heuristics"):
            candidates.extend(memory_slice.heuristics or [])
        if hasattr(memory_slice, "outcomes"):
            candidates.extend(memory_slice.outcomes or [])
        if hasattr(memory_slice, "knowledge"):
            candidates.extend(memory_slice.knowledge or [])
        if hasattr(memory_slice, "anti_patterns"):
            candidates.extend(memory_slice.anti_patterns or [])
        if hasattr(memory_slice, "preferences"):
            candidates.extend(memory_slice.preferences or [])

        # Fallback for list-like containers
        if not candidates and hasattr(memory_slice, "__iter__"):
            candidates = list(memory_slice)

        return candidates

    def _verify_candidates(
        self,
        candidates: List[Any],
        ground_truth_sources: Optional[List[str]],
        cross_verify: Optional[bool],
    ) -> VerifiedResults:
        """
        Verify each candidate and categorize.

        Args:
            candidates: List of memory candidates
            ground_truth_sources: Optional authoritative sources
            cross_verify: Whether to use cross-verification

        Returns:
            VerifiedResults with categorized memories
        """
        results = VerifiedResults()

        # Determine verification method
        use_cross_verify = (
            cross_verify
            if cross_verify is not None
            else self.config.default_method == "cross_verify"
        )

        for candidate in candidates:
            # Get retrieval score if available
            retrieval_score = self._get_retrieval_score(candidate)

            # Verify the candidate
            if ground_truth_sources:
                verification = self._verify_against_sources(
                    candidate, ground_truth_sources
                )
            elif use_cross_verify and self.llm:
                verification = self._cross_verify(candidate, candidates)
            else:
                verification = self._confidence_fallback(candidate)

            # Create verified memory
            vm = VerifiedMemory(
                memory=candidate,
                verification=verification,
                retrieval_score=retrieval_score,
            )

            # Categorize by status
            if verification.status == VerificationStatus.VERIFIED:
                results.verified.append(vm)
            elif verification.status == VerificationStatus.UNCERTAIN:
                results.uncertain.append(vm)
            elif verification.status == VerificationStatus.CONTRADICTED:
                results.contradicted.append(vm)
            else:
                results.unverifiable.append(vm)

        # Sort by combined score within each category
        results.sort_by_confidence()

        return results

    def _get_retrieval_score(self, memory: Any) -> float:
        """Extract retrieval/similarity score from memory."""
        # Try common attribute names
        for attr in ["similarity_score", "score", "relevance", "confidence"]:
            if hasattr(memory, attr):
                val = getattr(memory, attr)
                if isinstance(val, (int, float)):
                    return float(val)

        # Try metadata
        if hasattr(memory, "metadata") and isinstance(memory.metadata, dict):
            for key in ["similarity_score", "score", "relevance"]:
                if key in memory.metadata:
                    return float(memory.metadata[key])

        return 0.5  # Default middle score

    def _get_memory_content(self, memory: Any) -> str:
        """Extract content string from memory for verification."""
        # Try common content attributes
        if hasattr(memory, "content"):
            return str(memory.content)
        if hasattr(memory, "fact"):
            return str(memory.fact)
        if hasattr(memory, "strategy"):
            return f"{getattr(memory, 'condition', '')}: {memory.strategy}"
        if hasattr(memory, "task_description"):
            return str(memory.task_description)
        if hasattr(memory, "preference"):
            return str(memory.preference)
        if hasattr(memory, "pattern"):
            return str(memory.pattern)

        # Fallback to string representation
        return str(memory)

    def _get_memory_id(self, memory: Any) -> str:
        """Get memory ID for comparison."""
        if hasattr(memory, "id"):
            return str(memory.id)
        return str(id(memory))

    def _verify_against_sources(
        self,
        memory: Any,
        sources: List[str],
    ) -> Verification:
        """
        Verify memory against authoritative sources using LLM.

        Args:
            memory: Memory to verify
            sources: List of authoritative source strings

        Returns:
            Verification result
        """
        start_time = time.time()

        if not self.llm:
            return Verification(
                status=VerificationStatus.UNVERIFIABLE,
                confidence=0.5,
                reason="No LLM available for ground truth verification",
                method=VerificationMethod.NONE,
            )

        content = self._get_memory_content(memory)
        limited_sources = sources[: self.config.max_sources_for_verification]

        prompt = f"""Verify if this memory is consistent with the authoritative sources.

Memory to verify:
{content}

Authoritative sources:
{chr(10).join(f'- {s}' for s in limited_sources)}

Respond in this exact format (no other text):
STATUS: verified|contradicted|uncertain
CONFIDENCE: 0.0-1.0
REASON: Brief explanation (one sentence)
CONTRADICTION: (only if STATUS is contradicted) What specifically contradicts it"""

        try:
            response = self.llm.complete(
                prompt, timeout=self.config.llm_timeout_seconds
            )
            verification = self._parse_verification_response(response)
            verification.method = VerificationMethod.GROUND_TRUTH
            verification.verification_time_ms = int((time.time() - start_time) * 1000)
            return verification
        except Exception as e:
            logger.warning(f"Ground truth verification failed: {e}")
            return Verification(
                status=VerificationStatus.UNVERIFIABLE,
                confidence=0.5,
                reason=f"Verification failed: {str(e)[:100]}",
                method=VerificationMethod.GROUND_TRUTH,
                verification_time_ms=int((time.time() - start_time) * 1000),
            )

    def _cross_verify(
        self,
        memory: Any,
        all_candidates: List[Any],
    ) -> Verification:
        """
        Cross-verify memory against other retrieved memories.

        Args:
            memory: Memory to verify
            all_candidates: All candidate memories

        Returns:
            Verification result
        """
        start_time = time.time()

        if not self.llm:
            return self._confidence_fallback(memory)

        memory_id = self._get_memory_id(memory)
        content = self._get_memory_content(memory)

        # Get other memories for comparison
        others = [
            m
            for m in all_candidates
            if self._get_memory_id(m) != memory_id
        ][: self.config.max_memories_for_cross_verify]

        if not others:
            return Verification(
                status=VerificationStatus.UNVERIFIABLE,
                confidence=0.5,
                reason="No other memories to cross-verify against",
                method=VerificationMethod.CROSS_VERIFY,
                verification_time_ms=int((time.time() - start_time) * 1000),
            )

        other_contents = [self._get_memory_content(m) for m in others]

        prompt = f"""Check if this memory is consistent with related memories.

Memory to verify:
{content}

Related memories:
{chr(10).join(f'- {c}' for c in other_contents)}

Respond in this exact format (no other text):
STATUS: verified|contradicted|uncertain
CONFIDENCE: 0.0-1.0
REASON: Brief explanation (one sentence)"""

        try:
            response = self.llm.complete(
                prompt, timeout=self.config.llm_timeout_seconds
            )
            verification = self._parse_verification_response(response)
            verification.method = VerificationMethod.CROSS_VERIFY
            verification.verification_time_ms = int((time.time() - start_time) * 1000)
            return verification
        except Exception as e:
            logger.warning(f"Cross-verification failed: {e}")
            # Fallback to confidence-based
            fallback = self._confidence_fallback(memory)
            fallback.verification_time_ms = int((time.time() - start_time) * 1000)
            return fallback

    def _confidence_fallback(self, memory: Any) -> Verification:
        """
        Confidence-based verification fallback (no LLM required).

        Uses the memory's confidence score to determine verification status.

        Args:
            memory: Memory to verify

        Returns:
            Verification result
        """
        # Get confidence from memory
        confidence = 0.5
        if hasattr(memory, "confidence"):
            confidence = float(memory.confidence)
        elif hasattr(memory, "metadata") and isinstance(memory.metadata, dict):
            confidence = float(memory.metadata.get("confidence", 0.5))

        if confidence >= self.config.confidence_threshold:
            return Verification(
                status=VerificationStatus.VERIFIED,
                confidence=confidence,
                reason=f"High confidence score ({confidence:.2f} >= {self.config.confidence_threshold})",
                method=VerificationMethod.CONFIDENCE,
            )
        elif confidence >= self.config.confidence_threshold * 0.5:
            return Verification(
                status=VerificationStatus.UNCERTAIN,
                confidence=confidence,
                reason=f"Moderate confidence score ({confidence:.2f})",
                method=VerificationMethod.CONFIDENCE,
            )
        else:
            return Verification(
                status=VerificationStatus.UNCERTAIN,
                confidence=confidence,
                reason=f"Low confidence score ({confidence:.2f})",
                method=VerificationMethod.CONFIDENCE,
            )

    def _parse_verification_response(self, response: str) -> Verification:
        """
        Parse LLM verification response into Verification object.

        Expected format:
            STATUS: verified|contradicted|uncertain
            CONFIDENCE: 0.0-1.0
            REASON: Brief explanation
            CONTRADICTION: (optional) What contradicts it

        Args:
            response: LLM response string

        Returns:
            Verification object
        """
        lines = response.strip().split("\n")
        result = {
            "status": "uncertain",
            "confidence": 0.5,
            "reason": "Unable to parse verification response",
            "contradiction": None,
        }

        for line in lines:
            line = line.strip()
            if line.upper().startswith("STATUS:"):
                status_str = line.split(":", 1)[1].strip().lower()
                if status_str in ("verified", "contradicted", "uncertain"):
                    result["status"] = status_str
            elif line.upper().startswith("CONFIDENCE:"):
                try:
                    conf_str = line.split(":", 1)[1].strip()
                    # Extract number from string
                    match = re.search(r"[\d.]+", conf_str)
                    if match:
                        result["confidence"] = float(match.group())
                except (ValueError, IndexError):
                    pass
            elif line.upper().startswith("REASON:"):
                result["reason"] = line.split(":", 1)[1].strip()
            elif line.upper().startswith("CONTRADICTION:"):
                result["contradiction"] = line.split(":", 1)[1].strip()

        # Map status string to enum
        status_map = {
            "verified": VerificationStatus.VERIFIED,
            "contradicted": VerificationStatus.CONTRADICTED,
            "uncertain": VerificationStatus.UNCERTAIN,
        }

        return Verification(
            status=status_map.get(result["status"], VerificationStatus.UNCERTAIN),
            confidence=max(0.0, min(1.0, result["confidence"])),
            reason=result["reason"],
            contradicting_source=result["contradiction"],
        )


def create_verified_retriever(
    retrieval_engine: Any,
    llm_client: Optional[LLMClient] = None,
    config: Optional[Union[VerificationConfig, Dict[str, Any]]] = None,
) -> VerifiedRetriever:
    """
    Factory function to create a VerifiedRetriever.

    Args:
        retrieval_engine: RetrievalEngine or compatible retriever
        llm_client: Optional LLM client for verification
        config: Configuration dict or VerificationConfig

    Returns:
        Configured VerifiedRetriever
    """
    if isinstance(config, dict):
        config = VerificationConfig.from_dict(config)

    return VerifiedRetriever(
        retrieval_engine=retrieval_engine,
        llm_client=llm_client,
        config=config,
    )
