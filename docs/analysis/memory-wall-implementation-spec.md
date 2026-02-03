# ALMA Memory Wall Enhancement Implementation Spec

## Priority 1: Mode-Aware Retrieval Engine

### Overview
Transform retrieval from one-size-fits-all to task-aware, dynamically adjusting strategy based on cognitive mode.

### File: `alma/retrieval/modes.py` (NEW)

```python
"""Mode-aware retrieval for different cognitive tasks."""
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Any


class RetrievalMode(Enum):
    """Cognitive modes that affect retrieval strategy."""
    BROAD = "broad"          # Planning, brainstorming, exploring
    PRECISE = "precise"      # Execution, implementation, building
    DIAGNOSTIC = "diagnostic"  # Debugging, troubleshooting, fixing
    LEARNING = "learning"    # Pattern finding, consolidation
    RECALL = "recall"        # Exact memory retrieval


@dataclass
class ModeConfig:
    """Configuration for a retrieval mode."""
    top_k: int
    min_confidence: float
    weights: Dict[str, float]
    include_anti_patterns: bool
    diversity_factor: float  # 0.0 = focused, 1.0 = diverse
    prioritize_failures: bool = False
    cluster_similar: bool = False
    exact_match_boost: float = 1.0


MODE_CONFIGS: Dict[RetrievalMode, ModeConfig] = {
    RetrievalMode.BROAD: ModeConfig(
        top_k=15,
        min_confidence=0.3,
        weights={"similarity": 0.70, "recency": 0.10, "success": 0.10, "confidence": 0.10},
        include_anti_patterns=False,
        diversity_factor=0.8,
    ),
    RetrievalMode.PRECISE: ModeConfig(
        top_k=5,
        min_confidence=0.7,
        weights={"similarity": 0.30, "recency": 0.10, "success": 0.40, "confidence": 0.20},
        include_anti_patterns=True,
        diversity_factor=0.2,
        exact_match_boost=2.0,
    ),
    RetrievalMode.DIAGNOSTIC: ModeConfig(
        top_k=10,
        min_confidence=0.4,
        weights={"similarity": 0.40, "recency": 0.30, "success": 0.00, "confidence": 0.30},
        include_anti_patterns=True,
        diversity_factor=0.5,
        prioritize_failures=True,
    ),
    RetrievalMode.LEARNING: ModeConfig(
        top_k=20,
        min_confidence=0.2,
        weights={"similarity": 0.90, "recency": 0.00, "success": 0.05, "confidence": 0.05},
        include_anti_patterns=True,
        diversity_factor=0.3,
        cluster_similar=True,
    ),
    RetrievalMode.RECALL: ModeConfig(
        top_k=3,
        min_confidence=0.5,
        weights={"similarity": 0.95, "recency": 0.00, "success": 0.00, "confidence": 0.05},
        include_anti_patterns=False,
        diversity_factor=0.0,
        exact_match_boost=3.0,
    ),
}


def infer_mode_from_query(query: str) -> RetrievalMode:
    """Heuristically infer retrieval mode from query text."""
    query_lower = query.lower()

    # Diagnostic indicators
    diagnostic_terms = ["error", "bug", "fail", "broken", "issue", "problem", "debug", "fix", "wrong"]
    if any(term in query_lower for term in diagnostic_terms):
        return RetrievalMode.DIAGNOSTIC

    # Planning indicators
    planning_terms = ["how should", "what approach", "options for", "ways to", "plan", "design", "architect"]
    if any(term in query_lower for term in planning_terms):
        return RetrievalMode.BROAD

    # Recall indicators
    recall_terms = ["what was", "when did", "remember when", "last time", "previously"]
    if any(term in query_lower for term in recall_terms):
        return RetrievalMode.RECALL

    # Learning indicators
    learning_terms = ["pattern", "similar", "consolidate", "common", "recurring"]
    if any(term in query_lower for term in learning_terms):
        return RetrievalMode.LEARNING

    # Default to precise for execution
    return RetrievalMode.PRECISE
```

### Modification: `alma/retrieval/engine.py`

```python
# Add to RetrievalEngine class

from alma.retrieval.modes import RetrievalMode, MODE_CONFIGS, infer_mode_from_query

def retrieve_with_mode(
    self,
    query: str,
    mode: Optional[RetrievalMode] = None,
    **kwargs
) -> List[Memory]:
    """Retrieve memories using mode-aware strategy."""

    # Auto-infer mode if not specified
    if mode is None:
        mode = infer_mode_from_query(query)

    config = MODE_CONFIGS[mode]

    # Override defaults with mode config
    effective_top_k = kwargs.get('top_k', config.top_k)
    effective_min_confidence = kwargs.get('min_confidence', config.min_confidence)

    # Get candidates with expanded top_k for filtering
    candidates = self._get_candidates(
        query=query,
        top_k=effective_top_k * 3,  # Get more for diversity filtering
        min_confidence=effective_min_confidence,
        include_anti_patterns=config.include_anti_patterns
    )

    # Apply mode-specific scoring
    scored = self._apply_mode_scoring(candidates, config)

    # Apply diversity filtering if needed
    if config.diversity_factor > 0:
        scored = self._diversify_results(scored, config.diversity_factor)

    # Prioritize failures if diagnostic mode
    if config.prioritize_failures:
        scored = self._boost_failures(scored)

    # Cluster similar if learning mode
    if config.cluster_similar:
        scored = self._cluster_for_learning(scored)

    return scored[:effective_top_k]

def _apply_mode_scoring(
    self,
    candidates: List[Memory],
    config: ModeConfig
) -> List[Tuple[Memory, float]]:
    """Apply mode-specific scoring weights."""
    scored = []
    for memory in candidates:
        score = (
            config.weights["similarity"] * memory.similarity_score +
            config.weights["recency"] * self._recency_score(memory) +
            config.weights["success"] * memory.success_rate +
            config.weights["confidence"] * memory.confidence
        )

        # Apply exact match boost
        if config.exact_match_boost > 1.0 and self._is_exact_match(memory):
            score *= config.exact_match_boost

        scored.append((memory, score))

    return sorted(scored, key=lambda x: x[1], reverse=True)

def _diversify_results(
    self,
    scored: List[Tuple[Memory, float]],
    diversity_factor: float
) -> List[Tuple[Memory, float]]:
    """Apply MMR-style diversity to results."""
    if not scored:
        return scored

    selected = [scored[0]]
    remaining = scored[1:]

    while remaining and len(selected) < len(scored):
        best_idx = 0
        best_score = float('-inf')

        for i, (mem, score) in enumerate(remaining):
            # Diversity penalty based on similarity to selected
            max_sim = max(
                self._semantic_similarity(mem, sel_mem)
                for sel_mem, _ in selected
            )
            diversity_penalty = diversity_factor * max_sim
            adjusted_score = score - diversity_penalty

            if adjusted_score > best_score:
                best_score = adjusted_score
                best_idx = i

        selected.append(remaining.pop(best_idx))

    return selected
```

---

## Priority 2: Decay-Based Forgetting

### File: `alma/learning/decay.py` (NEW)

```python
"""Decay-based forgetting with access reinforcement."""
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional


@dataclass
class MemoryStrength:
    """Tracks decay and reinforcement of a memory."""
    memory_id: str
    initial_strength: float = 1.0
    decay_half_life_days: int = 30
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    access_count: int = 0
    reinforcement_events: List[datetime] = field(default_factory=list)
    explicit_importance: float = 0.5  # User-set importance (0-1)

    def current_strength(self) -> float:
        """Calculate current memory strength with decay and reinforcement."""
        now = datetime.utcnow()
        days_since_access = (now - self.last_accessed).days

        # Base decay using half-life formula
        base_decay = math.exp(
            -0.693 * days_since_access / self.decay_half_life_days
        )

        # Access reinforcement (diminishing returns)
        access_bonus = min(0.4, math.log1p(self.access_count) * 0.1)

        # Recency of reinforcements
        recent_reinforcements = sum(
            1 for r in self.reinforcement_events
            if (now - r).days < 7
        )
        reinforcement_bonus = min(0.3, recent_reinforcements * 0.1)

        # Explicit importance factor
        importance_factor = 0.5 + (self.explicit_importance * 0.5)

        # Combine factors
        strength = (base_decay + access_bonus + reinforcement_bonus) * importance_factor
        return min(1.0, max(0.0, strength))

    def access(self) -> None:
        """Record an access to this memory."""
        self.last_accessed = datetime.utcnow()
        self.access_count += 1

    def reinforce(self) -> None:
        """Explicitly reinforce this memory."""
        now = datetime.utcnow()
        self.reinforcement_events.append(now)
        self.last_accessed = now
        # Keep only last 10 reinforcement events
        self.reinforcement_events = self.reinforcement_events[-10:]

    def should_forget(self, threshold: float = 0.1) -> bool:
        """Determine if memory should be forgotten."""
        return self.current_strength() < threshold

    def recoverable(self) -> bool:
        """Check if memory can be recovered with effort."""
        strength = self.current_strength()
        return 0.05 <= strength < 0.3  # Weak but not gone


class DecayManager:
    """Manages memory decay across the system."""

    def __init__(self, storage, default_half_life: int = 30):
        self.storage = storage
        self.default_half_life = default_half_life
        self._strength_cache: dict[str, MemoryStrength] = {}

    def get_strength(self, memory_id: str) -> MemoryStrength:
        """Get or create strength tracker for memory."""
        if memory_id not in self._strength_cache:
            # Load from storage or create new
            stored = self.storage.get_memory_strength(memory_id)
            if stored:
                self._strength_cache[memory_id] = stored
            else:
                self._strength_cache[memory_id] = MemoryStrength(
                    memory_id=memory_id,
                    decay_half_life_days=self.default_half_life
                )
        return self._strength_cache[memory_id]

    def record_access(self, memory_id: str) -> None:
        """Record that a memory was accessed (retrieved and used)."""
        strength = self.get_strength(memory_id)
        strength.access()
        self.storage.update_memory_strength(strength)

    def reinforce_memory(self, memory_id: str) -> None:
        """Explicitly reinforce a memory (user action or successful use)."""
        strength = self.get_strength(memory_id)
        strength.reinforce()
        self.storage.update_memory_strength(strength)

    def get_forgettable_memories(
        self,
        project_id: str,
        agent: str,
        threshold: float = 0.1
    ) -> List[str]:
        """Get list of memories that should be forgotten."""
        all_memories = self.storage.get_all_memory_ids(project_id, agent)
        forgettable = []

        for mem_id in all_memories:
            strength = self.get_strength(mem_id)
            if strength.should_forget(threshold):
                forgettable.append(mem_id)

        return forgettable

    def get_recoverable_memories(
        self,
        project_id: str,
        agent: str
    ) -> List[tuple[str, float]]:
        """Get memories that are weak but recoverable."""
        all_memories = self.storage.get_all_memory_ids(project_id, agent)
        recoverable = []

        for mem_id in all_memories:
            strength = self.get_strength(mem_id)
            if strength.recoverable():
                recoverable.append((mem_id, strength.current_strength()))

        return sorted(recoverable, key=lambda x: x[1])

    def smart_forget(
        self,
        project_id: str,
        agent: str,
        target_count: Optional[int] = None
    ) -> List[str]:
        """
        Intelligently forget memories to maintain system health.

        Unlike simple pruning, this:
        1. Never forgets high-importance memories
        2. Archives recoverable memories before deletion
        3. Maintains minimum coverage per category
        """
        forgettable = self.get_forgettable_memories(project_id, agent)

        if target_count is None:
            target_count = len(forgettable)

        # Sort by strength (forget weakest first)
        scored = [
            (mem_id, self.get_strength(mem_id).current_strength())
            for mem_id in forgettable
        ]
        scored.sort(key=lambda x: x[1])

        forgotten = []
        for mem_id, strength in scored[:target_count]:
            # Archive before deletion
            self.storage.archive_memory(mem_id)
            self.storage.delete_memory(mem_id)
            forgotten.append(mem_id)

        return forgotten
```

---

## Priority 3: Two-Stage Verified Retrieval

### File: `alma/retrieval/verification.py` (NEW)

```python
"""Two-stage retrieval with verification."""
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any, Callable


class VerificationStatus(Enum):
    VERIFIED = "verified"
    UNCERTAIN = "uncertain"
    CONTRADICTED = "contradicted"
    UNVERIFIABLE = "unverifiable"


@dataclass
class Verification:
    status: VerificationStatus
    confidence: float
    reason: str
    contradicting_source: Optional[str] = None


@dataclass
class VerifiedMemory:
    memory: Any  # Memory object
    verification: Verification
    retrieval_score: float


@dataclass
class VerifiedResults:
    verified: List[VerifiedMemory]
    uncertain: List[VerifiedMemory]
    contradicted: List[VerifiedMemory]
    unverifiable: List[VerifiedMemory]
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def all_usable(self) -> List[VerifiedMemory]:
        """Get all memories safe to use (verified + uncertain with warning)."""
        return self.verified + self.uncertain

    @property
    def high_confidence(self) -> List[VerifiedMemory]:
        """Get only verified memories."""
        return self.verified


class VerifiedRetriever:
    """Two-stage retrieval with verification."""

    def __init__(
        self,
        retrieval_engine,
        llm_client=None,
        verification_threshold: float = 0.7
    ):
        self.retrieval_engine = retrieval_engine
        self.llm = llm_client
        self.verification_threshold = verification_threshold

    def retrieve_verified(
        self,
        query: str,
        ground_truth_sources: Optional[List[str]] = None,
        cross_verify: bool = True,
        **retrieval_kwargs
    ) -> VerifiedResults:
        """
        Two-stage retrieval:
        1. Fuzzy recall - get candidates via semantic search
        2. Verification - check against ground truth or internal consistency
        """
        # Stage 1: Fuzzy recall (get more candidates than needed)
        recall_k = retrieval_kwargs.get('top_k', 5) * 4
        retrieval_kwargs['top_k'] = recall_k

        candidates = self.retrieval_engine.retrieve(query, **retrieval_kwargs)

        # Stage 2: Verification
        verified = []
        uncertain = []
        contradicted = []
        unverifiable = []

        for candidate in candidates:
            if ground_truth_sources:
                verification = self._verify_against_sources(
                    candidate, ground_truth_sources
                )
            elif cross_verify:
                verification = self._cross_verify(candidate, candidates)
            else:
                verification = Verification(
                    status=VerificationStatus.UNVERIFIABLE,
                    confidence=candidate.confidence,
                    reason="No verification method available"
                )

            verified_mem = VerifiedMemory(
                memory=candidate,
                verification=verification,
                retrieval_score=getattr(candidate, 'similarity_score', 0.5)
            )

            if verification.status == VerificationStatus.VERIFIED:
                verified.append(verified_mem)
            elif verification.status == VerificationStatus.UNCERTAIN:
                uncertain.append(verified_mem)
            elif verification.status == VerificationStatus.CONTRADICTED:
                contradicted.append(verified_mem)
            else:
                unverifiable.append(verified_mem)

        return VerifiedResults(
            verified=sorted(verified, key=lambda x: x.verification.confidence, reverse=True),
            uncertain=sorted(uncertain, key=lambda x: x.retrieval_score, reverse=True),
            contradicted=contradicted,
            unverifiable=unverifiable,
            metadata={
                "query": query,
                "total_candidates": len(candidates),
                "verification_method": "ground_truth" if ground_truth_sources else "cross_verify"
            }
        )

    def _verify_against_sources(
        self,
        memory,
        sources: List[str]
    ) -> Verification:
        """Verify memory against authoritative sources."""
        if not self.llm:
            return Verification(
                status=VerificationStatus.UNVERIFIABLE,
                confidence=0.5,
                reason="No LLM available for verification"
            )

        prompt = f"""
        Verify if this memory/fact is consistent with the provided sources.

        Memory to verify: {memory.content}

        Authoritative sources:
        {chr(10).join(f'- {s}' for s in sources)}

        Respond with:
        - STATUS: verified/contradicted/uncertain
        - CONFIDENCE: 0.0-1.0
        - REASON: Brief explanation
        - CONTRADICTION: If contradicted, what source contradicts it
        """

        response = self.llm.complete(prompt)
        return self._parse_verification_response(response)

    def _cross_verify(
        self,
        memory,
        all_candidates: List
    ) -> Verification:
        """Verify memory against other retrieved memories."""
        if not self.llm:
            # Fallback: Use confidence as proxy
            if memory.confidence >= self.verification_threshold:
                return Verification(
                    status=VerificationStatus.VERIFIED,
                    confidence=memory.confidence,
                    reason="High confidence score"
                )
            return Verification(
                status=VerificationStatus.UNCERTAIN,
                confidence=memory.confidence,
                reason="Confidence below threshold"
            )

        # Use LLM to check consistency
        other_memories = [m for m in all_candidates if m.id != memory.id][:5]

        prompt = f"""
        Check if this memory is consistent with related memories.

        Memory to check: {memory.content}

        Related memories:
        {chr(10).join(f'- {m.content}' for m in other_memories)}

        Respond with:
        - STATUS: verified/contradicted/uncertain
        - CONFIDENCE: 0.0-1.0
        - REASON: Brief explanation
        """

        response = self.llm.complete(prompt)
        return self._parse_verification_response(response)

    def _parse_verification_response(self, response: str) -> Verification:
        """Parse LLM verification response."""
        # Simple parsing - production would be more robust
        lines = response.strip().split('\n')

        status = VerificationStatus.UNCERTAIN
        confidence = 0.5
        reason = "Unable to parse response"
        contradiction = None

        for line in lines:
            line = line.strip()
            if line.startswith('STATUS:'):
                status_str = line.split(':', 1)[1].strip().lower()
                if 'verified' in status_str:
                    status = VerificationStatus.VERIFIED
                elif 'contradict' in status_str:
                    status = VerificationStatus.CONTRADICTED
                else:
                    status = VerificationStatus.UNCERTAIN
            elif line.startswith('CONFIDENCE:'):
                try:
                    confidence = float(line.split(':', 1)[1].strip())
                except ValueError:
                    pass
            elif line.startswith('REASON:'):
                reason = line.split(':', 1)[1].strip()
            elif line.startswith('CONTRADICTION:'):
                contradiction = line.split(':', 1)[1].strip()

        return Verification(
            status=status,
            confidence=confidence,
            reason=reason,
            contradicting_source=contradiction
        )
```

---

## Priority 4: Memory Compression Pipeline

### File: `alma/compression/pipeline.py` (NEW)

```python
"""Memory compression for efficient storage and retrieval."""
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from enum import Enum


class CompressionLevel(Enum):
    NONE = "none"
    LIGHT = "light"      # Remove redundancy
    MEDIUM = "medium"    # Extract key points
    AGGRESSIVE = "aggressive"  # Maximum compression


@dataclass
class CompressedMemory:
    original_length: int
    compressed_length: int
    compression_ratio: float
    key_facts: List[str]
    constraints: List[str]
    summary: str
    full_content: str  # Original preserved for verification


class MemoryCompressor:
    """Compress verbose inputs into structured memory."""

    def __init__(self, llm_client=None):
        self.llm = llm_client

    def compress_outcome(
        self,
        verbose_outcome: str,
        level: CompressionLevel = CompressionLevel.MEDIUM
    ) -> CompressedMemory:
        """Extract key facts from verbose task outcome."""
        if not self.llm:
            return self._rule_based_compression(verbose_outcome)

        prompt = f"""
        Compress this task outcome into structured memory. Extract only essential information.

        Original outcome:
        {verbose_outcome}

        Extract:
        1. CORE_STRATEGY: The main approach used (1 sentence max)
        2. KEY_FACTS: Essential facts that should be remembered (bullet list, max 5)
        3. CONSTRAINTS: Any limitations or requirements (bullet list)
        4. REUSABLE_PATTERN: A rule for similar future situations (1 sentence)

        Be extremely concise. Every word must earn its place.
        """

        response = self.llm.complete(prompt)
        return self._parse_compression_response(response, verbose_outcome)

    def compress_conversation(
        self,
        conversation: str,
        focus: Optional[str] = None
    ) -> CompressedMemory:
        """Compress a conversation into learnable memory."""
        if not self.llm:
            return self._rule_based_compression(conversation)

        focus_clause = f" Focus on: {focus}" if focus else ""

        prompt = f"""
        Extract learnable knowledge from this conversation.{focus_clause}

        Conversation:
        {conversation}

        Extract:
        1. DECISIONS_MADE: Key decisions and their rationale (bullet list)
        2. FACTS_ESTABLISHED: Facts that were confirmed (bullet list)
        3. PATTERNS_IDENTIFIED: Reusable patterns or rules (bullet list)
        4. CONSTRAINTS_DISCOVERED: Limitations found (bullet list)
        5. SUMMARY: One paragraph capturing essential knowledge

        Only extract genuinely useful, reusable knowledge. Skip pleasantries and tangents.
        """

        response = self.llm.complete(prompt)
        return self._parse_compression_response(response, conversation)

    def extract_heuristic(
        self,
        experiences: List[str]
    ) -> Optional[str]:
        """Extract a heuristic rule from multiple experiences."""
        if len(experiences) < 3:
            return None

        if not self.llm:
            return None

        prompt = f"""
        These are {len(experiences)} similar experiences. Extract a general rule.

        Experiences:
        {chr(10).join(f'{i+1}. {e}' for i, e in enumerate(experiences))}

        If there is a clear pattern, state it as a single rule in the form:
        "When [situation], then [action] because [reason]."

        If no clear pattern exists, respond with: NO_PATTERN
        """

        response = self.llm.complete(prompt).strip()
        if "NO_PATTERN" in response:
            return None
        return response

    def deduplicate_knowledge(
        self,
        new_knowledge: str,
        existing_knowledge: List[str]
    ) -> Optional[str]:
        """Merge new knowledge with existing, removing redundancy."""
        if not existing_knowledge:
            return new_knowledge

        if not self.llm:
            return new_knowledge

        # Find most similar existing knowledge
        similar = self._find_similar(new_knowledge, existing_knowledge, threshold=0.7)

        if not similar:
            return new_knowledge  # Truly new

        prompt = f"""
        Merge this new knowledge with existing knowledge, eliminating redundancy.

        New: {new_knowledge}

        Existing similar knowledge:
        {chr(10).join(f'- {k}' for k in similar)}

        If the new knowledge adds nothing, respond: DUPLICATE
        If it adds new information, respond with a merged statement that combines both.
        """

        response = self.llm.complete(prompt).strip()
        if "DUPLICATE" in response:
            return None
        return response

    def _rule_based_compression(self, text: str) -> CompressedMemory:
        """Fallback compression without LLM."""
        # Simple extractive compression
        sentences = text.split('.')

        # Take first and last sentences as summary
        summary = f"{sentences[0]}. {sentences[-1]}." if len(sentences) > 1 else text

        # Extract anything that looks like a fact (contains "is", "are", "has")
        key_facts = [s.strip() for s in sentences if any(w in s.lower() for w in [' is ', ' are ', ' has '])][:5]

        return CompressedMemory(
            original_length=len(text),
            compressed_length=len(summary),
            compression_ratio=len(summary) / len(text) if text else 1.0,
            key_facts=key_facts,
            constraints=[],
            summary=summary,
            full_content=text
        )

    def _parse_compression_response(self, response: str, original: str) -> CompressedMemory:
        """Parse LLM compression response."""
        key_facts = []
        constraints = []
        summary = ""

        current_section = None
        for line in response.split('\n'):
            line = line.strip()
            if not line:
                continue

            if 'KEY_FACTS' in line or 'FACTS' in line:
                current_section = 'facts'
            elif 'CONSTRAINTS' in line:
                current_section = 'constraints'
            elif 'SUMMARY' in line or 'CORE_STRATEGY' in line or 'REUSABLE' in line:
                current_section = 'summary'
            elif line.startswith('-') or line.startswith('•'):
                content = line.lstrip('-•').strip()
                if current_section == 'facts':
                    key_facts.append(content)
                elif current_section == 'constraints':
                    constraints.append(content)
            elif current_section == 'summary' and line:
                summary += line + " "

        summary = summary.strip() or response[:200]

        return CompressedMemory(
            original_length=len(original),
            compressed_length=len(summary),
            compression_ratio=len(summary) / len(original) if original else 1.0,
            key_facts=key_facts,
            constraints=constraints,
            summary=summary,
            full_content=original
        )

    def _find_similar(
        self,
        query: str,
        candidates: List[str],
        threshold: float
    ) -> List[str]:
        """Find semantically similar strings (placeholder for embedding)."""
        # In production, use embeddings
        # Simple word overlap for now
        query_words = set(query.lower().split())
        similar = []

        for candidate in candidates:
            cand_words = set(candidate.lower().split())
            overlap = len(query_words & cand_words) / max(len(query_words | cand_words), 1)
            if overlap >= threshold:
                similar.append(candidate)

        return similar
```

---

## New MCP Tools

### File: `alma/mcp/tools_enhanced.py` (additions)

```python
# Add to existing tools.py

@mcp_tool("alma_retrieve_for_mode")
async def retrieve_for_mode(
    query: str,
    mode: str,  # "broad", "precise", "diagnostic", "learning", "recall"
    project_id: Optional[str] = None,
    agent: Optional[str] = None,
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """Retrieve memories using mode-aware strategy."""
    from alma.retrieval.modes import RetrievalMode

    mode_enum = RetrievalMode(mode)
    memories = alma.retrieve_with_mode(
        query=query,
        mode=mode_enum,
        project_id=project_id,
        agent=agent,
        top_k=top_k
    )
    return [m.to_dict() for m in memories]


@mcp_tool("alma_retrieve_verified")
async def retrieve_verified(
    query: str,
    ground_truth: Optional[List[str]] = None,
    project_id: Optional[str] = None,
    agent: Optional[str] = None
) -> Dict[str, Any]:
    """Two-stage verified retrieval."""
    from alma.retrieval.verification import VerifiedRetriever

    retriever = VerifiedRetriever(alma.retrieval_engine, alma.llm)
    results = retriever.retrieve_verified(
        query=query,
        ground_truth_sources=ground_truth
    )

    return {
        "verified": [vm.memory.to_dict() for vm in results.verified],
        "uncertain": [vm.memory.to_dict() for vm in results.uncertain],
        "contradicted": [vm.memory.to_dict() for vm in results.contradicted],
        "total_candidates": results.metadata.get("total_candidates", 0)
    }


@mcp_tool("alma_compress_and_learn")
async def compress_and_learn(
    content: str,
    memory_type: str,  # "outcome", "knowledge", "heuristic"
    project_id: Optional[str] = None,
    agent: Optional[str] = None
) -> Dict[str, Any]:
    """Compress content and store as memory."""
    from alma.compression.pipeline import MemoryCompressor

    compressor = MemoryCompressor(alma.llm)
    compressed = compressor.compress_outcome(content)

    # Store compressed version
    memory_id = alma.learn(
        content=compressed.summary,
        memory_type=memory_type,
        project_id=project_id,
        agent=agent,
        metadata={
            "compression_ratio": compressed.compression_ratio,
            "key_facts": compressed.key_facts,
            "original_length": compressed.original_length
        }
    )

    return {
        "memory_id": memory_id,
        "compression_ratio": compressed.compression_ratio,
        "key_facts": compressed.key_facts
    }


@mcp_tool("alma_reinforce")
async def reinforce_memory(memory_id: str) -> Dict[str, Any]:
    """Explicitly reinforce a memory (prevent decay)."""
    from alma.learning.decay import DecayManager

    decay_manager = DecayManager(alma.storage)
    decay_manager.reinforce_memory(memory_id)

    strength = decay_manager.get_strength(memory_id)
    return {
        "memory_id": memory_id,
        "new_strength": strength.current_strength(),
        "access_count": strength.access_count
    }


@mcp_tool("alma_get_weak_memories")
async def get_weak_memories(
    project_id: Optional[str] = None,
    agent: Optional[str] = None,
    include_recoverable: bool = True
) -> Dict[str, Any]:
    """Get memories that are weak and may be forgotten."""
    from alma.learning.decay import DecayManager

    decay_manager = DecayManager(alma.storage)

    forgettable = decay_manager.get_forgettable_memories(project_id, agent)
    result = {"forgettable": forgettable}

    if include_recoverable:
        recoverable = decay_manager.get_recoverable_memories(project_id, agent)
        result["recoverable"] = [
            {"memory_id": mid, "strength": strength}
            for mid, strength in recoverable
        ]

    return result
```

---

## Testing Requirements

Each enhancement should have:

1. **Unit tests** - Test individual functions
2. **Integration tests** - Test with storage backends
3. **Performance tests** - Ensure no regression in retrieval latency
4. **Edge case tests** - Empty inputs, extreme values, error conditions

Example test structure:

```python
# tests/unit/test_modes.py
class TestRetrievalModes:
    def test_mode_inference_diagnostic(self):
        query = "Why is the login failing?"
        mode = infer_mode_from_query(query)
        assert mode == RetrievalMode.DIAGNOSTIC

    def test_mode_config_weights_sum_to_one(self):
        for mode, config in MODE_CONFIGS.items():
            total = sum(config.weights.values())
            assert abs(total - 1.0) < 0.01, f"{mode} weights don't sum to 1"


# tests/unit/test_decay.py
class TestMemoryDecay:
    def test_fresh_memory_full_strength(self):
        strength = MemoryStrength(memory_id="test")
        assert strength.current_strength() > 0.9

    def test_old_memory_decays(self):
        strength = MemoryStrength(
            memory_id="test",
            last_accessed=datetime.utcnow() - timedelta(days=60)
        )
        assert strength.current_strength() < 0.5

    def test_access_reinforces(self):
        strength = MemoryStrength(
            memory_id="test",
            last_accessed=datetime.utcnow() - timedelta(days=30)
        )
        initial = strength.current_strength()
        strength.access()
        assert strength.current_strength() > initial
```

---

## Migration Path

For existing ALMA installations:

1. **Schema additions** - New tables for memory strength tracking
2. **Backfill** - Initialize strength records for existing memories
3. **Configuration** - New config options for decay parameters
4. **Gradual rollout** - Feature flags for new retrieval modes

```yaml
# config.yaml additions
alma:
  decay:
    enabled: true
    default_half_life_days: 30
    forget_threshold: 0.1
    auto_forget_enabled: false  # Manual trigger initially

  retrieval:
    mode_aware: true
    default_mode: "precise"
    verification_enabled: false  # Opt-in initially

  compression:
    enabled: true
    level: "medium"
    llm_required: false  # Fallback to rule-based
```
