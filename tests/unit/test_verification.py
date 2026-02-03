"""
Unit tests for the Two-Stage Verified Retrieval system.

Tests VerificationStatus, Verification, VerifiedMemory, VerifiedResults,
VerificationConfig, and VerifiedRetriever classes.
"""

from dataclasses import dataclass
from unittest.mock import MagicMock

from alma.retrieval.verification import (
    Verification,
    VerificationConfig,
    VerificationMethod,
    VerificationStatus,
    VerifiedMemory,
    VerifiedResults,
    VerifiedRetriever,
    create_verified_retriever,
)


class TestVerificationStatus:
    """Tests for VerificationStatus enum."""

    def test_verified_status(self):
        """Should have verified status."""
        assert VerificationStatus.VERIFIED.value == "verified"

    def test_uncertain_status(self):
        """Should have uncertain status."""
        assert VerificationStatus.UNCERTAIN.value == "uncertain"

    def test_contradicted_status(self):
        """Should have contradicted status."""
        assert VerificationStatus.CONTRADICTED.value == "contradicted"

    def test_unverifiable_status(self):
        """Should have unverifiable status."""
        assert VerificationStatus.UNVERIFIABLE.value == "unverifiable"


class TestVerificationMethod:
    """Tests for VerificationMethod enum."""

    def test_ground_truth_method(self):
        """Should have ground truth method."""
        assert VerificationMethod.GROUND_TRUTH.value == "ground_truth"

    def test_cross_verify_method(self):
        """Should have cross verify method."""
        assert VerificationMethod.CROSS_VERIFY.value == "cross_verify"

    def test_confidence_method(self):
        """Should have confidence method."""
        assert VerificationMethod.CONFIDENCE.value == "confidence"

    def test_none_method(self):
        """Should have none method."""
        assert VerificationMethod.NONE.value == "none"


class TestVerification:
    """Tests for Verification dataclass."""

    def test_create_verification(self):
        """Should create verification with required fields."""
        v = Verification(
            status=VerificationStatus.VERIFIED,
            confidence=0.95,
            reason="Confirmed by sources",
        )

        assert v.status == VerificationStatus.VERIFIED
        assert v.confidence == 0.95
        assert v.reason == "Confirmed by sources"
        assert v.method == VerificationMethod.NONE
        assert v.contradicting_source is None

    def test_create_with_all_fields(self):
        """Should create verification with all fields."""
        v = Verification(
            status=VerificationStatus.CONTRADICTED,
            confidence=0.3,
            reason="Conflicts with current API",
            method=VerificationMethod.GROUND_TRUTH,
            contradicting_source="API documentation v2",
            verification_time_ms=150,
        )

        assert v.status == VerificationStatus.CONTRADICTED
        assert v.contradicting_source == "API documentation v2"
        assert v.verification_time_ms == 150

    def test_confidence_clamped_to_range(self):
        """Should clamp confidence to 0-1 range."""
        v1 = Verification(
            status=VerificationStatus.VERIFIED,
            confidence=1.5,
            reason="Test",
        )
        assert v1.confidence == 1.0

        v2 = Verification(
            status=VerificationStatus.VERIFIED,
            confidence=-0.5,
            reason="Test",
        )
        assert v2.confidence == 0.0

    def test_is_usable(self):
        """Should return correct usability."""
        verified = Verification(
            status=VerificationStatus.VERIFIED,
            confidence=0.9,
            reason="Test",
        )
        assert verified.is_usable() is True

        uncertain = Verification(
            status=VerificationStatus.UNCERTAIN,
            confidence=0.5,
            reason="Test",
        )
        assert uncertain.is_usable() is True

        contradicted = Verification(
            status=VerificationStatus.CONTRADICTED,
            confidence=0.3,
            reason="Test",
        )
        assert contradicted.is_usable() is False

        unverifiable = Verification(
            status=VerificationStatus.UNVERIFIABLE,
            confidence=0.5,
            reason="Test",
        )
        assert unverifiable.is_usable() is False

    def test_needs_review(self):
        """Should identify memories needing review."""
        contradicted = Verification(
            status=VerificationStatus.CONTRADICTED,
            confidence=0.3,
            reason="Conflict detected",
        )
        assert contradicted.needs_review() is True

        verified = Verification(
            status=VerificationStatus.VERIFIED,
            confidence=0.9,
            reason="OK",
        )
        assert verified.needs_review() is False

    def test_to_dict(self):
        """Should convert to dictionary."""
        v = Verification(
            status=VerificationStatus.VERIFIED,
            confidence=0.85,
            reason="Confirmed",
            method=VerificationMethod.CROSS_VERIFY,
            verification_time_ms=100,
        )

        d = v.to_dict()

        assert d["status"] == "verified"
        assert d["confidence"] == 0.85
        assert d["reason"] == "Confirmed"
        assert d["method"] == "cross_verify"
        assert d["verification_time_ms"] == 100

    def test_from_dict(self):
        """Should create from dictionary."""
        data = {
            "status": "contradicted",
            "confidence": 0.4,
            "reason": "Conflict found",
            "method": "ground_truth",
            "contradicting_source": "docs",
            "verification_time_ms": 200,
        }

        v = Verification.from_dict(data)

        assert v.status == VerificationStatus.CONTRADICTED
        assert v.confidence == 0.4
        assert v.method == VerificationMethod.GROUND_TRUTH
        assert v.contradicting_source == "docs"


class TestVerifiedMemory:
    """Tests for VerifiedMemory dataclass."""

    @dataclass
    class MockMemory:
        id: str
        content: str
        confidence: float = 0.8

        def to_dict(self):
            return {"id": self.id, "content": self.content, "confidence": self.confidence}

    def test_create_verified_memory(self):
        """Should create verified memory wrapper."""
        memory = self.MockMemory(id="mem-1", content="Test content")
        verification = Verification(
            status=VerificationStatus.VERIFIED,
            confidence=0.9,
            reason="OK",
        )

        vm = VerifiedMemory(
            memory=memory,
            verification=verification,
            retrieval_score=0.85,
        )

        assert vm.memory == memory
        assert vm.verification == verification
        assert vm.retrieval_score == 0.85

    def test_status_shortcut(self):
        """Should provide status shortcut."""
        memory = self.MockMemory(id="mem-1", content="Test")
        verification = Verification(
            status=VerificationStatus.UNCERTAIN,
            confidence=0.6,
            reason="Test",
        )

        vm = VerifiedMemory(memory=memory, verification=verification)

        assert vm.status == VerificationStatus.UNCERTAIN

    def test_is_verified_property(self):
        """Should check if verified."""
        memory = self.MockMemory(id="mem-1", content="Test")

        verified = VerifiedMemory(
            memory=memory,
            verification=Verification(
                status=VerificationStatus.VERIFIED,
                confidence=0.9,
                reason="OK",
            ),
        )
        assert verified.is_verified is True

        uncertain = VerifiedMemory(
            memory=memory,
            verification=Verification(
                status=VerificationStatus.UNCERTAIN,
                confidence=0.5,
                reason="Test",
            ),
        )
        assert uncertain.is_verified is False

    def test_is_usable_property(self):
        """Should check if usable."""
        memory = self.MockMemory(id="mem-1", content="Test")

        vm = VerifiedMemory(
            memory=memory,
            verification=Verification(
                status=VerificationStatus.VERIFIED,
                confidence=0.9,
                reason="OK",
            ),
        )
        assert vm.is_usable is True

    def test_combined_score(self):
        """Should compute combined score."""
        memory = self.MockMemory(id="mem-1", content="Test")
        vm = VerifiedMemory(
            memory=memory,
            verification=Verification(
                status=VerificationStatus.VERIFIED,
                confidence=0.8,
                reason="OK",
            ),
            retrieval_score=0.6,
        )

        # Default 50/50 weight
        score = vm.combined_score()
        assert abs(score - 0.7) < 0.001  # 0.6 * 0.5 + 0.8 * 0.5 = 0.7

        # Custom weight
        score_custom = vm.combined_score(verification_weight=0.3)
        # 0.6 * 0.7 + 0.8 * 0.3 = 0.42 + 0.24 = 0.66
        assert abs(score_custom - 0.66) < 0.001

    def test_to_dict(self):
        """Should convert to dictionary."""
        memory = self.MockMemory(id="mem-1", content="Test content")
        vm = VerifiedMemory(
            memory=memory,
            verification=Verification(
                status=VerificationStatus.VERIFIED,
                confidence=0.9,
                reason="OK",
            ),
            retrieval_score=0.8,
        )

        d = vm.to_dict()

        assert d["memory"]["id"] == "mem-1"
        assert d["verification"]["status"] == "verified"
        assert d["retrieval_score"] == 0.8


class TestVerifiedResults:
    """Tests for VerifiedResults container."""

    @dataclass
    class MockMemory:
        id: str
        content: str
        confidence: float = 0.8

    def _create_vm(
        self, memory_id: str, status: VerificationStatus, confidence: float
    ) -> VerifiedMemory:
        memory = self.MockMemory(id=memory_id, content=f"Content {memory_id}")
        return VerifiedMemory(
            memory=memory,
            verification=Verification(
                status=status,
                confidence=confidence,
                reason=f"Reason for {memory_id}",
            ),
            retrieval_score=confidence,
        )

    def test_empty_results(self):
        """Should handle empty results."""
        results = VerifiedResults()

        assert len(results.verified) == 0
        assert len(results.uncertain) == 0
        assert len(results.contradicted) == 0
        assert len(results.unverifiable) == 0
        assert results.total_count == 0

    def test_all_usable_property(self):
        """Should return verified + uncertain."""
        results = VerifiedResults(
            verified=[
                self._create_vm("v1", VerificationStatus.VERIFIED, 0.9),
                self._create_vm("v2", VerificationStatus.VERIFIED, 0.85),
            ],
            uncertain=[
                self._create_vm("u1", VerificationStatus.UNCERTAIN, 0.6),
            ],
            contradicted=[
                self._create_vm("c1", VerificationStatus.CONTRADICTED, 0.3),
            ],
        )

        usable = results.all_usable
        assert len(usable) == 3

    def test_high_confidence_property(self):
        """Should return only verified."""
        results = VerifiedResults(
            verified=[
                self._create_vm("v1", VerificationStatus.VERIFIED, 0.9),
            ],
            uncertain=[
                self._create_vm("u1", VerificationStatus.UNCERTAIN, 0.6),
            ],
        )

        high_conf = results.high_confidence
        assert len(high_conf) == 1
        assert high_conf[0].memory.id == "v1"

    def test_needs_review_property(self):
        """Should return contradicted."""
        results = VerifiedResults(
            verified=[
                self._create_vm("v1", VerificationStatus.VERIFIED, 0.9),
            ],
            contradicted=[
                self._create_vm("c1", VerificationStatus.CONTRADICTED, 0.3),
            ],
        )

        needs_review = results.needs_review
        assert len(needs_review) == 1
        assert needs_review[0].memory.id == "c1"

    def test_total_count(self):
        """Should count all memories."""
        results = VerifiedResults(
            verified=[self._create_vm("v1", VerificationStatus.VERIFIED, 0.9)],
            uncertain=[self._create_vm("u1", VerificationStatus.UNCERTAIN, 0.6)],
            contradicted=[self._create_vm("c1", VerificationStatus.CONTRADICTED, 0.3)],
            unverifiable=[
                self._create_vm("x1", VerificationStatus.UNVERIFIABLE, 0.5)
            ],
        )

        assert results.total_count == 4

    def test_summary(self):
        """Should return summary statistics."""
        results = VerifiedResults(
            verified=[
                self._create_vm("v1", VerificationStatus.VERIFIED, 0.9),
                self._create_vm("v2", VerificationStatus.VERIFIED, 0.85),
            ],
            uncertain=[
                self._create_vm("u1", VerificationStatus.UNCERTAIN, 0.6),
            ],
            contradicted=[],
            unverifiable=[],
        )

        summary = results.summary()

        assert summary["verified"] == 2
        assert summary["uncertain"] == 1
        assert summary["contradicted"] == 0
        assert summary["unverifiable"] == 0
        assert summary["total"] == 3
        assert summary["usable_count"] == 3
        assert summary["usable_ratio"] == 1.0

    def test_get_by_status(self):
        """Should get memories by status."""
        results = VerifiedResults(
            verified=[self._create_vm("v1", VerificationStatus.VERIFIED, 0.9)],
            uncertain=[self._create_vm("u1", VerificationStatus.UNCERTAIN, 0.6)],
        )

        verified = results.get_by_status(VerificationStatus.VERIFIED)
        assert len(verified) == 1
        assert verified[0].memory.id == "v1"

        uncertain = results.get_by_status(VerificationStatus.UNCERTAIN)
        assert len(uncertain) == 1

    def test_sort_by_confidence(self):
        """Should sort all categories by confidence."""
        results = VerifiedResults(
            verified=[
                self._create_vm("v1", VerificationStatus.VERIFIED, 0.7),
                self._create_vm("v2", VerificationStatus.VERIFIED, 0.95),
                self._create_vm("v3", VerificationStatus.VERIFIED, 0.8),
            ],
        )

        results.sort_by_confidence(descending=True)

        assert results.verified[0].memory.id == "v2"
        assert results.verified[1].memory.id == "v3"
        assert results.verified[2].memory.id == "v1"

    def test_to_dict(self):
        """Should convert to dictionary."""
        results = VerifiedResults(
            verified=[self._create_vm("v1", VerificationStatus.VERIFIED, 0.9)],
            metadata={"query": "test"},
        )

        d = results.to_dict()

        assert len(d["verified"]) == 1
        assert d["metadata"]["query"] == "test"
        assert "summary" in d


class TestVerificationConfig:
    """Tests for VerificationConfig dataclass."""

    def test_default_config(self):
        """Should have sensible defaults."""
        config = VerificationConfig()

        assert config.enabled is False  # Opt-in
        assert config.default_method == "confidence"
        assert config.confidence_threshold == 0.7
        assert config.llm_timeout_seconds == 5.0
        assert config.expand_candidates_factor == 4

    def test_custom_config(self):
        """Should accept custom values."""
        config = VerificationConfig(
            enabled=True,
            default_method="cross_verify",
            confidence_threshold=0.8,
            llm_timeout_seconds=10.0,
        )

        assert config.enabled is True
        assert config.default_method == "cross_verify"
        assert config.confidence_threshold == 0.8
        assert config.llm_timeout_seconds == 10.0

    def test_from_dict(self):
        """Should create from dictionary."""
        data = {
            "enabled": True,
            "default_method": "ground_truth",
            "confidence_threshold": 0.75,
        }

        config = VerificationConfig.from_dict(data)

        assert config.enabled is True
        assert config.default_method == "ground_truth"
        assert config.confidence_threshold == 0.75

    def test_to_dict(self):
        """Should convert to dictionary."""
        config = VerificationConfig(enabled=True)

        d = config.to_dict()

        assert d["enabled"] is True
        assert "default_method" in d
        assert "confidence_threshold" in d


class TestVerifiedRetriever:
    """Tests for VerifiedRetriever class."""

    @dataclass
    class MockMemory:
        id: str
        content: str
        confidence: float = 0.8

    class MockMemorySlice:
        def __init__(self, heuristics=None, outcomes=None):
            self.heuristics = heuristics or []
            self.outcomes = outcomes or []
            self.knowledge = []
            self.anti_patterns = []
            self.preferences = []

    def test_create_retriever(self):
        """Should create retriever with engine."""
        engine = MagicMock()
        retriever = VerifiedRetriever(retrieval_engine=engine)

        assert retriever.retrieval_engine == engine
        assert retriever.llm is None
        assert retriever.config.enabled is False

    def test_create_with_llm(self):
        """Should create retriever with LLM."""
        engine = MagicMock()
        llm = MagicMock()

        retriever = VerifiedRetriever(
            retrieval_engine=engine,
            llm_client=llm,
            config=VerificationConfig(enabled=True),
        )

        assert retriever.llm == llm
        assert retriever.config.enabled is True

    def test_retrieve_verified_uses_expanded_candidates(self):
        """Should retrieve with expanded candidate set."""
        engine = MagicMock()
        engine.retrieve.return_value = self.MockMemorySlice(
            heuristics=[
                self.MockMemory(id="h1", content="Test 1", confidence=0.9),
                self.MockMemory(id="h2", content="Test 2", confidence=0.8),
            ]
        )

        retriever = VerifiedRetriever(
            retrieval_engine=engine,
            config=VerificationConfig(expand_candidates_factor=4),
        )

        retriever.retrieve_verified(
            query="test query",
            agent="test-agent",
            project_id="test-project",
            top_k=5,
        )

        # Should expand candidates by factor
        call_args = engine.retrieve.call_args
        assert call_args.kwargs["top_k"] == 20  # 5 * 4

    def test_confidence_fallback_without_llm(self):
        """Should use confidence fallback without LLM."""
        engine = MagicMock()
        engine.retrieve.return_value = self.MockMemorySlice(
            heuristics=[
                self.MockMemory(id="h1", content="High conf", confidence=0.9),
                self.MockMemory(id="h2", content="Low conf", confidence=0.3),
            ]
        )

        retriever = VerifiedRetriever(
            retrieval_engine=engine,
            llm_client=None,  # No LLM
            config=VerificationConfig(confidence_threshold=0.7),
        )

        results = retriever.retrieve_verified(
            query="test",
            agent="agent",
            project_id="proj",
        )

        # High confidence should be verified
        assert len(results.verified) == 1
        assert results.verified[0].memory.id == "h1"

        # Low confidence should be uncertain
        assert len(results.uncertain) == 1
        assert results.uncertain[0].memory.id == "h2"

    def test_ground_truth_verification_with_llm(self):
        """Should use ground truth verification when sources provided."""
        engine = MagicMock()
        engine.retrieve.return_value = self.MockMemorySlice(
            heuristics=[
                self.MockMemory(id="h1", content="Test memory", confidence=0.5),
            ]
        )

        llm = MagicMock()
        llm.complete.return_value = """STATUS: verified
CONFIDENCE: 0.95
REASON: Matches source documentation"""

        retriever = VerifiedRetriever(
            retrieval_engine=engine,
            llm_client=llm,
        )

        results = retriever.retrieve_verified(
            query="test",
            agent="agent",
            project_id="proj",
            ground_truth_sources=["Source doc 1", "Source doc 2"],
        )

        assert len(results.verified) == 1
        assert results.verified[0].verification.method == VerificationMethod.GROUND_TRUTH

    def test_cross_verification_with_llm(self):
        """Should use cross verification when enabled."""
        engine = MagicMock()
        engine.retrieve.return_value = self.MockMemorySlice(
            heuristics=[
                self.MockMemory(id="h1", content="Memory 1", confidence=0.5),
                self.MockMemory(id="h2", content="Memory 2", confidence=0.5),
            ]
        )

        llm = MagicMock()
        llm.complete.return_value = """STATUS: verified
CONFIDENCE: 0.85
REASON: Consistent with related memories"""

        retriever = VerifiedRetriever(
            retrieval_engine=engine,
            llm_client=llm,
            config=VerificationConfig(default_method="cross_verify"),
        )

        results = retriever.retrieve_verified(
            query="test",
            agent="agent",
            project_id="proj",
            cross_verify=True,
        )

        assert len(results.verified) == 2
        assert results.verified[0].verification.method == VerificationMethod.CROSS_VERIFY

    def test_handles_contradicted_response(self):
        """Should handle contradicted verification response."""
        engine = MagicMock()
        engine.retrieve.return_value = self.MockMemorySlice(
            heuristics=[
                self.MockMemory(id="h1", content="Outdated info", confidence=0.5),
            ]
        )

        llm = MagicMock()
        llm.complete.return_value = """STATUS: contradicted
CONFIDENCE: 0.3
REASON: Conflicts with current documentation
CONTRADICTION: API endpoint changed in v2"""

        retriever = VerifiedRetriever(
            retrieval_engine=engine,
            llm_client=llm,
        )

        results = retriever.retrieve_verified(
            query="test",
            agent="agent",
            project_id="proj",
            ground_truth_sources=["Current API docs"],
        )

        assert len(results.contradicted) == 1
        assert results.contradicted[0].verification.contradicting_source is not None

    def test_handles_llm_error_gracefully(self):
        """Should fallback when LLM fails."""
        engine = MagicMock()
        engine.retrieve.return_value = self.MockMemorySlice(
            heuristics=[
                self.MockMemory(id="h1", content="Test", confidence=0.8),
            ]
        )

        llm = MagicMock()
        llm.complete.side_effect = Exception("LLM timeout")

        retriever = VerifiedRetriever(
            retrieval_engine=engine,
            llm_client=llm,
            config=VerificationConfig(confidence_threshold=0.7),
        )

        results = retriever.retrieve_verified(
            query="test",
            agent="agent",
            project_id="proj",
            ground_truth_sources=["Source"],
        )

        # Should fallback to unverifiable due to error
        assert len(results.unverifiable) == 1

    def test_results_limited_to_top_k(self):
        """Should limit results to top_k per category."""
        engine = MagicMock()
        engine.retrieve.return_value = self.MockMemorySlice(
            heuristics=[
                self.MockMemory(id=f"h{i}", content=f"Test {i}", confidence=0.9)
                for i in range(10)
            ]
        )

        retriever = VerifiedRetriever(
            retrieval_engine=engine,
            config=VerificationConfig(confidence_threshold=0.7),
        )

        results = retriever.retrieve_verified(
            query="test",
            agent="agent",
            project_id="proj",
            top_k=3,
        )

        assert len(results.verified) <= 3

    def test_metadata_includes_timing(self):
        """Should include timing in metadata."""
        engine = MagicMock()
        engine.retrieve.return_value = self.MockMemorySlice(
            heuristics=[
                self.MockMemory(id="h1", content="Test", confidence=0.9),
            ]
        )

        retriever = VerifiedRetriever(retrieval_engine=engine)

        results = retriever.retrieve_verified(
            query="test",
            agent="agent",
            project_id="proj",
        )

        assert "total_verification_time_ms" in results.metadata
        assert "total_candidates" in results.metadata
        assert "query" in results.metadata


class TestParseVerificationResponse:
    """Tests for parsing LLM verification responses."""

    def test_parse_verified_response(self):
        """Should parse verified response."""
        engine = MagicMock()
        retriever = VerifiedRetriever(retrieval_engine=engine)

        response = """STATUS: verified
CONFIDENCE: 0.95
REASON: Matches documentation exactly"""

        result = retriever._parse_verification_response(response)

        assert result.status == VerificationStatus.VERIFIED
        assert abs(result.confidence - 0.95) < 0.01
        assert "documentation" in result.reason.lower()

    def test_parse_contradicted_response(self):
        """Should parse contradicted response with source."""
        engine = MagicMock()
        retriever = VerifiedRetriever(retrieval_engine=engine)

        response = """STATUS: contradicted
CONFIDENCE: 0.4
REASON: API has changed since this memory was created
CONTRADICTION: The endpoint /v1/users is now /v2/users"""

        result = retriever._parse_verification_response(response)

        assert result.status == VerificationStatus.CONTRADICTED
        assert result.contradicting_source is not None
        assert "/v2/users" in result.contradicting_source

    def test_parse_uncertain_response(self):
        """Should parse uncertain response."""
        engine = MagicMock()
        retriever = VerifiedRetriever(retrieval_engine=engine)

        response = """STATUS: uncertain
CONFIDENCE: 0.6
REASON: Partially matches but some details unclear"""

        result = retriever._parse_verification_response(response)

        assert result.status == VerificationStatus.UNCERTAIN
        assert abs(result.confidence - 0.6) < 0.01

    def test_parse_malformed_response(self):
        """Should handle malformed response gracefully."""
        engine = MagicMock()
        retriever = VerifiedRetriever(retrieval_engine=engine)

        response = "This is not a properly formatted response"

        result = retriever._parse_verification_response(response)

        assert result.status == VerificationStatus.UNCERTAIN
        assert result.confidence == 0.5


class TestCreateVerifiedRetriever:
    """Tests for factory function."""

    def test_create_with_engine_only(self):
        """Should create with just engine."""
        engine = MagicMock()
        retriever = create_verified_retriever(retrieval_engine=engine)

        assert retriever.retrieval_engine == engine
        assert retriever.llm is None

    def test_create_with_dict_config(self):
        """Should accept dict config."""
        engine = MagicMock()
        config = {"enabled": True, "confidence_threshold": 0.8}

        retriever = create_verified_retriever(
            retrieval_engine=engine,
            config=config,
        )

        assert retriever.config.enabled is True
        assert retriever.config.confidence_threshold == 0.8

    def test_create_with_config_object(self):
        """Should accept VerificationConfig object."""
        engine = MagicMock()
        config = VerificationConfig(enabled=True)

        retriever = create_verified_retriever(
            retrieval_engine=engine,
            config=config,
        )

        assert retriever.config.enabled is True


class TestExtractCandidates:
    """Tests for candidate extraction from memory slices."""

    class MockMemorySlice:
        def __init__(self):
            self.heuristics = [{"id": "h1"}]
            self.outcomes = [{"id": "o1"}]
            self.knowledge = [{"id": "k1"}]
            self.anti_patterns = []
            self.preferences = []

    def test_extract_from_memory_slice(self):
        """Should extract all memory types from slice."""
        engine = MagicMock()
        retriever = VerifiedRetriever(retrieval_engine=engine)

        slice_obj = self.MockMemorySlice()
        candidates = retriever._extract_candidates(slice_obj)

        assert len(candidates) == 3

    def test_extract_from_list(self):
        """Should handle list-like containers."""
        engine = MagicMock()
        retriever = VerifiedRetriever(retrieval_engine=engine)

        candidates = retriever._extract_candidates([1, 2, 3])

        assert candidates == [1, 2, 3]


class TestGetMemoryContent:
    """Tests for extracting content from various memory types."""

    def test_extract_content_attribute(self):
        """Should extract from content attribute."""
        engine = MagicMock()
        retriever = VerifiedRetriever(retrieval_engine=engine)

        class MemWithContent:
            content = "Test content"

        content = retriever._get_memory_content(MemWithContent())
        assert content == "Test content"

    def test_extract_fact_attribute(self):
        """Should extract from fact attribute."""
        engine = MagicMock()
        retriever = VerifiedRetriever(retrieval_engine=engine)

        class MemWithFact:
            fact = "Important fact"

        content = retriever._get_memory_content(MemWithFact())
        assert content == "Important fact"

    def test_extract_strategy_with_condition(self):
        """Should combine condition and strategy."""
        engine = MagicMock()
        retriever = VerifiedRetriever(retrieval_engine=engine)

        class Heuristic:
            condition = "when testing"
            strategy = "use mocks"

        content = retriever._get_memory_content(Heuristic())
        assert "when testing" in content
        assert "use mocks" in content

    def test_fallback_to_str(self):
        """Should fallback to string representation."""
        engine = MagicMock()
        retriever = VerifiedRetriever(retrieval_engine=engine)

        class CustomObj:
            def __str__(self):
                return "Custom string"

        content = retriever._get_memory_content(CustomObj())
        assert content == "Custom string"
