"""
ALMA Shared Test Fixtures.

Provides common fixtures for all test types:
- Unit tests
- Integration tests
- E2E tests
- Performance benchmarks

The fixtures follow a layered approach:
1. Base fixtures (storage, config)
2. ALMA instance fixtures
3. Agent-specific fixtures (Helena, Victor hooks)
4. Memory fixtures (pre-seeded data)
"""

import shutil
import tempfile
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Generator, List
from unittest.mock import MagicMock, patch

import pytest

from alma import (
    ALMA,
    AntiPattern,
    DomainKnowledge,
    Heuristic,
    MemoryScope,
    MemorySlice,
    Outcome,
    UserPreference,
)
from alma.integration.helena import (
    HELENA_CATEGORIES,
    HelenaHooks,
    UITestContext,
    UITestOutcome,
)
from alma.integration.victor import (
    VICTOR_CATEGORIES,
    APITestContext,
    APITestOutcome,
    VictorHooks,
)
from alma.learning.protocols import LearningProtocol
from alma.retrieval.engine import RetrievalEngine
from alma.storage.file_based import FileBasedStorage

# =============================================================================
# Base Fixtures - Storage and Configuration
# =============================================================================


@pytest.fixture
def temp_storage_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for storage."""
    temp_dir = tempfile.mkdtemp(prefix="alma_test_")
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def test_config(temp_storage_dir: Path) -> Dict[str, Any]:
    """Create a test configuration dict."""
    return {
        "project_id": "test-project",
        "storage": "file",
        "storage_dir": str(temp_storage_dir),
        "embedding_provider": "mock",
        "agents": {
            "helena": {
                "can_learn": HELENA_CATEGORIES,
                "cannot_learn": [
                    "backend_logic",
                    "database_queries",
                    "api_design",
                ],
                "min_occurrences_for_heuristic": 3,
            },
            "victor": {
                "can_learn": VICTOR_CATEGORIES,
                "cannot_learn": [
                    "frontend_styling",
                    "ui_testing",
                    "marketing_content",
                ],
                "min_occurrences_for_heuristic": 3,
            },
        },
    }


@pytest.fixture
def storage(temp_storage_dir: Path) -> FileBasedStorage:
    """Create a file-based storage instance for testing."""
    return FileBasedStorage(storage_dir=temp_storage_dir)


# =============================================================================
# ALMA Instance Fixtures
# =============================================================================


@pytest.fixture
def scopes() -> Dict[str, MemoryScope]:
    """Create memory scopes for Helena and Victor."""
    return {
        "helena": MemoryScope(
            agent_name="helena",
            can_learn=HELENA_CATEGORIES,
            cannot_learn=["backend_logic", "database_queries", "api_design"],
            min_occurrences_for_heuristic=3,
        ),
        "victor": MemoryScope(
            agent_name="victor",
            can_learn=VICTOR_CATEGORIES,
            cannot_learn=["frontend_styling", "ui_testing", "marketing_content"],
            min_occurrences_for_heuristic=3,
        ),
    }


@pytest.fixture
def retrieval_engine(storage: FileBasedStorage) -> RetrievalEngine:
    """Create a retrieval engine for testing."""
    return RetrievalEngine(
        storage=storage,
        embedding_provider="mock",
    )


@pytest.fixture
def learning_protocol(
    storage: FileBasedStorage,
    scopes: Dict[str, MemoryScope],
) -> LearningProtocol:
    """Create a learning protocol for testing."""
    return LearningProtocol(
        storage=storage,
        scopes=scopes,
    )


@pytest.fixture
def alma_instance(
    storage: FileBasedStorage,
    retrieval_engine: RetrievalEngine,
    learning_protocol: LearningProtocol,
    scopes: Dict[str, MemoryScope],
) -> ALMA:
    """Create a fully configured ALMA instance for testing."""
    return ALMA(
        storage=storage,
        retrieval_engine=retrieval_engine,
        learning_protocol=learning_protocol,
        scopes=scopes,
        project_id="test-project",
    )


@pytest.fixture
def mock_alma() -> MagicMock:
    """Create a mock ALMA instance for unit tests."""
    alma = MagicMock(spec=ALMA)
    alma.project_id = "test-project"

    # Default retrieve returns empty slice
    alma.retrieve.return_value = MemorySlice(
        heuristics=[],
        anti_patterns=[],
        outcomes=[],
        domain_knowledge=[],
        preferences=[],
    )

    # Default learn returns True
    alma.learn.return_value = True
    alma.add_user_preference.return_value = MagicMock(id="pref-1")
    alma.add_domain_knowledge.return_value = MagicMock(id="dk-1")
    alma.forget.return_value = 0
    alma.get_stats.return_value = {"total_count": 0}

    return alma


# =============================================================================
# Agent-Specific Fixtures - Helena
# =============================================================================


@pytest.fixture
def helena_hooks(alma_instance: ALMA) -> HelenaHooks:
    """Create Helena hooks with a real ALMA instance."""
    with patch("alma.integration.helena.CodingDomain") as mock_domain:
        mock_domain.create_helena.return_value = MagicMock()
        hooks = HelenaHooks(alma=alma_instance, auto_learn=True)
        return hooks


@pytest.fixture
def mock_helena_hooks(mock_alma: MagicMock) -> HelenaHooks:
    """Create Helena hooks with a mock ALMA instance."""
    with patch("alma.integration.helena.CodingDomain") as mock_domain:
        mock_domain.create_helena.return_value = MagicMock()
        hooks = HelenaHooks(alma=mock_alma, auto_learn=True)
        hooks.alma = mock_alma  # Ensure mock is used
        return hooks


@pytest.fixture
def helena_test_context() -> UITestContext:
    """Create a sample Helena test context."""
    return UITestContext(
        task_description="Test login form validation",
        task_type="form_testing",
        agent_name="helena",
        project_id="test-project",
        component_type="form",
        page_url="/login",
        viewport={"width": 1280, "height": 720},
    )


@pytest.fixture
def helena_test_outcome() -> UITestOutcome:
    """Create a sample Helena test outcome."""
    return UITestOutcome(
        success=True,
        strategy_used="validate inputs first, then submit",
        selectors_used=[
            "[data-testid='email']",
            "[data-testid='password']",
            "[data-testid='submit']",
        ],
        duration_ms=1500,
    )


# =============================================================================
# Agent-Specific Fixtures - Victor
# =============================================================================


@pytest.fixture
def victor_hooks(alma_instance: ALMA) -> VictorHooks:
    """Create Victor hooks with a real ALMA instance."""
    with patch("alma.integration.victor.CodingDomain") as mock_domain:
        mock_domain.create_victor.return_value = MagicMock()
        hooks = VictorHooks(alma=alma_instance, auto_learn=True)
        return hooks


@pytest.fixture
def mock_victor_hooks(mock_alma: MagicMock) -> VictorHooks:
    """Create Victor hooks with a mock ALMA instance."""
    with patch("alma.integration.victor.CodingDomain") as mock_domain:
        mock_domain.create_victor.return_value = MagicMock()
        hooks = VictorHooks(alma=mock_alma, auto_learn=True)
        hooks.alma = mock_alma  # Ensure mock is used
        return hooks


@pytest.fixture
def victor_test_context() -> APITestContext:
    """Create a sample Victor test context."""
    return APITestContext(
        task_description="Test user authentication endpoint",
        task_type="authentication_patterns",
        agent_name="victor",
        project_id="test-project",
        endpoint="/api/v1/auth/login",
        method="POST",
        is_auth_test=True,
    )


@pytest.fixture
def victor_test_outcome() -> APITestOutcome:
    """Create a sample Victor test outcome."""
    return APITestOutcome(
        success=True,
        strategy_used="validate request, check auth, process",
        response_status=200,
        response_time_ms=150,
    )


# =============================================================================
# Memory Fixtures - Pre-seeded Data
# =============================================================================


@pytest.fixture
def sample_heuristics() -> List[Heuristic]:
    """Create sample heuristics for testing."""
    now = datetime.now(timezone.utc)
    return [
        Heuristic(
            id=str(uuid.uuid4()),
            agent="helena",
            project_id="test-project",
            condition="form testing with multiple inputs",
            strategy="validate each input individually before full form submit",
            confidence=0.85,
            occurrence_count=10,
            success_count=9,
            last_validated=now,
            created_at=now - timedelta(days=30),
        ),
        Heuristic(
            id=str(uuid.uuid4()),
            agent="helena",
            project_id="test-project",
            condition="modal dialog testing",
            strategy="wait for animation before interacting",
            confidence=0.90,
            occurrence_count=15,
            success_count=14,
            last_validated=now,
            created_at=now - timedelta(days=20),
        ),
        Heuristic(
            id=str(uuid.uuid4()),
            agent="victor",
            project_id="test-project",
            condition="API endpoint testing",
            strategy="check authentication before payload validation",
            confidence=0.92,
            occurrence_count=20,
            success_count=19,
            last_validated=now,
            created_at=now - timedelta(days=15),
        ),
    ]


@pytest.fixture
def sample_domain_knowledge() -> List[DomainKnowledge]:
    """Create sample domain knowledge for testing."""
    now = datetime.now(timezone.utc)
    return [
        DomainKnowledge(
            id=str(uuid.uuid4()),
            agent="helena",
            project_id="test-project",
            domain="selector_patterns",
            fact="data-testid selectors are most stable for testing",
            source="test_run:stability=0.95",
            confidence=0.95,
            last_verified=now,
        ),
        DomainKnowledge(
            id=str(uuid.uuid4()),
            agent="helena",
            project_id="test-project",
            domain="accessibility_testing",
            fact="All form inputs must have associated labels",
            source="accessibility_audit",
            confidence=1.0,
            last_verified=now,
        ),
        DomainKnowledge(
            id=str(uuid.uuid4()),
            agent="victor",
            project_id="test-project",
            domain="error_handling",
            fact="Always return structured error responses with error codes",
            source="api_design_review",
            confidence=1.0,
            last_verified=now,
        ),
    ]


@pytest.fixture
def sample_anti_patterns() -> List[AntiPattern]:
    """Create sample anti-patterns for testing."""
    now = datetime.now(timezone.utc)
    return [
        AntiPattern(
            id=str(uuid.uuid4()),
            agent="helena",
            project_id="test-project",
            pattern="Using fixed sleep() for async waits",
            why_bad="Causes flaky tests, doesn't adapt to load",
            better_alternative="Use explicit waits with conditions",
            occurrence_count=5,
            last_seen=now,
            created_at=now - timedelta(days=10),
        ),
        AntiPattern(
            id=str(uuid.uuid4()),
            agent="victor",
            project_id="test-project",
            pattern="Testing with hardcoded auth tokens",
            why_bad="Tokens expire, tests become brittle",
            better_alternative="Use token generation in test setup",
            occurrence_count=3,
            last_seen=now,
            created_at=now - timedelta(days=5),
        ),
    ]


@pytest.fixture
def sample_user_preferences() -> List[UserPreference]:
    """Create sample user preferences for testing."""
    now = datetime.now(timezone.utc)
    return [
        UserPreference(
            id=str(uuid.uuid4()),
            user_id="test-user",
            category="code_style",
            preference="No emojis in documentation",
            source="explicit_instruction",
            confidence=1.0,
            timestamp=now,
        ),
        UserPreference(
            id=str(uuid.uuid4()),
            user_id="test-user",
            category="testing",
            preference="Prefer data-testid over CSS selectors",
            source="explicit_instruction",
            confidence=1.0,
            timestamp=now,
        ),
    ]


@pytest.fixture
def seeded_memory_slice(
    sample_heuristics: List[Heuristic],
    sample_domain_knowledge: List[DomainKnowledge],
    sample_anti_patterns: List[AntiPattern],
    sample_user_preferences: List[UserPreference],
) -> MemorySlice:
    """Create a pre-populated MemorySlice for testing."""
    return MemorySlice(
        heuristics=sample_heuristics,
        anti_patterns=sample_anti_patterns,
        domain_knowledge=sample_domain_knowledge,
        preferences=sample_user_preferences,
        query="test query",
        agent="test",
        retrieval_time_ms=10,
    )


@pytest.fixture
def seeded_storage(
    storage: FileBasedStorage,
    sample_heuristics: List[Heuristic],
    sample_domain_knowledge: List[DomainKnowledge],
    sample_anti_patterns: List[AntiPattern],
    sample_user_preferences: List[UserPreference],
) -> FileBasedStorage:
    """Create storage pre-seeded with test data."""
    for h in sample_heuristics:
        storage.save_heuristic(h)

    for dk in sample_domain_knowledge:
        storage.save_domain_knowledge(dk)

    for ap in sample_anti_patterns:
        storage.save_anti_pattern(ap)

    for pref in sample_user_preferences:
        storage.save_user_preference(pref)

    return storage


@pytest.fixture
def seeded_alma(
    seeded_storage: FileBasedStorage,
    scopes: Dict[str, MemoryScope],
) -> ALMA:
    """Create ALMA instance with pre-seeded data."""
    retrieval = RetrievalEngine(
        storage=seeded_storage,
        embedding_provider="mock",
    )

    learning = LearningProtocol(
        storage=seeded_storage,
        scopes=scopes,
    )

    return ALMA(
        storage=seeded_storage,
        retrieval_engine=retrieval,
        learning_protocol=learning,
        scopes=scopes,
        project_id="test-project",
    )


# =============================================================================
# Performance Testing Fixtures
# =============================================================================


@pytest.fixture
def large_memory_storage(temp_storage_dir: Path) -> FileBasedStorage:
    """Create storage with many memory items for performance testing."""
    storage = FileBasedStorage(storage_dir=temp_storage_dir)
    now = datetime.now(timezone.utc)

    # Create 1000 heuristics
    for i in range(1000):
        agent = "helena" if i % 2 == 0 else "victor"
        h = Heuristic(
            id=str(uuid.uuid4()),
            agent=agent,
            project_id="test-project",
            condition=f"condition pattern {i}",
            strategy=f"strategy approach {i}",
            confidence=0.5 + (i % 50) / 100,
            occurrence_count=i % 20 + 1,
            success_count=i % 15 + 1,
            last_validated=now,
            created_at=now - timedelta(days=i % 90),
        )
        storage.save_heuristic(h)

    # Create 5000 outcomes
    for i in range(5000):
        agent = "helena" if i % 2 == 0 else "victor"
        o = Outcome(
            id=str(uuid.uuid4()),
            agent=agent,
            project_id="test-project",
            task_type=f"task_type_{i % 10}",
            task_description=f"Task description {i}",
            success=i % 3 != 0,
            strategy_used=f"Strategy {i % 20}",
            duration_ms=100 + (i % 500),
            timestamp=now - timedelta(hours=i),
        )
        storage.save_outcome(o)

    # Create 500 domain knowledge items
    for i in range(500):
        agent = "helena" if i % 2 == 0 else "victor"
        dk = DomainKnowledge(
            id=str(uuid.uuid4()),
            agent=agent,
            project_id="test-project",
            domain=f"domain_{i % 20}",
            fact=f"Important fact number {i}",
            source="test_generation",
            confidence=0.7 + (i % 30) / 100,
            last_verified=now,
        )
        storage.save_domain_knowledge(dk)

    return storage


# =============================================================================
# Multi-Agent Fixtures
# =============================================================================


@pytest.fixture
def multi_agent_scopes() -> Dict[str, MemoryScope]:
    """Create scopes for multiple agents with overlapping permissions and sharing."""
    return {
        "helena": MemoryScope(
            agent_name="helena",
            can_learn=HELENA_CATEGORIES,
            cannot_learn=["backend_logic", "database_queries"],
            share_with=["shared_agent"],  # Helena shares with shared_agent
            min_occurrences_for_heuristic=3,
        ),
        "victor": MemoryScope(
            agent_name="victor",
            can_learn=VICTOR_CATEGORIES,
            cannot_learn=["frontend_styling", "ui_testing"],
            share_with=["shared_agent"],  # Victor shares with shared_agent
            min_occurrences_for_heuristic=3,
        ),
        "shared_agent": MemoryScope(
            agent_name="shared_agent",
            can_learn=HELENA_CATEGORIES + VICTOR_CATEGORIES,
            cannot_learn=[],
            inherit_from=["helena", "victor"],  # Shared agent can read from both
            min_occurrences_for_heuristic=5,
        ),
    }


# =============================================================================
# Utility Fixtures
# =============================================================================


@pytest.fixture
def timestamp_now() -> datetime:
    """Get current UTC timestamp."""
    return datetime.now(timezone.utc)


@pytest.fixture
def timestamp_past() -> datetime:
    """Get timestamp from 30 days ago."""
    return datetime.now(timezone.utc) - timedelta(days=30)


@pytest.fixture
def unique_id() -> str:
    """Generate a unique ID."""
    return str(uuid.uuid4())
