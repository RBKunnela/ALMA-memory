"""
Unit tests for Azure Cosmos DB storage backend.

Uses mocking since we don't have a real Cosmos DB instance in CI.
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

# Skip all tests if azure-cosmos is not installed
pytest.importorskip("azure.cosmos")


from alma.storage.azure_cosmos import AzureCosmosStorage
from alma.types import (
    AntiPattern,
    DomainKnowledge,
    Heuristic,
    Outcome,
    UserPreference,
)


class TestAzureCosmosStorageInit:
    """Tests for AzureCosmosStorage initialization."""

    @patch("alma.storage.azure_cosmos.CosmosClient")
    def test_init_creates_client(self, mock_cosmos_client):
        """Test that initialization creates a Cosmos client."""
        mock_client = MagicMock()
        mock_cosmos_client.return_value = mock_client
        mock_database = MagicMock()
        mock_client.create_database_if_not_exists.return_value = mock_database

        storage = AzureCosmosStorage(
            endpoint="https://test.documents.azure.com:443/",
            key="test-key",
            database_name="test-db",
        )

        mock_cosmos_client.assert_called_once_with(
            "https://test.documents.azure.com:443/",
            {"masterKey": "test-key"},
        )
        assert storage._database == mock_database

    @patch("alma.storage.azure_cosmos.CosmosClient")
    def test_init_creates_containers(self, mock_cosmos_client):
        """Test that initialization creates all required containers."""
        mock_client = MagicMock()
        mock_cosmos_client.return_value = mock_client
        mock_database = MagicMock()
        mock_client.create_database_if_not_exists.return_value = mock_database

        AzureCosmosStorage(
            endpoint="https://test.documents.azure.com:443/",
            key="test-key",
            create_if_not_exists=True,
        )

        # Should create 5 containers
        assert mock_database.create_container_if_not_exists.call_count == 5

        # Check container names
        container_names = [
            call.kwargs["id"]
            for call in mock_database.create_container_if_not_exists.call_args_list
        ]
        assert "alma-heuristics" in container_names
        assert "alma-outcomes" in container_names
        assert "alma-preferences" in container_names
        assert "alma-knowledge" in container_names
        assert "alma-antipatterns" in container_names


class TestHeuristicOperations:
    """Tests for heuristic CRUD operations."""

    @pytest.fixture
    def mock_storage(self):
        """Create a mocked storage instance."""
        with patch("alma.storage.azure_cosmos.CosmosClient") as mock_cosmos:
            mock_client = MagicMock()
            mock_cosmos.return_value = mock_client
            mock_database = MagicMock()
            mock_client.create_database_if_not_exists.return_value = mock_database

            # Mock containers
            mock_containers = {}
            for name in [
                "alma-heuristics",
                "alma-outcomes",
                "alma-preferences",
                "alma-knowledge",
                "alma-antipatterns",
            ]:
                mock_containers[name] = MagicMock()

            mock_database.create_container_if_not_exists.side_effect = (
                lambda **kwargs: mock_containers[kwargs["id"]]
            )

            storage = AzureCosmosStorage(
                endpoint="https://test.documents.azure.com:443/",
                key="test-key",
            )
            storage._mock_containers = mock_containers
            yield storage

    @pytest.fixture
    def sample_heuristic(self):
        """Create a sample heuristic."""
        return Heuristic(
            id="h-123",
            agent="helena",
            project_id="proj-1",
            condition="form testing",
            strategy="validate inputs first",
            confidence=0.85,
            occurrence_count=10,
            success_count=9,
            last_validated=datetime.now(timezone.utc),
            created_at=datetime.now(timezone.utc),
            embedding=[0.1] * 384,
        )

    def test_save_heuristic(self, mock_storage, sample_heuristic):
        """Test saving a heuristic."""
        container = mock_storage._mock_containers["alma-heuristics"]

        mock_storage.save_heuristic(sample_heuristic)

        container.upsert_item.assert_called_once()
        call_args = container.upsert_item.call_args
        doc = call_args[0][0]

        assert doc["id"] == "h-123"
        assert doc["agent"] == "helena"
        assert doc["project_id"] == "proj-1"
        assert doc["condition"] == "form testing"
        assert doc["strategy"] == "validate inputs first"
        assert doc["confidence"] == 0.85

    def test_get_heuristics_by_project(self, mock_storage):
        """Test getting heuristics by project ID."""
        container = mock_storage._mock_containers["alma-heuristics"]

        # Mock query response
        mock_items = [
            {
                "id": "h-1",
                "agent": "helena",
                "project_id": "proj-1",
                "condition": "test",
                "strategy": "do test",
                "confidence": 0.8,
                "occurrence_count": 5,
                "success_count": 4,
                "last_validated": datetime.now(timezone.utc).isoformat(),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "embedding": [0.1] * 384,
            }
        ]
        container.query_items.return_value = iter(mock_items)

        results = mock_storage.get_heuristics(project_id="proj-1")

        assert len(results) == 1
        assert results[0].id == "h-1"
        assert results[0].agent == "helena"
        container.query_items.assert_called_once()

    def test_update_heuristic(self, mock_storage, sample_heuristic):
        """Test updating a heuristic."""
        container = mock_storage._mock_containers["alma-heuristics"]

        # Mock finding the heuristic
        mock_doc = {
            "id": "h-123",
            "agent": "helena",
            "project_id": "proj-1",
            "condition": "form testing",
            "strategy": "validate inputs first",
            "confidence": 0.85,
            "occurrence_count": 10,
            "success_count": 9,
            "last_validated": datetime.now(timezone.utc).isoformat(),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "embedding": [0.1] * 384,
        }
        container.query_items.return_value = iter([mock_doc])

        mock_storage.update_heuristic(
            heuristic_id="h-123",
            confidence=0.90,
            occurrence_count=11,
            success_count=10,
        )

        container.upsert_item.assert_called_once()
        updated_doc = container.upsert_item.call_args[0][0]
        assert updated_doc["confidence"] == 0.90
        assert updated_doc["occurrence_count"] == 11
        assert updated_doc["success_count"] == 10

    def test_delete_heuristic(self, mock_storage):
        """Test deleting a heuristic."""
        container = mock_storage._mock_containers["alma-heuristics"]

        # Mock finding the heuristic
        mock_doc = {
            "id": "h-123",
            "project_id": "proj-1",
        }
        container.query_items.return_value = iter([mock_doc])

        result = mock_storage.delete_heuristic("h-123")

        assert result is True
        container.delete_item.assert_called_once_with(
            item="h-123",
            partition_key="proj-1",
        )


class TestOutcomeOperations:
    """Tests for outcome CRUD operations."""

    @pytest.fixture
    def mock_storage(self):
        """Create a mocked storage instance."""
        with patch("alma.storage.azure_cosmos.CosmosClient") as mock_cosmos:
            mock_client = MagicMock()
            mock_cosmos.return_value = mock_client
            mock_database = MagicMock()
            mock_client.create_database_if_not_exists.return_value = mock_database

            mock_containers = {}
            for name in [
                "alma-heuristics",
                "alma-outcomes",
                "alma-preferences",
                "alma-knowledge",
                "alma-antipatterns",
            ]:
                mock_containers[name] = MagicMock()

            mock_database.create_container_if_not_exists.side_effect = (
                lambda **kwargs: mock_containers[kwargs["id"]]
            )

            storage = AzureCosmosStorage(
                endpoint="https://test.documents.azure.com:443/",
                key="test-key",
            )
            storage._mock_containers = mock_containers
            yield storage

    @pytest.fixture
    def sample_outcome(self):
        """Create a sample outcome."""
        return Outcome(
            id="o-123",
            agent="victor",
            project_id="proj-1",
            task_type="api_testing",
            task_description="Test user endpoint",
            strategy_used="check auth first",
            success=True,
            duration_ms=500,
            timestamp=datetime.now(timezone.utc),
            embedding=[0.1] * 384,
        )

    def test_save_outcome(self, mock_storage, sample_outcome):
        """Test saving an outcome."""
        container = mock_storage._mock_containers["alma-outcomes"]

        mock_storage.save_outcome(sample_outcome)

        container.upsert_item.assert_called_once()
        doc = container.upsert_item.call_args[0][0]

        assert doc["id"] == "o-123"
        assert doc["agent"] == "victor"
        assert doc["success"] is True
        assert doc["duration_ms"] == 500

    def test_get_outcomes_by_agent(self, mock_storage):
        """Test getting outcomes filtered by agent."""
        container = mock_storage._mock_containers["alma-outcomes"]

        mock_items = [
            {
                "id": "o-1",
                "agent": "victor",
                "project_id": "proj-1",
                "task_type": "api_testing",
                "task_description": "test",
                "strategy_used": "strategy",
                "success": True,
                "duration_ms": 100,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "embedding": [0.1] * 384,
            }
        ]
        container.query_items.return_value = iter(mock_items)

        results = mock_storage.get_outcomes(project_id="proj-1", agent="victor")

        assert len(results) == 1
        assert results[0].agent == "victor"


class TestUserPreferenceOperations:
    """Tests for user preference CRUD operations."""

    @pytest.fixture
    def mock_storage(self):
        """Create a mocked storage instance."""
        with patch("alma.storage.azure_cosmos.CosmosClient") as mock_cosmos:
            mock_client = MagicMock()
            mock_cosmos.return_value = mock_client
            mock_database = MagicMock()
            mock_client.create_database_if_not_exists.return_value = mock_database

            mock_containers = {}
            for name in [
                "alma-heuristics",
                "alma-outcomes",
                "alma-preferences",
                "alma-knowledge",
                "alma-antipatterns",
            ]:
                mock_containers[name] = MagicMock()

            mock_database.create_container_if_not_exists.side_effect = (
                lambda **kwargs: mock_containers[kwargs["id"]]
            )

            storage = AzureCosmosStorage(
                endpoint="https://test.documents.azure.com:443/",
                key="test-key",
            )
            storage._mock_containers = mock_containers
            yield storage

    @pytest.fixture
    def sample_preference(self):
        """Create a sample user preference."""
        return UserPreference(
            id="p-123",
            user_id="user-1",
            category="code_style",
            preference="use_typescript",
            source="explicit_instruction",
            confidence=0.9,
            timestamp=datetime.now(timezone.utc),
        )

    def test_save_user_preference(self, mock_storage, sample_preference):
        """Test saving a user preference."""
        container = mock_storage._mock_containers["alma-preferences"]

        mock_storage.save_user_preference(sample_preference)

        container.upsert_item.assert_called_once()
        doc = container.upsert_item.call_args[0][0]

        assert doc["id"] == "p-123"
        assert doc["user_id"] == "user-1"
        assert doc["category"] == "code_style"
        assert doc["preference"] == "use_typescript"

    def test_get_user_preferences(self, mock_storage):
        """Test getting user preferences."""
        container = mock_storage._mock_containers["alma-preferences"]

        mock_items = [
            {
                "id": "p-1",
                "user_id": "user-1",
                "category": "code_style",
                "preference": "use_typescript",
                "strength": 0.9,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        ]
        container.query_items.return_value = iter(mock_items)

        results = mock_storage.get_user_preferences(user_id="user-1")

        assert len(results) == 1
        assert results[0].user_id == "user-1"
        assert results[0].preference == "use_typescript"


class TestDomainKnowledgeOperations:
    """Tests for domain knowledge CRUD operations."""

    @pytest.fixture
    def mock_storage(self):
        """Create a mocked storage instance."""
        with patch("alma.storage.azure_cosmos.CosmosClient") as mock_cosmos:
            mock_client = MagicMock()
            mock_cosmos.return_value = mock_client
            mock_database = MagicMock()
            mock_client.create_database_if_not_exists.return_value = mock_database

            mock_containers = {}
            for name in [
                "alma-heuristics",
                "alma-outcomes",
                "alma-preferences",
                "alma-knowledge",
                "alma-antipatterns",
            ]:
                mock_containers[name] = MagicMock()

            mock_database.create_container_if_not_exists.side_effect = (
                lambda **kwargs: mock_containers[kwargs["id"]]
            )

            storage = AzureCosmosStorage(
                endpoint="https://test.documents.azure.com:443/",
                key="test-key",
            )
            storage._mock_containers = mock_containers
            yield storage

    @pytest.fixture
    def sample_knowledge(self):
        """Create sample domain knowledge."""
        return DomainKnowledge(
            id="k-123",
            agent="helena",
            project_id="proj-1",
            domain="accessibility_testing",
            fact="ARIA labels improve screen reader compatibility",
            source="accessibility_audit",
            created_at=datetime.now(timezone.utc),
            embedding=[0.1] * 384,
        )

    def test_save_domain_knowledge(self, mock_storage, sample_knowledge):
        """Test saving domain knowledge."""
        container = mock_storage._mock_containers["alma-knowledge"]

        mock_storage.save_domain_knowledge(sample_knowledge)

        container.upsert_item.assert_called_once()
        doc = container.upsert_item.call_args[0][0]

        assert doc["id"] == "k-123"
        assert doc["domain"] == "accessibility_testing"
        assert "ARIA" in doc["fact"]


class TestAntiPatternOperations:
    """Tests for anti-pattern CRUD operations."""

    @pytest.fixture
    def mock_storage(self):
        """Create a mocked storage instance."""
        with patch("alma.storage.azure_cosmos.CosmosClient") as mock_cosmos:
            mock_client = MagicMock()
            mock_cosmos.return_value = mock_client
            mock_database = MagicMock()
            mock_client.create_database_if_not_exists.return_value = mock_database

            mock_containers = {}
            for name in [
                "alma-heuristics",
                "alma-outcomes",
                "alma-preferences",
                "alma-knowledge",
                "alma-antipatterns",
            ]:
                mock_containers[name] = MagicMock()

            mock_database.create_container_if_not_exists.side_effect = (
                lambda **kwargs: mock_containers[kwargs["id"]]
            )

            storage = AzureCosmosStorage(
                endpoint="https://test.documents.azure.com:443/",
                key="test-key",
            )
            storage._mock_containers = mock_containers
            yield storage

    @pytest.fixture
    def sample_antipattern(self):
        """Create a sample anti-pattern."""
        return AntiPattern(
            id="ap-123",
            agent="victor",
            project_id="proj-1",
            pattern="hardcoded credentials",
            why_bad="security vulnerability",
            better_alternative="use environment variables",
            occurrence_count=3,
            last_seen=datetime.now(timezone.utc),
            created_at=datetime.now(timezone.utc),
            embedding=[0.1] * 384,
        )

    def test_save_anti_pattern(self, mock_storage, sample_antipattern):
        """Test saving an anti-pattern."""
        container = mock_storage._mock_containers["alma-antipatterns"]

        mock_storage.save_anti_pattern(sample_antipattern)

        container.upsert_item.assert_called_once()
        doc = container.upsert_item.call_args[0][0]

        assert doc["id"] == "ap-123"
        assert doc["pattern"] == "hardcoded credentials"
        assert doc["why_bad"] == "security vulnerability"


class TestVectorSearch:
    """Tests for vector similarity search."""

    @pytest.fixture
    def mock_storage(self):
        """Create a mocked storage instance."""
        with patch("alma.storage.azure_cosmos.CosmosClient") as mock_cosmos:
            mock_client = MagicMock()
            mock_cosmos.return_value = mock_client
            mock_database = MagicMock()
            mock_client.create_database_if_not_exists.return_value = mock_database

            mock_containers = {}
            for name in [
                "alma-heuristics",
                "alma-outcomes",
                "alma-preferences",
                "alma-knowledge",
                "alma-antipatterns",
            ]:
                mock_containers[name] = MagicMock()

            mock_database.create_container_if_not_exists.side_effect = (
                lambda **kwargs: mock_containers[kwargs["id"]]
            )

            storage = AzureCosmosStorage(
                endpoint="https://test.documents.azure.com:443/",
                key="test-key",
            )
            storage._mock_containers = mock_containers
            yield storage

    def test_get_heuristics_with_embedding(self, mock_storage):
        """Test getting heuristics using vector similarity search."""
        container = mock_storage._mock_containers["alma-heuristics"]

        # Mock vector search results
        mock_items = [
            {
                "id": "h-1",
                "agent": "helena",
                "project_id": "proj-1",
                "condition": "form testing",
                "strategy": "validate first",
                "confidence": 0.9,
                "occurrence_count": 10,
                "success_count": 9,
                "last_validated": datetime.now(timezone.utc).isoformat(),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "embedding": [0.1] * 384,
                "similarity_score": 0.95,
            }
        ]
        container.query_items.return_value = iter(mock_items)

        query_embedding = [0.1] * 384
        results = mock_storage.get_heuristics(
            project_id="proj-1",
            embedding=query_embedding,
            top_k=5,
        )

        assert len(results) == 1
        # Check that vector search query was used
        query_call = container.query_items.call_args
        query_string = query_call.kwargs["query"]
        assert "VectorDistance" in query_string or "similarity" in query_string.lower()


class TestContainerNames:
    """Tests for container naming conventions."""

    def test_container_names_constant(self):
        """Test that container names are correctly defined."""
        assert AzureCosmosStorage.CONTAINER_NAMES["heuristics"] == "alma-heuristics"
        assert AzureCosmosStorage.CONTAINER_NAMES["outcomes"] == "alma-outcomes"
        assert AzureCosmosStorage.CONTAINER_NAMES["preferences"] == "alma-preferences"
        assert AzureCosmosStorage.CONTAINER_NAMES["knowledge"] == "alma-knowledge"
        assert AzureCosmosStorage.CONTAINER_NAMES["antipatterns"] == "alma-antipatterns"
