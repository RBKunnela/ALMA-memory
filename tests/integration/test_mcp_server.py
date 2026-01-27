"""
Integration tests for ALMA MCP Server.
"""

import json
from pathlib import Path

import pytest

from alma import ALMA, MemoryScope
from alma.integration.helena import HELENA_CATEGORIES
from alma.integration.victor import VICTOR_CATEGORIES
from alma.learning.protocols import LearningProtocol
from alma.mcp.resources import get_agents_resource, get_config_resource, list_resources
from alma.mcp.server import ALMAMCPServer
from alma.retrieval.engine import RetrievalEngine
from alma.storage.file_based import FileBasedStorage


@pytest.mark.asyncio
class TestALMAMCPServer:
    """Tests for ALMAMCPServer class."""

    @pytest.fixture
    def server_alma(self, temp_storage_dir: Path) -> ALMA:
        """Create ALMA instance for server testing."""
        storage = FileBasedStorage(storage_dir=temp_storage_dir)
        scopes = {
            "helena": MemoryScope(
                agent_name="helena",
                can_learn=HELENA_CATEGORIES,
                cannot_learn=["backend"],
                min_occurrences_for_heuristic=3,
            ),
            "victor": MemoryScope(
                agent_name="victor",
                can_learn=VICTOR_CATEGORIES,
                cannot_learn=["frontend"],
                min_occurrences_for_heuristic=3,
            ),
        }
        retrieval = RetrievalEngine(storage=storage, embedding_provider="mock")
        learning = LearningProtocol(storage=storage, scopes=scopes)

        return ALMA(
            storage=storage,
            retrieval_engine=retrieval,
            learning_protocol=learning,
            scopes=scopes,
            project_id="mcp-test",
        )

    @pytest.fixture
    def mcp_server(self, server_alma: ALMA) -> ALMAMCPServer:
        """Create MCP server instance."""
        return ALMAMCPServer(alma=server_alma)

    @pytest.mark.asyncio
    async def test_initialize_request(self, mcp_server: ALMAMCPServer):
        """Test MCP initialize request."""
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {},
        }

        response = await mcp_server.handle_request(request)

        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 1
        assert "result" in response
        assert response["result"]["protocolVersion"] == "2024-11-05"
        assert "serverInfo" in response["result"]

    @pytest.mark.asyncio
    async def test_tools_list_request(self, mcp_server: ALMAMCPServer):
        """Test MCP tools/list request."""
        request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {},
        }

        response = await mcp_server.handle_request(request)

        assert response["id"] == 2
        assert "result" in response
        tools = response["result"]["tools"]

        # Should have 8 tools (including alma_consolidate added in Phase 1)
        assert len(tools) == 8

        tool_names = [t["name"] for t in tools]
        assert "alma_retrieve" in tool_names
        assert "alma_learn" in tool_names
        assert "alma_add_preference" in tool_names
        assert "alma_add_knowledge" in tool_names
        assert "alma_forget" in tool_names
        assert "alma_stats" in tool_names
        assert "alma_health" in tool_names
        assert "alma_consolidate" in tool_names  # Added in Phase 1

    @pytest.mark.asyncio
    async def test_tool_call_retrieve(self, mcp_server: ALMAMCPServer):
        """Test MCP tools/call for alma_retrieve."""
        request = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "alma_retrieve",
                "arguments": {
                    "task": "Test login form",
                    "agent": "helena",
                    "top_k": 5,
                },
            },
        }

        response = await mcp_server.handle_request(request)

        assert response["id"] == 3
        assert "result" in response
        assert "content" in response["result"]

        # Parse the content
        content = response["result"]["content"][0]
        assert content["type"] == "text"
        result = json.loads(content["text"])
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_tool_call_learn(self, mcp_server: ALMAMCPServer):
        """Test MCP tools/call for alma_learn."""
        request = {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {
                "name": "alma_learn",
                "arguments": {
                    "agent": "helena",
                    "task": "Test form validation",
                    "outcome": "success",
                    "strategy_used": "validate inputs first",
                    "task_type": "form_testing",
                },
            },
        }

        response = await mcp_server.handle_request(request)

        assert response["id"] == 4
        content = response["result"]["content"][0]
        result = json.loads(content["text"])
        assert result["success"] is True
        assert result["learned"] is True

    @pytest.mark.asyncio
    async def test_tool_call_stats(self, mcp_server: ALMAMCPServer):
        """Test MCP tools/call for alma_stats."""
        request = {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": {
                "name": "alma_stats",
                "arguments": {
                    "agent": "helena",
                },
            },
        }

        response = await mcp_server.handle_request(request)

        assert response["id"] == 5
        content = response["result"]["content"][0]
        result = json.loads(content["text"])
        assert result["success"] is True
        assert "stats" in result

    @pytest.mark.asyncio
    async def test_tool_call_health(self, mcp_server: ALMAMCPServer):
        """Test MCP tools/call for alma_health."""
        request = {
            "jsonrpc": "2.0",
            "id": 6,
            "method": "tools/call",
            "params": {
                "name": "alma_health",
                "arguments": {},
            },
        }

        response = await mcp_server.handle_request(request)

        assert response["id"] == 6
        content = response["result"]["content"][0]
        result = json.loads(content["text"])
        assert result["success"] is True
        assert result["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_resources_list_request(self, mcp_server: ALMAMCPServer):
        """Test MCP resources/list request."""
        request = {
            "jsonrpc": "2.0",
            "id": 7,
            "method": "resources/list",
            "params": {},
        }

        response = await mcp_server.handle_request(request)

        assert response["id"] == 7
        resources = response["result"]["resources"]

        # Should have 2 resources
        assert len(resources) == 2

        uris = [r["uri"] for r in resources]
        assert "alma://config" in uris
        assert "alma://agents" in uris

    @pytest.mark.asyncio
    async def test_resource_read_config(self, mcp_server: ALMAMCPServer):
        """Test MCP resources/read for config."""
        request = {
            "jsonrpc": "2.0",
            "id": 8,
            "method": "resources/read",
            "params": {
                "uri": "alma://config",
            },
        }

        response = await mcp_server.handle_request(request)

        assert response["id"] == 8
        contents = response["result"]["contents"]
        assert len(contents) == 1
        assert contents[0]["uri"] == "alma://config"

        config = json.loads(contents[0]["text"])
        assert config["project_id"] == "mcp-test"
        assert "helena" in config["registered_agents"]

    @pytest.mark.asyncio
    async def test_resource_read_agents(self, mcp_server: ALMAMCPServer):
        """Test MCP resources/read for agents."""
        request = {
            "jsonrpc": "2.0",
            "id": 9,
            "method": "resources/read",
            "params": {
                "uri": "alma://agents",
            },
        }

        response = await mcp_server.handle_request(request)

        assert response["id"] == 9
        contents = response["result"]["contents"]
        content = json.loads(contents[0]["text"])

        assert content["agent_count"] == 2
        agent_names = [a["name"] for a in content["agents"]]
        assert "helena" in agent_names
        assert "victor" in agent_names

    @pytest.mark.asyncio
    async def test_unknown_method_error(self, mcp_server: ALMAMCPServer):
        """Test error response for unknown method."""
        request = {
            "jsonrpc": "2.0",
            "id": 10,
            "method": "unknown/method",
            "params": {},
        }

        response = await mcp_server.handle_request(request)

        assert response["id"] == 10
        assert "error" in response
        assert response["error"]["code"] == -32601
        assert "not found" in response["error"]["message"].lower()

    @pytest.mark.asyncio
    async def test_unknown_tool_error(self, mcp_server: ALMAMCPServer):
        """Test error response for unknown tool."""
        request = {
            "jsonrpc": "2.0",
            "id": 11,
            "method": "tools/call",
            "params": {
                "name": "unknown_tool",
                "arguments": {},
            },
        }

        response = await mcp_server.handle_request(request)

        assert response["id"] == 11
        assert "error" in response
        assert response["error"]["code"] == -32602

    @pytest.mark.asyncio
    async def test_ping_request(self, mcp_server: ALMAMCPServer):
        """Test MCP ping request."""
        request = {
            "jsonrpc": "2.0",
            "id": 12,
            "method": "ping",
            "params": {},
        }

        response = await mcp_server.handle_request(request)

        assert response["id"] == 12
        assert "result" in response


class TestMCPResources:
    """Tests for MCP resource functions."""

    @pytest.fixture
    def resource_alma(self, temp_storage_dir: Path) -> ALMA:
        """Create ALMA for resource testing."""
        storage = FileBasedStorage(storage_dir=temp_storage_dir)
        scopes = {
            "helena": MemoryScope(
                agent_name="helena",
                can_learn=["testing"],
                cannot_learn=["backend"],
                min_occurrences_for_heuristic=3,
            ),
        }
        retrieval = RetrievalEngine(storage=storage, embedding_provider="mock")
        learning = LearningProtocol(storage=storage, scopes=scopes)

        return ALMA(
            storage=storage,
            retrieval_engine=retrieval,
            learning_protocol=learning,
            scopes=scopes,
            project_id="resource-test",
        )

    def test_list_resources(self):
        """Test listing all resources."""
        resources = list_resources()

        assert len(resources) == 2
        assert resources[0]["uri"] == "alma://config"
        assert resources[1]["uri"] == "alma://agents"

    def test_get_config_resource(self, resource_alma: ALMA):
        """Test getting config resource."""
        resource = get_config_resource(resource_alma)

        assert resource["uri"] == "alma://config"
        assert resource["mimeType"] == "application/json"
        assert resource["content"]["project_id"] == "resource-test"
        assert "helena" in resource["content"]["registered_agents"]

    def test_get_agents_resource(self, resource_alma: ALMA):
        """Test getting agents resource."""
        resource = get_agents_resource(resource_alma)

        assert resource["uri"] == "alma://agents"
        assert resource["content"]["agent_count"] == 1

        helena = resource["content"]["agents"][0]
        assert helena["name"] == "helena"
        assert "testing" in helena["scope"]["can_learn"]
        assert "backend" in helena["scope"]["cannot_learn"]


@pytest.mark.asyncio
class TestMCPFullWorkflow:
    """Tests for complete MCP workflows."""

    @pytest.fixture
    def workflow_alma(self, temp_storage_dir: Path) -> ALMA:
        """Create ALMA for workflow testing."""
        storage = FileBasedStorage(storage_dir=temp_storage_dir)
        scopes = {
            "helena": MemoryScope(
                agent_name="helena",
                can_learn=["testing_strategies", "selector_patterns"],
                cannot_learn=["backend"],
                min_occurrences_for_heuristic=3,
            ),
        }
        retrieval = RetrievalEngine(storage=storage, embedding_provider="mock")
        learning = LearningProtocol(storage=storage, scopes=scopes)

        return ALMA(
            storage=storage,
            retrieval_engine=retrieval,
            learning_protocol=learning,
            scopes=scopes,
            project_id="workflow-test",
        )

    @pytest.fixture
    def workflow_server(self, workflow_alma: ALMA) -> ALMAMCPServer:
        """Create MCP server for workflow testing."""
        return ALMAMCPServer(alma=workflow_alma)

    @pytest.mark.asyncio
    async def test_learn_then_retrieve_workflow(
        self, workflow_server: ALMAMCPServer
    ):
        """Test complete learn -> retrieve workflow."""
        # Step 1: Learn something
        learn_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "alma_learn",
                "arguments": {
                    "agent": "helena",
                    "task": "Test login form",
                    "outcome": "success",
                    "strategy_used": "validate inputs first",
                    "task_type": "testing_strategies",
                },
            },
        }

        learn_response = await workflow_server.handle_request(learn_request)
        learn_result = json.loads(
            learn_response["result"]["content"][0]["text"]
        )
        assert learn_result["success"] is True

        # Step 2: Add knowledge
        knowledge_request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "alma_add_knowledge",
                "arguments": {
                    "agent": "helena",
                    "domain": "selector_patterns",
                    "fact": "Use data-testid for stable selectors",
                },
            },
        }

        knowledge_response = await workflow_server.handle_request(
            knowledge_request
        )
        knowledge_result = json.loads(
            knowledge_response["result"]["content"][0]["text"]
        )
        assert knowledge_result["success"] is True

        # Step 3: Retrieve memories
        retrieve_request = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "alma_retrieve",
                "arguments": {
                    "task": "Test registration form",
                    "agent": "helena",
                },
            },
        }

        retrieve_response = await workflow_server.handle_request(
            retrieve_request
        )
        retrieve_result = json.loads(
            retrieve_response["result"]["content"][0]["text"]
        )
        assert retrieve_result["success"] is True

        # Should have domain knowledge
        assert len(retrieve_result["memories"]["domain_knowledge"]) >= 1

        # Step 4: Check stats
        stats_request = {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {
                "name": "alma_stats",
                "arguments": {"agent": "helena"},
            },
        }

        stats_response = await workflow_server.handle_request(stats_request)
        stats_result = json.loads(
            stats_response["result"]["content"][0]["text"]
        )
        assert stats_result["success"] is True
        assert stats_result["stats"]["outcomes_count"] >= 1
        assert stats_result["stats"]["domain_knowledge_count"] >= 1
