"""
ALMA MCP Server Implementation.

Provides the main server class that handles MCP protocol communication.
Supports both stdio (for Claude Code) and HTTP modes.
"""

import asyncio
import json
import logging
import sys
from typing import Any, Dict, List, Optional

from alma import ALMA
from alma.mcp.resources import (
    get_agents_resource,
    get_config_resource,
    list_resources,
)
from alma.mcp.tools import (
    alma_add_knowledge,
    alma_add_preference,
    alma_consolidate,
    alma_forget,
    alma_health,
    alma_learn,
    alma_retrieve,
    alma_stats,
    # Workflow tools (v0.6.0)
    alma_checkpoint,
    alma_resume,
    alma_merge_states,
    alma_workflow_learn,
    alma_link_artifact,
    alma_get_artifacts,
    alma_cleanup_checkpoints,
    alma_retrieve_scoped,
    # Async variants
    async_alma_add_knowledge,
    async_alma_add_preference,
    async_alma_forget,
    async_alma_health,
    async_alma_learn,
    async_alma_retrieve,
    async_alma_stats,
    # Async workflow variants
    async_alma_checkpoint,
    async_alma_resume,
    async_alma_workflow_learn,
    async_alma_link_artifact,
    async_alma_retrieve_scoped,
)

logger = logging.getLogger(__name__)


class ALMAMCPServer:
    """
    MCP Server for ALMA.

    Exposes ALMA functionality via the Model Context Protocol,
    allowing any MCP-compatible client (like Claude Code) to
    interact with the memory system.
    """

    def __init__(
        self,
        alma: ALMA,
        server_name: str = "alma-memory",
        server_version: str = "0.6.0",
    ):
        """
        Initialize the MCP server.

        Args:
            alma: Configured ALMA instance
            server_name: Server identifier
            server_version: Server version
        """
        self.alma = alma
        self.server_name = server_name
        self.server_version = server_version

        # Register tools
        self.tools = self._register_tools()

        # Register resources
        self.resources = list_resources()

    def _register_tools(self) -> List[Dict[str, Any]]:
        """Register available MCP tools."""
        return [
            {
                "name": "alma_retrieve",
                "description": "Retrieve relevant memories for a task. Returns heuristics, domain knowledge, anti-patterns, and user preferences.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "task": {
                            "type": "string",
                            "description": "Description of the task to perform",
                        },
                        "agent": {
                            "type": "string",
                            "description": "Name of the agent requesting memories (e.g., 'helena', 'victor')",
                        },
                        "user_id": {
                            "type": "string",
                            "description": "Optional user ID for preference retrieval",
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Maximum items per memory type (default: 5)",
                            "default": 5,
                        },
                    },
                    "required": ["task", "agent"],
                },
            },
            {
                "name": "alma_learn",
                "description": "Record a task outcome for learning. Use after completing a task to help improve future performance.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "agent": {
                            "type": "string",
                            "description": "Name of the agent that executed the task",
                        },
                        "task": {
                            "type": "string",
                            "description": "Description of the task",
                        },
                        "outcome": {
                            "type": "string",
                            "enum": ["success", "failure"],
                            "description": "Whether the task succeeded or failed",
                        },
                        "strategy_used": {
                            "type": "string",
                            "description": "What approach was taken",
                        },
                        "task_type": {
                            "type": "string",
                            "description": "Category of task (for grouping)",
                        },
                        "duration_ms": {
                            "type": "integer",
                            "description": "How long the task took in milliseconds",
                        },
                        "error_message": {
                            "type": "string",
                            "description": "Error details if failed",
                        },
                        "feedback": {
                            "type": "string",
                            "description": "User feedback if provided",
                        },
                    },
                    "required": ["agent", "task", "outcome", "strategy_used"],
                },
            },
            {
                "name": "alma_add_preference",
                "description": "Add a user preference to memory. Preferences persist across sessions.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "user_id": {
                            "type": "string",
                            "description": "User identifier",
                        },
                        "category": {
                            "type": "string",
                            "description": "Category (communication, code_style, workflow)",
                        },
                        "preference": {
                            "type": "string",
                            "description": "The preference text",
                        },
                        "source": {
                            "type": "string",
                            "description": "How this was learned (default: explicit_instruction)",
                            "default": "explicit_instruction",
                        },
                    },
                    "required": ["user_id", "category", "preference"],
                },
            },
            {
                "name": "alma_add_knowledge",
                "description": "Add domain knowledge within agent's scope. Knowledge is facts, not strategies.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "agent": {
                            "type": "string",
                            "description": "Agent this knowledge belongs to",
                        },
                        "domain": {
                            "type": "string",
                            "description": "Knowledge domain",
                        },
                        "fact": {
                            "type": "string",
                            "description": "The fact to remember",
                        },
                        "source": {
                            "type": "string",
                            "description": "How this was learned (default: user_stated)",
                            "default": "user_stated",
                        },
                    },
                    "required": ["agent", "domain", "fact"],
                },
            },
            {
                "name": "alma_forget",
                "description": "Prune stale or low-confidence memories to keep the system clean.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "agent": {
                            "type": "string",
                            "description": "Specific agent to prune, or omit for all",
                        },
                        "older_than_days": {
                            "type": "integer",
                            "description": "Remove outcomes older than this (default: 90)",
                            "default": 90,
                        },
                        "below_confidence": {
                            "type": "number",
                            "description": "Remove heuristics below this confidence (default: 0.3)",
                            "default": 0.3,
                        },
                    },
                },
            },
            {
                "name": "alma_stats",
                "description": "Get memory statistics for monitoring and debugging.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "agent": {
                            "type": "string",
                            "description": "Specific agent or omit for all",
                        },
                    },
                },
            },
            {
                "name": "alma_health",
                "description": "Health check for the ALMA server.",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                },
            },
            {
                "name": "alma_consolidate",
                "description": "Consolidate similar memories to reduce redundancy. Merges near-duplicate memories based on semantic similarity. Use dry_run=true first to preview what would be merged.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "agent": {
                            "type": "string",
                            "description": "Agent whose memories to consolidate",
                        },
                        "memory_type": {
                            "type": "string",
                            "enum": [
                                "heuristics",
                                "outcomes",
                                "domain_knowledge",
                                "anti_patterns",
                            ],
                            "description": "Type of memory to consolidate (default: heuristics)",
                            "default": "heuristics",
                        },
                        "similarity_threshold": {
                            "type": "number",
                            "description": "Minimum cosine similarity to group memories (0.0-1.0, default: 0.85). Higher values are more conservative.",
                            "default": 0.85,
                        },
                        "dry_run": {
                            "type": "boolean",
                            "description": "If true, preview what would be merged without modifying storage (default: true)",
                            "default": True,
                        },
                    },
                    "required": ["agent"],
                },
            },
            # ==================== WORKFLOW TOOLS (v0.6.0) ====================
            {
                "name": "alma_checkpoint",
                "description": "Create a checkpoint for crash recovery. Persists workflow state at key execution points.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "run_id": {
                            "type": "string",
                            "description": "The workflow run identifier",
                        },
                        "node_id": {
                            "type": "string",
                            "description": "The node creating this checkpoint",
                        },
                        "state": {
                            "type": "object",
                            "description": "The state to persist",
                        },
                        "branch_id": {
                            "type": "string",
                            "description": "Optional branch identifier for parallel execution",
                        },
                        "parent_checkpoint_id": {
                            "type": "string",
                            "description": "Previous checkpoint in the chain",
                        },
                        "metadata": {
                            "type": "object",
                            "description": "Additional checkpoint metadata",
                        },
                        "skip_if_unchanged": {
                            "type": "boolean",
                            "description": "Skip if state hasn't changed (default: true)",
                            "default": True,
                        },
                    },
                    "required": ["run_id", "node_id", "state"],
                },
            },
            {
                "name": "alma_resume",
                "description": "Get the checkpoint to resume from after a crash.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "run_id": {
                            "type": "string",
                            "description": "The workflow run identifier",
                        },
                        "branch_id": {
                            "type": "string",
                            "description": "Optional branch to filter by",
                        },
                    },
                    "required": ["run_id"],
                },
            },
            {
                "name": "alma_merge_states",
                "description": "Merge multiple branch states after parallel execution using configurable reducers.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "states": {
                            "type": "array",
                            "items": {"type": "object"},
                            "description": "List of state dicts from parallel branches",
                        },
                        "reducer_config": {
                            "type": "object",
                            "description": "Mapping of key -> reducer (append, merge_dict, last_value, first_value, sum, max, min, union)",
                        },
                    },
                    "required": ["states"],
                },
            },
            {
                "name": "alma_workflow_learn",
                "description": "Record learnings from a completed workflow execution.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "agent": {
                            "type": "string",
                            "description": "The agent that executed the workflow",
                        },
                        "workflow_id": {
                            "type": "string",
                            "description": "The workflow definition identifier",
                        },
                        "run_id": {
                            "type": "string",
                            "description": "The specific run identifier",
                        },
                        "result": {
                            "type": "string",
                            "enum": ["success", "failure", "partial", "cancelled", "timeout"],
                            "description": "Result status",
                        },
                        "summary": {
                            "type": "string",
                            "description": "Human-readable summary of what happened",
                        },
                        "strategies_used": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of strategies attempted",
                        },
                        "successful_patterns": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Patterns that worked well",
                        },
                        "failed_patterns": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Patterns that didn't work",
                        },
                        "duration_seconds": {
                            "type": "number",
                            "description": "How long the workflow took",
                        },
                        "node_count": {
                            "type": "integer",
                            "description": "Number of nodes executed",
                        },
                        "error_message": {
                            "type": "string",
                            "description": "Error details if failed",
                        },
                        "tenant_id": {
                            "type": "string",
                            "description": "Multi-tenant isolation identifier",
                        },
                        "metadata": {
                            "type": "object",
                            "description": "Additional outcome metadata",
                        },
                    },
                    "required": ["agent", "workflow_id", "run_id", "result", "summary"],
                },
            },
            {
                "name": "alma_link_artifact",
                "description": "Link an external artifact (file, screenshot, log) to a memory.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "memory_id": {
                            "type": "string",
                            "description": "The memory to link the artifact to",
                        },
                        "artifact_type": {
                            "type": "string",
                            "description": "Type (screenshot, log, report, file, document, image, etc.)",
                        },
                        "storage_url": {
                            "type": "string",
                            "description": "URL or path to the artifact in storage",
                        },
                        "filename": {
                            "type": "string",
                            "description": "Original filename",
                        },
                        "mime_type": {
                            "type": "string",
                            "description": "MIME type",
                        },
                        "size_bytes": {
                            "type": "integer",
                            "description": "Size in bytes",
                        },
                        "checksum": {
                            "type": "string",
                            "description": "SHA256 checksum for integrity",
                        },
                        "metadata": {
                            "type": "object",
                            "description": "Additional artifact metadata",
                        },
                    },
                    "required": ["memory_id", "artifact_type", "storage_url"],
                },
            },
            {
                "name": "alma_get_artifacts",
                "description": "Get all artifacts linked to a memory.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "memory_id": {
                            "type": "string",
                            "description": "The memory to get artifacts for",
                        },
                    },
                    "required": ["memory_id"],
                },
            },
            {
                "name": "alma_cleanup_checkpoints",
                "description": "Clean up old checkpoints for a completed workflow run.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "run_id": {
                            "type": "string",
                            "description": "The workflow run identifier",
                        },
                        "keep_latest": {
                            "type": "integer",
                            "description": "Number of latest checkpoints to keep (default: 1)",
                            "default": 1,
                        },
                    },
                    "required": ["run_id"],
                },
            },
            {
                "name": "alma_retrieve_scoped",
                "description": "Retrieve memories with workflow scope filtering. Supports hierarchical scoping: node -> run -> workflow -> agent -> tenant -> global.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "task": {
                            "type": "string",
                            "description": "Description of the task to perform",
                        },
                        "agent": {
                            "type": "string",
                            "description": "Name of the agent requesting memories",
                        },
                        "scope": {
                            "type": "string",
                            "enum": ["node", "run", "workflow", "agent", "tenant", "global"],
                            "description": "Scope level for filtering (default: agent)",
                            "default": "agent",
                        },
                        "tenant_id": {
                            "type": "string",
                            "description": "Tenant identifier for multi-tenant",
                        },
                        "workflow_id": {
                            "type": "string",
                            "description": "Workflow definition identifier",
                        },
                        "run_id": {
                            "type": "string",
                            "description": "Specific run identifier",
                        },
                        "node_id": {
                            "type": "string",
                            "description": "Current node identifier",
                        },
                        "user_id": {
                            "type": "string",
                            "description": "Optional user ID for preferences",
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Maximum items per memory type (default: 5)",
                            "default": 5,
                        },
                    },
                    "required": ["task", "agent"],
                },
            },
        ]

    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle an incoming MCP request.

        Args:
            request: The MCP request

        Returns:
            MCP response
        """
        method = request.get("method", "")
        params = request.get("params", {})
        request_id = request.get("id")

        try:
            if method == "initialize":
                return self._handle_initialize(request_id, params)
            elif method == "tools/list":
                return self._handle_tools_list(request_id)
            elif method == "tools/call":
                return await self._handle_tool_call(request_id, params)
            elif method == "resources/list":
                return self._handle_resources_list(request_id)
            elif method == "resources/read":
                return self._handle_resource_read(request_id, params)
            elif method == "ping":
                return self._success_response(request_id, {})
            else:
                return self._error_response(
                    request_id,
                    -32601,
                    f"Method not found: {method}",
                )

        except Exception as e:
            logger.exception(f"Error handling request: {e}")
            return self._error_response(request_id, -32603, str(e))

    def _handle_initialize(
        self,
        request_id: Optional[int],
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Handle initialize request."""
        return self._success_response(
            request_id,
            {
                "protocolVersion": "2024-11-05",
                "serverInfo": {
                    "name": self.server_name,
                    "version": self.server_version,
                },
                "capabilities": {
                    "tools": {},
                    "resources": {},
                },
            },
        )

    def _handle_tools_list(self, request_id: Optional[int]) -> Dict[str, Any]:
        """Handle tools/list request."""
        return self._success_response(request_id, {"tools": self.tools})

    async def _handle_tool_call(
        self,
        request_id: Optional[int],
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Handle tools/call request.

        Uses async tool variants for better concurrency in the async server.
        """
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})

        # Map tool names to async functions for non-blocking execution
        # All tools now use async variants for consistency in async server
        async_tool_handlers = {
            "alma_retrieve": lambda: async_alma_retrieve(
                self.alma,
                task=arguments.get("task", ""),
                agent=arguments.get("agent", ""),
                user_id=arguments.get("user_id"),
                top_k=arguments.get("top_k", 5),
            ),
            "alma_learn": lambda: async_alma_learn(
                self.alma,
                agent=arguments.get("agent", ""),
                task=arguments.get("task", ""),
                outcome=arguments.get("outcome", ""),
                strategy_used=arguments.get("strategy_used", ""),
                task_type=arguments.get("task_type"),
                duration_ms=arguments.get("duration_ms"),
                error_message=arguments.get("error_message"),
                feedback=arguments.get("feedback"),
            ),
            "alma_add_preference": lambda: async_alma_add_preference(
                self.alma,
                user_id=arguments.get("user_id", ""),
                category=arguments.get("category", ""),
                preference=arguments.get("preference", ""),
                source=arguments.get("source", "explicit_instruction"),
            ),
            "alma_add_knowledge": lambda: async_alma_add_knowledge(
                self.alma,
                agent=arguments.get("agent", ""),
                domain=arguments.get("domain", ""),
                fact=arguments.get("fact", ""),
                source=arguments.get("source", "user_stated"),
            ),
            "alma_forget": lambda: async_alma_forget(
                self.alma,
                agent=arguments.get("agent"),
                older_than_days=arguments.get("older_than_days", 90),
                below_confidence=arguments.get("below_confidence", 0.3),
            ),
            "alma_stats": lambda: async_alma_stats(
                self.alma,
                agent=arguments.get("agent"),
            ),
            "alma_health": lambda: async_alma_health(self.alma),
            "alma_consolidate": lambda: alma_consolidate(
                self.alma,
                agent=arguments.get("agent", ""),
                memory_type=arguments.get("memory_type", "heuristics"),
                similarity_threshold=arguments.get("similarity_threshold", 0.85),
                dry_run=arguments.get("dry_run", True),
            ),
            # Workflow tools (v0.6.0)
            "alma_checkpoint": lambda: async_alma_checkpoint(
                self.alma,
                run_id=arguments.get("run_id", ""),
                node_id=arguments.get("node_id", ""),
                state=arguments.get("state", {}),
                branch_id=arguments.get("branch_id"),
                parent_checkpoint_id=arguments.get("parent_checkpoint_id"),
                metadata=arguments.get("metadata"),
                skip_if_unchanged=arguments.get("skip_if_unchanged", True),
            ),
            "alma_resume": lambda: async_alma_resume(
                self.alma,
                run_id=arguments.get("run_id", ""),
                branch_id=arguments.get("branch_id"),
            ),
            "alma_merge_states": lambda: alma_merge_states(
                self.alma,
                states=arguments.get("states", []),
                reducer_config=arguments.get("reducer_config"),
            ),
            "alma_workflow_learn": lambda: async_alma_workflow_learn(
                self.alma,
                agent=arguments.get("agent", ""),
                workflow_id=arguments.get("workflow_id", ""),
                run_id=arguments.get("run_id", ""),
                result=arguments.get("result", ""),
                summary=arguments.get("summary", ""),
                strategies_used=arguments.get("strategies_used"),
                successful_patterns=arguments.get("successful_patterns"),
                failed_patterns=arguments.get("failed_patterns"),
                duration_seconds=arguments.get("duration_seconds"),
                node_count=arguments.get("node_count"),
                error_message=arguments.get("error_message"),
                tenant_id=arguments.get("tenant_id"),
                metadata=arguments.get("metadata"),
            ),
            "alma_link_artifact": lambda: async_alma_link_artifact(
                self.alma,
                memory_id=arguments.get("memory_id", ""),
                artifact_type=arguments.get("artifact_type", ""),
                storage_url=arguments.get("storage_url", ""),
                filename=arguments.get("filename"),
                mime_type=arguments.get("mime_type"),
                size_bytes=arguments.get("size_bytes"),
                checksum=arguments.get("checksum"),
                metadata=arguments.get("metadata"),
            ),
            "alma_get_artifacts": lambda: alma_get_artifacts(
                self.alma,
                memory_id=arguments.get("memory_id", ""),
            ),
            "alma_cleanup_checkpoints": lambda: alma_cleanup_checkpoints(
                self.alma,
                run_id=arguments.get("run_id", ""),
                keep_latest=arguments.get("keep_latest", 1),
            ),
            "alma_retrieve_scoped": lambda: async_alma_retrieve_scoped(
                self.alma,
                task=arguments.get("task", ""),
                agent=arguments.get("agent", ""),
                scope=arguments.get("scope", "agent"),
                tenant_id=arguments.get("tenant_id"),
                workflow_id=arguments.get("workflow_id"),
                run_id=arguments.get("run_id"),
                node_id=arguments.get("node_id"),
                user_id=arguments.get("user_id"),
                top_k=arguments.get("top_k", 5),
            ),
        }

        if tool_name not in async_tool_handlers:
            return self._error_response(
                request_id,
                -32602,
                f"Unknown tool: {tool_name}",
            )

        result = async_tool_handlers[tool_name]()

        # All handlers now return coroutines
        if asyncio.iscoroutine(result):
            result = await result

        return self._success_response(
            request_id,
            {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(result, indent=2),
                    }
                ],
            },
        )

    def _handle_resources_list(
        self,
        request_id: Optional[int],
    ) -> Dict[str, Any]:
        """Handle resources/list request."""
        return self._success_response(request_id, {"resources": self.resources})

    def _handle_resource_read(
        self,
        request_id: Optional[int],
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Handle resources/read request."""
        uri = params.get("uri", "")

        if uri == "alma://config":
            resource = get_config_resource(self.alma)
        elif uri == "alma://agents":
            resource = get_agents_resource(self.alma)
        else:
            return self._error_response(
                request_id,
                -32602,
                f"Unknown resource: {uri}",
            )

        return self._success_response(
            request_id,
            {
                "contents": [
                    {
                        "uri": resource["uri"],
                        "mimeType": resource["mimeType"],
                        "text": json.dumps(resource["content"], indent=2),
                    }
                ],
            },
        )

    def _success_response(
        self,
        request_id: Optional[int],
        result: Any,
    ) -> Dict[str, Any]:
        """Create a success response."""
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": result,
        }

    def _error_response(
        self,
        request_id: Optional[int],
        code: int,
        message: str,
    ) -> Dict[str, Any]:
        """Create an error response."""
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {
                "code": code,
                "message": message,
            },
        }

    async def run_stdio(self):
        """Run the server in stdio mode for Claude Code integration."""
        logger.info("Starting ALMA MCP Server (stdio mode)")

        reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(reader)
        await asyncio.get_event_loop().connect_read_pipe(lambda: protocol, sys.stdin)

        (
            writer_transport,
            writer_protocol,
        ) = await asyncio.get_event_loop().connect_write_pipe(
            asyncio.streams.FlowControlMixin, sys.stdout
        )
        writer = asyncio.StreamWriter(
            writer_transport, writer_protocol, None, asyncio.get_event_loop()
        )

        while True:
            try:
                # Read Content-Length header
                header_line = await reader.readline()
                if not header_line:
                    break

                header = header_line.decode().strip()
                if not header.startswith("Content-Length:"):
                    continue

                content_length = int(header.split(":")[1].strip())

                # Read empty line
                await reader.readline()

                # Read content
                content = await reader.read(content_length)
                request = json.loads(content.decode())

                # Handle request
                response = await self.handle_request(request)

                # Send response
                response_str = json.dumps(response)
                response_bytes = response_str.encode()
                header_bytes = f"Content-Length: {len(response_bytes)}\r\n\r\n".encode()

                writer.write(header_bytes + response_bytes)
                await writer.drain()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Error in stdio loop: {e}")

    async def run_http(self, host: str = "0.0.0.0", port: int = 8765):
        """
        Run the server in HTTP mode for remote access.

        Note: Requires aiohttp (optional dependency).
        """
        try:
            from aiohttp import web
        except ImportError:
            logger.error(
                "aiohttp required for HTTP mode. Install with: pip install aiohttp"
            )
            return

        async def handle_post(request: web.Request) -> web.Response:
            """Handle HTTP POST requests."""
            try:
                data = await request.json()
                response = await self.handle_request(data)
                return web.json_response(response)
            except Exception as e:
                return web.json_response(
                    {"error": str(e)},
                    status=500,
                )

        async def handle_health(request: web.Request) -> web.Response:
            """Handle health check endpoint."""
            result = alma_health(self.alma)
            return web.json_response(result)

        app = web.Application()
        app.router.add_post("/", handle_post)
        app.router.add_get("/health", handle_health)

        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, host, port)
        await site.start()

        logger.info(f"ALMA MCP Server running on http://{host}:{port}")

        # Keep running
        while True:
            await asyncio.sleep(3600)
