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
        server_version: str = "0.2.0",
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
        """Handle tools/call request."""
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})

        # Map tool names to functions
        tool_handlers = {
            "alma_retrieve": lambda: alma_retrieve(
                self.alma,
                task=arguments.get("task", ""),
                agent=arguments.get("agent", ""),
                user_id=arguments.get("user_id"),
                top_k=arguments.get("top_k", 5),
            ),
            "alma_learn": lambda: alma_learn(
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
            "alma_add_preference": lambda: alma_add_preference(
                self.alma,
                user_id=arguments.get("user_id", ""),
                category=arguments.get("category", ""),
                preference=arguments.get("preference", ""),
                source=arguments.get("source", "explicit_instruction"),
            ),
            "alma_add_knowledge": lambda: alma_add_knowledge(
                self.alma,
                agent=arguments.get("agent", ""),
                domain=arguments.get("domain", ""),
                fact=arguments.get("fact", ""),
                source=arguments.get("source", "user_stated"),
            ),
            "alma_forget": lambda: alma_forget(
                self.alma,
                agent=arguments.get("agent"),
                older_than_days=arguments.get("older_than_days", 90),
                below_confidence=arguments.get("below_confidence", 0.3),
            ),
            "alma_stats": lambda: alma_stats(
                self.alma,
                agent=arguments.get("agent"),
            ),
            "alma_health": lambda: alma_health(self.alma),
            "alma_consolidate": lambda: alma_consolidate(
                self.alma,
                agent=arguments.get("agent", ""),
                memory_type=arguments.get("memory_type", "heuristics"),
                similarity_threshold=arguments.get("similarity_threshold", 0.85),
                dry_run=arguments.get("dry_run", True),
            ),
        }

        if tool_name not in tool_handlers:
            return self._error_response(
                request_id,
                -32602,
                f"Unknown tool: {tool_name}",
            )

        result = tool_handlers[tool_name]()

        # Handle async functions (like alma_consolidate)
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
