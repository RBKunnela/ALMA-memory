"""
ALMA MCP Server CLI Entry Point.

Usage:
    # stdio mode (for Claude Code)
    python -m alma.mcp --config .alma/config.yaml

    # HTTP mode (for remote access)
    python -m alma.mcp --http --port 8765

    # With verbose logging
    python -m alma.mcp --config .alma/config.yaml --verbose
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from alma import ALMA
from alma.mcp.server import ALMAMCPServer


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stderr,  # Log to stderr to avoid interfering with stdio
    )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="ALMA MCP Server - Memory for AI Agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Start in stdio mode for Claude Code
    python -m alma.mcp --config .alma/config.yaml

    # Start in HTTP mode on port 8765
    python -m alma.mcp --http --port 8765

    # Use with verbose logging
    python -m alma.mcp --config .alma/config.yaml --verbose
        """,
    )

    parser.add_argument(
        "--config",
        type=str,
        default=".alma/config.yaml",
        help="Path to ALMA config file (default: .alma/config.yaml)",
    )

    parser.add_argument(
        "--http",
        action="store_true",
        help="Run in HTTP mode instead of stdio",
    )

    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="HTTP server host (default: 0.0.0.0)",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="HTTP server port (default: 8765)",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    # Load ALMA from config
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        logger.info("Creating default config...")

        # Create minimal default config
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w") as f:
            f.write("""# ALMA Configuration
alma:
  project_id: "default"
  storage: "file"
  storage_dir: ".alma"
  embedding_provider: "local"

  agents:
    helena:
      can_learn:
        - testing_strategies
        - selector_patterns
        - form_testing
        - accessibility_testing
      cannot_learn:
        - backend_logic
        - database_queries
      min_occurrences_for_heuristic: 3

    victor:
      can_learn:
        - api_design_patterns
        - authentication_patterns
        - error_handling
        - database_query_patterns
      cannot_learn:
        - frontend_styling
        - ui_testing
      min_occurrences_for_heuristic: 3
""")
        logger.info(f"Created default config at {config_path}")

    try:
        alma = ALMA.from_config(str(config_path))
        logger.info(f"Loaded ALMA from {config_path}")
        logger.info(f"Project: {alma.project_id}")
        logger.info(f"Agents: {list(alma.scopes.keys())}")

    except Exception as e:
        logger.exception(f"Failed to load ALMA: {e}")
        sys.exit(1)

    # Create and run server
    server = ALMAMCPServer(alma=alma)

    if args.http:
        logger.info(f"Starting HTTP server on {args.host}:{args.port}")
        asyncio.run(server.run_http(host=args.host, port=args.port))
    else:
        logger.info("Starting stdio server for Claude Code integration")
        asyncio.run(server.run_stdio())


if __name__ == "__main__":
    main()
