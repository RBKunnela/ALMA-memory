# ALMA Openness Squad

> Make ALMA the universal personal brain backend -- zero switching cost, any AI tool, any platform.

## Purpose

The Openness Squad builds the protocol and integration layer that turns ALMA-memory into the definitive "Open Brain" backend. When a user captures a thought in Claude Code, that same thought is instantly available in ChatGPT, Cursor, Obsidian, or any MCP-compatible client. No vendor lock-in. No data silos. One persistent brain across every AI tool.

## Goals

1. **Personal Brain Domain Schema** -- A new `personal_brain` domain for `alma/domains/` that models how humans actually think: thoughts, insights, decisions, people, projects, lessons, questions.
2. **Multi-Client MCP Protocol** -- Patterns for concurrent access when Claude Code, ChatGPT, and Cursor all connect to the same ALMA instance simultaneously.
3. **45-Minute Quickstart** -- `alma init --open-brain` CLI command that generates config, runs migrations, configures MCP, and verifies the setup in under 45 minutes.
4. **Memory Migration Tools** -- Importers for Claude memory export, ChatGPT memory export, Obsidian vaults, and Notion exports.

## Agents

| Agent | Persona | Role |
|-------|---------|------|
| `@openness-chief` | Nova | Squad leader. Coordinates design, implementation, and integration across the team. |
| `@protocol-architect` | Sync | Designs the cross-tool memory protocol. Multi-client MCP, conflict resolution, deduplication. |
| `@quickstart-dev` | Bolt | Builds the "45-minute setup" experience. CLI commands, config generation, getting-started flow. |
| `@domain-designer` | Schema | Designs and implements the Personal Brain domain schema for `alma/domains/`. |

## Structure

```
alma-openness/
  README.md                              # This file
  agents/
    openness-chief.md                    # Squad leader (Nova)
    protocol-architect.md                # Cross-tool protocol design (Sync)
    quickstart-dev.md                    # Quickstart CLI and setup (Bolt)
    domain-designer.md                   # Personal Brain domain schema (Schema)
  tasks/
    design-personal-brain-schema.md      # Implement personal_brain domain
    build-quickstart-cli.md              # Build alma init --open-brain
    implement-multi-client-protocol.md   # Multi-client MCP access
    build-memory-migration.md            # Import from Claude/ChatGPT/Obsidian/Notion
  data/
    openness-kb.md                       # Knowledge base and design rationale
  templates/
    open-brain-config.md                 # Template for .alma/config.yaml
  workflows/
    open-brain-setup.yaml                # End-to-end setup workflow
```

## Workflows

### Open Brain Setup (Primary)

`init project -> generate config -> run migrations -> configure MCP -> verify setup -> capture first thought -> verify retrieval`

This is the golden path. A user goes from zero to a working personal brain in 45 minutes.

### Memory Migration

`select source -> export data -> run importer -> deduplicate -> verify import -> report`

Import existing memories from Claude, ChatGPT, Obsidian, or Notion into ALMA.

## Technical Context

ALMA-memory is a Python library (`pip install alma-memory`) with:

- **7 storage backends**: SQLite+FAISS, PostgreSQL+pgvector, Azure Cosmos, Qdrant, Chroma, Pinecone, File
- **MCP server**: stdio + HTTP modes via `alma.mcp.server.ALMAMCPServer`
- **6 pre-built domain schemas**: coding, research, sales, general, customer_support, content_creation
- **YAML config**: `alma.config.loader.ConfigLoader` with `${ENV_VAR}` and `${KEYVAULT:secret}` expansion
- **Domain factory**: `alma.domains.factory.DomainMemoryFactory` with `get_schema()`, `create_schema()`, `create_alma()`

The squad's output is Python code that extends these existing modules.

## Key Constraints

- ALMA is a library -- users bring their own database infrastructure
- All new code follows `alma/` package conventions (ruff, mypy, Google docstrings)
- New domain schemas follow the `DomainSchema` pattern from `alma/domains/types.py`
- Tests use `alma.testing.MockStorage` and `alma.testing.MockEmbedder`
- Python 3.10+ required

## Getting Started

1. Read `data/openness-kb.md` for design rationale and protocol specifications
2. Start with `@openness-chief *status` to see current progress
3. Use `@domain-designer *design-schema` to begin Personal Brain domain design
4. Use `@quickstart-dev *build-quickstart` to build the CLI flow

---

*ALMA Openness Squad v1.0.0*
*Built for ALMA-memory (pip install alma-memory)*
