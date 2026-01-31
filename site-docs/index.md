# ALMA - Agent Learning Memory Architecture

<div style="text-align: center; margin: 2rem 0;">
  <h2 style="font-size: 1.5rem; color: #666;">Persistent memory for AI agents that learn and improve over time</h2>
  <p><strong>No fine-tuning. No model changes. Just smarter prompts.</strong></p>
</div>

<div class="grid cards" markdown>

-   :material-brain:{ .lg .middle } __Learn from Outcomes__

    ---

    Agents remember what worked and what didn't. Success becomes heuristics, failures become anti-patterns.

-   :material-share-variant:{ .lg .middle } __Multi-Agent Sharing__

    ---

    Hierarchical knowledge sharing with `inherit_from` and `share_with` scopes.

-   :material-shield-check:{ .lg .middle } __Scoped Learning__

    ---

    Define exactly what each agent can and cannot learn. Prevent domain confusion.

-   :material-webhook:{ .lg .middle } __Event System__

    ---

    React to memory changes with webhooks and callbacks in real-time.

</div>

## Why ALMA?

ALMA isn't just another memory framework. Here's what sets it apart:

| Feature | ALMA | Mem0 | LangChain Memory |
|---------|------|------|------------------|
| **Memory Scoping** | `can_learn` / `cannot_learn` | Basic isolation | None |
| **Anti-Pattern Learning** | Yes | No | No |
| **Multi-Agent Sharing** | Yes | No | No |
| **Memory Consolidation** | LLM-powered | Basic | None |
| **Event System** | Webhooks + callbacks | No | No |
| **MCP Integration** | Native | No | No |
| **TypeScript SDK** | Full-featured | No | No |
| **Vector DB Backends** | 6 options | Limited | Limited |

[See full Mem0 comparison](comparison/mem0-vs-alma.md){ .md-button }
[See full LangChain comparison](comparison/langchain-memory-vs-alma.md){ .md-button }

## Quick Start

=== "Python"

    ```bash
    pip install alma-memory
    ```

    ```python
    from alma import ALMA

    # Initialize
    alma = ALMA.from_config(".alma/config.yaml")

    # Before task: Get relevant memories
    memories = alma.retrieve(
        task="Test the login form validation",
        agent="helena",
        top_k=5
    )

    # Inject into your prompt
    prompt = f"""
    ## Your Task
    Test the login form validation

    ## Knowledge from Past Runs
    {memories.to_prompt()}
    """

    # After task: Learn from outcome
    alma.learn(
        agent="helena",
        task="Test login form",
        outcome="success",
        strategy_used="Tested empty fields, invalid email, valid submission",
    )
    ```

=== "TypeScript"

    ```bash
    npm install @rbkunnela/alma-memory
    ```

    ```typescript
    import { ALMA } from '@rbkunnela/alma-memory';

    // Create client
    const alma = new ALMA({
      baseUrl: 'http://localhost:8765',
      projectId: 'my-project'
    });

    // Retrieve memories
    const memories = await alma.retrieve({
      query: 'authentication flow',
      agent: 'dev-agent',
      topK: 5
    });

    // Learn from outcomes
    await alma.learn({
      agent: 'dev-agent',
      task: 'Implement OAuth',
      outcome: 'success',
      strategyUsed: 'Used passport.js middleware'
    });
    ```

=== "MCP (Claude Code)"

    ```json
    // .mcp.json
    {
      "mcpServers": {
        "alma-memory": {
          "command": "python",
          "args": ["-m", "alma.mcp", "--config", ".alma/config.yaml"]
        }
      }
    }
    ```

    16 MCP tools available: `alma_retrieve`, `alma_learn`, `alma_checkpoint`, and more.

## Five Memory Types

| Type | What It Stores | Example |
|------|----------------|---------|
| **Heuristic** | Learned strategies | "For forms with >5 fields, test validation incrementally" |
| **Outcome** | Task results | "Login test succeeded using JWT token strategy" |
| **Preference** | User constraints | "User prefers verbose test output" |
| **Domain Knowledge** | Accumulated facts | "Login uses OAuth 2.0 with 24h token expiry" |
| **Anti-pattern** | What NOT to do | "Don't use sleep() for async waits - causes flaky tests" |

## Storage Backends

Deploy anywhere with your preferred database:

| Backend | Use Case | Vector Search |
|---------|----------|---------------|
| **SQLite + FAISS** | Local development | Yes |
| **PostgreSQL + pgvector** | Production | Yes (HNSW) |
| **Qdrant** | Managed vector DB | Yes (HNSW) |
| **Pinecone** | Serverless | Yes |
| **Chroma** | Lightweight local | Yes |
| **Azure Cosmos DB** | Enterprise | Yes (DiskANN) |

## v0.6.0 - Workflow Context Layer

The latest release adds multi-agent workflow support:

- **Checkpoint & Resume** - Save and restore workflow state
- **State Reducers** - Merge outputs from parallel agents
- **Artifact Linking** - Track code, tests, and documents per workflow
- **Scoped Retrieval** - Filter memories by workflow context
- **8 New MCP Tools** - Full workflow support in Claude Code

[View Changelog](about/changelog.md){ .md-button .md-button--primary }

## Get Started

<div class="grid cards" markdown>

-   :material-download:{ .lg .middle } __Installation__

    ---

    Install ALMA with pip or npm and configure your first project.

    [:octicons-arrow-right-24: Installation Guide](getting-started/installation.md)

-   :material-rocket-launch:{ .lg .middle } __Quick Start__

    ---

    Build your first memory-powered agent in 5 minutes.

    [:octicons-arrow-right-24: Quick Start](getting-started/quickstart.md)

-   :material-cog:{ .lg .middle } __Configuration__

    ---

    Configure storage backends, agents, and scopes.

    [:octicons-arrow-right-24: Configuration](getting-started/configuration.md)

-   :fontawesome-brands-github:{ .lg .middle } __GitHub__

    ---

    Star the repo, report issues, contribute.

    [:octicons-arrow-right-24: GitHub](https://github.com/RBKunnela/ALMA-memory)

</div>

---

<div style="text-align: center; margin-top: 3rem;">
  <p><strong>Built for AI agents that get better with every task.</strong></p>
  <p>
    <a href="https://pypi.org/project/alma-memory/">PyPI</a> ·
    <a href="https://github.com/RBKunnela/ALMA-memory">GitHub</a> ·
    <a href="https://github.com/RBKunnela/ALMA-memory/issues">Issues</a>
  </p>
</div>
