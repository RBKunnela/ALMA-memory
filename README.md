# ALMA - Agent Learning Memory Architecture

> A reusable harness pattern for creating AI agents that learn and improve over time through structured memory - without model weight updates.

## The Harness Pattern

ALMA isn't just agent memory - it's a **generalized framework** for any tool-using workflow:

```
┌─────────────────────────────────────────────────────────────────┐
│  1. SETTING        Fixed environment: tools, constraints        │
├─────────────────────────────────────────────────────────────────┤
│  2. CONTEXT        Ephemeral per-run inputs: task, user         │
├─────────────────────────────────────────────────────────────────┤
│  3. AGENT          The executor with scoped intelligence        │
├─────────────────────────────────────────────────────────────────┤
│  4. MEMORY SCHEMA  Domain-specific learning structure           │
└─────────────────────────────────────────────────────────────────┘
```

**The Flow:**
1. **Pre-run**: Inject relevant memory slices ("Past successes in similar tasks")
2. **Run**: Agent acts using tools, logs reflections
3. **Post-run**: Update memory schema
4. **Repeat**: Agent appears to "learn" without weight changes

## Why This Matters

```
Code exists       ≠ Knowledge retained
Knowledge retained ≠ Knowledge scoped
Knowledge scoped   ≠ Knowledge retrieved efficiently
```

ALMA solves all three through **scoped memory injection**. Agents get smarter via better-informed prompts, not model changes.

## Supported Domains

ALMA works for ANY tool-using workflow:

| Domain | Agents | Use Case |
|--------|--------|----------|
| **Coding** | Helena, Victor | Testing, API development |
| **Research** | Researcher | Market analysis, competitive intelligence |
| **Content** | Copywriter, Documenter | Marketing, documentation |
| **Operations** | Support | Customer service, automation |

## Quick Start

### Installation

```bash
pip install alma-memory
# or from source
pip install git+https://github.com/RBKunnela/ALMA-memory.git
```

### Using the Harness Pattern

```python
from alma import ALMA, create_harness, Context

# Initialize ALMA
alma = ALMA.from_config(".alma/config.yaml")

# Create a domain-specific harness
harness = create_harness("coding", "helena", alma)

# Define task context
context = Context(
    task="Test the login form validation",
    project_id="my-app",
    user_id="developer-1",
    inputs={"component": "LoginForm", "priority": "high"}
)

# Run with memory injection
result = harness.run(context)

# The harness automatically:
# 1. Retrieved relevant memories (testing strategies, past outcomes)
# 2. Built the prompt with injected knowledge
# 3. Will log the outcome for future learning
```

### Creating Custom Agents

```python
from alma import (
    ALMA, Harness, Setting, Agent, MemorySchema, Tool, ToolType
)

# Define the environment
setting = Setting(
    name="Bio Research Environment",
    description="Tools for biological data analysis",
    tools=[
        Tool(
            name="sequence_search",
            description="Search genomic databases",
            tool_type=ToolType.SEARCH,
        ),
        Tool(
            name="structure_analysis",
            description="Analyze protein structures",
            tool_type=ToolType.ANALYSIS,
        ),
    ],
    global_constraints=[
        "Cite all data sources",
        "Note confidence levels",
    ],
)

# Define what this agent can learn
schema = MemorySchema(
    domain="bioinformatics",
    description="Patterns for biological data analysis",
    learnable_categories=[
        "search_refinements",
        "analysis_patterns",
        "data_interpretation",
    ],
    forbidden_categories=[
        "medical_diagnosis",  # Out of scope
    ],
    min_occurrences=5,
)

# Create the agent
agent = Agent(
    name="bio_researcher",
    role="Bioinformatics Analyst",
    description="Expert in genomic analysis and protein structure prediction",
    memory_schema=schema,
)

# Assemble the harness
alma = ALMA.from_config(".alma/config.yaml")
harness = Harness(setting=setting, agent=agent, alma=alma)
```

### Basic Memory Operations

```python
from alma import ALMA

alma = ALMA.from_config(".alma/config.yaml")

# Retrieve relevant memories
memories = alma.retrieve(
    task="Test the login form validation",
    agent="helena",
    top_k=5
)

# Inject into prompt
prompt = f"""
## Your Task
Test the login form validation

## Relevant Knowledge (from past runs)
{memories.to_prompt()}
"""

# After task completion, learn from the outcome
alma.learn(
    agent="helena",
    task="Test login form",
    outcome="success",
    strategy_used="Tested empty fields, invalid email, valid submission",
    feedback="User confirmed tests were thorough"
)
```

## Memory Types

| Type | What It Stores | Example |
|------|----------------|---------|
| **Heuristic** | Learned strategies | "For forms with >5 fields, test validation incrementally" |
| **Outcome** | Task results | "Login test succeeded using JWT token strategy" |
| **Preference** | User constraints | "User prefers verbose test output" |
| **Domain Knowledge** | Accumulated facts | "Login uses OAuth 2.0 with 24h token expiry" |
| **Anti-pattern** | What NOT to do | "Don't use sleep() for async waits - causes flaky tests" |

## Configuration

Create `.alma/config.yaml`:

```yaml
alma:
  project_id: "my-project"
  storage: sqlite  # or "azure" for production

  domains:
    coding:
      enabled: true
      agents: [helena, victor]
    research:
      enabled: true
      agents: [researcher]

  agents:
    helena:
      domain: coding
      can_learn:
        - testing_strategies
        - selector_patterns
      cannot_learn:
        - backend_logic
      min_occurrences_for_heuristic: 3

    researcher:
      domain: research
      can_learn:
        - trend_patterns
        - source_reliability
      cannot_learn:
        - code_implementation
      min_occurrences_for_heuristic: 5
```

## Storage Backends

| Backend | Use Case | Vector Search |
|---------|----------|---------------|
| `azure` | Production | Cosmos DB with vector search |
| `sqlite` | Local dev | SQLite + FAISS |
| `file` | Testing | JSON files (no vector search) |

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      HARNESS PATTERN                            │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐    │
│  │ Setting  │  │ Context  │  │  Agent   │  │MemorySchema  │    │
│  │ (tools)  │  │ (task)   │  │(executor)│  │  (learning)  │    │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                        ALMA CORE                                │
│  ┌────────────┐  ┌────────────┐  ┌────────────────────────┐    │
│  │ Retrieval  │  │  Learning  │  │      Storage           │    │
│  │  Engine    │  │  Protocol  │  │ (Azure/SQLite/File)    │    │
│  └────────────┘  └────────────┘  └────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                     MEMORY TYPES                                │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌──────────┐  │
│  │ Heuristics │  │  Outcomes  │  │Preferences │  │Anti-patt.│  │
│  └────────────┘  └────────────┘  └────────────┘  └──────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Documentation

- [PRD](docs/architecture/PRD.md) - Full product requirements
- [Harness Pattern](docs/guides/harness-pattern.md) - Deep dive on the pattern
- [API Reference](docs/api/) - Coming soon

## Status

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Core Abstractions | Done |
| 2 | Local Storage (SQLite + FAISS) | Done |
| 3 | Retrieval Engine | In Progress |
| 4 | Learning Protocols | Todo |
| 5 | Agent Integration (Helena + Victor) | Todo |
| 6 | Azure Cosmos DB | Todo |
| 7 | Cache Layer | Todo |
| 8 | Forgetting Mechanism | Todo |

## License

MIT

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
