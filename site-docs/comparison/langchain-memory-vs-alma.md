# ALMA vs LangChain Memory: When to Use Each

> Comparing ALMA's persistent agent memory with LangChain's built-in memory modules.

## Different Use Cases

| Aspect | ALMA | LangChain Memory |
|--------|------|------------------|
| **Primary Use** | Long-term agent learning across sessions | Conversation context within sessions |
| **Persistence** | Always persistent (DB-backed) | Optional, often ephemeral |
| **Learning** | Agents improve from outcomes | No learning mechanism |
| **Scope** | Multi-agent with scoped access | Single chain/agent |
| **Memory Types** | 5 types (heuristics, outcomes, preferences, knowledge, anti-patterns) | Buffer, summary, entity, etc. |

## When to Use ALMA

- Agents that need to **remember across sessions**
- Multi-agent systems with **shared knowledge**
- When agents should **learn from mistakes** (anti-patterns)
- Complex workflows with **checkpoints and state**
- MCP integration with Claude Code

## When to Use LangChain Memory

- Simple chatbots needing **conversation history**
- Single-session interactions
- Quick prototypes
- Already using LangChain ecosystem

## Using ALMA with LangChain

ALMA complements LangChain - use both together:

```python
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from alma import ALMA

# LangChain for conversation context
langchain_memory = ConversationBufferMemory()

# ALMA for persistent learning
alma = ALMA.from_config(".alma/config.yaml")

async def run_agent(task: str, agent: str):
    # Get long-term memories from ALMA
    memories = alma.retrieve(task=task, agent=agent)

    # Inject into prompt
    system_prompt = f"""
    {memories.to_prompt()}

    Previous conversation:
    {langchain_memory.buffer}
    """

    # Run LLM
    llm = ChatOpenAI()
    result = await llm.ainvoke(system_prompt + task)

    # Learn from outcome (ALMA)
    alma.learn(agent=agent, task=task, outcome="success",
               strategy_used=result.content[:200])

    return result
```

## Feature Comparison

| Feature | ALMA | LangChain Memory |
|---------|------|------------------|
| Conversation buffer | Via session manager | `ConversationBufferMemory` |
| Summary memory | Via consolidation | `ConversationSummaryMemory` |
| Entity extraction | Via graph store | `ConversationEntityMemory` |
| Vector search | 6 backends | `VectorStoreRetrieverMemory` |
| **Learning from outcomes** | ✅ Yes | ❌ No |
| **Anti-pattern tracking** | ✅ Yes | ❌ No |
| **Multi-agent scoping** | ✅ Yes | ❌ No |
| **MCP server** | ✅ Yes | ❌ No |
| **Workflow checkpoints** | ✅ Yes | ❌ No |

## Installation

```bash
# ALMA
pip install alma-memory

# With LangChain
pip install alma-memory langchain langchain-openai
```

## Links

- [ALMA GitHub](https://github.com/RBKunnela/ALMA-memory)
- [LangChain Memory Docs](https://python.langchain.com/docs/modules/memory/)

---

*ALMA and LangChain serve different purposes - use them together for the best results.*
