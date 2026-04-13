# 4-Layer MemoryStack

**Module:** `alma/context/`
**Since:** v0.9.0
**Origin:** Inspired by MemPalace layers.py (MIT License)

## Overview

The MemoryStack provides token-efficient context loading by organizing memories into 4 layers with different loading strategies. Wake-up cost is under 900 tokens, leaving 95%+ of the context window free for conversation.

## The 4 Layers

| Layer | Name | Token Budget | Loading Strategy |
|-------|------|-------------|-----------------|
| L0 | Identity | ~100 tokens | Always loaded from text file. "Who am I?" |
| L1 | Essential Story | ~500-800 tokens | Always loaded. Top memories by confidence. |
| L2 | On-Demand | ~200-500 each | Loaded when a topic/domain comes up. |
| L3 | Deep Search | Unlimited | Full semantic search via ALMA retrieve(). |

## Quick Start

```python
from alma import ALMA
from alma.context import MemoryStack

# Initialize
alma = ALMA.from_config("config.yaml")
stack = MemoryStack(alma)

# Session start: inject L0 + L1 (~600-900 tokens)
system_context = stack.wake_up()

# Mid-conversation: topic-specific recall (L2)
auth_context = stack.recall("authentication flow", layer=2)

# Complex query: deep semantic search (L3)
results = stack.recall("JWT token expiry edge cases", layer=3)

# Format everything for prompt injection
prompt = stack.to_prompt(max_tokens=2000)
```

## API Reference

### `MemoryStack.__init__(alma_instance, identity_path=None, agent="default", l1_max_tokens=800)`

Create a new MemoryStack wrapping an ALMA instance.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `alma_instance` | `ALMA` | required | A configured ALMA instance for retrieval |
| `identity_path` | `str` or `None` | `~/.alma/identity.txt` | Path to the identity file for L0 |
| `agent` | `str` | `"default"` | Agent name for retrieval scoping |
| `l1_max_tokens` | `int` | `800` | Token budget for Layer 1 |

### `wake_up(domain=None, user_id=None) -> str`

Load L0 (Identity) + L1 (Essential Story) and return combined text. Call this at session start.

```python
context = stack.wake_up(domain="backend-api")
# Returns: "## Identity\n...\n\n## Essential Story\n..."
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `domain` | `str` or `None` | `None` | Optional domain filter for L1 retrieval |
| `user_id` | `str` or `None` | `None` | Optional user ID for preference retrieval |

### `recall(query, layer=None, top_k=5, domain=None, user_id=None) -> str`

Retrieve memories from a specified layer, or auto-select based on query characteristics.

Auto-selection logic:
- Short queries (< 30 chars) or queries with a domain hint -> L2 (On-Demand)
- Complex queries or no domain hint -> L3 (Deep Search)

```python
# Explicit layer selection
auth_mem = stack.recall("auth patterns", layer=2)

# Auto-select (short query -> L2)
quick = stack.recall("error handling")

# Auto-select (long query -> L3)
deep = stack.recall("How should we handle JWT token expiry when the refresh token is also expired?")
```

### `to_prompt(max_tokens=2000) -> str`

Format all loaded layers into a single string for prompt injection. Respects the token budget and includes only layers that have been loaded.

## Identity File

L0 loads from a plain text file (default: `~/.alma/identity.txt`). This file should contain a brief description of the agent's role and purpose:

```text
I am a backend API developer working on the ALMA project.
My focus areas: Python, FastAPI, PostgreSQL, vector search.
I prefer explicit error handling and comprehensive tests.
```

## ContextLayer

Each layer is represented by a `ContextLayer` object:

```python
from alma.context import ContextLayer, LAYER_IDENTITY

layer = ContextLayer(level=0, name="Identity")
layer.set_content("I am a developer.")
print(layer.is_loaded)     # True
print(layer.token_count)   # ~5
```

## Design Rationale

Traditional approaches load all available memories into the context window, consuming thousands of tokens. The MemoryStack defers loading until needed:

1. **L0 + L1 at startup** -- Essential context is always available (~900 tokens)
2. **L2 on topic change** -- When conversation shifts to a new domain, load relevant memories
3. **L3 on complex queries** -- Full semantic search only when the question warrants it

This keeps the context window lean while ensuring relevant memories are available when needed.
