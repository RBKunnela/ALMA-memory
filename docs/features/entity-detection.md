# Entity Detection

**Module:** `alma/extraction/entity_detector.py`
**Since:** v0.9.0
**Origin:** Adapted from MemPalace entity_detector.py (MIT License)

## Overview

The entity detector identifies people and projects mentioned in text using regex heuristics. It uses a two-pass approach: first extracting entity candidates with signal counts, then scoring and classifying each candidate. Returns ALMA `Entity` objects for direct integration with the graph memory system.

No LLM required. Pure pattern matching.

## Quick Start

```python
from alma.extraction.entity_detector import detect_entities

entities = detect_entities(
    "Alice said hello to Bob. They discussed the ALMA project."
)
for entity in entities:
    print(f"{entity.name} ({entity.entity_type})")

# Output:
# Alice (person)
# Bob (person)
# ALMA (project)
```

## Detection Methodology

### Pass 1: Signal Extraction

The detector scans text for capitalized words and collects signal counts from multiple pattern categories:

**Person signals:**
- Verb patterns: "{Name} said", "{Name} asked", "{Name} told", etc.
- Dialogue markers: "> Name:", "[Name]", "Name: ..."
- Greeting/address patterns: "hey Name", "thanks Name", "dear Name"
- Nearby pronoun resolution: he/she/they near a capitalized word

**Project signals:**
- Technical context patterns (to be matched against the candidate name)
- Uppercase/acronym patterns (e.g., "ALMA", "API")

### Pass 2: Scoring and Classification

Each candidate is scored based on the ratio of person vs. project signals. The classification thresholds determine whether a candidate is labeled as `person`, `project`, or `uncertain`.

## Supported Entity Types

| Type | Description | Detection Signals |
|------|-------------|-------------------|
| `person` | Human names mentioned in text | Verb actions, dialogue markers, pronouns |
| `project` | Project/product names | Technical context, acronym patterns |
| `uncertain` | Ambiguous entities | Insufficient signal strength |

## Integration with Graph Memory

Detected entities are returned as `alma.graph.store.Entity` objects, ready for storage in ALMA's graph backends (Neo4j, Memgraph, Kuzu, or In-Memory):

```python
from alma.extraction.entity_detector import detect_entities
from alma.graph import create_graph_backend

# Detect entities
entities = detect_entities(text)

# Store in graph
graph = create_graph_backend("in_memory")
for entity in entities:
    graph.create_entity(entity)
```

## Limitations

- Regex-based: does not use NER models, so accuracy is lower than spaCy/Transformers
- English-centric: patterns are designed for English text
- No coreference resolution: "she" near "Alice" is a signal, not a definitive link
- Common words in title case may produce false positives
