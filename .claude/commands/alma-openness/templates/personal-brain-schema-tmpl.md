# Personal Brain Schema Template

> Template for implementing the Personal Brain domain schema in `alma/domains/personal_brain.py`.

---

## Entity Types

### Overview Table

| Entity | Description | Key Attributes | Required Attributes | Validation |
|--------|-------------|----------------|---------------------|------------|
| `thought` | A raw idea, observation, or reflection | content, context, mood, source_tool, tags | content | content must be non-empty |
| `insight` | A refined understanding from thoughts | summary, confidence, evidence, domain, source_thoughts | summary | confidence in 0.0-1.0 if set |
| `decision` | A choice with rationale and alternatives | choice, alternatives, rationale, outcome, status | choice, rationale | status in {active, reversed, superseded} |
| `person` | A person in the knowledge graph | name, role, relationship, context, last_interaction | name | name must be non-empty |
| `project` | A project or area of work | name, status, goals, domain, priority | name | status in {active, paused, completed, abandoned} |
| `lesson` | A reusable lesson from experience | summary, context, applicability, cost_of_ignoring, source_experience | summary | summary must be non-empty |

### Thought Entity

```python
schema.add_entity_type(
    name="thought",
    description="A raw idea, observation, or reflection captured in the moment",
    attributes=[
        "content",          # str: The thought itself (required)
        "context",          # str: Where/when the thought occurred
        "mood",             # str: Emotional context (curious, frustrated, excited, neutral)
        "source_tool",      # str: Which AI tool captured this thought
        "tags",             # list[str]: User-defined tags
    ],
)
```

**Example thought:**
```json
{
    "entity_type": "thought",
    "content": "I wonder if we should use PostgreSQL instead of SQLite for production",
    "context": "Reviewing storage backend options for ALMA v1.0",
    "mood": "curious",
    "source_tool": "claude-code",
    "tags": ["alma", "storage", "architecture"]
}
```

### Insight Entity

```python
schema.add_entity_type(
    name="insight",
    description="A refined understanding derived from one or more thoughts",
    attributes=[
        "summary",          # str: The insight statement (required)
        "confidence",       # float: Confidence level 0.0-1.0
        "evidence",         # list[str]: Supporting observations
        "domain",           # str: Knowledge domain
        "source_thoughts",  # list[str]: IDs of thoughts that formed this insight
    ],
)
```

**Example insight:**
```json
{
    "entity_type": "insight",
    "summary": "PostgreSQL+pgvector is the best production storage for ALMA because it supports concurrent writes and vector search natively",
    "confidence": 0.85,
    "evidence": ["Benchmarked 3 backends", "pgvector handles 10k vectors well", "Row-level locking enables multi-client"],
    "domain": "infrastructure",
    "source_thoughts": ["thought_001", "thought_002", "thought_003"]
}
```

### Decision Entity

```python
schema.add_entity_type(
    name="decision",
    description="A choice made with explicit rationale and considered alternatives",
    attributes=[
        "choice",           # str: What was decided (required)
        "alternatives",     # list[str]: Options considered
        "rationale",        # str: Why this choice was made (required)
        "outcome",          # str: What happened as a result
        "status",           # str: active | reversed | superseded
    ],
)
```

**Example decision:**
```json
{
    "entity_type": "decision",
    "choice": "Use PostgreSQL+pgvector as the recommended production backend",
    "alternatives": ["Pinecone (too expensive)", "Qdrant (extra infrastructure)", "SQLite (no concurrent writes)"],
    "rationale": "Best balance of cost, features, and operational simplicity for most users",
    "outcome": null,
    "status": "active"
}
```

### Person Entity

```python
schema.add_entity_type(
    name="person",
    description="A person referenced in the user's knowledge graph",
    attributes=[
        "name",             # str: Person's name (required)
        "role",             # str: Professional role or title
        "relationship",     # str: How the user knows them
        "context",          # str: Primary context of interaction
        "last_interaction", # str: When last interacted with
    ],
)
```

**Example person:**
```json
{
    "entity_type": "person",
    "name": "Sarah Chen",
    "role": "Engineering Lead at Acme",
    "relationship": "colleague",
    "context": "Works on the infrastructure team, strong advocate for microservices",
    "last_interaction": "2025-11-15"
}
```

### Project Entity

```python
schema.add_entity_type(
    name="project",
    description="A project, initiative, or area of ongoing work",
    attributes=[
        "name",             # str: Project name (required)
        "status",           # str: active | paused | completed | abandoned
        "goals",            # list[str]: What this project aims to achieve
        "domain",           # str: Domain (work, personal, health, etc.)
        "priority",         # str: high | medium | low
    ],
)
```

**Example project:**
```json
{
    "entity_type": "project",
    "name": "ALMA v1.0 Launch",
    "status": "active",
    "goals": ["Production-ready storage", "Multi-client MCP", "45-minute quickstart"],
    "domain": "work",
    "priority": "high"
}
```

### Lesson Entity

```python
schema.add_entity_type(
    name="lesson",
    description="A reusable lesson learned from experience",
    attributes=[
        "summary",           # str: The lesson statement (required)
        "context",           # str: Situation where this was learned
        "applicability",     # str: When this lesson applies
        "cost_of_ignoring",  # str: What happens if ignored
        "source_experience", # str: The experience that taught this lesson
    ],
)
```

**Example lesson:**
```json
{
    "entity_type": "lesson",
    "summary": "Never deploy on Friday afternoon",
    "context": "Deployed a critical fix on Friday 4 PM, spent the weekend debugging a cascading failure",
    "applicability": "Any production deployment to user-facing systems",
    "cost_of_ignoring": "Weekend work, user impact, team morale damage",
    "source_experience": "ALMA v0.5 hotfix deployment incident, 2025-09-12"
}
```

---

## Relationship Types

### Overview Table

| Relationship | Description | Source Type | Target Type | Cardinality |
|-------------|-------------|------------|-------------|-------------|
| `inspires` | One thought/insight was inspired by another | thought | thought | many-to-many |
| `contradicts` | Two memories are in tension | insight | insight | many-to-many |
| `builds_on` | An insight deepens a prior thought/insight | insight | thought | many-to-many |
| `involves` | A memory references a person | thought | person | many-to-many |
| `belongs_to` | A memory is part of a project | thought | project | many-to-many |
| `leads_to` | An experience leads to a lesson | decision | lesson | many-to-many |

### Implementation

```python
# Primary relationships (one per ALMA RelationshipType)
schema.add_relationship_type(
    name="inspires",
    description="One thought was inspired by another thought",
    source_type="thought",
    target_type="thought",
)

schema.add_relationship_type(
    name="contradicts",
    description="Two insights are in tension or conflict",
    source_type="insight",
    target_type="insight",
)

schema.add_relationship_type(
    name="builds_on",
    description="An insight deepens or refines a prior thought",
    source_type="insight",
    target_type="thought",
)

schema.add_relationship_type(
    name="involves",
    description="A thought references a specific person",
    source_type="thought",
    target_type="person",
)

schema.add_relationship_type(
    name="belongs_to",
    description="A thought is part of a project context",
    source_type="thought",
    target_type="project",
)

schema.add_relationship_type(
    name="leads_to",
    description="A decision leads to a lesson learned",
    source_type="decision",
    target_type="lesson",
)
```

### Extended Relationships (Optional)

For richer graph coverage, add these additional relationship types:

```python
# Cross-entity inspires
schema.add_relationship_type(
    name="insight_inspires_insight",
    description="One insight inspired another insight",
    source_type="insight",
    target_type="insight",
)

# Cross-entity involves
schema.add_relationship_type(
    name="decision_involves_person",
    description="A decision involved or affected a person",
    source_type="decision",
    target_type="person",
)

# Cross-entity belongs_to
schema.add_relationship_type(
    name="decision_belongs_to_project",
    description="A decision belongs to a project context",
    source_type="decision",
    target_type="project",
)

schema.add_relationship_type(
    name="insight_belongs_to_project",
    description="An insight belongs to a project context",
    source_type="insight",
    target_type="project",
)

# Lesson to person
schema.add_relationship_type(
    name="lesson_involves_person",
    description="A lesson was learned in the context of a person",
    source_type="lesson",
    target_type="person",
)
```

---

## Graph Representation Guidelines

### Node Types = Entity Types

Each entity type becomes a node type in the graph:
- Nodes are colored/shaped by type for visual distinction
- All nodes carry their full attribute set as properties
- Nodes are linked by relationships defined in the schema

### Edge Types = Relationship Types

Each relationship type becomes an edge type:
- Edges are directional (source -> target)
- Edge labels match the relationship name
- Edges can carry metadata (created_at, source_tool, confidence)

### Traversal Patterns

Common graph queries the Personal Brain should support:

1. **"What do I know about X?"** -- Find all entities connected to a person, project, or topic
2. **"How did I reach this decision?"** -- Trace `inspires` and `builds_on` chains backward from a decision
3. **"What contradicts this?"** -- Find all `contradicts` relationships from an insight
4. **"What lessons from project X?"** -- Follow `belongs_to` to project, then `leads_to` to lessons
5. **"What has changed since last week?"** -- Time-filtered query across all entity types

### Integration with ALMA Graph Backends

ALMA supports 4 graph backends in `alma/graph/`:
- **Neo4j** -- Full Cypher query support
- **Memgraph** -- Compatible with Neo4j patterns
- **Kuzu** -- Embedded graph database
- **In-Memory** -- No external dependencies

The Personal Brain schema should work with all 4 backends. Entity types map to node labels, relationship types map to edge types.

---

## Metadata Mapping: Capture to Entity

When a raw thought is captured (e.g., via MCP `alma_learn` tool), it goes through this pipeline:

```
Raw Input
  |
  v
Content Classification (thought/insight/decision/lesson)
  |
  v
Entity Creation (with appropriate attributes)
  |
  v
Relationship Detection (references to people, projects, prior memories)
  |
  v
Storage (entity stored, relationships created, graph updated)
```

### Classification Heuristics

| Signal | Entity Type |
|--------|-------------|
| Contains "I learned", "lesson", "never again", "always" | lesson |
| Contains "I decided", "chose", "went with", "picked" | decision |
| Contains "I realized", "it seems", "pattern", "noticed that" | insight |
| Contains a person's name (detected via NER or explicit mention) | thought + person (with `involves` relationship) |
| References a known project name | thought + `belongs_to` relationship |
| Default (no strong signals) | thought |

### Example: Raw Capture to Typed Entity

**Input (via Claude Code MCP):**
```
"After reviewing the performance benchmarks, I decided to use PostgreSQL+pgvector
instead of Pinecone. The benchmarks showed 3x better latency for our query patterns,
and Sarah from the infra team confirmed they can support a managed PostgreSQL instance."
```

**Output:**

Entity 1 (Decision):
```json
{
    "entity_type": "decision",
    "choice": "Use PostgreSQL+pgvector instead of Pinecone",
    "alternatives": ["Pinecone"],
    "rationale": "3x better latency for our query patterns, managed PostgreSQL available",
    "status": "active",
    "metadata": {
        "source_tool": "claude-code",
        "captured_at": "2025-11-15T10:30:00Z"
    }
}
```

Entity 2 (Person -- if not already exists):
```json
{
    "entity_type": "person",
    "name": "Sarah",
    "role": "infra team",
    "context": "Confirmed managed PostgreSQL support"
}
```

Relationships:
- Decision `involves` Person (Sarah)
- Decision `belongs_to` Project (if a project context is active)

---

*Personal Brain Schema Template v1.0.0*
*For use by @domain-designer when implementing alma/domains/personal_brain.py*
