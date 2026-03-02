# Task: Design Personal Brain Domain Schema

**Agent:** `@domain-designer` (Schema)
**Squad:** alma-openness
**Priority:** P0 (foundational -- all other tasks depend on this)
**Requires:** None (no dependencies)

---

## Goal

Design and implement the Personal Brain domain schema for ALMA. This schema models how humans actually think and organize knowledge: thoughts that spark insights, insights that refine decisions, decisions that apply to projects, and lessons learned from experience. The schema becomes the 7th pre-built domain alongside coding, research, sales, general, customer_support, and content_creation.

## Output

- `alma/domains/personal_brain.py` -- Schema definition with `get_personal_brain_schema()` function
- Updated `alma/domains/factory.py` -- Registration in `DomainMemoryFactory._builtin_schemas`
- Updated `alma/domains/__init__.py` -- New exports
- `tests/unit/test_personal_brain.py` -- Unit tests for the schema

## Steps

### 1. Study ALMA's Existing DomainSchema Pattern

Read and understand the pattern used by all 6 existing schemas:

- **Types**: `alma/domains/types.py` -- `DomainSchema`, `EntityType`, `RelationshipType` dataclasses
- **Schemas**: `alma/domains/schemas.py` -- 6 existing schemas (`get_coding_schema()`, `get_research_schema()`, etc.)
- **Factory**: `alma/domains/factory.py` -- `DomainMemoryFactory` with `_builtin_schemas` registry and convenience functions
- **Init**: `alma/domains/__init__.py` -- Public API exports

Key API pattern:
```python
def get_personal_brain_schema() -> DomainSchema:
    schema = DomainSchema.create(
        name="personal_brain",
        description="...",
        learning_categories=[...],
    )
    schema.add_entity_type(name="...", description="...", attributes=[...])
    schema.add_relationship_type(name="...", description="...",
                                  source_type="...", target_type="...")
    return schema
```

### 2. Design Entity Types

Define 6 core entity types that model human cognition:

| Entity | Description | Attributes | Required |
|--------|-------------|------------|----------|
| `thought` | A raw idea, observation, or reflection captured in the moment | `content`, `context`, `mood`, `source_tool`, `tags` | `content` |
| `insight` | A refined understanding derived from one or more thoughts | `summary`, `confidence`, `evidence`, `domain`, `source_thoughts` | `summary` |
| `decision` | A choice made with rationale and considered alternatives | `choice`, `alternatives`, `rationale`, `outcome`, `status` | `choice`, `rationale` |
| `person` | A person referenced in the user's knowledge graph | `name`, `role`, `relationship`, `context`, `last_interaction` | `name` |
| `project` | A project, initiative, or area of ongoing work | `name`, `status`, `goals`, `domain`, `priority` | `name` |
| `lesson` | A lesson learned from experience, reusable across contexts | `summary`, `context`, `applicability`, `cost_of_ignoring`, `source_experience` | `summary` |

### 3. Design Relationship Types

Define relationships that capture how knowledge connects:

| Relationship | Description | Source -> Target |
|-------------|-------------|------------------|
| `inspires` | One thought or insight was inspired by another | thought -> thought, insight -> thought |
| `contradicts` | Two memories are in tension or conflict | insight -> insight, decision -> decision |
| `builds_on` | An insight or decision deepens/refines a prior one | insight -> thought, insight -> insight |
| `involves` | A memory references a specific person | thought -> person, decision -> person, lesson -> person |
| `belongs_to` | A memory is part of a project context | thought -> project, insight -> project, decision -> project |
| `leads_to` | One experience or decision leads to a lesson | decision -> lesson, thought -> lesson |

Note: ALMA's `RelationshipType` supports a single source_type/target_type pair. For relationships that cross multiple entity pairs (e.g., `inspires` works for both thought->thought and insight->thought), create separate relationship instances with descriptive names: `thought_inspires_thought`, `insight_inspires_thought` -- or pick the primary direction and document the flexibility.

### 4. Design Learning Categories

Define what patterns the Personal Brain schema can learn from:

```python
learning_categories = [
    "thought_patterns",           # Recurring themes in captured thoughts
    "decision_frameworks",        # How the user makes decisions
    "insight_synthesis",          # Patterns in how insights form from thoughts
    "relationship_dynamics",      # Patterns in people interactions
    "project_management",         # How the user manages project work
    "lesson_application",         # When and how lessons get reapplied
    "knowledge_connections",      # Cross-domain knowledge linking patterns
    "reflection_habits",          # Meta-patterns in thinking style
]
```

### 5. Implement the Schema

Create `alma/domains/personal_brain.py` following the exact pattern from `alma/domains/schemas.py`:

```python
"""
Personal Brain Domain Schema.

Domain schema for personal knowledge management and cognitive modeling.
Models how humans think: thoughts, insights, decisions, people, projects, and lessons.
"""

from alma.domains.types import DomainSchema


def get_personal_brain_schema() -> DomainSchema:
    """
    Pre-built schema for personal brain / open brain workflows.

    Models human cognition as a graph of interconnected knowledge entities.
    Suitable for: Personal knowledge management, cross-tool memory, open brain.
    """
    schema = DomainSchema.create(
        name="personal_brain",
        description="Memory schema for personal knowledge management and cognitive modeling",
        learning_categories=[...],
    )

    # Entity types (6 types)
    schema.add_entity_type(name="thought", ...)
    schema.add_entity_type(name="insight", ...)
    schema.add_entity_type(name="decision", ...)
    schema.add_entity_type(name="person", ...)
    schema.add_entity_type(name="project", ...)
    schema.add_entity_type(name="lesson", ...)

    # Relationship types (6+ relationships)
    schema.add_relationship_type(name="inspires", ...)
    schema.add_relationship_type(name="contradicts", ...)
    schema.add_relationship_type(name="builds_on", ...)
    schema.add_relationship_type(name="involves", ...)
    schema.add_relationship_type(name="belongs_to", ...)
    schema.add_relationship_type(name="leads_to", ...)

    return schema
```

### 6. Register Schema with ALMA's Domain Registry

Update `alma/domains/factory.py`:

```python
# Add import
from alma.domains.personal_brain import get_personal_brain_schema

# Add to _builtin_schemas dict
_builtin_schemas: Dict[str, callable] = {
    "coding": get_coding_schema,
    "research": get_research_schema,
    "sales": get_sales_schema,
    "general": get_general_schema,
    "customer_support": get_customer_support_schema,
    "content_creation": get_content_creation_schema,
    "personal_brain": get_personal_brain_schema,  # NEW
}
```

Add convenience function:

```python
def create_personal_brain_alma(
    project_id: str,
    agent: str = "thinker",
    storage: Optional[Any] = None,
    **config,
) -> Any:
    """Create ALMA configured for personal brain / open brain workflows."""
    factory = DomainMemoryFactory()
    return factory.create_alma_for_agent(
        schema_name="personal_brain",
        agent=agent,
        project_id=project_id,
        storage=storage,
        **config,
    )
```

Update `alma/domains/__init__.py` with new exports:
```python
from alma.domains.personal_brain import get_personal_brain_schema
```

### 7. Write Tests

Create `tests/unit/test_personal_brain.py`:

```python
def test_personal_brain_schema_creation():
    """Schema can be created with all entity and relationship types."""

def test_personal_brain_entity_types():
    """All 6 entity types are present with correct attributes."""

def test_personal_brain_relationship_types():
    """All relationship types reference valid entity types."""

def test_personal_brain_learning_categories():
    """Learning categories cover personal knowledge patterns."""

def test_personal_brain_schema_validation():
    """Schema passes validation (no orphan relationships, no duplicate entities)."""

def test_personal_brain_factory_registration():
    """Schema is available via DomainMemoryFactory.get_schema('personal_brain')."""

def test_create_personal_brain_alma():
    """Convenience function creates a working ALMA instance."""
```

Use `alma.testing.MockStorage` and `alma.testing.MockEmbedder` for test fixtures.

## Gate (Definition of Done)

- [ ] All 6 entity types defined: thought, insight, decision, person, project, lesson
- [ ] All 6 relationship types defined with valid source/target constraints
- [ ] Learning categories cover personal knowledge patterns (8 categories)
- [ ] Schema follows the exact `DomainSchema` factory pattern from `alma/domains/schemas.py`
- [ ] Schema registered in `DomainMemoryFactory._builtin_schemas` as `"personal_brain"`
- [ ] Convenience function `create_personal_brain_alma()` added to `alma/domains/factory.py`
- [ ] Exports updated in `alma/domains/__init__.py`
- [ ] `schema.validate()` returns empty list (no errors)
- [ ] All unit tests pass: `.venv/bin/python -m pytest tests/unit/test_personal_brain.py -v`
- [ ] Linting passes: `.venv/bin/python -m ruff check alma/domains/personal_brain.py`

## References

- `alma/domains/types.py` -- DomainSchema, EntityType, RelationshipType dataclasses
- `alma/domains/schemas.py` -- 6 existing schemas (follow same pattern)
- `alma/domains/factory.py` -- DomainMemoryFactory._builtin_schemas registry
- `alma/domains/__init__.py` -- Public API exports
- `templates/personal-brain-schema-tmpl.md` -- Entity and relationship design template
- `data/openness-patterns-kb.md` -- Entity type definitions and examples
