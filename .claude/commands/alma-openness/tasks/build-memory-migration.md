# Task: Build Memory Migration Tools

**Agent:** `@domain-designer` (Schema)
**Squad:** alma-openness
**Priority:** P1
**Requires:** `design-personal-brain-schema` (completed)

---

## Goal

Build memory migration tools that let users import their existing memories from other AI tools and knowledge stores into ALMA's Personal Brain schema. Support Claude memory export (JSON), ChatGPT memory export (JSON), and Obsidian vault (Markdown + YAML frontmatter). Preserve >90% of source metadata during import.

## Output

- `alma/migration/__init__.py` -- Migration module init with public API
- `alma/migration/base.py` -- Base importer ABC
- `alma/migration/claude_importer.py` -- Claude memory export importer
- `alma/migration/chatgpt_importer.py` -- ChatGPT memory export importer
- `alma/migration/obsidian_importer.py` -- Obsidian vault importer
- `alma/migration/types.py` -- Migration data types (ImportResult, SourceMemory, etc.)
- `tests/unit/test_migration.py` -- Unit tests for all importers

## Steps

### 1. Research Claude Memory Export Format

Claude's memory export is a JSON file containing user memories:

```json
{
  "memories": [
    {
      "id": "mem_abc123",
      "content": "User prefers TypeScript over JavaScript for new projects",
      "created_at": "2025-11-15T10:30:00Z",
      "source": "conversation",
      "context": "Discussion about project setup"
    }
  ]
}
```

Key fields to map:
- `content` -> ALMA heuristic content or Personal Brain thought/insight/lesson
- `created_at` -> ALMA timestamp
- `source` -> metadata.source_tool = "claude"
- `context` -> metadata.context

### 2. Research ChatGPT Memory Export Format

ChatGPT memory exports are available via Settings > Data Controls > Export Data. The memories appear in `model_comparisons.json` or `memories.json`:

```json
{
  "memories": [
    {
      "id": "...",
      "text": "Prefers dark mode in all editors",
      "created_at": "2025-10-20",
      "updated_at": "2025-11-01"
    }
  ]
}
```

Key fields to map:
- `text` -> ALMA content
- `created_at` / `updated_at` -> ALMA timestamps
- Source tool = "chatgpt"

### 3. Research Obsidian Vault Structure

Obsidian vaults are directories of Markdown files with optional YAML frontmatter:

```
vault/
  daily/
    2025-11-15.md
  projects/
    alma-memory.md
  people/
    john-smith.md
```

Each file:
```markdown
---
title: ALMA Memory Project
tags: [project, python, library]
created: 2025-11-01
status: active
---

# ALMA Memory Project

A Python library for persistent AI agent memory.

## Key Decisions
- Use PostgreSQL+pgvector for production storage
- Support 7 storage backends for flexibility

## Lessons Learned
- ABC pattern works well for storage backends
- Domain schemas should be user-extensible
```

Key mapping:
- File path -> entity type (projects/, people/, daily/ -> project, person, thought)
- YAML frontmatter -> metadata
- `[[wiki links]]` -> relationships
- `#tags` -> ALMA tags/categories
- Content sections -> structured entity attributes

### 4. Design Base Importer

```python
"""alma/migration/base.py"""

from abc import ABC, abstractmethod
from typing import List
from alma.migration.types import ImportResult, SourceMemory


class BaseImporter(ABC):
    """Base class for memory importers."""

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Name of the source system (e.g., 'claude', 'chatgpt', 'obsidian')."""
        ...

    @abstractmethod
    def parse(self, source_path: str) -> List[SourceMemory]:
        """
        Parse source data into normalized SourceMemory objects.

        Args:
            source_path: Path to the source file or directory

        Returns:
            List of parsed memories
        """
        ...

    @abstractmethod
    def map_to_alma(self, source_memory: SourceMemory) -> dict:
        """
        Map a source memory to ALMA's Personal Brain schema.

        Returns:
            Dict with keys: entity_type, content, metadata, relationships
        """
        ...

    def run_import(
        self,
        source_path: str,
        alma_instance,
        dry_run: bool = False,
    ) -> ImportResult:
        """
        Run the full import pipeline.

        Steps:
        1. Parse source data
        2. Map to ALMA schema
        3. Deduplicate against existing memories
        4. Store in ALMA (unless dry_run)
        5. Return results with statistics
        """
        memories = self.parse(source_path)
        mapped = [self.map_to_alma(m) for m in memories]

        result = ImportResult(
            source=self.source_name,
            total_found=len(memories),
            imported=0,
            skipped_duplicates=0,
            skipped_errors=0,
            metadata_preserved_pct=0.0,
        )

        if dry_run:
            result.imported = len(mapped)
            return result

        for item in mapped:
            try:
                # Store via ALMA
                alma_instance.learn(
                    content=item["content"],
                    metadata=item["metadata"],
                )
                result.imported += 1
            except Exception as e:
                result.skipped_errors += 1
                result.errors.append(str(e))

        result.metadata_preserved_pct = self._calculate_preservation(memories, mapped)
        return result
```

### 5. Build Claude Importer

```python
"""alma/migration/claude_importer.py"""

class ClaudeImporter(BaseImporter):
    """Import memories from Claude's memory export JSON."""

    source_name = "claude"

    def parse(self, source_path: str) -> List[SourceMemory]:
        """Parse Claude memory export JSON."""
        with open(source_path) as f:
            data = json.load(f)

        memories = []
        for item in data.get("memories", []):
            memories.append(SourceMemory(
                id=item.get("id", str(uuid.uuid4())),
                content=item["content"],
                created_at=parse_datetime(item.get("created_at")),
                source_tool="claude",
                metadata={
                    "source": item.get("source"),
                    "context": item.get("context"),
                },
            ))
        return memories

    def map_to_alma(self, source_memory: SourceMemory) -> dict:
        """Map Claude memory to Personal Brain entity."""
        # Classify: is this a thought, insight, lesson, or decision?
        entity_type = self._classify_content(source_memory.content)

        return {
            "entity_type": entity_type,
            "content": source_memory.content,
            "metadata": {
                "source_tool": "claude",
                "original_id": source_memory.id,
                "imported_at": datetime.now(timezone.utc).isoformat(),
                **source_memory.metadata,
            },
        }

    def _classify_content(self, content: str) -> str:
        """Classify content into a Personal Brain entity type."""
        content_lower = content.lower()
        if any(word in content_lower for word in ["learned", "lesson", "never again", "always"]):
            return "lesson"
        elif any(word in content_lower for word in ["decided", "chose", "picked", "went with"]):
            return "decision"
        elif any(word in content_lower for word in ["realized", "noticed", "pattern"]):
            return "insight"
        else:
            return "thought"
```

### 6. Build ChatGPT Importer

```python
"""alma/migration/chatgpt_importer.py"""

class ChatGPTImporter(BaseImporter):
    """Import memories from ChatGPT's data export."""

    source_name = "chatgpt"

    def parse(self, source_path: str) -> List[SourceMemory]:
        """Parse ChatGPT memory export."""
        # Handle both memories.json and model_comparisons.json formats
        ...

    def map_to_alma(self, source_memory: SourceMemory) -> dict:
        """Map ChatGPT memory to Personal Brain entity."""
        ...
```

### 7. Build Obsidian Importer

```python
"""alma/migration/obsidian_importer.py"""

class ObsidianImporter(BaseImporter):
    """Import memories from an Obsidian vault."""

    source_name = "obsidian"

    def parse(self, source_path: str) -> List[SourceMemory]:
        """
        Parse all markdown files in an Obsidian vault.

        Handles:
        - YAML frontmatter extraction
        - [[wiki link]] detection for relationships
        - #tag extraction
        - Directory-based entity type inference
        """
        ...

    def map_to_alma(self, source_memory: SourceMemory) -> dict:
        """
        Map Obsidian note to Personal Brain entity.

        Mapping rules:
        - people/ directory -> person entity
        - projects/ directory -> project entity
        - daily/ directory -> thought entity
        - Notes with 'decision' tag -> decision entity
        - Notes with 'lesson' tag -> lesson entity
        - Other notes -> insight or thought based on content
        """
        ...

    def _extract_relationships(self, content: str) -> List[dict]:
        """Extract [[wiki links]] as potential relationships."""
        import re
        links = re.findall(r'\[\[(.*?)\]\]', content)
        return [{"target": link, "type": "references"} for link in links]

    def _parse_frontmatter(self, content: str) -> dict:
        """Extract YAML frontmatter from markdown file."""
        ...
```

### 8. Validate Metadata Preservation

Calculate what percentage of source metadata was preserved during import:

```python
def _calculate_preservation(
    self,
    source: List[SourceMemory],
    mapped: List[dict],
) -> float:
    """
    Calculate metadata preservation percentage.

    Checks:
    - Content preserved (always 100% unless truncated)
    - Timestamps preserved
    - Tags/categories mapped
    - Source-specific metadata carried over
    - Relationships detected and mapped

    Target: >90% preservation
    """
    ...
```

## Gate (Definition of Done)

- [ ] `BaseImporter` ABC defined in `alma/migration/base.py` with parse/map/run_import methods
- [ ] `ClaudeImporter` parses Claude memory export JSON and maps to Personal Brain entities
- [ ] `ChatGPTImporter` parses ChatGPT memory export and maps to Personal Brain entities
- [ ] `ObsidianImporter` parses Obsidian vault (markdown + frontmatter + wiki links)
- [ ] Content classification assigns appropriate entity types (thought/insight/decision/lesson)
- [ ] `source_tool` metadata tracked on every imported memory
- [ ] Obsidian wiki links `[[...]]` extracted as relationships
- [ ] Obsidian YAML frontmatter preserved as metadata
- [ ] Metadata preservation measured and reported (target: >90%)
- [ ] Dry-run mode works (parse and map without storing)
- [ ] Import is idempotent (re-running does not create duplicates)
- [ ] All unit tests pass: `.venv/bin/python -m pytest tests/unit/test_migration.py -v`
- [ ] Linting passes: `.venv/bin/python -m ruff check alma/migration/`

## References

- `alma/domains/personal_brain.py` -- Personal Brain entity and relationship types
- `alma/consolidation/deduplication.py` -- DeduplicationEngine for import dedup
- `alma/types.py` -- Heuristic, Outcome data structures
- `alma/core.py` -- ALMA.learn() method for storing imported memories
- `alma/storage/base.py` -- StorageBackend ABC (all importers write through ALMA)
- `tasks/design-personal-brain-schema.md` -- Required: Personal Brain schema defines target entity types
- `data/openness-patterns-kb.md` -- Memory migration format specifications
