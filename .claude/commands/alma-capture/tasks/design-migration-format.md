# Task: Design Migration Format

## Goal

Design the memory migration format and importer architecture for importing memories from external tools (Claude, ChatGPT, Obsidian) into ALMA.

## Agent

**migration-engineer** ([agents/migration-engineer.md](../agents/migration-engineer.md))

## Requires

None -- this task can run in parallel with the pipeline design. However, the intermediate format must align with ALMA's metadata schema, so coordinate with **metadata-extractor** on field definitions.

## Steps

### Step 1: Research Claude Memory Export Format

Investigate the Claude memory export format:
- What format does Claude use for memory exports? (JSON, CSV, other)
- What fields are included? (content, timestamp, tags, source)
- What metadata is available vs what must be inferred?
- What are the limitations? (no embeddings, minimal structure)

Document the format with example data.

### Step 2: Research ChatGPT Memory Export Format

Investigate the ChatGPT memory export format:
- ChatGPT memory is part of the full data export (Settings > Data controls > Export data)
- What is the file structure? (JSON within a ZIP)
- What fields are available for memory entries?
- How are memories stored vs conversations?
- What metadata exists?

Document the format with example data.

### Step 3: Research Obsidian Vault Structure

Investigate the Obsidian vault format:
- Directory structure (flat vs nested, special folders)
- Markdown file format with YAML frontmatter
- Link types: `[[wikilinks]]`, `[markdown](links)`, tags (`#tag`)
- Attachments and media handling
- Common frontmatter fields (title, date, tags, aliases)
- Plugin-generated metadata (Dataview, Templater, etc.)

Document the format with example data.

### Step 4: Design Common Intermediate Format

Design a canonical intermediate representation that any source can be mapped to:

```python
@dataclass
class MigrationRecord:
    """Intermediate format for memory migration."""
    content: str                          # The thought/note text
    source_format: str                    # "claude" | "chatgpt" | "obsidian"
    source_id: str | None                 # Original ID from source system
    created_at: datetime | None           # Original creation timestamp
    updated_at: datetime | None           # Original modification timestamp
    metadata: ThoughtMetadata | None      # Extracted/mapped metadata
    raw_metadata: dict                    # Original metadata preserved as-is
    links: list[str]                      # References to other records (for Obsidian)
    tags: list[str]                       # Original tags from source
    import_notes: list[str]              # Notes about what was lost/inferred during mapping
```

The intermediate format must:
- Preserve all available source metadata in `raw_metadata`
- Map what it can to ALMA's `ThoughtMetadata` schema
- Track what information was lost or inferred via `import_notes`
- Support linking between records (for Obsidian wikilinks)

### Step 5: Design Importer Plugin Architecture

Design an extensible importer architecture following ALMA's existing patterns:

```python
class MemoryImporter(ABC):
    """Base class for memory importers."""

    @abstractmethod
    def source_name(self) -> str:
        """Return the name of the source format."""
        ...

    @abstractmethod
    def parse(self, source_path: Path) -> Iterator[MigrationRecord]:
        """Parse the source data and yield MigrationRecords."""
        ...

    @abstractmethod
    def validate_source(self, source_path: Path) -> ValidationResult:
        """Validate the source data before import."""
        ...

    def import_to_alma(
        self,
        source_path: Path,
        alma_instance: ALMA,
        on_progress: Callable | None = None,
        on_error: Callable | None = None,
    ) -> ImportResult:
        """Import all records from source into ALMA."""
        ...
```

Design decisions to document:
- Plugin discovery: registration pattern (like storage backends) vs auto-discovery
- Error handling: skip-and-report vs fail-fast
- Progress reporting: callback-based for large imports
- Deduplication: how to detect and handle duplicate imports
- Metadata enrichment: should we run LLM extraction on imported records?

### Step 6: Document Metadata Mapping for Each Source

For each of the 3 sources, document the metadata mapping:

| Source Field | ALMA Field | Mapping Type | Notes |
|-------------|-----------|--------------|-------|
| (example) | type | Inferred | LLM extraction needed |
| (example) | topics | Direct | Map from tags |
| (example) | people | Inferred | Extract from content |
| (example) | created_at | Direct | Timestamp available |

Calculate the metadata preservation rate for each source:
- Direct mappings: 100% preservation
- Inferred mappings: depends on extraction accuracy
- Lost fields: 0% preservation
- Target: >90% overall preservation

## Output

- **Architecture document:** Migration architecture covering intermediate format, plugin pattern, metadata mapping
- **Module skeleton:** `alma/migration/` package with:
  - `alma/migration/__init__.py` -- package init with public API
  - `alma/migration/base.py` -- abstract `MemoryImporter` base class
  - `alma/migration/intermediate.py` -- `MigrationRecord` dataclass
  - `alma/migration/claude_importer.py` -- skeleton for Claude importer
  - `alma/migration/chatgpt_importer.py` -- skeleton for ChatGPT importer
  - `alma/migration/obsidian_importer.py` -- skeleton for Obsidian importer

## Gate

- [ ] Intermediate format handles all 3 source formats (Claude, ChatGPT, Obsidian)
- [ ] Metadata mapping is documented for each source with preservation rates
- [ ] Overall metadata preservation target is >90%
- [ ] Importer plugin architecture is extensible (new importers without modifying core)
- [ ] Architecture follows ALMA patterns (ABC, error handling, async support)
- [ ] Module skeleton is created in `alma/migration/`
- [ ] Deduplication strategy is defined
- [ ] Progress reporting pattern is defined for large imports
