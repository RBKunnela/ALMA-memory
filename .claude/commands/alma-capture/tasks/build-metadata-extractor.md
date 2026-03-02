# Task: Build Metadata Extractor

## Goal

Build the Python module for LLM-powered metadata extraction. This module takes raw thought text and returns structured metadata (type, topics, people, action_items, importance) using a configurable LLM provider.

## Agent

**metadata-extractor** ([agents/metadata-extractor.md](../agents/metadata-extractor.md))

## Requires

- **design-capture-pipeline** (completed) -- the architecture document defines the extraction prompt design, data shapes, and error handling strategy

## Steps

### Step 1: Design Extraction Prompt

Create the LLM prompt that extracts metadata from raw thought text. The prompt must:

- Clearly define the 8 thought types: decision, person_note, insight, meeting, task, idea, reference, lesson
- Specify the JSON output schema with all 5 fields
- Include 3-5 few-shot examples covering different types
- Define the importance scale: 1 (trivial/passing thought), 2 (useful context), 3 (notable/worth remembering), 4 (important decision or insight), 5 (critical/career-defining)
- Handle edge cases: very short inputs, multi-topic inputs, inputs with no people mentioned
- Be token-efficient -- minimize prompt size without sacrificing accuracy

Reference: [templates/metadata-schema-tmpl.md](../templates/metadata-schema-tmpl.md)

### Step 2: Build Configurable LLM Provider Interface

Design a provider interface that allows any LLM to be used for extraction:

```python
class MetadataExtractor:
    """Extract structured metadata from thought text using a configurable LLM."""

    def __init__(self, llm_provider: LLMProvider, prompt_template: str | None = None):
        ...

    def extract(self, content: str) -> ThoughtMetadata:
        """Extract metadata from thought text. Returns validated metadata."""
        ...

    async def async_extract(self, content: str) -> ThoughtMetadata:
        """Async variant of extract."""
        ...
```

The LLM provider interface should follow ALMA's existing Protocol pattern (see `alma/learning/protocols.py`). It must be:
- Not hardcoded to any specific LLM (OpenAI, Anthropic, etc.)
- Configurable via ALMA's YAML config
- Testable with a mock provider

### Step 3: Implement Extraction with JSON Schema Validation

Build the extraction pipeline:

1. Format the prompt with the input content
2. Call the LLM provider
3. Parse the JSON response
4. Validate against the metadata schema:
   - `type` must be one of the 8 valid types
   - `topics` must be a list of strings (non-empty)
   - `people` must be a list of strings (can be empty)
   - `action_items` must be a list of strings (can be empty)
   - `importance` must be an integer 1-5 (clamp if out of range)
5. Handle malformed LLM output:
   - Missing fields: use sensible defaults
   - Invalid JSON: retry once, then return partial extraction
   - Wrong types: coerce where possible, log warnings

### Step 4: Create Test Corpus

Build a test corpus of 20+ examples covering:

- All 8 thought types (at least 2 examples each)
- Short inputs (< 20 words)
- Long inputs (> 200 words)
- Inputs with multiple people
- Inputs with no action items
- Inputs with ambiguous type classification
- Inputs with explicit importance signals ("this is critical", "just a random thought")
- Edge cases: empty topics, single-word inputs, code snippets

Each test example should have:
- Input text
- Expected metadata (human-annotated ground truth)
- Difficulty rating (easy, medium, hard)

### Step 5: Measure Accuracy

Run extraction on the full test corpus and measure:

| Field | Metric | Target |
|-------|--------|--------|
| type | Exact match accuracy | >85% |
| topics | F1 score | >80% |
| people | Recall | >90% |
| action_items | F1 score | >75% |
| importance | Within +/-1 accuracy | >70% |
| **Overall** | **Weighted average** | **>80%** |

### Step 6: Optimize Prompt for Edge Cases

Analyze failures from the test corpus:
- Identify systematic failure patterns
- Adjust prompt with additional examples or clarifications
- Re-run test corpus after each optimization
- Stop when overall accuracy exceeds 80%

## Output

- **Module:** `alma/extraction/metadata.py`
- **Data classes:** `ThoughtMetadata` dataclass in `alma/types.py` (or `alma/extraction/metadata.py`)
- **Tests:** `tests/unit/test_metadata_extraction.py`
- **Test corpus:** `tests/fixtures/metadata_extraction_corpus.json`

## Gate

- [ ] Extracts all 5 metadata fields: type, topics, people, action_items, importance
- [ ] Unit tests pass with >80% overall accuracy on test corpus
- [ ] Works with configurable LLM provider (not hardcoded to one model)
- [ ] Handles malformed LLM output gracefully (no crashes, sensible defaults)
- [ ] Has both sync `extract()` and async `async_extract()` methods
- [ ] JSON schema validation catches invalid outputs
- [ ] Code follows ALMA standards: ruff, type hints, Google docstrings
- [ ] Test corpus has 20+ diverse examples covering all 8 types
