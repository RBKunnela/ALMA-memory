# ALMA Developer Experience (DX) & API Design Specification

**Version**: 1.0
**Date**: 2026-01-28
**Audit Type**: Developer Experience (DX) Focus
**Target Audience**: API Consumers, SDK Developers, Integration Engineers

---

## Executive Summary

This document provides a comprehensive audit of ALMA's public API surface, MCP tools interface, configuration UX, documentation quality, and overall developer experience. Since ALMA is a backend library without a frontend UI, this specification focuses on the API design and developer ergonomics.

**Key Findings:**
- Strong public API surface with well-designed dataclasses
- Good MCP integration for Claude Code compatibility
- Configuration UX needs improvement (missing template file)
- Documentation is comprehensive but lacks practical examples
- Type hints are present but inconsistent across modules
- Error handling could be more developer-friendly

---

## 1. Public API Surface Analysis

### 1.1 Module Export Structure

**File**: `/Users/friendlyaifi/Documents/GitHub/ALMA-memory/alma/__init__.py`

The `__init__.py` exports 48 symbols across 8 logical groupings:

| Category | Exports | Lines |
|----------|---------|-------|
| Core | `ALMA`, `MemorySlice`, `MemoryScope`, + 4 types | 19-29 |
| Harness Pattern | `Setting`, `Context`, `Agent`, + 6 types | 31-48 |
| Progress Tracking | `WorkItem`, `ProgressTracker`, + 3 types | 50-57 |
| Session Management | `SessionHandoff`, `SessionManager`, + 2 types | 59-65 |
| Domain Factory | `DomainMemoryFactory`, + 6 types/functions | 67-77 |
| Session Initializer | `SessionInitializer`, + 3 types | 79-85 |
| Confidence Engine | `ConfidenceEngine`, + 3 signal types | 87-93 |

**Strengths:**
- Logical grouping with clear phase comments (`Phase 10`, `Phase 11`, `Phase 12`)
- All symbols properly listed in `__all__` for IDE autocomplete
- Imports are from submodules, not star imports

**Issues Identified:**

1. **Redundant Harness Pattern Exports** (lines 42-48):
   The `create_harness` function is exported alongside domain-specific classes (`CodingDomain`, `ResearchDomain`, etc.), but there's no clear documentation on when to use one vs. the other.

2. **Missing Version Deprecation Hints**:
   Symbols like `create_harness` should indicate if they're the preferred API or legacy.

3. **Inconsistent Naming Convention**:
   ```python
   # Inconsistent: Some use "get_" prefix, some don't
   get_coding_schema()    # Function with get_ prefix
   get_research_schema()  # Function with get_ prefix
   DomainMemoryFactory    # Class without create_ in name
   create_harness         # Function with create_ prefix
   ```

**Recommendation**: Standardize on `get_*` for retrieval and `create_*` for factory functions.

---

### 1.2 Core ALMA Class API

**File**: `/Users/friendlyaifi/Documents/GitHub/ALMA-memory/alma/core.py`

The `ALMA` class is the primary interface with 7 public methods:

| Method | Parameters | Return Type | Line |
|--------|------------|-------------|------|
| `from_config()` | `config_path: str` | `ALMA` | 51-108 |
| `retrieve()` | `task, agent, user_id?, top_k?` | `MemorySlice` | 128-158 |
| `learn()` | `agent, task, outcome, strategy_used, ...` | `bool` | 160-206 |
| `add_user_preference()` | `user_id, category, preference, source?` | `UserPreference` | 208-237 |
| `add_domain_knowledge()` | `agent, domain, fact, source?` | `Optional[DomainKnowledge]` | 239-278 |
| `forget()` | `agent?, older_than_days?, below_confidence?` | `int` | 280-310 |
| `get_stats()` | `agent?` | `Dict[str, Any]` | 312-325 |

**DX Analysis:**

1. **Inconsistent Return Types** (DX Pain Point):
   ```python
   # Returns UserPreference directly
   def add_user_preference(...) -> UserPreference:

   # Returns Optional - may be None on scope violation
   def add_domain_knowledge(...) -> Optional[DomainKnowledge]:

   # Returns bool for success/failure
   def learn(...) -> bool:
   ```

   **Issue**: Developers must handle different return semantics for similar operations.

   **Recommendation**: Use a `Result` type or consistently return the created object (with exceptions for failures).

2. **String-Typed Outcome Parameter** (line 164):
   ```python
   outcome: str,  # "success" or "failure"
   ```

   **Issue**: Magic strings instead of enum. Developer must know valid values.

   **Recommendation**: Accept `Literal["success", "failure"]` or create `OutcomeStatus` enum.

3. **Missing Async API**:
   All methods are synchronous, but storage backends may benefit from async I/O. No `async def` variants exist.

4. **Constructor Complexity** (lines 37-49):
   ```python
   def __init__(
       self,
       storage: StorageBackend,
       retrieval_engine: RetrievalEngine,
       learning_protocol: LearningProtocol,
       scopes: Dict[str, MemoryScope],
       project_id: str,
   ):
   ```

   **Issue**: Requires 5 dependencies. Most users should use `from_config()` instead.

   **Recommendation**: Make `__init__` private or add builder pattern.

---

### 1.3 Type Definitions Quality

**File**: `/Users/friendlyaifi/Documents/GitHub/ALMA-memory/alma/types.py`

**Strengths:**
- Uses `dataclass` for all types (good IDE support)
- Includes docstrings with examples (lines 43-49, 71-77, etc.)
- `MemorySlice.to_prompt()` method is very useful (lines 167-205)
- Computed properties like `success_rate` (lines 63-68)

**Issues:**

1. **Deprecated `datetime.utcnow()`** (lines 88, 106, 124, 144):
   ```python
   timestamp: datetime = field(default_factory=datetime.utcnow)
   ```

   **Issue**: `datetime.utcnow()` is deprecated since Python 3.12.

   **Recommendation**: Use `datetime.now(timezone.utc)`.

2. **Token Estimation is Rough** (lines 201-203):
   ```python
   # Basic token estimation (rough: 1 token ~ 4 chars)
   if len(result) > max_tokens * 4:
   ```

   **Issue**: This estimation is inaccurate for non-ASCII text and modern tokenizers.

   **Recommendation**: Use tiktoken or similar for accurate counting.

3. **Missing `__str__` or `__repr__`** on dataclasses:
   Debugging would be easier with readable string representations.

---

## 2. MCP Tools Interface Review

**Files**:
- `/Users/friendlyaifi/Documents/GitHub/ALMA-memory/alma/mcp/__init__.py`
- `/Users/friendlyaifi/Documents/GitHub/ALMA-memory/alma/mcp/tools.py`
- `/Users/friendlyaifi/Documents/GitHub/ALMA-memory/alma/mcp/server.py`

### 2.1 Available MCP Tools

| Tool | Description | Required Params | Optional Params |
|------|-------------|-----------------|-----------------|
| `alma_retrieve` | Get memories for task | `task`, `agent` | `user_id`, `top_k` |
| `alma_learn` | Record task outcome | `agent`, `task`, `outcome`, `strategy_used` | `task_type`, `duration_ms`, `error_message`, `feedback` |
| `alma_add_preference` | Add user preference | `user_id`, `category`, `preference` | `source` |
| `alma_add_knowledge` | Add domain knowledge | `agent`, `domain`, `fact` | `source` |
| `alma_forget` | Prune stale memories | (none) | `agent`, `older_than_days`, `below_confidence` |
| `alma_stats` | Get statistics | (none) | `agent` |
| `alma_health` | Health check | (none) | (none) |

### 2.2 MCP Tool Schema Quality

**Strengths:**
- JSON Schema definitions are complete (lines 67-238 in server.py)
- Descriptions are clear and actionable
- Default values documented

**Issues:**

1. **Enum Constraint on `outcome`** (lines 111-115 in server.py):
   ```json
   "outcome": {
       "type": "string",
       "enum": ["success", "failure"],
       "description": "Whether the task succeeded or failed"
   }
   ```

   **Good**: Properly constrained. This should be mirrored in the Python API.

2. **Missing Input Validation** (tools.py lines 100-119):
   ```python
   def alma_retrieve(
       alma: ALMA,
       task: str,
       agent: str,
       ...
   ) -> Dict[str, Any]:
       try:
           memories = alma.retrieve(...)
   ```

   **Issue**: No validation of empty strings. If `task=""` is passed, behavior is undefined.

   **Recommendation**: Add input validation before calling ALMA methods.

3. **Error Response Format** (tools.py):
   ```python
   return {
       "success": False,
       "error": str(e),
   }
   ```

   **Issue**: Only error message is returned, not error type or suggestions for remediation.

   **Recommendation**: Include `error_type` and `suggestion` fields.

### 2.3 MCP Server Configuration

**File**: `/Users/friendlyaifi/Documents/GitHub/ALMA-memory/alma/mcp/__init__.py` (lines 7-22)

```python
"""
Usage:
    # stdio mode (for Claude Code integration)
    python -m alma.mcp --config .alma/config.yaml

    # HTTP mode (for remote access)
    python -m alma.mcp --http --port 8765
"""
```

**Strength**: Clear usage documentation in module docstring.

**Issue**: The `--http` mode requires `aiohttp` but this isn't clearly documented as optional.

---

## 3. Configuration UX Assessment

### 3.1 Configuration Files

**Expected Files:**
- `.alma/config.yaml` - Main configuration
- `.env` - Environment variables
- `.env.template` - Template for environment variables

**Findings:**

1. **Missing `config.yaml.template`**:
   The file `/Users/friendlyaifi/Documents/GitHub/ALMA-memory/config.yaml.template` does not exist (file read returned error), despite being referenced in the git commit history (`cdb3c4f feat: Add config.yaml.template`).

   **Impact**: New developers have no template to copy.

   **Recommendation**: Ensure `config.yaml.template` exists with all options documented.

2. **`.env.template` is Excellent** (lines 1-103):
   - Comprehensive documentation for each variable
   - Clear sections for each provider
   - Includes dimension reference table
   - Explains pros/cons of each option

   **This is a DX best practice to emulate elsewhere.**

### 3.2 Configuration Loading

**File**: `/Users/friendlyaifi/Documents/GitHub/ALMA-memory/alma/config/loader.py`

**Strengths:**
- Supports `${ENV_VAR}` expansion (line 91)
- Supports `${KEYVAULT:secret-name}` for Azure Key Vault (line 88)
- Falls back to defaults if config not found (lines 41-43)

**Issues:**

1. **Silent Failure on Missing Env Vars** (lines 93-94):
   ```python
   if env_value is None:
       logger.warning(f"Environment variable {ref} not set")
       return match.group(0)  # Keep original if not found
   ```

   **Issue**: Returns the unexpanded `${VAR}` string instead of failing fast.

   **Recommendation**: Raise `ConfigurationError` for missing required variables.

2. **No Schema Validation**:
   Config is loaded as raw dict with no validation of required keys or value types.

   **Recommendation**: Use Pydantic `BaseSettings` or validate with JSON Schema.

3. **Default Config is Minimal** (lines 136-143):
   ```python
   return {
       "project_id": "default",
       "storage": "file",
       "embedding_provider": "local",
       "agents": {},
   }
   ```

   **Issue**: No agents configured by default means `retrieve()` will always warn.

---

## 4. Documentation Quality Audit

### 4.1 README.md Analysis

**File**: `/Users/friendlyaifi/Documents/GitHub/ALMA-memory/README.md` (593 lines)

**Strengths:**
- Professional badges (PyPI, Python version, License)
- ASCII architecture diagrams (lines 20-31, 284-293, 427-457)
- Feature comparison tables (lines 77-84, 86-93)
- Copy-paste code examples throughout
- API Reference section (lines 476-580)

**Issues:**

1. **Quick Start Missing Error Handling** (lines 39-70):
   ```python
   # Initialize
   alma = ALMA.from_config(".alma/config.yaml")
   ```

   **Issue**: What happens if config doesn't exist? No try/catch shown.

2. **Outdated Architecture Diagram** (line 429):
   ```
   │                        ALMA v0.3.0                              │
   ```

   **Issue**: Version shown is v0.3.0 but `__version__ = "0.4.0"` in `__init__.py` line 17.

3. **Missing Troubleshooting Section**:
   Common errors and solutions are not documented.

4. **No Migration Guide**:
   Breaking changes between versions not documented.

### 4.2 Additional Documentation

**Found:**
- `/Users/friendlyaifi/Documents/GitHub/ALMA-memory/docs/ROADMAP_TO_FUNDING.md` - Business strategy (not developer docs)
- `/Users/friendlyaifi/Documents/GitHub/ALMA-memory/docs/architecture/PRD.md` - Product requirements
- `/Users/friendlyaifi/Documents/GitHub/ALMA-memory/docs/prd/ALMA_EXPANSION_PRD.md` - Expansion plans

**Missing:**
- API Reference as separate doc
- Integration guides for specific frameworks
- Performance tuning guide
- Storage backend comparison
- Example projects directory

---

## 5. Error Handling & Messages Review

### 5.1 Error Message Quality

**core.py Error Handling:**

```python
# Line 148-149
if agent not in self.scopes:
    logger.warning(f"Agent '{agent}' has no defined scope, using defaults")
```

**Issue**: Warning only, no exception. Behavior continues with undefined results.

```python
# Lines 261-263
if scope and not scope.is_allowed(domain):
    logger.warning(f"Agent '{agent}' not allowed to learn in domain '{domain}'")
    return None
```

**Issue**: Returns `None` instead of raising an exception. Caller must check for `None`.

### 5.2 Exception Hierarchy

**Finding**: No custom exception classes defined.

**Recommendation**: Create exception hierarchy:
```python
class ALMAError(Exception): pass
class ConfigurationError(ALMAError): pass
class ScopeViolationError(ALMAError): pass
class StorageError(ALMAError): pass
class RetrievalError(ALMAError): pass
```

### 5.3 MCP Error Responses

**tools.py** (lines 114-119):
```python
except Exception as e:
    logger.exception(f"Error in alma_retrieve: {e}")
    return {
        "success": False,
        "error": str(e),
    }
```

**Issue**: Generic `Exception` catch loses stack trace for debugging. Error type not included.

---

## 6. Type Hints & IDE Support

### 6.1 Type Hint Coverage

| File | Full Hints | Partial | Missing |
|------|------------|---------|---------|
| `alma/__init__.py` | Yes | - | - |
| `alma/core.py` | Yes | - | - |
| `alma/types.py` | Yes | - | - |
| `alma/mcp/tools.py` | Yes | - | - |
| `alma/config/loader.py` | Yes | - | `_expand_config` generic |

### 6.2 IDE Support Features

**Present:**
- `__all__` exports for autocomplete
- Docstrings on public methods
- Type hints on parameters and returns

**Missing:**
- `py.typed` marker file for PEP 561 compliance
- Stub files (`.pyi`) for complex types
- `typing.overload` for methods with variable returns

**Recommendation**: Add `alma/py.typed` empty marker file to indicate the package supports type checking.

---

## 7. Getting Started Experience

### 7.1 Installation

**From README** (line 36):
```bash
pip install alma-memory
```

**Issues:**
1. Package name `alma-memory` vs import name `alma` may confuse developers
2. Optional dependencies not explained in quick start
3. No `pip install alma-memory[all]` mentioned for full features

### 7.2 First-Run Experience

**Ideal Flow:**
1. `pip install alma-memory`
2. Copy template config
3. Run minimal example

**Current Flow:**
1. `pip install alma-memory`
2. Create `.alma/config.yaml` manually (no template to copy!)
3. Figure out required keys from README

**Recommendation**: Add CLI command `alma init` to generate config.

### 7.3 Minimal Working Example

**README Quick Start** (lines 42-70) is good but could be simpler:

```python
# SUGGESTED: Simpler 3-line example
from alma import ALMA
alma = ALMA.from_config(".alma/config.yaml")
memories = alma.retrieve("Test login form", agent="default")
```

---

## 8. Common Use Case Patterns

### 8.1 Pattern: Memory Injection into Prompts

**Documented** (README lines 54-61):
```python
memories = alma.retrieve(task="...", agent="helena")
prompt = f"""
## Knowledge from Past Runs
{memories.to_prompt()}
"""
```

**Strength**: `to_prompt()` method is well-designed for this use case.

### 8.2 Pattern: Auto-Learning from Conversations

**File**: `/Users/friendlyaifi/Documents/GitHub/ALMA-memory/alma/extraction/auto_learner.py`

**Usage** (lines 35-46):
```python
auto_learner = AutoLearner(alma)
results = auto_learner.learn_from_conversation(
    messages=[...],
    agent="helena",
)
```

**Issue**: `AutoLearner` not exported from main `alma` package.

**Recommendation**: Add to `__all__` in `alma/__init__.py`.

### 8.3 Pattern: Multi-Agent Memory Scoping

**Documented** in README (lines 313-336):
```yaml
agents:
  helena:
    can_learn: [testing_strategies]
    cannot_learn: [backend_logic]
```

**Strength**: Clear configuration for agent boundaries.

**Issue**: No runtime API to query agent scopes.

---

## 9. API Consistency Issues

### 9.1 Parameter Naming Inconsistencies

| Method | Parameter | Issue |
|--------|-----------|-------|
| `retrieve()` | `top_k` | Underscore style |
| `learn()` | `strategy_used` | Past tense implies already happened |
| `add_user_preference()` | `user_id` | Consistent |
| `add_domain_knowledge()` | `agent` | Different from `add_user_preference` |

### 9.2 Return Type Inconsistencies

```python
def add_user_preference(...) -> UserPreference:    # Always returns
def add_domain_knowledge(...) -> Optional[...]:    # May return None
def learn(...) -> bool:                            # Returns status
```

**Recommendation**: Standardize to always return the created object or raise exception.

### 9.3 Async/Sync Inconsistency

- Core ALMA class: All sync
- MCP Server: Async (`async def handle_request`)
- Storage backends: Sync

**Recommendation**: Provide `ALMMA.aio` namespace for async methods or async context manager.

---

## 10. Developer Pain Points

### 10.1 Configuration Discovery

**Pain**: Developers don't know what config options are available.

**Evidence**: No `config.yaml.template` file exists despite being in commit history.

**Solution**: Generate config with CLI or provide comprehensive template.

### 10.2 Debugging Memory Retrieval

**Pain**: Hard to understand why certain memories are returned.

**Evidence**: `retrieve()` returns `MemorySlice` with no explanation of ranking/filtering.

**Solution**: Add `explain: bool` parameter to return ranking scores.

### 10.3 Testing Integration

**Pain**: No test utilities provided.

**Evidence**: No `alma.testing` module with mock backends.

**Solution**: Add `MockStorage`, `MockEmbedder` for unit tests.

### 10.4 Version Mismatches

**Pain**: README says v0.3.0, code says v0.4.0.

**Evidence**:
- `alma/__init__.py` line 17: `__version__ = "0.4.0"`
- `README.md` line 429: `ALMA v0.3.0`

**Solution**: Use `__version__` in documentation generation.

---

## 11. Recommendations for DX Improvements

### 11.1 High Priority

| ID | Recommendation | Effort | Impact |
|----|----------------|--------|--------|
| DX-1 | Create `config.yaml.template` with all options documented | Low | High |
| DX-2 | Add custom exception hierarchy | Low | Medium |
| DX-3 | Add `alma init` CLI command for project setup | Medium | High |
| DX-4 | Export `AutoLearner` from main package | Low | Medium |
| DX-5 | Fix version mismatch in README | Low | Low |

### 11.2 Medium Priority

| ID | Recommendation | Effort | Impact |
|----|----------------|--------|--------|
| DX-6 | Add `py.typed` marker for PEP 561 | Low | Medium |
| DX-7 | Create `alma.testing` module with mocks | Medium | High |
| DX-8 | Add troubleshooting section to README | Low | Medium |
| DX-9 | Use `datetime.now(timezone.utc)` instead of deprecated `utcnow()` | Low | Low |
| DX-10 | Add input validation to MCP tools | Medium | Medium |

### 11.3 Low Priority / Future

| ID | Recommendation | Effort | Impact |
|----|----------------|--------|--------|
| DX-11 | Add async API variants | High | Medium |
| DX-12 | Create example projects directory | Medium | Medium |
| DX-13 | Add `explain` parameter to `retrieve()` | Medium | Low |
| DX-14 | Use Pydantic for config validation | Medium | Medium |
| DX-15 | Standardize naming conventions across API | Medium | Low |

---

## 12. Appendix: File References

### Key Files Analyzed

| File | Purpose | Lines |
|------|---------|-------|
| `/Users/friendlyaifi/Documents/GitHub/ALMA-memory/alma/__init__.py` | Public API exports | 151 |
| `/Users/friendlyaifi/Documents/GitHub/ALMA-memory/alma/core.py` | Main ALMA class | 326 |
| `/Users/friendlyaifi/Documents/GitHub/ALMA-memory/alma/types.py` | Data structures | 217 |
| `/Users/friendlyaifi/Documents/GitHub/ALMA-memory/alma/mcp/tools.py` | MCP tool functions | 375 |
| `/Users/friendlyaifi/Documents/GitHub/ALMA-memory/alma/mcp/server.py` | MCP server | 534 |
| `/Users/friendlyaifi/Documents/GitHub/ALMA-memory/alma/config/loader.py` | Config loading | 157 |
| `/Users/friendlyaifi/Documents/GitHub/ALMA-memory/README.md` | Main documentation | 593 |
| `/Users/friendlyaifi/Documents/GitHub/ALMA-memory/.env.template` | Env var template | 103 |
| `/Users/friendlyaifi/Documents/GitHub/ALMA-memory/pyproject.toml` | Package metadata | 108 |

### Storage Backend Interface

**File**: `/Users/friendlyaifi/Documents/GitHub/ALMA-memory/alma/storage/base.py`

The `StorageBackend` abstract class defines 21 methods that all backends must implement, including:
- `save_*` methods for each memory type
- `get_*` methods with filtering options
- `update_*` and `delete_*` methods
- `from_config()` class method for factory pattern

---

## 13. Conclusion

ALMA demonstrates a well-architected API for agent memory management. The public surface is thoughtfully designed with clear separation between core operations (retrieve, learn) and auxiliary features (progress tracking, session management).

**Key Strengths:**
- Comprehensive type system with dataclasses
- MCP integration is production-ready
- `.env.template` is exemplary documentation
- Multiple storage backend support

**Key Areas for Improvement:**
- Configuration setup experience (missing template)
- Error handling (silent failures, generic exceptions)
- Consistency (return types, naming conventions)
- Testing utilities (no mock backends provided)

The recommendations in this document, if implemented, would significantly improve the developer experience for teams adopting ALMA in their AI agent architectures.

---

*This specification was generated as part of a DX audit for ALMA-memory v0.4.0.*
