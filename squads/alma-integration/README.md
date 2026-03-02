# ALMA Integration Squad

**Squad ID**: `alma-integration`
**Version**: 1.0.0
**Purpose**: Complete ALMA + Veritas integration with task-first architecture

---

## Overview

This squad completes the integration of:
- **ALMA-Memory**: Adaptive Long-term Memory Agent (5 memory types)
- **Veritas-Framework**: Evidence-based trust verification (5 behaviors)

### What This Squad Does

1. **Enables Cross-Agent Learning**: Agents share memories based on scope rules
2. **Enforces Trust Behaviors**: All claims require verifiable evidence
3. **Provides Memory Persistence**: Learnings persist across sessions
4. **Implements Trust Profiles**: Dynamic trust scores per agent

---

## Task-First Architecture

This squad uses a **task-first** approach where each task has:

```yaml
task:
  id: unique_identifier
  agent: specialized_agent
  inputs: [required_artifacts]
  output: produced_artifact
  gates:
    must_implement: [features]
    evidence_required: [tests, coverage]
```

### Benefits

- Clear progress tracking
- Parallel execution where possible
- Rollback at task boundaries
- Verification gates prevent incomplete work

---

## Squad Agents

| Agent | Role | Specialization |
|--------|------|----------------|
| **@architect-alma** | Architecture | Memory design, scope configuration, MCP integration |
| **@dev-alma** | Implementation | ALMA adapter, Veritas hook, CLI commands |
| **@qa-veritas** | Verification | Trust behavior verification, evidence validation |

---

## Workflow Phases

### Phase 1: Foundation (INT-001, INT-002)
- MCP Integration Layer
- ALMA Memory Adapter

### Phase 2: Trust Framework (INT-003, INT-004)
- Veritas Trust Hook
- Enhanced Context Injection

### Phase 3: Cross-Agent Learning (INT-005)
- Memory Scope Configuration

### Phase 4: Agent Integration (INT-006, INT-007)
- Agent Invoker Updates
- CLI Commands

### Phase 5: Verification (INT-008)
- Integration Testing
- Final Verification

---

## ALMA Memory Types

| Type | Description | Lifecycle |
|-------|-------------|------------|
| **Heuristics** | Successful verified patterns | Long-lived |
| **Outcomes** | Task results for learning | Permanent |
| **Preferences** | User communication settings | Until changed |
| **Domain Knowledge** | Project-specific facts | Medium-lived |
| **Anti-Patterns** | Gotchas to avoid | Until resolved |

---

## Veritas Trust Behaviors

| Behavior | Description | Severity |
|-----------|-------------|------------|
| **Verification Before Claim** | Claims require test evidence | High |
| **Loud Failure** | Errors explicitly reported | Medium |
| **Honest Uncertainty** | Low confidence acknowledged | Low |
| **Paper Trail** | Documentation/artifacts exist | Medium |
| **Diligent Execution** | Tests run before complete | High |

---

## Quick Start

### 1. Activate Architect Agent
```bash
# Design memory architecture
@architect-alma *design-memory-schema --types all
```

### 2. Activate Dev Agent
```bash
# Implement ALMA adapter
@dev-alma *implement-alma-adapter --types all

# Implement Veritas hook
@dev-alma *implement-veritas-hook --behaviors all
```

### 3. Activate QA Agent
```bash
# Verify integration
@qa-veritas *run-integration-tests

# Verify trust behaviors
@qa-veritas *verify-behaviors --agent dev
```

---

## Files Structure

```
squads/alma-integration/
├── squad.yaml                          # Squad configuration
├── README.md                           # This file
├── agents/
│   ├── architect-alma.md                # @architect specialized
│   ├── dev-alma.md                     # @dev specialized
│   └── qa-veritas.md                   # @qa specialized
├── workflows/
│   ├── alma-integration-workflow.yaml   # Main workflow
│   ├── memory-learning-workflow.yaml     # Cross-agent learning
│   └── trust-verification-workflow.yaml # Trust verification
└── config/
    ├── memory-scopes.yaml              # Memory scope rules
    ├── trust-thresholds.yaml          # Trust level thresholds
    └── mental-models.md              # Applied mental models
```

---

## Integration Status

| Phase | Tasks | Status |
|-------|--------|--------|
| INT-001: MCP Integration Layer | 3 tasks | Completed |
| INT-002: ALMA Memory Integration | 4 tasks | Completed |
| INT-003: Veritas Framework Integration | 4 tasks | Completed |
| INT-004: Context Injection Enhancement | 2 tasks | Completed |
| INT-005: Cross-Agent Learning | 3 tasks | In Progress |
| INT-006: Agent Integration | 3 tasks | Pending |
| INT-007: CLI Integration | 3 tasks | Pending |
| INT-008: Testing & Documentation | 4 tasks | Pending |

---

## Configuration

### Memory Scopes
See `config/memory-scopes.yaml` for which agents can learn, share with, and inherit from.

### Trust Thresholds
See `config/trust-thresholds.yaml` for permission thresholds by trust level.

### Mental Models
See `config/mental-models.md` for the design principles applied.

---

## CLI Commands

### Memory Commands
```bash
*alma heuristic "Test happy path first" --context "API testing" --confidence 0.9
*alma outcome task-123 --success --learnings "Tests pass"
*alma query "api testing" --limit 5
*alma stats
```

### Trust Commands
```bash
*trust report dev --verbose
*trust guidance dev
*trust violations dev --limit 10
*trust list
```

---

## Rollback Plan

If issues occur:

1. **Disable MCP**: Edit `~/.aios/mcp/global-config.json` → set `disabled: true`
2. **Restore Gotchas**: Copy `.aios/gotchas.backup.json` to `.aios/gotchas.json`
3. **Revert Code**: `git checkout` modified files
4. **Clear Trust**: Delete `.aios/veritas/profiles.json`

---

## Contributing

When adding tasks to this squad:

1. Follow task-first pattern (inputs/outputs/gates)
2. Add Veritas behavior requirements
3. Specify evidence needed for verification
4. Update workflow YAML files

---

## Related Documentation

- ALMA Adapter: `.aios-core/core/memory/alma-adapter.js`
- Veritas Hook: `.aios-core/core/hooks/veritas-trust-hook.js`
- Scope Config: `.aios-core/core/memory/scope-config.js`
- Integration Summary: `.aios-core/core/INTEGRATION_SUMMARY.md`

---

*ALMA Integration Squad - Task-First Architecture*
*Version: 1.0.0*
