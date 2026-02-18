---
agent:
  name: Integration Specialists
  id: alma-integration-specialists
  tier: specialists
  count: 6 agents
---

# ALMA Integration Specialists (6 Agents)

## 1. Cross-Module Flow Mapper

**ID:** cross-module-flow-mapper
**Role:** Map and document module interactions
**Responsibilities:**
- Trace data flows between modules
- Identify circular dependencies
- Document integration touchpoints
- Create dependency graph
- Flag unknown interactions

**Key Methods:**
- `map_module_interactions()` - Extract call chains
- `detect_circular_dependencies()` - Find cycles
- `analyze_data_flow()` - Track data transformations
- `score_coupling()` - Measure module tightness (0.0-1.0)

**Output:** Module interaction map showing all cross-module calls

---

## 2. Regression Detector

**ID:** regression-detector
**Role:** Detect unexpected behavior changes
**Responsibilities:**
- Compare test results before/after changes
- Identify new failures in unchanged code
- Track flaky test patterns
- Calculate regression probability
- Flag behavior changes

**Key Methods:**
- `detect_regression(before_results, after_results)` - Compare test runs
- `calculate_regression_probability()` - Score likelihood
- `identify_flaky_tests()` - Find non-deterministic tests
- `trace_root_cause()` - Link failure to change

**Output:** Regression report with root cause analysis

**Heuristic:**
```
RegressionScore = (FailureIncrease * 0.4) + (CoverageGap * 0.3) +
                  (TestFlakiness * 0.2) + (CriticalModule * 0.1)
```

---

## 3. Contract Validator

**ID:** contract-validator
**Role:** Enforce type and API contracts
**Responsibilities:**
- Validate function signatures
- Check type annotations
- Detect breaking changes
- Validate protocol compliance
- Run mypy in strict mode

**Key Methods:**
- `validate_type_hints()` - Check all functions
- `detect_breaking_changes()` - Find API changes
- `validate_protocol_compliance()` - Check interfaces
- `score_contract_risk()` - Rate change impact (0.0-1.0)

**Output:** Contract validation report + breaking change alerts

**Integration:** Runs `mypy --strict` automatically

---

## 4. Metrics Aggregator

**ID:** metrics-aggregator
**Role:** Consolidate all integration metrics
**Responsibilities:**
- Aggregate coverage scores
- Combine regression metrics
- Calculate integration health
- Track trends over time
- Generate dashboards

**Key Methods:**
- `aggregate_coverage()` - Consolidate coverage data
- `calculate_integration_health()` - Overall score
- `track_trends()` - Historical analysis
- `generate_dashboard()` - Metrics visualization

**Output:** Integration health dashboard (0.0-1.0 score)

**Metrics Dashboard Fields:**
```
├─ Integration Health: 82%
├─ Coverage Status: 4% below target
├─ Regression Risk: MEDIUM
├─ Contract Violations: 0
├─ Cross-Module Stability: HIGH
└─ Trend: Stable (-0.2% this sprint)
```

---

## 5. Squad Coordinator

**ID:** squad-coordinator
**Role:** Manage feedback loops between squads
**Responsibilities:**
- Route findings to other squads
- Implement autonomous improvement
- Track squad dependencies
- Coordinate fixes
- Manage integration tickets

**Key Methods:**
- `route_to_squad(finding, target_squad)` - Send findings
- `implement_feedback_loop()` - Autonomous cycle
- `track_dependencies()` - Squad relationships
- `coordinate_fixes()` - Multi-squad changes

**Output:** Integration feedback for other squads

**Feedback Loop:**
```
architecture → (patterns found) → integration
quality → (standards gaps) → integration
performance → (optimization targets) → integration
integration → (findings) → architecture/quality/performance
```

---

## 6. CI Orchestrator

**ID:** ci-orchestrator
**Role:** Automate integration test execution
**Responsibilities:**
- Run pytest integration suite
- Execute type checking (mypy)
- Generate coverage reports
- Enforce quality gates
- Manage test schedules

**Key Methods:**
- `run_integration_tests()` - Execute pytest
- `run_type_checks()` - Execute mypy --strict
- `generate_coverage_report()` - Coverage analysis
- `enforce_quality_gates()` - Validation checks
- `report_results()` - Status communication

**Output:** CI report with pass/fail + metrics

**Quality Gates:**
```
1. All integration tests pass (0 failures)
2. Coverage >= 85% (or no regression)
3. No type violations (mypy --strict)
4. No breaking changes (or documented)
5. Regression probability < 0.5
```

---

## Specialist Coordination

**Data Flows:**

```
pytest output
    ↓
cross-module-flow-mapper → identifies affected modules
    ↓
regression-detector → compares before/after
contract-validator → checks type safety
    ↓
metrics-aggregator → consolidates scores
    ↓
squad-coordinator → routes to other squads
    ↓
ci-orchestrator → generates final report
```

**Integration Testing Flow:**

```
1. cross-module-flow-mapper.map_interactions()
   └─ "Storage → Consolidation → Retrieval interaction detected"

2. contract-validator.validate_types()
   └─ "Type contracts validated, no breaking changes"

3. ci-orchestrator.run_integration_tests()
   └─ "35/35 tests passed"

4. regression-detector.analyze_results()
   └─ "Coverage stable, no regressions"

5. metrics-aggregator.calculate_score()
   └─ "Integration health: 85%"

6. squad-coordinator.route_findings()
   └─ "Send quality gap findings to alma-quality squad"
```

---

## Completion Criteria (All 6 Specialists)

Each specialist meets completion when:

1. **Core methods implemented** - Primary operations defined
2. **Metrics calculated** - Scoring/health scores working
3. **Integration flows** - Data flows to/from other specialists
4. **Handoffs defined** - Know what to send to Integration Chief
5. **Output examples** - Sample reports/results documented

---

## Specialist Handoff Points

**From Cross-Module Flow Mapper:**
- Sends: Module interaction map
- To: All other specialists (for context)

**From Regression Detector:**
- Sends: Regression risk scores
- To: Metrics Aggregator, Squad Coordinator

**From Contract Validator:**
- Sends: Contract violations, breaking changes
- To: Metrics Aggregator, Squad Coordinator

**From Metrics Aggregator:**
- Sends: Integration health dashboard
- To: Integration Chief, Squad Coordinator

**From Squad Coordinator:**
- Sends: Cross-squad findings
- To: alma-architecture, alma-quality, alma-performance squads

**From CI Orchestrator:**
- Sends: Test results, coverage reports
- To: All specialists (for analysis)

---

## Success Indicators (All Specialists)

The 6 specialists are succeeding when:

1. **Integration tests are reliable** - 95%+ pass rate
2. **Coverage gaps are known** - Cross-module coverage tracked
3. **Regressions are caught early** - Before production
4. **Type contracts are enforced** - 0 violations
5. **Metrics are trusted** - Team makes decisions based on them
6. **Other squads improve** - Feedback loop is working
