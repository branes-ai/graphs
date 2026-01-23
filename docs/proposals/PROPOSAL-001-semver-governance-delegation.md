# PROPOSAL-001: SEMVER, Governance, and Task Delegation Framework

**Status**: Draft
**Author**: Claude Opus 4.5 (AI Assistant)
**Date**: 2026-01-23
**Review Requested From**: Project Lead

---

## Executive Summary

This proposal addresses three interconnected needs:

1. **Semantic Versioning (SEMVER)** - Establish version numbering aligned with roadmap milestones
2. **Task Delegation Framework** - Enable parallel work streams across multiple resources (human and AI)
3. **AI Governance & Accountability** - Document decision rationale for audit, review, and accountability

These frameworks are essential for:
- Communicating stability and compatibility to users
- Scaling development across multiple contributors
- Maintaining accountability for AI-assisted code changes
- Establishing trust in automated contributions

---

## Part 1: Semantic Versioning Strategy

### 1.1 Version Numbering Scheme

```
MAJOR.MINOR.PATCH[-PRERELEASE][+BUILD]

Examples:
  0.9.0-alpha.1    Pre-release before v1.0
  0.9.0-beta.1     Beta testing phase
  0.9.0-rc.1       Release candidate
  1.0.0            First stable release
  1.1.0            New features, backward compatible
  1.1.1            Bug fixes only
  2.0.0            Breaking changes
```

### 1.2 Milestone-to-Version Mapping

| Milestone | Version | Release Type | Key Deliverables |
|-----------|---------|--------------|------------------|
| M1: Foundation Consolidation | 0.8.0 | Alpha | Package structure, confidence tracking |
| M2: Calibration Infrastructure | 0.9.0-alpha | Alpha | Auto-calibration, registry sync |
| M3: Estimation Accuracy | 0.9.0-beta | Beta | <15% MAPE latency, <20% MAPE energy |
| M4: Multi-Platform Support | 0.9.0-rc | RC | 30+ calibrated platforms |
| M5: Production Hardening | 1.0.0 | Stable | Full test coverage, documentation |
| M6: Advanced Estimation | 1.1.0 | Feature | Thermal modeling, power states |
| M7: Integration APIs | 1.2.0 | Feature | REST API, language bindings |
| M8: Ecosystem | 2.0.0 | Major | Plugin system, breaking changes OK |

### 1.3 Version 1.0.0 Release Criteria

**Functional Requirements:**
- [ ] All estimation modules return confidence levels
- [ ] Calibration profiles for 30+ hardware platforms
- [ ] Latency estimates within 15% MAPE (Mean Absolute Percentage Error)
- [ ] Energy estimates within 20% MAPE
- [ ] Memory estimates within 10% of actual

**Quality Gates:**
- [ ] 100% of public API has docstrings
- [ ] Test coverage >= 80% for core modules
- [ ] Zero known critical/high severity bugs
- [ ] All deprecation shims documented with removal timeline
- [ ] Migration guide from 0.x to 1.0

**Documentation Requirements:**
- [ ] Complete API reference
- [ ] Getting started guide
- [ ] Hardware calibration guide
- [ ] Architecture decision records (ADRs) for major decisions

### 1.4 Deprecation Policy

```
Version   Action
-------   ------
1.0.0     Deprecation warning added (ir/, analysis/ shims)
1.1.0     Deprecation warning remains
1.2.0     Deprecation warning remains, docs updated
2.0.0     Deprecated code removed
```

### 1.5 CHANGELOG Format

```markdown
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- New feature description

### Changed
- Modification description

### Deprecated
- Soon-to-be removed feature

### Removed
- Removed feature

### Fixed
- Bug fix description

### Security
- Security fix description

## [0.8.0] - 2026-01-23

### Added
- Package structure consolidation (core/, estimation/, calibration/, benchmarks/, frontends/)
- Confidence tracking for all estimates (ConfidenceLevel, EstimationConfidence)
- Deprecation shims for backward compatibility (ir/ -> core/, analysis/ -> estimation/)

### Removed
- Duplicate files from ir/ and analysis/ directories
- Deprecated hw/ package
```

---

## Part 2: Task Delegation Framework

### 2.1 Purpose

Enable multiple resources (developers, AI assistants, contractors) to work in parallel on well-defined tasks with clear interfaces and acceptance criteria.

### 2.2 Task Specification Format

Each delegated task MUST include:

```yaml
# task-spec.yaml
task_id: "TASK-2026-001"
title: "Implement TPU v5 calibration profile"
version: "1.0"

# Ownership
assignee: "@developer-handle or AI-Assistant-ID"
reviewer: "@lead-developer"
created_by: "@requester"
created_at: "2026-01-23T10:00:00Z"

# Scope Definition
objective: |
  Create calibration profile for Google TPU v5 hardware
  to enable accurate latency/energy estimation.

deliverables:
  - path: "src/graphs/calibration/profiles/tpu_v5.json"
    type: "calibration_profile"
    schema: "CalibrationProfile v2"
  - path: "src/graphs/hardware/mappers/accelerators/tpu.py"
    type: "code_modification"
    description: "Add create_tpu_v5_mapper() factory function"
  - path: "tests/hardware/test_tpu_v5.py"
    type: "test"
    coverage_target: "90%"

# Interface Contracts
dependencies:
  - "CalibrationProfile schema from calibration/schema.py"
  - "HardwareMapper base class from hardware/mappers/base.py"

interfaces_provided:
  - name: "create_tpu_v5_mapper"
    signature: "() -> HardwareMapper"
    module: "graphs.hardware.mappers.accelerators.tpu"

interfaces_consumed:
  - name: "HardwareMapper"
    module: "graphs.hardware.mappers.base"
  - name: "CalibrationProfile"
    module: "graphs.calibration.schema"

# Acceptance Criteria
acceptance_criteria:
  - "Mapper passes all existing TPU test patterns"
  - "Calibration profile validated against schema"
  - "Latency estimates within 20% of TPU v4 baseline (until real calibration)"
  - "All new code has type hints"
  - "No new deprecation warnings introduced"

# Constraints
constraints:
  - "Must not modify public API signatures"
  - "Must maintain backward compatibility"
  - "Must not add new dependencies"

# Testing Requirements
test_requirements:
  unit_tests: true
  integration_tests: true
  validation_tests: "validation/hardware/test_tpu_v5.py"

# Estimated Effort
estimated_effort: "4 hours"
priority: "medium"
milestone: "M4"
target_version: "0.9.0-rc"
```

### 2.3 Task States

```
DRAFT -> READY -> ASSIGNED -> IN_PROGRESS -> REVIEW -> APPROVED -> MERGED
                                    |            |
                                    v            v
                              BLOCKED      CHANGES_REQUESTED
```

### 2.4 Task Registry

Maintain a central registry of tasks:

```
docs/tasks/
  index.md              # Task index with status
  templates/
    task-spec.yaml      # Template for new tasks
    code-task.yaml      # Code implementation task
    calibration-task.yaml  # Hardware calibration task
    documentation-task.yaml  # Documentation task
  active/
    TASK-2026-001.yaml
    TASK-2026-002.yaml
  completed/
    TASK-2025-xxx.yaml
  blocked/
    TASK-2026-003.yaml
```

### 2.5 Interface Contract Validation

Before merging any task:

1. **Schema Validation**: All data files validate against declared schemas
2. **Type Checking**: `mypy --strict` passes on modified files
3. **API Stability**: Public API signatures unchanged (unless explicitly allowed)
4. **Test Coverage**: Coverage targets met for deliverables
5. **Integration**: Dependent tasks notified of interface changes

---

## Part 3: AI Governance and Accountability Framework

### 3.1 Purpose

Establish transparency, traceability, and accountability for AI-assisted code contributions. Enable:
- Audit trails for compliance
- Decision review and override
- Learning from AI decisions (good and bad)
- Trust building through transparency

### 3.2 Decision Documentation Requirements

Every AI-generated code change MUST include a **Decision Record** documenting:

#### 3.2.1 Decision Record Format

```yaml
# .claude/decisions/DECISION-2026-01-23-001.yaml
decision_id: "DECISION-2026-01-23-001"
timestamp: "2026-01-23T14:30:00Z"
ai_model: "claude-opus-4-5-20251101"
session_id: "70e70b3b-9e19-4492-8728-f471097888bc"

# What was decided
summary: "Removed duplicate files from analysis/ directory"

# The request that triggered this decision
trigger:
  type: "user_request"
  content: "definitely focus on completing milestone 1"
  context: "User reviewed roadmap showing package consolidation as M1 goal"

# Alternatives considered
alternatives:
  - option: "Keep duplicates, update imports gradually"
    pros: ["Lower risk", "Incremental change"]
    cons: ["Confusion from two locations", "Maintenance burden"]
    rejected_reason: "Prolongs technical debt, conflicts with M1 goals"

  - option: "Remove duplicates, keep deprecation shims"
    pros: ["Clean structure", "Backward compatible", "Matches roadmap"]
    cons: ["Requires import updates in CLI tools"]
    selected: true
    selected_reason: "Best balance of cleanup and compatibility"

  - option: "Remove duplicates and shims immediately"
    pros: ["Cleanest result"]
    cons: ["Breaks existing imports", "No migration path"]
    rejected_reason: "Breaking change inappropriate before v1.0"

# Impact assessment
impact:
  files_modified: 12
  files_deleted: 13
  lines_removed: 9135
  breaking_changes: false
  deprecations_added: false  # Already existed
  tests_affected: 0  # All tests pass

# Verification performed
verification:
  - action: "Ran full test suite"
    result: "295 passed, 4 skipped"
  - action: "Checked deprecation shims still work"
    result: "Imports via old paths emit warnings but function"

# Risk assessment
risks:
  - risk: "External code using analysis/ imports will see warnings"
    mitigation: "Deprecation warnings guide users to new paths"
    severity: "low"

# References
references:
  - type: "roadmap"
    path: "docs/architecture/ROADMAP.md"
    section: "Milestone 1: Foundation Consolidation"
  - type: "plan"
    path: ".claude/plans/valiant-shimmying-abelson.md"
```

#### 3.2.2 Inline Decision Comments

For smaller decisions within code, use structured comments:

```python
# DECISION: Use ConfidenceLevel.UNKNOWN as default rather than THEORETICAL
# RATIONALE: Unknown is more honest when no calibration data exists.
#            THEORETICAL implies we computed from specs, which we didn't.
# ALTERNATIVES: THEORETICAL (rejected: misleading),
#               raise ValueError (rejected: breaks existing code)
# DATE: 2026-01-23
# AI: claude-opus-4-5
confidence: ConfidenceLevel = ConfidenceLevel.UNKNOWN
```

### 3.3 Contribution Tracking

#### 3.3.1 Commit Attribution

All AI-assisted commits include co-author attribution:

```
Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
```

#### 3.3.2 Contribution Log

Maintain a machine-readable contribution log:

```json
// .claude/contributions/2026-01.jsonl (one JSON object per line)
{"date": "2026-01-23", "commit": "7ada23a", "type": "refactor", "files": 25, "lines_added": 26, "lines_removed": 9135, "decision_id": "DECISION-2026-01-23-001", "human_approved": true, "approver": "stillwater"}
```

#### 3.3.3 Monthly Summary Report

```markdown
# AI Contribution Summary - January 2026

## Statistics
- Total commits with AI assistance: 15
- Total lines added: 2,340
- Total lines removed: 12,456
- Net change: -10,116 lines (code cleanup)

## Decision Categories
- Refactoring: 8 decisions
- Bug fixes: 4 decisions
- New features: 2 decisions
- Documentation: 1 decision

## Review Outcomes
- Approved as-is: 12
- Approved with modifications: 2
- Rejected: 1 (reason: approach too aggressive)

## Lessons Learned
- Decision DECISION-2026-01-15-003 was later revised; AI over-optimized
- Pattern: AI suggestions for test coverage consistently valuable
```

### 3.4 Review and Approval Workflow

```
AI Generates Code
       |
       v
Decision Record Created -----> Stored in .claude/decisions/
       |
       v
Human Reviews Code + Decision Record
       |
       +---> Approve: Merge with attribution
       |
       +---> Request Changes: AI revises, new decision record
       |
       +---> Reject: Decision record marked rejected with reason
```

### 3.5 Audit Trail Requirements

For compliance and accountability, maintain:

1. **Session Transcripts**: Full conversation logs (already in `.claude/projects/`)
2. **Decision Records**: Structured YAML for each significant decision
3. **Contribution Logs**: JSONL files for machine processing
4. **Git History**: Commits with co-author attribution
5. **Review Records**: Approval/rejection with human reviewer ID

### 3.6 Governance Roles

| Role | Responsibility |
|------|----------------|
| **AI Assistant** | Generate code, document decisions, flag uncertainties |
| **Human Developer** | Review decisions, approve/reject, provide guidance |
| **Tech Lead** | Review high-impact decisions, set policy |
| **Auditor** | Periodic review of AI contributions, identify patterns |

### 3.7 Escalation Criteria

AI MUST escalate to human review (not proceed autonomously) when:

1. **Breaking Changes**: Any change that breaks public API
2. **Security-Sensitive**: Authentication, authorization, crypto, input validation
3. **Data Migration**: Schema changes, data transformations
4. **External Dependencies**: Adding new dependencies
5. **Performance-Critical**: Changes to hot paths or algorithms
6. **Uncertainty**: AI confidence in approach is low
7. **Novel Patterns**: First use of a pattern in codebase

---

## Part 4: Implementation Plan

### Phase 1: Foundation (This Week)

1. **Create CHANGELOG.md** - Initialize with current state (v0.8.0)
2. **Update pyproject.toml** - Add version = "0.8.0"
3. **Create decision record template** - `.claude/decisions/template.yaml`
4. **Create task spec template** - `docs/tasks/templates/task-spec.yaml`
5. **Document current milestone as v0.8.0** - Update ROADMAP.md

### Phase 2: Process (Next 2 Weeks)

1. **Create task registry** - `docs/tasks/index.md`
2. **Define 5-10 delegatable tasks** - For M2-M3 work
3. **Implement contribution logging** - `.claude/contributions/`
4. **Train on decision documentation** - Practice with next 3 changes

### Phase 3: Tooling (Following Month)

1. **Decision record generator** - CLI tool or script
2. **Contribution report generator** - Monthly summary automation
3. **Task status dashboard** - Simple markdown or HTML
4. **Version bump automation** - Script for releases

---

## Part 5: Questions for Review

1. **Version Numbering**: Is the milestone-to-version mapping appropriate? Should we start at 0.8.0 or different?

2. **Decision Granularity**: What level of decisions require formal records?
   - Every commit?
   - Every PR?
   - Only significant architectural decisions?

3. **Task Delegation Scope**: Who are the anticipated "other resources"?
   - Other AI assistants (Claude, GPT, etc.)?
   - Human contractors?
   - Internal team members?

4. **Storage Location**: Where should governance artifacts live?
   - In-repo (`.claude/`, `docs/`)?
   - Separate governance repo?
   - External system (Notion, Confluence)?

5. **Review Burden**: How to balance accountability with velocity?
   - All AI changes require human review?
   - Only certain categories?
   - Post-hoc audit acceptable for low-risk?

6. **Retroactive Application**: Should we document decisions for recent work retroactively?

---

## Appendix A: File Structure

```
graphs/
  CHANGELOG.md                    # NEW: Version history
  pyproject.toml                  # UPDATE: Add version field
  docs/
    architecture/
      ROADMAP.md                  # UPDATE: Add version targets
      ADR/                        # NEW: Architecture Decision Records
        ADR-001-package-structure.md
    tasks/
      index.md                    # NEW: Task registry
      templates/
        task-spec.yaml
        code-task.yaml
        calibration-task.yaml
      active/
      completed/
    governance/
      AI-CONTRIBUTION-POLICY.md   # NEW: Formal policy document
      DECISION-RECORD-GUIDE.md    # NEW: How to write decision records
  .claude/
    decisions/
      template.yaml               # NEW: Decision record template
      2026-01/                    # NEW: Monthly decision records
        DECISION-2026-01-23-001.yaml
    contributions/
      2026-01.jsonl               # NEW: Contribution log
    reports/
      2026-01-summary.md          # NEW: Monthly summary
```

## Appendix B: Example Task Breakdown for M2

```yaml
# Milestone 2: Calibration Infrastructure
# Target Version: 0.9.0-alpha

tasks:
  - id: "M2-001"
    title: "Implement CalibrationOrchestrator"
    assignee: "TBD"
    dependencies: []

  - id: "M2-002"
    title: "Add NVIDIA GPU auto-detection"
    assignee: "TBD"
    dependencies: ["M2-001"]

  - id: "M2-003"
    title: "Add Intel CPU auto-detection"
    assignee: "TBD"
    dependencies: ["M2-001"]

  - id: "M2-004"
    title: "Implement registry sync protocol"
    assignee: "TBD"
    dependencies: ["M2-001"]

  - id: "M2-005"
    title: "Create calibration CLI tool"
    assignee: "TBD"
    dependencies: ["M2-001", "M2-002", "M2-003"]
```

---

**End of Proposal**

*This proposal was generated by Claude Opus 4.5 on 2026-01-23 in response to a request for SEMVER integration, task delegation framework, and AI governance structure.*
