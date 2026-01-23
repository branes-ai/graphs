# AI Contribution Policy

This document defines the governance framework for AI-assisted code contributions
to the graphs repository.

## Purpose

Establish transparency, traceability, and accountability for AI-assisted
development while maximizing productivity and code quality.

## Scope

This policy applies to all code changes generated or assisted by AI systems,
including but not limited to:
- Claude (Anthropic)
- GitHub Copilot
- Other AI coding assistants

## Core Principles

### 1. Transparency

All AI contributions must be clearly attributed:

```
Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
```

### 2. Traceability

Significant decisions must be documented in decision records that capture:
- What was decided
- Why it was decided (alternatives considered)
- Impact assessment
- Verification performed

### 3. Accountability

- Human reviewers are accountable for approving AI contributions
- AI assistants must flag uncertainty and escalate when appropriate
- All changes undergo standard review process

### 4. Quality

AI contributions must meet the same quality standards as human contributions:
- Pass all tests
- Follow code style guidelines
- Include appropriate documentation
- Meet acceptance criteria

## Decision Documentation Requirements

### When to Create a Decision Record

Create a decision record (`.claude/decisions/DECISION-YYYY-MM-DD-NNN.yaml`) when:

1. **Architectural Changes** - Modifying package structure, interfaces, or patterns
2. **Multiple Valid Approaches** - Choosing between reasonable alternatives
3. **Breaking Changes** - Any change affecting public API
4. **Significant Refactoring** - Changes affecting multiple files
5. **New Dependencies** - Adding external libraries
6. **Security-Relevant** - Authentication, authorization, input validation

### Decision Record Contents

Each decision record must include:
- Summary of the decision
- Trigger (what prompted the decision)
- Alternatives considered with pros/cons
- Impact assessment
- Verification performed
- Risk assessment

See `.claude/decisions/template.yaml` for full template.

### Inline Decision Comments

For smaller decisions within code:

```python
# DECISION: Use ConfidenceLevel.UNKNOWN as default
# RATIONALE: More honest when no calibration data exists
# ALTERNATIVES: THEORETICAL (rejected: misleading)
# DATE: 2026-01-23
# AI: claude-opus-4-5
```

## Contribution Logging

All AI contributions are logged in `.claude/contributions/YYYY-MM.jsonl`:

```json
{
  "date": "2026-01-23",
  "commit": "7ada23a",
  "type": "refactor",
  "summary": "Complete Milestone 1 package structure consolidation",
  "files_changed": 25,
  "lines_added": 26,
  "lines_removed": 9135,
  "decision_id": "DECISION-2026-01-23-001",
  "human_approved": true,
  "approver": "stillwater"
}
```

## Escalation Requirements

AI assistants MUST escalate to human review (not proceed autonomously) when:

1. **Breaking Changes** - Any modification to public API signatures
2. **Security-Sensitive** - Authentication, authorization, cryptography
3. **Data Migration** - Schema changes, data transformations
4. **External Dependencies** - Adding or upgrading dependencies
5. **Performance-Critical** - Changes to hot paths or algorithms
6. **Uncertainty** - When confidence in approach is low
7. **Novel Patterns** - First use of a pattern in codebase
8. **Destructive Operations** - File deletion, database drops

## Review Process

```
AI Generates Code
       |
       v
Decision Record Created (if applicable)
       |
       v
Human Reviews Code + Decision Record
       |
       +---> Approve: Merge with attribution
       |
       +---> Request Changes: AI revises
       |
       +---> Reject: Document reason
```

## Roles and Responsibilities

### AI Assistant

- Generate high-quality code following project standards
- Document decisions with rationale
- Flag uncertainties and escalate appropriately
- Update contribution logs
- Follow task specifications precisely

### Human Developer

- Review AI-generated code critically
- Approve or reject changes
- Provide guidance and context
- Update task assignments

### Tech Lead

- Review architectural decisions
- Set policy and standards
- Audit contribution patterns periodically

## Audit and Compliance

### Monthly Review

Generate monthly summary reports:
- Total AI-assisted commits
- Decision categories
- Review outcomes
- Lessons learned

### Periodic Audit

Quarterly review of:
- Decision quality
- Escalation appropriateness
- Code quality trends
- Policy effectiveness

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-01-23 | Initial policy |

---

*This policy is part of PROPOSAL-001: SEMVER, Governance, and Task Delegation Framework*
