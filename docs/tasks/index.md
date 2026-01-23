# Task Registry

This directory contains task specifications for delegatable work items.

## Task States

| State | Description |
|-------|-------------|
| `draft` | Task spec being written, not ready for assignment |
| `ready` | Task spec complete, ready to be assigned |
| `assigned` | Assigned to a resource, not yet started |
| `in_progress` | Actively being worked on |
| `review` | Work complete, under review |
| `blocked` | Cannot proceed due to dependency |
| `completed` | Merged and done |
| `cancelled` | No longer needed |

## Active Tasks

| ID | Title | Assignee | Status | Target Version |
|----|-------|----------|--------|----------------|
| - | No active tasks | - | - | - |

## Completed Tasks

| ID | Title | Completed | Version |
|----|-------|-----------|---------|
| - | No completed tasks | - | - |

## Templates

- [task-spec.yaml](templates/task-spec.yaml) - General task template

## Creating a New Task

1. Copy the appropriate template to `active/TASK-YYYY-NNN.yaml`
2. Fill in all required fields
3. Set status to `draft`
4. Request review from tech lead
5. Once approved, set status to `ready`

## Task Naming Convention

```
TASK-YYYY-NNN

Where:
  YYYY = Year
  NNN  = Sequential number within year (001, 002, etc.)

Examples:
  TASK-2026-001
  TASK-2026-002
```

## Milestone Mapping

| Milestone | Target Version | Focus Area |
|-----------|----------------|------------|
| M1 | 0.8.0 | Foundation Consolidation (COMPLETE) |
| M2 | 0.9.0-alpha | Calibration Infrastructure |
| M3 | 0.9.0-beta | Estimation Accuracy |
| M4 | 0.9.0-rc | Multi-Platform Support |
| M5 | 1.0.0 | Production Hardening |
| M6 | 1.1.0 | Advanced Estimation |
| M7 | 1.2.0 | Integration APIs |
| M8 | 2.0.0 | Ecosystem |
