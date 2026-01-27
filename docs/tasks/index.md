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

### M2: Benchmarking Infrastructure (0.9.0-alpha)

| ID | Title | Priority | Status | Blocked By | Target |
|----|-------|----------|--------|------------|--------|
| TASK-2026-004 | Implement NVIDIA power measurement collector | medium | ready | - | 0.9.0-alpha |

### M3: Calibration Framework (0.9.0-beta)

| ID | Title | Priority | Status | Blocked By | Target |
|----|-------|----------|--------|------------|--------|
| TASK-2026-006 | Implement roofline parameter fitting | high | **completed** | - | 0.9.0-beta |
| TASK-2026-008 | Implement energy coefficient fitting | high | ready | - | 0.9.0-beta |
| TASK-2026-009 | Implement utilization factor calibration | high | ready | - | 0.9.0-beta |
| TASK-2026-010 | Create calibration CLI tool | high | blocked | 008,009 | 0.9.0-beta |
| TASK-2026-011 | Run initial calibration on hardware | high | blocked | 010 | 0.9.0-beta |

## Task Dependency Graph

```
M2: Benchmarking Infrastructure (MOSTLY COMPLETE)
==================================================
TASK-2026-001 (Benchmark Schema) [DONE]
    |
    +---> TASK-2026-002 (GEMM Benchmark) [DONE]
    |
    +---> TASK-2026-003 (Runner Interface) [DONE]
    |         |
    |         +---> TASK-2026-004 (Power Collector) [ready]
    |
    +---> TASK-2026-007 (Conv2d Benchmark) [DONE]
    |
    +---> TASK-2026-005 (Benchmark CLI) [DONE]


M3: Calibration Framework
==================================================
TASK-2026-006 (Roofline Fitting) [DONE] -+
                                         |
TASK-2026-008 (Energy Fitting) ----------+---> TASK-2026-010 (Calibrate CLI)
                                         |              |
TASK-2026-009 (Utilization Fitting) -----+              v
                                              TASK-2026-011 (Initial Calibration)
```

## Completed Tasks

| ID | Title | Completed | Version |
|----|-------|-----------|---------|
| TASK-2026-001 | Define benchmark specification schema | 2026-01-23 | 0.9.0-alpha |
| TASK-2026-002 | Implement GEMM microbenchmark suite | 2026-01-23 | 0.9.0-alpha |
| TASK-2026-006 | Implement roofline parameter fitting | 2026-01-26 | 0.9.0-beta |
| TASK-2026-003 | Define benchmark runner interface | 2026-01-23 | 0.9.0-alpha |
| TASK-2026-005 | Create benchmark CLI tool | 2026-01-23 | 0.9.0-alpha |
| TASK-2026-007 | Implement Conv2d microbenchmark suite | 2026-01-23 | 0.9.0-alpha |

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

| Milestone | Target Version | Focus Area | Tasks |
|-----------|----------------|------------|-------|
| M1 | 0.8.0 | Foundation Consolidation | COMPLETE |
| M2 | 0.9.0-alpha | Benchmarking Infrastructure | 001-005, 007 (5/6 done) |
| M3 | 0.9.0-beta | Calibration Framework | 006, 008-011 |
| M4 | 0.9.0-rc | Validation Pipeline | TBD |
| M5 | 1.0.0 | Hardware Coverage | TBD |
| M6 | 1.1.0 | Frontend Expansion | TBD |
| M7 | 1.2.0 | Advanced Analysis | TBD |
| M8 | 2.0.0 | Production Readiness | TBD |

## Assignment Guidelines

When assigning tasks:

1. **Check dependencies** - Ensure blocked_by tasks are complete
2. **Match skills** - Assign based on expertise required
3. **Balance load** - Avoid overloading single resources
4. **Update status** - Change to `assigned` and set assignee

## For AI Assistants

When working on a task:

1. Read the full task specification
2. Check interface contracts carefully
3. Create a decision record for significant choices
4. Update task progress log as you work
5. Ensure all acceptance criteria are met before requesting review
