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
| TASK-2026-008 | Implement energy coefficient fitting | high | **completed** | - | 0.9.0-beta |
| TASK-2026-009 | Implement utilization factor calibration | high | **completed** | - | 0.9.0-beta |
| TASK-2026-010 | Create calibration CLI tool | high | **completed** | - | 0.9.0-beta |
| TASK-2026-011 | Run initial calibration on hardware | high | ready | - | 0.9.0-beta |

### M4.5: Bottom-Up Validation Infrastructure (0.9.0-rc)

| ID | Title | Priority | Status | Blocked By | Target |
|----|-------|----------|--------|------------|--------|
| TASK-2026-012 | Bottom-up validation infrastructure (umbrella) | high | in_progress | - | 0.9.0-rc |
| TASK-2026-013 | Phase 0: foundation (LayerTag, fitters pkg, provenance, composition scaffold) | high | review | - | 0.9.0-rc |
| TASK-2026-014 | Phase 0.5: PowerMeter abstraction + CI gate | high | ready | 013 | 0.9.0-rc |
| TASK-2026-015 | Phase 1: Layer-1 ALU throughput and energy | high | ready | 013, 014 | 0.9.0-rc |
| TASK-2026-016 | Phase 2: Layer-2 register / SIMD / warp / systolic fill | high | ready | 015 | 0.9.0-rc |
| TASK-2026-017 | Phase 3: Layer-3 tile / scratchpad / L1-L2 | high | ready | 016 | 0.9.0-rc |
| TASK-2026-018 | Phase 4: Layer-4 on-chip L3 / LLC / distributed L3 | high | ready | 017 | 0.9.0-rc |
| TASK-2026-019 | Phase 5: Layer-5 DRAM by technology | high | ready | 018 | 0.9.0-rc |
| TASK-2026-020 | Phase 6: Layer-6 PCIe / NVLink / NUMA / collectives | medium | ready | 019 | 0.9.0-rc |

> **Note (2026-04-20):** Tasks TASK-2026-012..020 are from the pre-pivot 6-layer
> validation plan. The plan was revised to a 9-layer framing and re-sequenced:
> measurement work now runs *after* the model-delivery epic below. These tasks
> remain valid in structure but need re-sequencing (and partial supersession by
> TASK-2026-022 which subsumes TASK-2026-013's scaffolding work). Deferring
> cleanup until after plan-owner review of the new task set.

### M_DELIVERY: Micro-architectural Model Delivery (0.10.0-microarch)

| ID | Title | Priority | Status | Blocked By | Target |
|----|-------|----------|--------|------------|--------|
| TASK-2026-021 | Micro-arch model delivery (umbrella, Layers 1-7) | high | ready | - | 0.10.0-microarch |
| TASK-2026-022 | M0: Scaffolding (schema, dataclass stubs, branded CLI+HTML) | high | ready | - | 0.10.0-microarch |
| TASK-2026-023 | M0.5: KPU domain-flow-tile model + GPU/TPU/KPU comparison harness | high | ready | 022 | 0.10.0-microarch |
| TASK-2026-024 | M1: Layer 1 (ALU) model population + panel | high | ready | 022, 023 | 0.10.0-microarch |
| TASK-2026-025 | M2: Layer 2 (register file) model population + panel | high | ready | 024 | 0.10.0-microarch |
| TASK-2026-026 | M3: Layer 3 (L1 cache) model population + panel | high | ready | 025 | 0.10.0-microarch |
| TASK-2026-027 | M4: Layer 4 (L2 cache) model population + panel | high | ready | 026 | 0.10.0-microarch |
| TASK-2026-028 | M5: Layer 5 (L3 / LLC) model population + panel | high | ready | 027 | 0.10.0-microarch |
| TASK-2026-029 | M6: Layer 6 (SoC fabric) model population + panel | high | ready | 028 | 0.10.0-microarch |
| TASK-2026-030 | M7: Layer 7 (external memory) model population + panel | high | ready | 029 | 0.10.0-microarch |
| TASK-2026-031 | M8: Reporting polish + engineering PPT deck | high | ready | 030 | 0.10.0-microarch |

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
TASK-2026-006 (Roofline Fitting) [DONE] -----+
                                             |
TASK-2026-008 (Energy Fitting) [DONE] -------+---> TASK-2026-010 (Calibrate CLI) [DONE]
                                             |              |
TASK-2026-009 (Utilization Fitting) [DONE] --+              v
                                                  TASK-2026-011 (Initial Calibration) [ready]


M4.5: Bottom-Up Validation Infrastructure
==================================================
TASK-2026-012 (Umbrella) [in_progress]
     |
     v
TASK-2026-013 (Phase 0: Foundation) [review, PR #1]
     |
     v
TASK-2026-014 (Phase 0.5: PowerMeter + CI) [ready]
     |
     v
TASK-2026-015 (Phase 1: Layer-1 ALU) [ready]
     |
     v
TASK-2026-016 (Phase 2: Layer-2 Register/SIMD) [ready]
     |
     v
TASK-2026-017 (Phase 3: Layer-3 Scratchpad) [ready]
     |
     v
TASK-2026-018 (Phase 4: Layer-4 On-chip L3) [ready]
     |
     v
TASK-2026-019 (Phase 5: Layer-5 DRAM) [ready]
     |
     v
TASK-2026-020 (Phase 6: Layer-6 Cluster) [ready]


M_DELIVERY: Micro-architectural Model Delivery (0.10.0-microarch)
==================================================
TASK-2026-021 (Umbrella / epic) [ready]
     |
     +---> TASK-2026-022 (M0: Scaffolding) [ready]
                |
                +---> TASK-2026-023 (M0.5: KPU domain-flow-tile + comparison harness) [ready]
                           |
                           v
                      TASK-2026-024 (M1: Layer 1 ALU) [ready]
                           |
                           v
                      TASK-2026-025 (M2: Layer 2 Register) [ready]
                           |
                           v
                      TASK-2026-026 (M3: Layer 3 L1 cache) [ready]
                           |
                           v
                      TASK-2026-027 (M4: Layer 4 L2 cache) [ready]
                           |
                           v
                      TASK-2026-028 (M5: Layer 5 L3 / LLC) [ready]
                           |
                           v
                      TASK-2026-029 (M6: Layer 6 SoC fabric) [ready]
                           |
                           v
                      TASK-2026-030 (M7: Layer 7 External memory) [ready]
                           |
                           v
                      TASK-2026-031 (M8: Reporting polish + PPT deck) [ready]
```

## Milestone Mapping

| Milestone | Version | Scope |
|-----------|---------|-------|
| M2 | 0.9.0-alpha | Benchmarking infrastructure |
| M3 | 0.9.0-beta | Calibration framework |
| M4.5 | 0.9.0-rc | Bottom-up validation (measurement-driven; re-sequenced to run after M_DELIVERY) |
| M_DELIVERY | 0.10.0-microarch | Micro-architectural model delivery (analytical; TASK-2026-021..031) |

## Completed Tasks

| ID | Title | Completed | Version |
|----|-------|-----------|---------|
| TASK-2026-001 | Define benchmark specification schema | 2026-01-23 | 0.9.0-alpha |
| TASK-2026-002 | Implement GEMM microbenchmark suite | 2026-01-23 | 0.9.0-alpha |
| TASK-2026-006 | Implement roofline parameter fitting | 2026-01-26 | 0.9.0-beta |
| TASK-2026-003 | Define benchmark runner interface | 2026-01-23 | 0.9.0-alpha |
| TASK-2026-005 | Create benchmark CLI tool | 2026-01-23 | 0.9.0-alpha |
| TASK-2026-007 | Implement Conv2d microbenchmark suite | 2026-01-23 | 0.9.0-alpha |
| TASK-2026-008 | Implement energy coefficient fitting | 2026-01-26 | 0.9.0-beta |
| TASK-2026-009 | Implement utilization factor calibration | 2026-01-26 | 0.9.0-beta |
| TASK-2026-010 | Create calibration CLI tool | 2026-01-26 | 0.9.0-beta |

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
| M4.5 | 0.9.0-rc | Bottom-Up Validation Infrastructure | 012-020 |
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
