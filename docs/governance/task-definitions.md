# TASK definitions and methodology

The task definitions were created by me (Claude) as part of implementing the governance and delegation framework you requested. Here's the process:

## Origin:

  1. You asked for a proposal covering SEMVER versioning, task delegation, and AI governance
  2. I created docs/proposals/PROPOSAL-001-semver-governance-delegation.md with a 5-phase implementation plan
  3. In Phase 2, I defined 7 delegatable tasks based on the ROADMAP.md milestone structure

## Process:

  1. Template creation - Created docs/tasks/templates/task-spec.yaml as a standard format
  2. Task decomposition - Analyzed M2 (Benchmarking Infrastructure) from ROADMAP.md and broke it into discrete, implementable units
  3. Dependency mapping - Identified which tasks block others (e.g., schema must exist before runner)
  4. Task spec authoring - Created individual YAML files in docs/tasks/active/

Let me show you the template structure:

Read(docs/tasks/templates/task-spec.yaml)
Read(docs/governance/AI-CONTRIBUTION-POLICY.md)
Read(.claude/contributions/2026-01.jsonl)

## Summary of the Process:

```text
  ┌────────────────────┬─────────────┬─────────────────────────────────────────────────────────┐
  │        Step        │     Who     │                          What                           │
  ├────────────────────┼─────────────┼─────────────────────────────────────────────────────────┤
  │ 1. Request         │ You (human) │ Asked for SEMVER, governance, and delegation framework  │
  ├────────────────────┼─────────────┼─────────────────────────────────────────────────────────┤
  │ 2. Proposal        │ Claude      │ Created PROPOSAL-001 with task delegation approach      │
  ├────────────────────┼─────────────┼─────────────────────────────────────────────────────────┤
  │ 3. Approval        │ You (human) │ Approved and said "start implementing phase 1"          │
  ├────────────────────┼─────────────┼─────────────────────────────────────────────────────────┤
  │ 4. Template        │ Claude      │ Created task-spec.yaml template                         │
  ├────────────────────┼─────────────┼─────────────────────────────────────────────────────────┤
  │ 5. Task Definition │ Claude      │ Decomposed ROADMAP M2 into 7 tasks                      │
  ├────────────────────┼─────────────┼─────────────────────────────────────────────────────────┤
  │ 6. Review          │ You (human) │ Implicit approval via "assign task and start executing" │
  ├────────────────────┼─────────────┼─────────────────────────────────────────────────────────┤
  │ 7. Execution       │ Claude      │ Self-assigned and implemented tasks                     │
  ├────────────────────┼─────────────┼─────────────────────────────────────────────────────────┤
  │ 8. Commits         │ Claude      │ Each commit co-authored and logged                      │
  └────────────────────┴─────────────┴─────────────────────────────────────────────────────────┘
```

## Key Governance Elements:

  1. Contribution Logging (.claude/contributions/2026-01.jsonl) - Tracks all commits with metadata
  2. Decision Records (.claude/decisions/) - Documents significant architectural choices
  3. Task Specs (docs/tasks/active/) - Define work items with acceptance criteria
  4. Co-Author Attribution - Every commit includes Co-Authored-By: Claude Opus 4.5

## Who Can Create Tasks:

Per the AI-CONTRIBUTION-POLICY.md, tasks can be created by AI assistants but significant decisions require human review. The task creation in this session
was done by me based on the existing ROADMAP.md structure, with your implicit approval when you said "proceed to phase 2" and "assign task and start
executing."

The goal is to enable parallel delegation to multiple resources (human or AI) while maintaining accountability and traceability.

## Milestones and new tasks

The milestones are defined in docs/architecture/ROADMAP.md. Here's the current status:

  M2 Status (mostly complete):
  ┌──────────────────────────┬─────────────┬────────────────────────────────────────────────────────┐
  │         Section          │   Status    │                         Notes                          │
  ├──────────────────────────┼─────────────┼────────────────────────────────────────────────────────┤
  │ 2.1 Benchmark Definition │ Mostly done │ Schema, GEMM, Conv2d done; Attention/workloads pending │
  ├──────────────────────────┼─────────────┼────────────────────────────────────────────────────────┤
  │ 2.2 Benchmark Runners    │ Done        │ Runner, collectors implemented                         │
  ├──────────────────────────┼─────────────┼────────────────────────────────────────────────────────┤
  │ 2.3 CLI Integration      │ Done        │ cli/benchmark.py created                               │
  ├──────────────────────────┼─────────────┼────────────────────────────────────────────────────────┤
  │ 2.4 Initial Collection   │ Not started │ Run benchmarks on hardware                             │
  └──────────────────────────┴─────────────┴────────────────────────────────────────────────────────┘
  M3: Calibration Framework (v0.9.0-beta):
  3.1 Roofline Calibration
      - Measure achieved memory bandwidth
      - Measure achieved compute throughput
      - Fit efficiency curves by operation type

  3.2 Energy Model Calibration
      - Fit pJ/op for compute operations
      - Fit pJ/byte for memory operations
      - Fit static power from idle measurements

  3.3 Utilization Factor Calibration
      - Measure actual vs theoretical utilization
      - Fit utilization curves (vs problem size, batch size)

  3.4 CLI Integration
      - Create cli/calibrate.py command

  To proceed with governance process, I should create task definitions for M3. Shall I:

  1. Create M3 task specs following the template (TASK-2026-008 through TASK-2026-012 or so)
  2. Update the task index with M3 tasks and dependencies
  3. Update ROADMAP.md to mark M2 sections as complete

  This would produce tasks like:
  - TASK-2026-008: Implement roofline parameter fitting
  - TASK-2026-009: Implement energy coefficient fitting
  - TASK-2026-010: Implement utilization factor calibration
  - TASK-2026-011: Create calibration CLI tool
  - TASK-2026-012: Run initial calibration on available hardware

  Want me to proceed with creating the M3 task definitions?

## M3 Task Definitions Created

  Committed as c6b7702. Here's the M3 task structure:

  M3: Calibration Framework (v0.9.0-beta)
  =======================================

  Parallel (can be done simultaneously):
    TASK-2026-006: Roofline parameter fitting     [ready]
    TASK-2026-008: Energy coefficient fitting     [ready]
    TASK-2026-009: Utilization factor calibration [ready]
                             |
                             v
    TASK-2026-010: Calibration CLI tool           [blocked by 006,008,009]
                             |
                             v
    TASK-2026-011: Initial calibration on hardware [blocked by 010]

  Task Summary:
  ┌──────┬──────────────────────────────────────────────┬────────┬──────────────────────────────────────────────┐
  │ Task │                 Description                  │ Effort │                 Deliverables                 │
  ├──────┼──────────────────────────────────────────────┼────────┼──────────────────────────────────────────────┤
  │ 006  │ Fit roofline params from benchmarks          │ 8h     │ roofline_fitter.py, efficiency_curves.py     │
  ├──────┼──────────────────────────────────────────────┼────────┼──────────────────────────────────────────────┤
  │ 008  │ Fit energy coefficients from power data      │ 8h     │ energy_fitter.py, power_model.py             │
  ├──────┼──────────────────────────────────────────────┼────────┼──────────────────────────────────────────────┤
  │ 009  │ Fit utilization curves from benchmark sweeps │ 6h     │ utilization_fitter.py, utilization_curves.py │
  ├──────┼──────────────────────────────────────────────┼────────┼──────────────────────────────────────────────┤
  │ 010  │ CLI tool for calibration workflows           │ 6h     │ cli/calibrate.py                             │
  ├──────┼──────────────────────────────────────────────┼────────┼──────────────────────────────────────────────┤
  │ 011  │ Run calibration on available hardware        │ 4h     │ Calibration profiles, methodology docs       │
  └──────┴──────────────────────────────────────────────┴────────┴──────────────────────────────────────────────┘
  Next Steps:

  Tasks 006, 008, and 009 are all ready and can be worked on in parallel. Would you like me to:
  1. Assign and start executing one of the M3 tasks?
  2. Assign tasks to different resources (if you have other AI assistants or humans available)?
  3. Something else?

## Samples of calibration

Do we have a tool that allows us to see how many samples of calibration are present in the registry?

● Let me check what tools exist for viewing calibration data.

● Bash(ls -la cli/*.py | head -20)
  ⎿  -rwxrwxr-x 1 stillwater stillwater 25141 Jan 23 07:53 cli/analyze_batch.py
     -rwxrwxr-x 1 stillwater stillwater 10058 Dec  3 09:08 cli/analyze_cgra_energy.py
     -rwxrwxr-x 1 stillwater stillwater 23244 Jan 23 07:53 cli/analyze_comprehensive.py
     … +17 lines (ctrl+o to expand)

● Bash(python cli/show_calibration_efficiency.py --help 2>/dev/null)
  ⎿  usage: show_calibration_efficiency.py [-h] [--id ID] [--power-mode POWER_MODE]
                                           [--framework FRAMEWORK] [--all] [--list]
                                           [--summary] [--table]
     … +18 lines (ctrl+o to expand)

● Bash(python cli/show_calibration_efficiency.py --summary 2>/dev/null)
  ⎿  Hardware Registry Summary
     ========================================================================================================================
     ID                                  Type     TDP   Peak Compute    Peak BW Best Prec Calibrated
     … +52 lines (ctrl+o to expand)

● Yes, show_calibration_efficiency.py has this built in:

  Per hardware profile counts - use --summary:
  Calibrated column shows count: "Yes (6)" means 6 calibration samples

  Total count - use --table:
  Total: 18 calibration records

  Current calibration inventory:
  ┌────────────────────────────────────────┬──────────────┐
  │                Hardware                │ Calibrations │
  ├────────────────────────────────────────┼──────────────┤
  │ intel_12th_gen_intelr_coretm_i7_12700k │ 6            │
  ├────────────────────────────────────────┼──────────────┤
  │ jetson_orin_nano_cpu                   │ 4            │
  ├────────────────────────────────────────┼──────────────┤
  │ jetson_orin_nano_gpu                   │ 4            │
  ├────────────────────────────────────────┼──────────────┤
  │ ryzen_7_8845hs_w_radeon_780m_graphics  │ 2            │
  ├────────────────────────────────────────┼──────────────┤
  │ amd_ryzen_7_2700x_eight_core_processor │ 1            │
  ├────────────────────────────────────────┼──────────────┤
  │ ryzen_9_8945hs_w_radeon_780m_graphics  │ 1            │
  ├────────────────────────────────────────┼──────────────┤
  │ Total                                  │ 18           │
  └────────────────────────────────────────┴──────────────┘
  Would you like a dedicated --stats option that shows just the calibration counts in a more focused way?

