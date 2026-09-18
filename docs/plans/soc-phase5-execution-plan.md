# SoC study Phase 5: fidelity where the studies need it -- execution plan

Tracking: branes-ai/graphs#269, Phase 5. Parent plan:
`docs/plans/soc-microarchitecture-study-plan.md`, section 7 ("driven by what
the first studies expose"). The strict rule of Phases 2-4 carries over.
Started 2026-09-18.

## What the first study exposed

`docs/assessments/soc-orin-vs-kpu-heterogeneous.md` (PR 4.3) could decide
only DRAM feasibility. In order of leverage, what blocks the rest:

1. **No per-engine service time exists for any stage.** `default_v1` has two
   measured pairs, and neither is a format the stages run in.
2. The KPU has no FP32, so 14 of 19 stages fall to the CPU.
3. Unpriced silicon: Orin's SM, DLA, PVA and CPU logic, and the H64's
   systolic, LNS and fixed-function tiles.
4. Far flight needs about 470 GB/s of peak DRAM.

Items 2 to 4 are data, or design choices; Phase 5 builds no model for them.
Item 1 is where modelling or measurement is needed.

## The parent plan's Phase 5 items, against the strict rule

| Item | What it produces | Under the strict rule |
|---|---|---|
| Measured stage-kernel efficiencies | CALIBRATED `default_v1` entries from benchmarks run on the Orin AGX / Nano | Consistent: data, not fill |
| L1 DNN stages via `UnifiedAnalyzer` | Roofline service times for `det`, `sdfenc`, `policy`, `vlm`, `vla` | Model output, THEORETICAL. The Orin AGX mapper reports THEORETICAL (50%) even on ResNet-50. The stages' own networks (YOLOe-class, a 2B VLM, the SDF encoder, the DRL policy) are not in the model factory, so stand-ins would carry their efficiency over to the stage's ops. |
| L1 KPU stages (domain-flow wavefront cost model) | KPU service times from the SURE/SARE schedule | Model output, THEORETICAL. A research-scale item. |
| ILP mapper, split mappings | Better assignments of stages to engines | Consistent: infrastructure |
| RM/EDF response time, DRAM contention | Replaces the "owns the engine" lower bound | Consistent: infrastructure. It only bites once stages are priced. |
| Multi-block floorplan; die cost and yield | Area layout; cost per good die | Needs wafer cost and D0 per node, which are unsourced today (gaps) |
| Upstream schemas to embodied-schemas | IP block, SoC design and PipelineWorkload schemas | Consistent: housekeeping |

## Decision needed first

Which of these goes first is the user's call (P5-D1). It sets whether Phase
5 prices the pipeline through measurement, through labelled model output,
or leaves it priced by nothing while the infrastructure matures.

**P5-D1, decided 2026-09-18: the measurement path first.** Per-engine
efficiencies come from stage-kernel benchmarks run on the target hardware.
Neither model output from stand-in networks nor the KPU cost model is used
to fill `default_v1`.

## PR sequence (measurement path)

| PR | Content |
|---|---|
| 5.1 | `graphs.benchmarks.soc_kernels` and `cli/benchmark_soc_kernels.py`. See the notes below. |
| 5.2 | `tools/ingest_soc_kernel_benchmarks.py`, plus efficiency-table layering (`base:`). See the notes below. |
| 5.3 | The Orin Nano and AGX runs (the user runs them on the hardware), the regenerated table, and the 4.3 study re-run on measured data. |

**5.1, the benchmark harness:**
- It measures the kernel classes it has a representative kernel for, on
  the target, and writes a JSON document per run.
- **The device clock is sampled throughout every timed loop,** and a run
  is `verified` only when enough samples agree within 5%. This fixes the
  2026-02 calibrations' idle-clock readings.
- **CPU kernels run pinned to one core, single-threaded,** matching the
  analyzer's one-core server.
- Seven kernel classes have no kernel yet and are listed as not covered:
  cost volume DP, feature tracking, kNN, raycast, wavefront, graph search,
  and pixel fixed-function.

**5.2, the ingest and table layering:**
- The ingest turns runs into a measured table layered over `default_v1`.
- Efficiency is attained ops/s over the composed block's per-server peak
  *at the measured clock*.
- An entry is CALIBRATED when the clock was verified and INTERPOLATED
  otherwise.
- A format the IP template states no peak for (Orin's SM has no FP16 rate)
  stays a gap, with the measurement kept alongside it.
