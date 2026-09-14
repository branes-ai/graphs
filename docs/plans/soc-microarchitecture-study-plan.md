# SoC Micro-Architecture Study Framework -- Implementation Plan

Status: DRAFT for review
Date: 2026-09-13
Tracking issue: TBD (open after review)
Inputs:
  - `docs/BranesAI-Autonomy-Compute-Requirements.pdf` (7 tiers, 17 stages, 5 regimes)
  - `docs/designs/kpu-sku-and-process-node-plan.md` (ProcessNode, silicon_bin, validators, floorplan)
  - `docs/designs/v8/v10/v13 ComputeProduct unification` (multi-block `Die`, `IOBlock`)

## 1. Goal

Build a design-space-exploration tool for heterogeneous SoCs aimed at embodied
autonomy. A study takes three inputs:

1. An **SoC design**: a silicon allocation across CPU clusters, GPU clusters,
   NPU/DLA tiles, vector DSP/PVA, ISP, video codec, KPU tiles, the system cache,
   the NoC, memory PHYs and IO.
2. A **process node**: any `ProcessNodeEntry` in embodied-schemas, for example
   `tsmc_n16`, `tsmc_n7`, `tsmc_n5` or `samsung_8lpp`.
3. A **workload**: the 7-tier / 17-stage autonomy pipeline, evaluated under one
   of 5 operating regimes.

For each study the tool reports:

- **Die size**: total, per block and per circuit class. Transistor count.
  Pad/PHY shoreline feasibility.
- **Performance**: per-stage service time against its latency budget, per-engine
  utilization, DRAM bandwidth demand against supply, glass-to-actuator latency,
  and a feasible/infeasible verdict per regime.
- **Energy**: average power per regime, split into dynamic (by engine),
  memory-hierarchy, leakage and DRAM. Energy per second of flight, useful
  TOPS/W, and a thermal check against a `CoolingSolutionEntry`.

Every number carries `EstimationConfidence`.

The headline use is the argument in PDF sections 6 and 9, made quantitative:
no fixed allocation serves every regime, so we need to be able to compare
allocations (area, power, feasibility) regime by regime, across process nodes.

## 2. Should SoC specs live in `hardware_registry/soc/`? (Decision D1)

**Recommendation: no. Put them in a new sibling data tree, `soc_designs/`, and
emit ComputeProduct-shaped output, the same way the KPU generator does.**

- **Different semantics.** `hardware_registry/<type>/<id>/spec.json` holds
  *as-built* devices plus their calibration measurements. An SoC study
  design is hypothetical, parameterized and retargetable across nodes, so a
  node-specific id like `..._16nm_tsmc_ffp` does not apply.
- **Loader collision.** `HardwareRegistry.load_all()` globs `*/*/spec.json` and
  parses each match as a `HardwareProfile`. SoC compositions are a different
  shape, so they would need a special case in that loader.
- **A third source of truth.** The registry already disagrees with the catalog.
  `hardware_registry/accelerator/kpu_t256_32x32_lp5x16_16nm_tsmc_ffp/spec.json`
  says 426 TOPS at 1.4 GHz. The ComputeProduct YAML says 314.6 TOPS at 600 MHz,
  and nothing reconciles the two. Adding SoC specs to the registry would add a
  third source of truth.
- **An established pattern exists.** The KPU path is: architect-authored input
  spec (`KPUSKUInputSpec`, owned by graphs) -> `generate_kpu_sku()` ->
  `ComputeProduct` -> validators, floorplan and mappers. The SoC path should be
  the same: `SoCDesign` -> `compose_soc(design, node)` -> `SoCInstance`, which
  emits a multi-block `ComputeProduct` -> the same validators.

**Fallback if the team prefers one hardware tree:** use `hardware_registry/soc/`
with `design.yaml` files (never `spec.json`), and add an explicit skip in
`HardwareRegistry.load_all()`. That works, but it mixes as-built devices with
design candidates.

## 3. What exists and what is missing

| Capability | Exists today | Gap for SoC studies |
|---|---|---|
| Process node data | `ProcessNodeEntry` (embodied-schemas): per-circuit-class density, leakage/mm^2, `energy_per_op_pj["class:prec"]`, SRAM/NoC/DRAM pJ. 13 nodes, all THEORETICAL. | No frequency/speed model per node, so clocks cannot be retargeted. Vdd exponent exists only on n16/n7; SRAM/NoC pJ only on 5 nodes. No N3/N2/18A. |
| Multi-block die schema | `ComputeProduct.dies[].blocks: list[AnyBlock]` with 9 block kinds (KPU, GPU, CPU, NPU, CGRA, DPU, TPU, DSP, IO). `Interconnect`. | No `ISP` or `VIDEO_CODEC` block kind (both reserved in the `BlockKind` docstring). The Orin ComputeProduct has only its GPU block; CPU/DLA/PVA/ISP are lumped into one `fixed` silicon_bin line. |
| Area math | `sku_validators/silicon_math.py`: `resolve_block_area` = Mtx / density(node, class). | KPU-only (`_kpu_block` raises on non-KPU) and reads only `dies[0]`. |
| Power model | `kpu_power_model.py`: 5-term TDP roll-up, V^2 plus leakage Vdd exponent. `architectural_energy.py` and `operand_fetch.py` have per-architecture-class overheads (stored-program, data-parallel, systolic, domain-flow). | No per-block power for CPU/GPU/NPU/ISP inside one SoC. No regime-level power from a stage mapping. |
| Validators | `ValidatorRegistry`, 7 categories, 14 validators, `ValidatorContext(sku, process_node, cooling_solutions)`. | Assumes a KPU die. Needs multi-block and SoC-aware checks: PHY shoreline, per-block W/mm^2. |
| Floorplan | `silicon_floorplan.py`: KPU checkerboard, ASCII view. | KPU-only. A multi-block SoC floorplan is Phase 5. |
| Workload | `SubgraphDescriptor` (DNN op types). `embodied_schemas` `SoftwareArchitecture` / `OperatorEntry` (topology plus a scalar `compute_flops`). `scripts/robot_estimator_v8.py` (flat op list). | No stage record with per-precision ops, bytes, working set, rate, latency budget or regime. No op types for solver, FFT, sparse or QP. The v8 script has a watts-plus-joules power-clamp bug, so do not port numbers from it. |
| Estimators | `RooflineAnalyzer` / `EnergyAnalyzer` on one `HardwareResourceModel`. | No multi-engine mapping or scheduler. Roofline silently falls back from FP64 to the FP32 peak, and non-matmul ops take a deprecated bandwidth fallback. |
| CLI output | About 11 copy-pasted `_detect_format` helpers. | Extract one shared helper before adding more CLIs. |

## 4. Architecture overview

```
 soc_designs/ip/*.yaml          soc_designs/designs/*.yaml      embodied-schemas
 (node-independent IP)  ---->   (SoC composition)               process-nodes/, cooling-solutions/
            \                          |                                |
             \                         v                                v
              +------------> compose_soc(design, node, profile) <-------+
                                        |
                                        v
                          SoCInstance  (areas, clocks, peaks,
                                        per-block power model)
                               |                  |
                 emits ComputeProduct             |
                 -> ValidatorRegistry             |
                    (AREA/THERMAL/ENERGY/GEOMETRY)|
                                                  v
 workloads/pipelines/autonomy/*.yaml -> PipelineWorkload --regime--> StageDemand[]
                                                  |
                   soc_designs/efficiency/*.yaml -+-> Mapper (explicit | greedy | ILP)
                   (kernel_class x engine x prec) |
                                                  v
                                          Scheduler / roofline per stage
                                          (service time, utilization, DRAM BW,
                                           DAG critical path, deadlines)
                                                  |
                                                  v
                                          SoC energy roll-up
                                          (dynamic + memory + leakage + DRAM,
                                           idle gating, thermal clamp)
                                                  |
                                                  v
                                   SoCAnalysisResult (JSON / MD / CSV / text)
                                   analyze_soc.py | sweep_soc.py | show_soc.py
```

Package layout, following the `CLAUDE.md` structure:

```
src/graphs/core/pipeline_workload.py        # Stage, Regime, PipelineWorkload, aggregation
src/graphs/hardware/soc/
    ip_block.py                             # IPBlockTemplate + loader
    design.py                               # SoCDesign (composition) + loader
    clocking.py                             # fmax(node, class, Vdd), operating points
    area.py                                 # multi-block area (generalized silicon_math)
    power.py                                # per-block power; KPU blocks delegate to kpu_power_model
    compose.py                              # compose_soc() -> SoCInstance (+ to_compute_product())
src/graphs/estimation/soc/
    efficiency.py                           # kernel_class x engine_kind x precision table
    mapping.py                              # explicit / greedy / ILP stage->engine assignment
    schedule.py                             # utilization, deadlines, DRAM BW, critical path
    energy.py                               # regime-level energy roll-up
    analyzer.py                             # SoCAnalyzer.analyze() -> SoCAnalysisResult
src/graphs/reporting/output_format.py       # shared --output format detection + writers
soc_designs/{ip,designs,efficiency,studies}/
workloads/pipelines/autonomy/
cli/{analyze_soc.py, sweep_soc.py, show_soc.py, show_pipeline_workload.py}
```

## 5. Abstractions

### 5.1 IP block templates (process-node independent)

An IP template describes one reusable block in terms that do not depend on the
process node, so any node can be applied later.

- **Silicon.** A list of `(name, circuit_class, transistor_source)`, reusing the
  `SiliconBinBlock` / `TransistorSource` vocabulary (`fixed`, `per_kib`,
  `per_unit`). Die-shot areas can be given instead as
  `reference_area_mm2 @ reference_node`. The loader converts them to
  transistors with the reference node's density for that library, so analog, IO
  and SRAM correctly scale worse than logic on newer nodes.
- **Compute.** `units`, `ops_per_clock_per_unit[precision]`,
  `architecture_class` (the key into `architectural_energy.ArchitectureClass`),
  `engine_kind`, and `dispatch_overhead_us`, which captures the kernel-launch
  cost the PDF (section 8) says dominates MPC/CBF.
- **Fixed-function blocks** (ISP, codec) declare
  `throughput_mpix_per_s_per_ghz` and `energy_pj_per_pixel_ref` instead of
  ops/clock.
- **Local memory.** L1/L2 per unit and shared cache KiB. This feeds both area
  and the working-set fit test in 5.5.
- **Clock.** `fmax_ghz_ref` at `reference_node` and nominal Vdd.
- **KPU blocks** reference the existing `KPUArchitecture` / `KPUSKUInputSpec`
  tile definitions and delegate to `generate_kpu_sku()` / `kpu_power_model`.
  The KPU's own memory PHYs are stripped, because the SoC shared memory
  subsystem replaces them. The KPU is modeled as a domain-flow engine, not
  generic dataflow.
- **Heterogeneous KPU tiles.** Per `docs/plans/kpu-heterogeneous-tile-refactor-plan.md`,
  the KPU checkerboard can host `pe_fabric`, `systolic` and `fixed_function`
  tiles. Each KPU tile class is a separate engine in this analyzer.
  - ISP, codec, SGM and VIO engines are defined once as a `FunctionCore`, then
    placed either in the KPU mesh (as a tile) or on the SoC die (as a
    standalone block). That placement is a sweepable allocation choice.
  - Workload `segments` (named stage sets) let a fixed-function tile absorb
    several stages, removing their intermediate traffic.

Initial IP set:

- CPU: Cortex-A78AE x4 cluster, Cortex-A720AE x4 cluster.
- GPU: Ampere SM (Orin), Blackwell SM (Thor).
- NPU: NVDLA v2.
- DSP: a PVA-class vector DSP.
- ISP: 3-camera, 1440x1080 at 60 fps.
- Video codec: H.265 4K60 encode/decode.
- KPU: tile classes from the existing generator (INT8, BF16, Matrix).
- Shared blocks: system-level cache (SLC) SRAM, NoC, LPDDR5/5X controller+PHY
  per channel, MIPI CSI, PCIe, Ethernet, always-on/security/control island.

### 5.2 SoC design (composition)

```yaml
id: branes_hetero_a
process_node: tsmc_n7                 # default; --node overrides
blocks:
  - {instance: cpu,  ip: arm_cortex_a78ae_x4, count: 2}
  - {instance: gpu,  ip: nvidia_ampere_sm,    count: 4}
  - {instance: npu,  ip: nvdla_v2,            count: 1}
  - {instance: pva,  ip: vector_dsp_pva2,     count: 1}
  - {instance: kpu,  ip: kpu_t64_32x32}       # delegates to the KPU generator
  - {instance: isp,  ip: isp_3cam_1440p60}
  - {instance: venc, ip: codec_h265_4k60}
shared:
  system_cache_kib: 8192
  noc: {topology: mesh, bisection_gbps: 512}
  memory: {type: lpddr5x, channels: 8, bus_bits: 128, data_rate_mtps: 8533}
  io: {mipi_csi_lanes: 16, pcie: {gen: 4, lanes: 4}, eth_10g: 1}
layout: {whitespace_fraction: 0.15, io_ring_mm: 0.30}
power_profiles:
  - name: 25W
    cooling_solution_id: passive_heatsink_large
    operating_points: {cpu: {clock_ghz: 1.5}, gpu: {clock_ghz: 0.9}, kpu: {profile: 15W}}
# Any scalar may be a list -> sweep axis (for example kpu.tiles: [32, 64, 128]).
```

`compose_soc(design, node, profile)` produces an `SoCInstance` with:

- **Area.** Block area = sum over circuit classes of transistors /
  density(node, class). Die area = sum of blocks x (1 + whitespace), plus the IO
  ring.
- **Shoreline check.** Also compute the minimum die edge needed to fit the
  LPDDR/MIPI/PCIe PHYs along the edge. PHYs barely shrink, so a small-node SoC
  can end up pad-limited.
- **Peak performance** per precision, per engine.
- **Operating points** per block: clock and Vdd.
- **Per-block power model.**

`SoCInstance.to_compute_product()` emits a multi-block `Die`, so the existing
AREA, THERMAL, ENERGY and GEOMETRY validators run unchanged once `silicon_math`
is generalized.

### 5.3 Process-node retargeting

- **Area and leakage.** Already keyed on `(node, circuit_class)` in
  `ProcessNodeEntry`, so these retarget directly.
- **Dynamic energy.** The base is `energy_per_op_pj["<class>:<prec>"]` (ALU
  only). The architectural overhead comes from the existing
  `ArchitecturalEnergyModel` / `OperandFetchEnergyModel` for the block's
  `architecture_class`. Without that overhead a CPU looks 10-50x too efficient.
  Scale by (V/Vnom)^2.
- **Clock: gap.** `ProcessNodeEntry` has no speed data.
  - Proposed: add optional fields `logic_speed_factor` per circuit class
    (relative to a reference node) and `vdd_min_v` / `vdd_max_v` to embodied-schemas.
  - Derive fmax(node, Vdd) with an alpha-power law, anchored at
    `fmax_ghz_ref`.
  - Values are THEORETICAL, taken from foundry iso-power speed disclosures.
  - An explicit `clock_ghz` in the design always wins.
- **Missing figures.** When a node lacks SRAM/NoC pJ or a leakage Vdd exponent,
  use the power model's existing defaults. Downgrade that number's confidence
  and list the substitutions in the report.

### 5.4 Workload: `PipelineWorkload` (stages x regimes)

**Stage fields:**

- Identity: `id`, `pipeline_tier` (`T1`..`T7`; do not call this field `tier`,
  because that name already means power class and memory tier elsewhere),
  `name`, `algorithms`.
- `kernel_class`, used for engine efficiency:
  - `dense_conv_gemm`, `attention_prefill`, `weight_stream_decode`, `elementwise_norm`
  - `cost_volume_dp` (SGM), `fft`, `feature_track`
  - `sparse_hash_scatter`, `knn_tree`, `raycast`, `wavefront`
  - `small_dense_linalg` (Cholesky/Schur/Gauss-Newton), `small_qp`, `graph_search`
  - `pixel_fixed_function`
- `ops_per_call[precision]`: the split across int8/fp16/bf16/fp32/fp64.
  Precision class A/B/C is derived from the split, not stored.
- `bytes_per_call`: `ingress`, `egress`, `internal` (DRAM traffic under
  competent blocking), `weights`.
- `working_set_bytes`: decides whether internal traffic stays on chip.
- `serial_kernels`: the dispatch-overhead count.
- `depends_on`: stage DAG edges, used for glass-to-actuator latency.

**Regime fields:**

- `id`, `governing_constraint`, `loop_deadline_ms`, `power_budget_w`.
- Per stage: `{enabled, rate_hz, latency_budget_ms}`.

**Aggregation** (pure data, no hardware): sustained ops/s, class A/B/C shares,
ingress and DRAM GB/s per regime.

**Acceptance test: reproduce the annex tables.**

| Metric | Minimum usable | Contested | Far flight | Fast flight | Air superiority |
|---|---|---|---|---|---|
| Sustained TOP/s | 1.15 | 5.34 | 10.24 | 7.23 | 12.81 |
| Class A share | 82.1% | 69.4% | 69.0% | 72.6% | 83.4% |
| DRAM traffic (GB/s) | 16 | 156 | 289 | 182 | 100 |

**Extension stages** (Decision D5), flagged `extension: true` so the
reproduction test excludes them:

- ISP RAW->YUV for 3 cameras.
- Video encode for the datalink in Minimum usable, where semantics are
  offloaded.

These give the ISP and codec silicon real work instead of being dead area.

### 5.5 Engine efficiency and the stage service-time model (L0)

`soc_designs/efficiency/default_v1.yaml` maps
`(kernel_class, engine_kind, precision)` to `{supported, compute_eff, bw_eff}`,
each with a source and a confidence.

- **Unsupported pairs are infeasible, never silently promoted.** This avoids
  the roofline FP64 -> FP32 fallback.
- **Dense conv on GPU/DLA** is INTERPOLATED from the existing Orin calibrations
  in `hardware_registry/gpu/jetson_orin_agx_gpu/calibrations` and
  `accelerator/nvidia_dla_orin/calibrations`.
- **KPU entries:**
  - SURE-expressible kernels (`small_dense_linalg`, `cost_volume_dp`,
    `wavefront`, `dense_conv_gemm`, `fft`) get domain-flow efficiencies.
  - Irregular kernels (`sparse_hash_scatter`, `knn_tree`, `graph_search`) are
    marked unsupported or low.
  - All KPU entries are THEORETICAL until the wavefront cost model (Phase 5)
    replaces them.
- **`annex_v1` table.** A second table reproduces the PDF's stated effective
  throughputs: Class A 2000, Class B 300 and Class C 15 GOP/s on an Orin-class
  part. This is a regression target (Phase 3).

Stage `s` on engine `e`:

```
dram_bytes = ingress + egress + weights + (internal if working_set > sram_avail(e) else 0)
t_compute  = sum_p ops_p / (peak_e,p * compute_eff)
t_memory   = dram_bytes / (bw_share_e * bw_eff)
t_service  = max(t_compute, t_memory) + serial_kernels * dispatch_overhead_e
bound      = COMPUTE | MEMORY | DISPATCH
```

### 5.6 Mapping and scheduling

- **Mapping**, in order of delivery:
  - Explicit (the design or study file names an engine per stage).
  - Greedy: lowest energy among engines that stay feasible.
  - ILP on `scipy.optimize.milp` (Phase 5; scipy is installed but not declared,
    so add it as the `soc` extra).
  - Phase 1 maps each stage to a single engine. Split mappings (INT8 trunk on
    NPU, FP16 head on GPU) with a transfer cost come in Phase 5.
- **Per-regime checks:**
  - Engine utilization: `U_e = sum rate_s * t_service <= 1`. CPU clusters are
    N parallel servers.
  - Shared DRAM: `sum rate_s * dram_bytes_s <= BW_peak * sustained_fraction`
    (default 0.65 for scattered access, per the PDF).
  - Constraint ratio `t_service / latency_budget`, the same metric as the PDF,
    so "stages over" counts compare directly.
  - DAG critical path against `loop_deadline_ms`.
- **Phase 5:** response-time schedulability (RM/EDF) and bandwidth-contention
  inflation, which replace the "owns the whole engine" lower bound.

### 5.7 Energy roll-up per regime

```
P_dyn     = sum_s rate_s * sum_p ops_p * E_op(node, class_e, p) * overhead(arch_class_e) * (V_e/Vnom)^2
P_mem     = SRAM bytes * sram_pj + NoC flits * hops * noc_pj + DRAM bytes * (phy_pj + device_pj)
P_leak    = sum_blocks area * leakage_w_per_mm2(class) * (V/Vnom)^exp   [x0 for gated idle engines if --gate-idle]
P_fixed   = sum rate * pixels * E_pixel(node)
P_total   = P_dyn + P_mem + P_leak + P_fixed    -> checked against profile TDP and cooling max_total_w / W per mm^2
```

- On-die energy and off-die DRAM device energy are reported separately.
- "Useful TOPS/W" uses the workload's ops, not the nameplate peak.
- The idle-gating toggle quantifies PDF section 9: the area and leakage paid
  for engines that sit idle in a given regime.

### 5.8 Result schema (`SoCAnalysisResult`, JSON-first for the orchestrator)

```
design, node, profile, regime, confidence_summary
die:      area_mm2, transistors_b, by_block[], by_circuit_class[], shoreline_ok, findings[]
peak:     ops_per_s[precision] per engine
stages[]: id, pipeline_tier, engine, t_service_ms, budget_ms, ratio, bound, dram_gbs, energy_mj_per_s, confidence
engines[]: utilization, dynamic_w, leakage_w, gated
memory:   dram_demand_gbs, dram_supply_gbs, headroom
power:    dynamic_w, memory_w, leakage_w, fixed_w, dram_device_w, total_w, tdp_w, cooling_ok
summary:  feasible, stages_over, e2e_latency_ms, useful_tops_per_w
```

## 6. CLI tools

All four use the shared `reporting/output_format.py` for `--output`
auto-detection (JSON/CSV/MD/text). They deliberately bypass `UnifiedAnalyzer`,
because there is no PyTorch model. Document that exception in the tool
docstring, the same way `analyze_operator_roofline.py` does.

| Tool | Purpose |
|---|---|
| `cli/analyze_soc.py --design D --workload W [--regime R\|all] [--node N] [--profile P] [--mapping auto\|explicit\|file] [--efficiency default_v1\|annex_v1] [--gate-idle] -o out.json` | One design, one node: die, performance and energy per regime |
| `cli/sweep_soc.py --study S` or `--designs a,b --nodes tsmc_n16,tsmc_n7,tsmc_n5 --regime all -o sweep.csv [--pareto area,power]` | Allocation x node x regime sweeps, Pareto front, "union of regimes" minimum design |
| `cli/show_soc.py --design D --node N [--list-ip]` | Area breakdown, peak performance and clocks at a node; lists IP |
| `cli/show_pipeline_workload.py --workload W [--regime R]` | Stage table and regime aggregates, reproducing the annex numbers |

`cli/validate_sku.py` gains `--soc D --node N`, which validates the emitted
ComputeProduct.

Python API for the Embodied-AI-Architect tool call:
`SoCAnalyzer().analyze(design, workload, node, regime, profile)`, with a
`to_dict()` result. Register it with the MCP server (`docs/mcp-server.md`) in
Phase 4.

## 7. Phased delivery (PR sequence)

The phases are ordered breadth first: a thin, end-to-end number by Phase 3,
then depth. Validation against measurements is a separate later pass. The only
anchors gated here are the paper reproductions and the Orin reconstruction.

**Phase 0: plan and data** (this document)
- PR 0.1: this plan, plus a tracking issue.
- Obtain the "Autonomy Workload -- Data Annex" (D3). It has per-call ops and
  bytes and per-regime rates. Without it, only the sizing regime can be
  populated.

**Phase 1: workload as data**
- PR 1.1: `core/pipeline_workload.py` (Stage, Regime, PipelineWorkload,
  aggregation) and unit tests.
- PR 1.2: `workloads/pipelines/autonomy/branes_7tier_v1.yaml`, with 17 stages
  x 5 regimes and provenance on every field.
- PR 1.3: `cli/show_pipeline_workload.py` and `reporting/output_format.py`
  (the shared helper).
- **Accept:** the annex regression test passes. Totals match to within 1%, and
  the class shares and DRAM GB/s match as shown in 5.4.

**Phase 2: IP library, composition, die area**
- PR 2.1: `hardware/soc/ip_block.py`, `design.py` and loaders. Initial IP YAMLs
  (5.1).
- PR 2.2: generalize `silicon_math.py` to multi-block, multi-die, keeping the
  KPU path byte-identical (pinned by the existing tests).
  `compose_soc()` and `to_compute_product()`.
- PR 2.3: `clocking.py`, plus an embodied-schemas PR adding the node speed
  fields (5.3).
- PR 2.4: `cli/show_soc.py`, SoC validators (PHY shoreline, per-block W/mm^2),
  and `validate_sku.py --soc`.
- **Accept:**
  - An `orin_class_reference` design at `samsung_8lpp` lands within 15% of the
    455 mm^2 / 17 B transistors in the catalog.
  - Its GPU and DLA peak INT8 TOPS match the datasheet.
  - Re-running it at `tsmc_n7` and `tsmc_n5` gives monotonic area and energy,
    with IO/analog scaling visibly worse than logic.

**Phase 3: L0 analyzer, end to end**
- PR 3.1: `estimation/soc/efficiency.py`, plus the `default_v1` and `annex_v1`
  tables.
- PR 3.2: `mapping.py` (explicit and greedy) and `schedule.py` (utilization,
  DRAM, ratios, critical path).
- PR 3.3: `power.py` / `energy.py` roll-up with `--gate-idle` and the thermal
  check.
- PR 3.4: `SoCAnalyzer`, `SoCAnalysisResult`, `cli/analyze_soc.py`, and
  confidence propagation.
- **Accept:**
  - The `orin_class_reference` design with `--efficiency annex_v1` reproduces
    the annex's 2.7 / 9.2 / 16.2 / 12.9 / 17.3 s of compute per second of
    flight to within 5%.
  - Its "stages over" counts match the annex:

    | Regime | Stages over |
    |---|---|
    | Minimum usable | 2 of 16 |
    | Contested | 4 of 17 |
    | Far flight | 4 of 17 |
    | Fast flight | 7 of 17 |
    | Air superiority | 8 of 15 |

  - The same design at 25 W sits in the same order of magnitude as the measured
    101.8 GOPS/W.

**Phase 4: studies**
- PR 4.1: `cli/sweep_soc.py`. List-valued design fields become sweep axes.
  Study YAMLs go in `soc_designs/studies/`.
- PR 4.2: Pareto and "union of regimes" reports, plus plots (a matplotlib
  optional extra).
- PR 4.3: first study, "Orin-class vs KPU-heterogeneous at N16 / N7 / N5 across
  5 regimes". Write it up in `docs/assessments/`.
- PR 4.4: MCP tool registration.

**Phase 5: fidelity refinements** (driven by what the first studies expose)
- Mapping and scheduling: ILP mapper, split mappings, RM/EDF response time,
  DRAM contention, DVFS per regime.
- L1 DNN stages: trace representative models (a YOLOe-class detector, a 2B VLM,
  the SDF encoder, the DRL policy) through `UnifiedAnalyzer` on a per-block
  synthesized `HardwareResourceModel`.
- L1 KPU stages: the domain-flow wavefront/polytope cost model.
- A multi-block SoC floorplan, extending `silicon_floorplan.py`.
- Die cost and yield (D7).
- Upstream the stabilized schemas (IP block, SoC design, `PipelineWorkload`, and
  the ISP/VIDEO_CODEC block kinds) into embodied-schemas.

## 8. Confidence and validation anchors

| Quantity | Initial confidence | Anchor |
|---|---|---|
| Node density, leakage, pJ/op | THEORETICAL (as in the catalog) | Sharpens automatically when PDK overlays land (`PROCESS_NODE_DATA_DIR`) |
| IP transistor budgets | THEORETICAL (die-shot estimates) | Orin reconstruction against 455 mm^2 / 17 B |
| Dense conv efficiency on GPU/DLA | INTERPOLATED | Existing Orin calibration JSONs |
| All other kernel-class efficiencies | THEORETICAL | `annex_v1` reproduction. Microbenchmarks later, under the hardware-centric validation plan. |
| Workload ops, bytes, rates | THEORETICAL (structural estimates, per the annex) | Data Annex provenance per field |

A result's confidence is the minimum of its inputs' confidences, and the report
lists which input set it.

## 9. Cross-repo impact

- **embodied-schemas:**
  - Optional node speed fields on `ProcessNodeEntry` (Phase 2).
  - `ISP` and `VIDEO_CODEC` `BlockKind`s (Phase 2 if we want the emitted
    ComputeProduct to carry them as first-class blocks; otherwise they go in as
    `fixed` silicon_bin lines, as Orin does today).
  - Later: IP, SoC design and pipeline-workload schemas.
- **Embodied-AI-Architect:** consumes `SoCAnalysisResult.to_dict()` through the
  MCP tool. Run the `cross-repo-checker` agent before PR 3.4 and PR 4.4.
- **Existing tests:** the KPU roundtrip invariant
  (`test_roundtrip_power_default_tdp_is_derived`) and the silicon_math tests
  must stay green through the PR 2.2 generalization.

## 10. Decisions for review

| # | Decision | Recommendation |
|---|---|---|
| D1 | Where SoC specs live | New `soc_designs/` tree, not `hardware_registry/soc/` (section 2) |
| D2 | Schema home | Graphs-local first, as `KPUSKUInputSpec` is. Upstream to embodied-schemas in Phase 5 once stable. |
| D3 | Workload source data | Need the Data Annex (per-call ops/bytes, per-regime rates, class split inside mixed stages). Where is it, and can it be committed? |
| D4 | Process nodes in scope | The 13 existing nodes, plus TSMC N3E / N2 and Samsung SF4 if leading-edge studies are wanted. Each needs density, leakage, pJ and speed figures. |
| D5 | ISP and codec workload | Add flagged extension stages (camera ISP, datalink encode) so those blocks carry load |
| D6 | First fidelity level | L0 table-driven (Phase 3) before L1 traced DNN stages |
| D7 | Die cost and yield | Defer to Phase 5 (wafer cost per node, D0, negative-binomial yield). Say if it should come earlier. |
| D8 | CPU FP64 path | Model FP64 explicitly per engine (Orin GPU FP64 is a small fraction of FP32). Class C stages on engines without FP64 are infeasible, not approximated. |

## 11. Risks and pitfalls

- **Overhead-free energy.** Using `energy_per_op_pj` without architectural
  overhead makes CPU and GPU look an order of magnitude too efficient. Always
  compose with `architectural_energy` / `operand_fetch`.
- **Silent precision promotion.** The existing roofline treats a missing FP64
  profile as the FP32 peak. The SoC path must fail closed.
- **Stale registry data.** Never read `hardware_registry/accelerator/kpu_*`
  spec.json for KPU numbers. Go through the ComputeProduct and the generator.
- **Transistor budgets for third-party IP are not public.** Keep them
  THEORETICAL and let the Orin reconstruction bound the error. Expose every
  budget in `show_soc` so reviewers can challenge it.
- **The per-stage ratio is a lower bound**, because it assumes the stage owns
  the engine, just as in the PDF. Label it that way until Phase 5 adds
  contention.
- **`robot_estimator_v8.py`** has a power-clamp unit bug (it adds watts to
  joules). Reuse none of its numbers. Its `OP_EFF_MOD` idea is superseded by the
  efficiency table.
