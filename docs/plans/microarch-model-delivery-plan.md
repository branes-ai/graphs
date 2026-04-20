# Plan: Micro-architectural Model Delivery (Layers 1–7)

**Status:** Draft for review
**Date:** 2026-04-20
**Related:**
- [`docs/plans/bottom-up-microbenchmark-plan.md`](bottom-up-microbenchmark-plan.md) — long-term validation roadmap (9 layers). **Re-sequenced**: its measurement work runs *after* this plan completes.
- [`docs/assessments/bottom-up-microbenchmark-coverage.md`](../assessments/bottom-up-microbenchmark-coverage.md) — current-state assessment (unchanged).

## Purpose

Deliver analytical energy and latency models for Layers 1–7 of the micro-architectural hierarchy, end-to-end through the reporting tool (HTML + PowerPoint + JSON), on a compressed schedule to support **investor presentations and product planning**. System-level validation (microbenchmark writing, power-meter integration, measurement campaigns, CI residual gates) is explicitly deferred to a follow-on phase that runs after this plan ships.

Layers 8 (distributed memory, intra-server) and Layer 9 (cluster interconnect, inter-server) are **out of scope** for this delivery push; they will be addressed after M8.

## Principles

1. **Ship models before measurements.** Expert judgment is the acceptance gate during this push. `THEORETICAL` or `INTERPOLATED` confidence ships visibly; no stakeholder is told a number is measured when it isn't.
2. **No validation interleaving.** No power-meter work, no microbenchmark kernels, no composition tests, no hardware runs gate any milestone M0–M8.
3. **Shippable at each milestone.** M0 produces a running tool with panels stubbed. Each M1–M7 adds a fully populated layer panel across all target SKUs.
4. **Validation-ready handoff.** Every milestone leaves the schema slots, JSON fields, and CLI flags the later validation pass will need, so measurement drops in without refactors.
5. **Investor-readable artifacts.** The report is the deliverable, not the code. Every milestone ends with the HTML/PPT artifact regenerated and reviewed.

## Target SKUs for v1 (locked 2026-04-20)

Populate all seven layers for each:

- **GPU:** Jetson Orin AGX (`jetson_orin_agx_64gb`).
- **CPU:** Intel i7-12700K, AMD Ryzen 9 8945HS.
- **KPU accelerators:** T64, T128 (**new SKU — configuration to be built**), T256. T64 and T256 configurations are refreshed alongside T128 because the KPU tile structure needs a redo. KPU T768 is out of scope for this delivery.
- **Edge TPU:** Coral Edge TPU (`coral_edge_tpu`). The `tpu_edge_pro` variant is out of scope unless promoted during plan review.
- **Edge DSP:** Hailo-8 (`hailo8`), Hailo-10H (`hailo10h`).

~9 SKUs; per-milestone effort scales roughly linearly with SKU count.

---

## Milestones

### M0 — Scaffolding (3–5 days)

Foundation that every subsequent milestone depends on. No layer content yet.

**Deliverables:**
- `LayerTag` enum with all 9 entries in `src/graphs/benchmarks/schema.py` (Layers 8 and 9 defined but unused this phase).
- `HardwareResourceModel.field_provenance: Dict[str, EstimationConfidence]`.
- Empty dataclass stubs: `SoCFabricModel` (`hardware/fabric_model.py`), `IntraServerFabricModel` (`hardware/intra_server_fabric_model.py`), `ClusterInterconnectModel` (`hardware/cluster_interconnect_model.py`) — field shapes only; values in later phases or later plans.
- `cli/microarch_validation_report.py` skeleton with flags `--hardware`, `--layer`, `--precision`, `--output`, `--format {html,pptx,json}`.
- HTML template with per-SKU page shell plus 7 empty layer panels and a cross-SKU comparison page.
- **Branes branding:** template header and footer integrate `docs/img/Branes_Logo.jpg`; colors and typography chosen to match brand. Every generated HTML page and PPT slide carries the logo — this report ships externally.
- JSON data contract: `reports/microarch_model/<date>/data/<sku>.json`.
- Legend/README explaining the confidence ladder (`THEORETICAL` / `INTERPOLATED` / `CALIBRATED`) so readers don't misread `THEORETICAL` as "wrong".

**Explicitly not in M0:**
- Power meter (`PowerMeter` abstraction).
- Composition test runner.
- Any benchmark kernel.
- CI residual-gate hooks.

**Exit criteria:**
- `./cli/microarch_validation_report.py --hardware jetson_orin_agx_64gb --output tmp/` produces HTML with `"NOT YET POPULATED"` placeholders in all 7 panels, Branes-branded chrome visible, and a working PowerPoint export.
- Unit tests cover JSON round-trip and the dataclass-stub serialization.

---

### M0.5 — KPU Tile Model Refinement + Compute-Archetype Comparison Harness (~7–10 days)

Prerequisite for any KPU coverage in M1–M7. This milestone does two things: (1) refines the KPU tile abstraction to reflect what the architecture actually is, and (2) stands up the three-way GPU vs. TPU vs. KPU comparison view that the product positioning lives or dies by.

**Context — why the KPU tile model needs refinement.**
The current KPU tile model was set up as a matrix tile to reach peak-ops/s parity with NVIDIA Tensor Cores. Tensor Cores are concurrent dot products with reduction trees; modeling the KPU tile the same way was a convenient apples-to-apples shortcut but misrepresents the architecture. The KPU is a **distributed dataflow fabric**: each tile executes **dot-product wavefronts** across a PE array and relies on **perfect pipelining** for high utilization and low energy per op. The competitive story follows from that: the KPU **does not win on peak ops/s** (Tensor Cores do; we compensate with larger PE arrays such as 32×32), it **wins on energy per op** in the steady-state pipelined regime. Modeling the tile as "big matrix multiplier" erases that story. Modeling it as "dataflow pipeline with fill/drain separated from per-PE steady-state MAC energy" surfaces it.

This milestone is partly exploratory — PE array shape and tile count for T64 / T128 / T256 are design levers, and we need the comparison harness in place to play with them before committing.

**Part A — KPU tile abstraction refinement:**
- Replace the matrix-tile abstraction in `src/graphs/hardware/models/accelerators/kpu_*.py` with a dataflow-tile abstraction carrying:
  - PE array shape per tile (design knob; 16×16 is today's default, 32×32 is the likely target).
  - Wavefront scheduling model: dot-product wavefronts streaming through the PE array, with **per-wavefront fill and drain cycles accounted separately** from the steady-state MAC energy.
  - Per-PE steady-state MAC energy — the field where the KPU energy advantage is expressed and defended.
  - **`pipeline_utilization` surfaced as an explicit parameter** (perfect = 1.0). The report shows how much of the efficiency claim depends on this assumption and what realistic values across workload shapes look like. If the advantage collapses under realistic utilization, that is a genuine product finding that must be visible in the artifact, not hidden.
  - Per-tile scratchpad semantics (carried over).
  - Tile-to-tile NoC slots for inter-tile wavefronts (Layer 6 concern; hooks added here for M6 to populate).
- Document the new tile model in `docs/hardware/` so the story is written down explicitly, not implicit in code.

**Part B — Configure T64 / T128 / T256 under the refined abstraction:**
- Re-parameterize T64 and T256 against the dataflow-tile abstraction. Larger PE arrays (32×32) are on the table as a design-space lever; the three SKUs may shift shape during exploration.
- Build T128 as a new SKU sized between T64 and T256 in dataflow-tile terms.
- Register all three in `hardware/mappers/__init__.py` and the factory registry.
- T768 remains out of scope.

**Part C — GPU vs. TPU vs. KPU comparison harness:**
Add a dedicated **compute-archetype comparison page** (`compare_archetypes.html`) to the M0 report skeleton, representing the three archetypes with Jetson Orin AGX Tensor Cores (SIMT + reduction trees), Coral Edge TPU (systolic weight-stationary), and KPU T128 (distributed dataflow). The page stands up at M0.5 with the limited data available at this stage and gains detail as M1–M7 land.

Five visualizations, each designed around the product narrative:
1. **Energy per op** at fixed precision across archetypes — KPU's argument.
2. **Peak ops/s** at fixed precision — Tensor Cores' argument; shown honestly so the positioning is credible.
3. **Ops/W** (energy efficiency) — where the three curves cross; the deciding chart.
4. **Pipeline / utilization sensitivity** — KPU ops/W as a function of `pipeline_utilization` across 0.5 → 1.0; Tensor Core ops/W as a function of warp-divergence rate; TPU ops/W as a function of matmul shape relative to the systolic array. Makes load-bearing assumptions visible.
5. **Array-size scaling** — KPU ops/s and ops/W as PE array grows from 16×16 to 32×32 to 64×64; shows where the "compensate with larger arrays" story has headroom and where it runs into diminishing returns.

The comparison harness is the exploration tool we use during M0.5 to decide the T64/T128/T256 array sizes, not just a deliverable to be admired after the fact.

**Part D — Internal-consistency tests:**
- Unit tests asserting the expected comparative behavior: KPU energy per op below Tensor Core at matched precision in the steady-state regime; pipeline fill/drain dominant at small workload sizes; per-op energy decreases (within a regime) as PE array grows; KPU peak ops/s below Tensor Core at matched silicon area.
- These are **internal-consistency** tests, not silicon validation. They catch sign errors, unit mistakes, and regressions in the comparative claims as the model is refined interactively.

**Explicitly not in M0.5:**
- KPU T768.
- Any silicon measurement of KPU, TPU, or GPU energy.
- **TPU mapper redo.** The TPU mapper stays as the current systolic-array weight-stationary model. If the comparison harness surfaces that the TPU model lacks the accounting granularity needed for a fair three-way comparison (e.g., per-PE MAC energy or fill/drain separation), that becomes a separate tracked issue — a detour into TPU refactoring is not in this milestone's budget.
- Layer 1–7 numeric population — starts in M1. M0.5 produces the structural KPU config plus the comparison scaffolding; per-layer numeric content fills in as the layer milestones progress.

**Exit criteria:**
- `get_mapper_by_name('kpu_t64')`, `kpu_t128`, `kpu_t256` resolve under the refined dataflow-tile abstraction with `pipeline_utilization` surfaced.
- `docs/hardware/` carries a written description of the new tile model and the positioning rationale.
- `compare_archetypes.html` renders with all five visualizations, populated from whatever analytical data is available at this point (stub values where M1–M7 haven't landed yet); every visualization has working interactivity (e.g., the utilization slider for chart 4, the array-size toggle for chart 5).
- Plan-owner review confirms (a) the tile model captures the distributed-dataflow pipeline story, (b) the comparison views are useful for design-space exploration, and (c) the chosen PE array sizes for T64 / T128 / T256 are the ones to lock before M1 populates numeric content.

**Risks specific to this milestone:**
- **Array-size exploration open-ended.** Trying 16×16 vs. 32×32 vs. 64×64 for each SKU can consume unbounded time. Timebox to the ~7–10 day budget; exploration beyond that spills to a post-M8 follow-up issue.
- **TPU accounting granularity.** If the TPU mapper doesn't expose the right fields for a fair comparison, the M0.5 deliverable is an *unfair* comparison with a tracked follow-up issue rather than an M0.5 slip. Flag at end of Part C.
- **Perfect-pipelining assumption is load-bearing.** Part C chart 4 is the mitigation: make the assumption visible so the KPU story survives scrutiny.

---

### M1 — Layer 1: ALU / MAC / Tensor Core (3–4 days)

**Model content:**
- Populate `ComputeFabric.ops_per_unit_per_clock[precision]` per SKU for all supported precisions (FP64 / FP32 / TF32 / BF16 / FP16 / FP8 / INT8 / INT4).
- Populate per-precision energy coefficients via `PROCESS_NODE_ENERGY × CIRCUIT_TYPE_MULTIPLIER`. Attach `field_provenance = THEORETICAL` to each entry.
- Integrate DVFS curve from existing `gpu_clock.py` / `cpu_clock.py` as `INTERPOLATED` where it spans a known family.

**Report content:**
- Layer 1 panel per SKU: predicted ops/s and pJ/op for each precision, source tag (datasheet vs. tech profile), confidence badge.
- Cross-SKU comparison chart: ops/W at FP16 and INT8.

**Tests:** model self-consistency (ranges sane, no sign errors, monotone with precision narrowing). No hardware measurement.

**Exit criteria:** domain expert reviews the Layer 1 panel for one GPU, one CPU, one accelerator SKU; signs off or queues refinements. Refinements folded before M2 starts.

---

### M2 — Layer 2: Register File (3–4 days)

**Model content:**
- `StoredProgramEnergyModel.register_file_read/write_energy` parameterized from `TechnologyProfile.register_{read,write}_energy_pj`.
- `DataParallelEnergyModel.warp_divergence_penalty` and `warp_divergence_rate` per-SKU defaults (keep analytical 5% for now; mark `THEORETICAL`).
- SIMD-efficiency coefficients in `CPUMapper._analyze_vectorization` exposed with `field_provenance` tags (currently hard-coded 0.95 / 0.80 / 0.70).
- Systolic pipeline-fill overhead coefficients for TPU/KPU mappers (documented analytical formula, tagged `THEORETICAL`).

**Report content:** Layer 2 panel per SKU; register-read energy as fraction of ALU energy chart.

**Exit criteria:** expert review on representative GPU, CPU, TPU, and KPU SKU; sign-off or refinement queue.

---

### M3 — Layer 3: L1 Cache / Scratchpad (3–4 days)

**Model content:**
- Populate `HardwareResourceModel.l1_cache_per_unit` from datasheets for every target SKU.
- Per-op L1 hit-rate model in `DataParallelEnergyModel.l1_hit_rate` with documented per-op-type defaults (replacing the single 0.95 constant with a small lookup table; tagged `THEORETICAL`).
- L1 read/write energy coefficients from the technology profile.
- KPU scratchpad residency model (256 KB tile, L1 viewed as scratchpad semantically).

**Report content:** Layer 3 panel per SKU; per-op hit-rate table; scratchpad-vs-cache annotation for accelerators.

---

### M4 — Layer 4: L2 Cache (3–4 days)

**Model content:**
- Populate `HardwareResourceModel.l2_cache_total` and, where applicable, `l2_cache_per_unit`.
- Per-op L2 hit-rate lookup (replacing 0.90 constant).
- L2 read/write energy coefficients.
- Prefetch-effectiveness coefficient (analytical, tagged `THEORETICAL`).

**Report content:** Layer 4 panel per SKU; L2 topology annotation (per-core vs. shared).

---

### M5 — Layer 5: L3 / LLC / Shared Cache (3–4 days)

**Model content:**
- Populate `HardwareResourceModel.l3_cache_total` for SKUs with a distinct L3.
- GPU LLC (often L2-as-LLC) hit-rate model per op type.
- Coherence-state overhead coefficients for multi-core CPU (true-sharing vs. private-line penalty; analytical).
- Annotate SKUs where L3 is merged into L2 (most discrete GPUs).

**Report content:** Layer 5 panel per SKU; architecture-specific notes about the cache hierarchy shape.

---

### M6 — Layer 6: SoC Data Movement / On-Chip Fabric (5–7 days)

Highest-risk milestone in this plan because the modeling concept is the newest and datasheet sources are thinner. Flag early if M6 slips.

**Model content:**
- Populate `SoCFabricModel` per SKU: topology (crossbar / ring / 2D-mesh / CLOS), hop count formula, per-hop latency and pJ/flit, bisection bandwidth.
- Tie `KPUTileEnergyModel.l3_routing_distance_factor` to `SoCFabricModel` (remove the constant).
- CPU ring/mesh hop coefficients for x86 SKUs.
- GPU SM-to-L2 crossbar coefficients.
- Memory-controller contention coefficient (analytical; tagged `THEORETICAL`).

**Report content:** Layer 6 panel per SKU with topology diagram (if feasible in HTML — otherwise a textual description) and hop-count curve.

**Exit criteria:** expert review across the full SKU set (fabric topology varies most dramatically across architectures, so sign-off must span every shipping SKU).

---

### M7 — Layer 7: External Memory (3–4 days)

**Model content:**
- `HardwareResourceModel.peak_bandwidth` and `energy_per_byte` populated from `TechnologyProfile.dram_{read,write}_energy_per_byte_pj` per memory technology (HBM2e / HBM3 / HBM3e / GDDR6 / DDR5 / LPDDR5 / LPDDR5X).
- Read/write asymmetry coefficients.
- Access-pattern overhead multipliers (sequential vs. strided vs. random; analytical, tagged `THEORETICAL`).

**Report content:** Layer 7 panel per SKU; memory-technology comparison chart; bandwidth vs. pJ/byte scatter across SKUs.

---

### M8 — Reporting Polish + Engineering Preview (3–5 days)

**Deliverables:**
- `compare.html` cross-SKU comparison page with interactive Plotly filters (checkbox per SKU, per precision).
- `index.html` landing with the 7-layer coverage matrix and a confidence-ladder legend front and center.
- PowerPoint export path fully wired via existing `cli/generate_*_slides.py` patterns. **One deck** at M8, pitched to an internal engineering audience: layer-by-layer walkthrough with the full confidence and provenance detail engineers need. The investor due-diligence slide is a subsequent derivation from this deck (not in M8 scope; a separate pull-down afterward).
- Branes logo + brand styling on every slide and every HTML page (integrated from M0); final CSS polish pass so the artifact is presentable externally.
- README in `reports/microarch_model/` describing how to regenerate, what the confidence tags mean, and which SKUs are in scope.

**Exit criteria:** plan owner sign-off on the engineering deck; artifacts committed to a tagged branch or release.

---

## Out of scope for this plan

- Microbenchmark kernels (`layer<N>/*.py` executables).
- Power meter (`PowerMeter` abstraction).
- Any empirical CSV under `validation/empirical/results/layer<N>/`.
- Composition tests.
- CI residual-gate hooks.
- Layers 8 and 9 (intra-server and inter-server fabric).
- Workload decomposition (Hierarchy B).

All of these are tracked by the bottom-up validation plan and its future follow-on; they do not gate any milestone here.

## Handoff to the validation plan

Once M8 ships, the bottom-up microbenchmark plan re-activates. Its Phase 0 scaffolding is already in place (schema, `LayerTag`, `field_provenance`, dataclass stubs, CLI harness), so validation picks up at Phase 1 with the first microbenchmark kernel. The report tool from M0 gains a "measured" overlay on each layer panel as Phases 1–7 land; `THEORETICAL` entries graduate to `CALIBRATED` in place.

## Timeline

~7–9 weeks for one developer end to end. The plan owner is the dedicated reviewer, so milestone sign-offs are not rate-limited by external schedule. M0.5 is now the widest milestone (7–10 days) because it absorbs both the KPU dataflow-tile refinement and the compute-archetype comparison harness that drives product-positioning exploration. Parallelizable across two developers at the M1 / M2 split and again at M4 / M5; M0.5's harness work can partially overlap with M1 on non-KPU SKUs if a second developer is available.

| Week | Milestones                         |
|------|------------------------------------|
| 1    | M0 + start M0.5                    |
| 2    | M0.5 (continue; tile model + harness) |
| 3    | M0.5 close + M1                    |
| 4    | M2 + M3                            |
| 5    | M4 + M5                            |
| 6    | M6                                 |
| 7    | M7 + M8                            |
| 8    | Buffer / review                    |

## Risks

1. **M0.5 scope and exploration budget.** M0.5 combines the KPU dataflow-tile refinement (Parts A–B), the compute-archetype comparison harness (Part C), and internal-consistency tests (Part D). Array-size exploration and the pipeline-utilization sensitivity view are inherently open-ended. If the KPU mapper or energy model needs changes beyond configuration to support the dataflow-tile abstraction, or if the TPU mapper needs refactoring for fair comparison, M0.5 slips. Mitigate by timeboxing to the 7–10 day budget, by spilling excess array-size exploration to a post-M8 follow-up, and by handling any TPU-mapper gap as a separate tracked issue rather than an M0.5 detour.
2. **"THEORETICAL" misread by external audiences.** The artifact ships externally with Branes branding. Mitigate with a prominent confidence-ladder legend on the landing page and a provenance badge next to every number.
3. **M6 (SoC fabric) datasheet scarcity.** Topology and per-hop numbers are thinner in public sources, especially for the accelerator SKUs (KPU, Hailo, Edge TPU). Mitigate by allocating the 5–7 day budget (double other cache layers) and flagging slippage at the end of week 4 if sources aren't pinned down.
4. **SKU list drift.** Stakeholders may ask for additional SKUs mid-plan (T768, datacenter GPUs, etc.). The list is locked as of 2026-04-20; add-ons become a post-M8 follow-up.
5. **Scope creep into Layers 8–9.** Easy to slip from "DRAM" into "and the NVLink to the other card" — keep the scope boundary strict; intra-server and inter-server fabric remain out until the delivery artifact ships.

## Issues to file (after plan approval)

Proposed epic and child issues (not yet filed — pending review):

- **Epic:** Micro-architectural model delivery M0–M8 (Layers 1–7).
  - M0: Scaffolding (schema + dataclasses + CLI skeleton + branded HTML template + JSON contract).
  - M0.5: KPU dataflow-tile refinement (T64/T128/T256) + compute-archetype comparison harness (GPU/TPU/KPU).
  - M1: Layer 1 (ALU) model population + panel.
  - M2: Layer 2 (register file) model population + panel.
  - M3: Layer 3 (L1 cache) model population + panel.
  - M4: Layer 4 (L2 cache) model population + panel.
  - M5: Layer 5 (L3 / LLC) model population + panel.
  - M6: Layer 6 (SoC fabric) `SoCFabricModel` + population + panel.
  - M7: Layer 7 (external memory) model population + panel.
  - M8: Reporting polish + engineering-preview deck.

Each child issue carries an acceptance-criteria block explicitly stating: "No measurement or microbenchmark work. Acceptance is plan-owner review of the regenerated report artifact."

## Decisions locked during plan review (2026-04-20)

- **SKU list:** Jetson Orin AGX, i7-12700K, Ryzen 9 8945HS, KPU T64 / T128 (new) / T256, Coral Edge TPU, Hailo-8, Hailo-10H. T768 out of scope.
- **KPU dataflow-tile refinement is a prerequisite** (M0.5), sequenced between M0 and M1. KPU tiles are modeled as distributed dataflow fabrics with perfect-pipelining semantics, not matrix tiles; product positioning targets energy per op, not peak ops/s. M0.5 also delivers the GPU/TPU/KPU compute-archetype comparison harness that makes the positioning legible.
- **Reviewer:** plan owner is the dedicated reviewer; no bandwidth bottleneck.
- **M8 output:** one deck, internal engineering audience. Investor due-diligence slide is a downstream derivation from the engineering deck.
- **Branding:** Branes logo (`docs/img/Branes_Logo.jpg`) integrated in M0 HTML template and M8 PPT master; artifacts ship externally with Branes identity.
