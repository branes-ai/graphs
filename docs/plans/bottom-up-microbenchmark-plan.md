# Plan: Bottom-Up Microbenchmark Validation Infrastructure

**Status:** Draft
**Date:** 2026-04-16
**Companion:** [`docs/assessments/bottom-up-microbenchmark-coverage.md`](../assessments/bottom-up-microbenchmark-coverage.md)

## Purpose

Stand up a six-layer bottom-up benchmark suite that measures each
physical layer of the modeled micro-architectures in isolation, fits
that layer's coefficients from measurement, and composes upward so the
composite workload (ResNet, MobileNet, ViT, MLP) becomes a **cross-
check** of the layered model rather than the source of calibration.

## Principles

1. **One layer per benchmark family.** A benchmark must isolate a
   single modeling parameter (or a small set) so measurement error
   maps cleanly to a specific field in `HardwareResourceModel`,
   `ComputeFabric`, or the architectural energy model.
2. **Bottom-up composition.** Layer N is validated against
   measurements from Layer N; the composed prediction of Layers 1..N
   is then used as a prior for Layer N+1's fit. The top layer's
   composite sweep becomes a residual check.
3. **Confidence-first.** Every benchmark emits an
   `EstimationConfidence` tag per coefficient it fits, and the
   resource model field's provenance becomes `CALIBRATED` only when
   the fit is backed by real silicon at a known thermal point.
4. **Reproducible via YAML specs.** Each benchmark reads a
   `benchmarks/specs/layer<N>/*.yaml` and writes a CSV / JSON under
   `validation/empirical/results/layer<N>/`. The fitter under
   `calibration/` consumes the CSV and produces a resource-model
   patch plus a confidence annotation.
5. **Run on what we have.** Ship with coverage for the hardware we can
   physically measure (Intel i7-12700K, Jetson Orin AGX, H100, A100
   where accessible, Coral Edge TPU). KPU / TPU / DSP / CGRA stay
   `THEORETICAL` until silicon is available, but the scaffolding
   accepts future measurements without code changes.
6. **No premature abstractions.** Start with the first two layers
   end-to-end on one SKU before generalizing.

## Target Directory Layout

```
src/graphs/benchmarks/
├── layer1_alu/                     # FMA loops, precision rate probes
├── layer2_register_simd/           # register file, SIMD, warp, systolic fill
├── layer3_scratchpad/              # L1/L2 and scratchpad residency
├── layer4_onchip/                  # L3 / LLC / distributed L3 reuse
├── layer5_dram/                    # HBM / GDDR / DDR / LPDDR
├── layer6_cluster/                 # NVLink / ICI / PCIe / NUMA
├── specs/layer{1..6}/              # YAML specs per layer
└── runner.py, collectors.py, schema.py  # shared harness (reuse)

src/graphs/calibration/fitters/
├── layer1_alu_fitter.py
├── layer2_register_fitter.py
├── …

validation/empirical/results/layer{1..6}/
└── <hw>_<profile>_<precision>.csv
```

---

## Phase 0 — Foundation (1 week)

Scope: get the harness, result schema, and confidence annotation path
in place before writing any new benchmark.

### Tasks
1. Extend `benchmarks/schema.py` with a `LayerTag` enum (`ALU`,
   `REGISTER_SIMD`, `SCRATCHPAD`, `ONCHIP`, `DRAM`, `CLUSTER`) and
   make every `BenchmarkResult` carry one.
2. Add `calibration/fitters/` subpackage; move existing
   `energy_fitter.py`, `roofline_fitter.py`,
   `efficiency_curves.py` into `fitters/` namespace and re-export for
   backward compat.
3. Add `HardwareResourceModel.field_provenance: Dict[str,
   EstimationConfidence]` so each field's source is queryable.
4. Define a **composition test** helper:
   `validation/composition/test_layer_composition.py` that, given
   layer 1..N fits for a SKU, predicts layer N+1 behavior and diffs
   against that layer's measurements.
5. Power-measurement plumbing: factor out a `PowerMeter` abstraction
   (Intel RAPL for x86, `tegrastats` for Jetson, `nvml` for discrete
   NVIDIA) so benchmarks can request Joules, not just seconds.
6. CI hook that re-runs the composition test on every PR that
   modifies `resource_model.py`, `architectural_energy.py`, or any
   `hardware/models/**/*.py`.

### Exit criteria
- `python -m graphs.benchmarks.schema --self-check` passes.
- A dummy Layer 1 benchmark can emit a `BenchmarkResult(layer=ALU,
  confidence=CALIBRATED)` and the fitter updates the resource model's
  `field_provenance`.
- Composition test runs (even if trivially empty) in CI.

---

## Phase 1 — Layer 1: ALU / MAC / Tensor-Core Rate and Energy (2 weeks)

### Measurement goals
- FP64 / FP32 / TF32 / BF16 / FP16 / FP8 / INT8 / INT4 **ops/sec** per
  compute unit.
- Per-op **pJ** for each precision on each compute fabric.
- Clock-frequency ↔ throughput relationship (validate
  `sm_sustained_clock_hz`, `sustained_ops_per_sec`).

### Benchmarks
- `layer1_alu/fma_rate.py` — tight FMA loop, empty-loop subtraction,
  unrolled to amortize loop overhead below 1% of ALU time. One
  function per precision and fabric (CPU scalar, CPU SIMD, GPU CUDA
  core, GPU tensor core, KPU tile PE simulator — until silicon).
- `layer1_alu/precision_sweep.py` — sweep precision with fixed op
  count, record throughput and Joules via `PowerMeter`.
- `layer1_alu/clock_dvfs_probe.py` — run the FMA loop at multiple
  thermal-governor settings, capture clock frequency per trial,
  validate DVFS throttle model.

### Fitter
- `layer1_alu_fitter.py` fits:
  - `ComputeFabric.ops_per_unit_per_clock[precision]`
  - `ComputeFabric.energy_per_flop_fp32` (derived via reverse lookup
    of measured pJ/op vs. `PROCESS_NODE_ENERGY × CIRCUIT_TYPE_MULTIPLIER`)
  - Marks field provenance `CALIBRATED`.

### Specs
- `specs/layer1/fma_rate.yaml` — per-SKU op counts per precision
  (small enough to fit in register file, large enough to hide timer
  resolution).

### Hardware scope at ship
- Intel i7-12700K (AVX-2, VNNI): all precisions.
- Jetson Orin AGX GPU (Ampere SMs, Tensor Cores).
- H100 if access is arranged; otherwise keep analytical and flag.

### Exit criteria
- Measured pJ/op for FP32 on i7-12700K within 20% of the analytical
  `get_base_alu_energy(7, 'standard_cell')` — if not, investigate
  whether register-file energy is leaking in (Layer 2 concern).
- Composition test: predicted FP32 GEMM rate at large M/N (Layer 3)
  matches measured rate within 10% when Layer 1 is the only measured
  input.

---

## Phase 2 — Layer 2: Register File, SIMD, Warp, Systolic Fill (2 weeks)

### Measurement goals
- Register-file read/write energy per access.
- SIMD-width effective throughput vs. scalar (validate the 0.70
  SIMD efficiency constant and the 0.90 `simd_packed` multiplier).
- GPU warp issue rate and divergence penalty.
- Systolic array fill-and-drain overhead (TPU / KPU once silicon is
  available).

### Benchmarks
- `layer2_register_simd/register_rw.py` — tight load-use chains that
  force register pressure, measure throughput ceiling; compare
  against spill-to-L1 case to isolate register-file energy.
- `layer2_register_simd/simd_width.py` — same arithmetic at scalar /
  128-bit / 256-bit / 512-bit SIMD widths, on the same precision.
- `layer2_register_simd/warp_divergence.py` — GPU branch-divergence
  probe, varying divergence rate from 0% to 50% in 5% steps.
- `layer2_register_simd/systolic_fill.py` — matmul with M = N small
  (below array size) and K sweeping across the pipeline depth
  threshold.

### Fitter
- `layer2_register_fitter.py` fits:
  - `StoredProgramEnergyModel.register_file_read/write_energy` (from
    `TechnologyProfile`; updates the profile rather than overriding)
  - `DataParallelEnergyModel.warp_divergence_penalty` +
    `warp_divergence_rate`
  - Systolic pipeline-fill overhead (feeds
    `TPUMapper._analyze_operation_type.pipeline_fill_overhead`)
  - `CPUMapper._analyze_vectorization.vectorization_efficiency`
    (currently hard-coded 0.95 / 0.80 / 0.70).

### Exit criteria
- Measured register-file energy within 30% of
  `TechnologyProfile.register_{read,write}_energy_pj`.
- GPU warp-divergence penalty curve fitted; the fixed 5% assumption
  in `DataParallelEnergyModel.warp_divergence_rate` is replaced with
  a per-op estimate.
- Composition test: Layer 1+2 → Layer 3 GEMM prediction within 8%.

---

## Phase 3 — Layer 3: Tile / Scratchpad / L1–L2 (2 weeks)

This layer upgrades the existing GEMM/Conv suite rather than replacing
it.

### Measurement goals
- Cache-line stride thresholds (L1, L2 capacity boundaries).
- Tile residency validation for accelerators (KPU 256 KB scratchpad,
  TPU UB).
- Validate `_analyze_tiling` overhead formula
  (`1.0 + 0.10 × (iterations − 1)`) against measurement.
- Per-precision memory-traffic scaling (currently derived by scaling
  FP32 bytes).

### Benchmarks
- `layer3_scratchpad/stride_walk.py` — pointer-chase / strided-read
  that sweeps stride across powers of 2 crossing L1 and L2 sizes.
- `layer3_scratchpad/working_set_sweep.py` — GEMM with M = N = K
  chosen so working set is 0.25×, 0.5×, 1×, 2×, 4× of L1 then L2.
- Migrate `microbench/gemm.py` and `microbench/conv2d.py` to the new
  harness and tag their results `layer=SCRATCHPAD`.
- `layer3_scratchpad/tile_overhead.py` — for accelerators: sweep tile
  count from "fits in scratchpad" (1 iteration) to "needs 8
  iterations" and fit the per-iteration overhead.

### Fitter
- `layer3_scratchpad_fitter.py` fits:
  - `HardwareResourceModel.l1_cache_per_unit` validation check.
  - `KPUMapper._analyze_tiling` overhead coefficients (replacing the
    0.80 / 0.10 constants with measured values).
  - GPU L1/L2 hit rate (currently 95% / 90% in
    `DataParallelEnergyModel`).

### Exit criteria
- Cache-capacity corners measured within 10% of spec L1/L2 sizes.
- KPU tiling overhead formula re-fit from measurement (once silicon
  or cycle-accurate simulator available).
- Composition test: Layer 1–3 → ResNet-18 Conv layer prediction
  within 15%.

---

## Phase 4 — Layer 4: On-Chip L3 / Last-Level Cache / Distributed L3 (1.5 weeks)

### Measurement goals
- Chip-wide cache capacity threshold (x86 L3, GPU L2, KPU distributed
  L3).
- NoC-hop cost for tile-mesh architectures (KPU T256 / T768, Cerebras
  reference).
- Coherence traffic cost on multi-core CPU.

### Benchmarks
- `layer4_onchip/working_set_l3.py` — working-set sweep crossing L3
  boundary.
- `layer4_onchip/noc_distance.py` — on mesh architectures, measure
  near-tile vs. far-tile access latency and energy.
- `layer4_onchip/coherence_traffic.py` — multi-core CPU benchmark
  that forces true sharing on one cache line, then false sharing,
  then padded (no coherence).
- `layer4_onchip/gpu_l2_hitrate.py` — CUPTI / Nsight-counter-backed
  measurement of GPU L2 hit rate across op types.

### Fitter
- Updates `HardwareResourceModel.l2_cache_total`,
  `KPUTileEnergyModel.l3_routing_distance_factor`,
  `DataParallelEnergyModel.l2_hit_rate`.

### Exit criteria
- GPU L2 hit rate measured per op type, replacing the single 0.90
  constant.
- Coherence-traffic cost quantified on x86; feeds a future CPU
  multi-core scaling refinement.
- Composition test: Layer 1–4 → ViT MLP block prediction within 15%.

---

## Phase 5 — Layer 5: External Memory by Technology (1.5 weeks)

### Measurement goals
- Achieved bandwidth per memory technology (HBM2e, HBM3, HBM3e, GDDR6,
  DDR5, LPDDR5, LPDDR5X).
- pJ/byte per technology (requires power telemetry in the loop).
- Read vs. write asymmetry.
- Sequential vs. strided vs. random access cost.

### Benchmarks
- Upgrade existing `memory_bench.py` and `multicore_stream.py`:
  - Tag as `layer=DRAM`.
  - Add power-meter integration so energy is recorded alongside
    bandwidth.
- `layer5_dram/access_pattern.py` — sequential / strided / random
  sweeps at fixed working set size above L3.
- `layer5_dram/read_write_asymmetry.py` — 100% read, 100% write, 50/50
  mix.
- `layer5_dram/technology_matrix.py` — same benchmark across every
  memory technology we have access to; the CSV output feeds a
  per-technology fit.

### Fitter
- Updates `HardwareResourceModel.peak_bandwidth`,
  `HardwareResourceModel.energy_per_byte`,
  `TechnologyProfile.dram_{read,write}_energy_per_byte_pj`.
- Validates `KPUTileEnergyModel.dram_read_energy_per_byte` vs.
  `dram_write_energy_per_byte` asymmetry.

### Exit criteria
- Measured pJ/byte for DDR5 on i7-12700K within 20% of the
  `DATACENTER_7NM_DDR5` profile.
- Composition test: Layer 1–5 → ResNet-18 end-to-end latency and
  energy within 15% on Jetson Orin AGX MAXN.

---

## Phase 6 — Layer 6: Distributed / Cluster (2 weeks)

### Measurement goals
- PCIe bandwidth and latency for host ↔ device.
- NVLink / NV-HBI bandwidth for multi-GPU and dual-die B100.
- NUMA cross-socket latency for multi-socket CPU.
- TPU ICI / inter-pod bandwidth (if access is arranged).
- Collective-op cost (AllReduce, AllGather) at small / medium / large
  message sizes.

### Benchmarks
- `layer6_cluster/pcie_bench.py` — host ↔ device memcpy sweep.
- `layer6_cluster/nvlink_bench.py` — peer-to-peer memcpy between
  NVLink-connected GPUs.
- `layer6_cluster/numa_bench.py` — cross-socket vs. same-socket
  memory access.
- `layer6_cluster/allreduce_bench.py` — NCCL AllReduce across ring /
  tree topologies at message sizes 4 KB → 1 GB.

### Model extensions
- Add `ClusterModel` dataclass in `hardware/cluster_model.py` (new
  file) describing topology, inter-chip bandwidth, link energy.
- Extend `GraphHardwareAllocation` with cluster-level metrics
  (per-link bandwidth used, collective-op time).

### Exit criteria
- PCIe and NVLink fits feed a new `multi_chip_overhead` field on
  relevant GPU resource models.
- Distributed end-to-end ResNet-50 training-step latency on a 2-GPU
  H100 system within 20% (if access available).

---

## Confidence Graduation

After each phase completes, the affected `field_provenance` entries
graduate:

- Pre-measurement: `THEORETICAL` from analytical model.
- Post-layer-N fit on one SKU: `INTERPOLATED` across the family.
- Post-layer-N fit on the specific SKU: `CALIBRATED`.
- After composition test passes within target error: composite
  prediction inherits the **minimum** confidence across inputs
  (already enforced by the aggregate rule).

---

## Cross-Cutting Work (runs alongside phases)

1. **Power-meter abstraction** — land in Phase 0, expand coverage as
   each phase adds a new SKU.
2. **CI composition tests** — extend after each phase; PRs that
   modify a resource-model field fail CI if the composition test on
   the affected SKU regresses > 5%.
3. **Calibration database schema** — ensure `calibration_data/<hw>/`
   gains a `layer<N>/` subdirectory that mirrors the benchmark
   hierarchy, so the v2 layout already documented in memory extends
   naturally.
4. **Documentation** — each phase closes with a short `docs/designs/`
   update describing which resource-model fields are now `CALIBRATED`
   vs. `THEORETICAL`.

---

## Hardware Coverage Matrix at Completion

| Layer | i7-12700K | Xeon SR | EPYC | Jetson Orin AGX | H100 (if access) | KPU | TPU v4 | Coral |
|-------|-----------|---------|------|------------------|-------------------|-----|--------|-------|
| 1 | ✓ | — | — | ✓ | △ | sim | — | ✓ |
| 2 | ✓ | — | — | ✓ | △ | sim | — | △ |
| 3 | ✓ | — | — | ✓ | △ | sim | — | △ |
| 4 | ✓ | — | — | ✓ | △ | sim | — | — |
| 5 | ✓ | — | — | ✓ | △ | — | — | — |
| 6 | — | — | — | — | △ | — | — | — |

✓ = shipped calibrated; △ = shipped if hardware access; sim =
cycle-accurate simulator; — = out of scope for v1.

---

## Ordering, Effort, and Risks

### Ordering rationale
Layers must be built 1 → 6 because later layers' composition tests
depend on earlier layers being fitted. Skipping to Layer 5 without
Layers 1–4 repeats today's top-down failure mode.

### Effort estimate
- Phase 0: ~1 week (harness + schema + CI).
- Phases 1–5: ~2 weeks each (10 weeks).
- Phase 6: ~2 weeks.
- **Total: ~13 weeks of focused work** for one developer, assuming
  hardware access to at least the i7-12700K and a Jetson Orin AGX
  throughout.

### Key risks
1. **Power-meter variance** — Intel RAPL and Jetson `tegrastats` have
   sampling intervals ≥ 10 ms. Benchmarks must run long enough to
   drown this out; add a minimum-duration check to the harness.
2. **KPU / TPU silicon access** — without hardware, Layers 1–4 stay
   simulator-based. Plan explicitly flags this and the architecture
   docs' `⚠ ESTIMATED` warnings remain in force.
3. **Compiler obliteration** — ALU microbenchmarks are famously
   susceptible to dead-code elimination. Every Layer 1 benchmark must
   include a result-sink check (volatile accumulator touched by the
   timer code).
4. **Timer resolution on Jetson** — the Tegra kernel launch overhead
   (~80 µs) dominates short kernels. Layer 1 benchmarks on Jetson
   must bundle enough work per launch to hide dispatch.
5. **Scope creep** — temptation to add workload benchmarks (BERT,
   LLaMA) once the Layer-3 harness is live. Resist until all 6
   layers have a v1 fit.

---

## Definition of Done

The infrastructure is considered complete when, on at least one fully-
covered SKU (target: Jetson Orin AGX MAXN or Intel i7-12700K):

1. Every resource-model field used by the roofline and energy models
   has a `field_provenance` entry backed by a benchmark in the
   corresponding layer.
2. The composition test predicts ResNet-18 (or equivalent) end-to-end
   latency and energy from Layer 1–5 fits within 15% of the
   measured composite.
3. A regression in any layer's fit triggers a CI failure before it
   can reach `main`.
4. The confidence column in `compare_architectures.py` shows
   `CALIBRATED` for the SKU's workload instead of the current
   inheriting-from-composite `CALIBRATED` that masks individual
   `THEORETICAL` inputs.
