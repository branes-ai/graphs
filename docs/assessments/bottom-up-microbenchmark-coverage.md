# Assessment: Bottom-Up Microbenchmark Coverage

**Date:** 2026-04-16
**Scope:** Do the current benchmarks validate the latency and energy models
from the bottom up — ALU → register file → tile/scratchpad → on-chip
L3 → external memory → distributed / cluster?

**Short answer:** Partially. About 3 of 6 layers have measurement
coverage; 3 layers are analytical-only. Today's stack is **top-down
calibrated** (composite GEMM/Conv sweeps → fitted efficiency curves →
back-propagated into resource models). A **bottom-up validation path**
that probes each physical layer independently does not exist, and that
is where undetected model drift hides.

---

## 1. Current Inventory

### 1.1 Benchmark harness

```
src/graphs/benchmarks/
├── collectors.py, runner.py, schema.py           # framework-agnostic harness
├── microbench/
│   ├── gemm.py                                   # Layer 3 (composite)
│   ├── conv2d.py                                 # Layer 3 (composite)
│   ├── conv_bn_resnet.py                         # Layer 3 (fused composite)
│   └── dispatch_overhead.py                      # cross-cutting: launch overhead
├── specs/
│   ├── gemm/{standard_sizes,batched}.yaml
│   ├── conv2d/{standard_sizes,batched,depthwise}.yaml
│   ├── memory/stream.yaml                        # Layer 5
│   └── workload/
├── pytorch_benchmarks/{matmul,memory,blas}_bench.py
├── numpy_benchmarks/{matmul,memory,blas,multicore_stream}_bench.py
├── tensorrt_benchmarks/{dla_models,dla_synthetic,nsys_profiler}.py
├── matmul_bench{,_multi}.py                      # legacy top-level
├── memory_bench.py                               # Layer 5
├── fused_{linear,conv,attention}_bench.py        # fusion-pattern composites
└── concurrent_stream.py                          # Layer 5 (multi-stream)
```

### 1.2 Calibration / fitting plumbing

```
src/graphs/calibration/
├── calibrator.py, calibration_db.py, schema.py
├── efficiency_curves.py                          # fits size → efficiency
├── energy_fitter.py                              # fits energy coefficients
├── roofline_fitter.py                            # fits AI breakpoint
├── utilization_curves.py, utilization_fitter.py
├── gpu_calibration.py, gpu_clock.py, cpu_clock.py
└── power_model.py, precision_detector.py
```

### 1.3 Validation tests

```
validation/
├── empirical/
│   ├── sweep_{gemm,conv2d,mlp}.py                # sweeps that feed calibration
│   └── results/
│       ├── gemm_sweep_full_cuda_50w.csv
│       ├── mlp_sweep_full_cuda_maxn.csv
│       ├── conv2d_sweep_full_cuda_50w.csv
│       └── …
├── energy_model/test_energy_consistency.py       # single-file consistency check
├── estimators/test_{resnet18,mobilenet,efficientnet,…}.py
└── hardware/test_{all_hardware,cross_architecture_physics,…}.py
```

---

## 2. Coverage vs. the 6-Layer Bottom-Up Hierarchy

### Layer 1 — ALU (per-op latency and energy)

**Goal:** Measure pJ/op and ns/op for a single ALU / MAC / tensor-core
slot in isolation. Validate `PROCESS_NODE_ENERGY × CIRCUIT_TYPE_MULTIPLIER`
and `ops_per_unit_per_clock × clock`.

**What we have:**
- Analytical baseline only: `resource_model.PROCESS_NODE_ENERGY` (3 nm →
  28 nm) × `CIRCUIT_TYPE_MULTIPLIER` (`standard_cell` / `tensor_core`
  / `simd_packed` / `custom_datacenter`).
- `dispatch_overhead.py` isolates CUDA launch / CUDA event / profiler
  overhead — adjacent to Layer 1 but above it.

**What is missing:**
- FMA-only loops (empty-loop subtraction) for each precision.
- Per-precision throughput probes (FP64 / FP32 / TF32 / BF16 / FP16 /
  FP8 / INT8 / INT4).
- A physics-aware energy probe that attributes measured Joules to a
  known op count.

**Validation status:** `THEORETICAL` only.

---

### Layer 2 — Register file + SIMD / warp / systolic composition

**Goal:** Measure the cost of feeding operands into the ALU array —
register-file read/write energy, SIMD fill, warp issue, systolic
row/column load.

**What we have:**
- Modeled, not measured: `register_file_read_energy` and
  `register_file_write_energy` in `StoredProgramEnergyModel` and
  `DataParallelEnergyModel`; analogous PE-local storage in
  `KPUTileEnergyModel` (`l1_read/write_energy_per_byte`) and
  `TPUTileEnergyModel`.
- Analytical note in `StoredProgramEnergyModel`: *"Register file energy
  is comparable to ALU energy"* — asserted, not measured.

**What is missing:**
- SIMD-width probe (AVX-2 vs. AVX-512 vs. NEON dotprod vs. AMX).
- Warp issue rate / divergence-penalty probe (GPU).
- Systolic fill-and-drain benchmark isolating steady-state vs. startup
  on TPU / KPU.
- Register spill-vs.-in-register benchmark.

**Validation status:** `THEORETICAL` only.

---

### Layer 3 — Tile / scratchpad / L1–L2

**Goal:** Measure data-movement cost and hit/miss behavior at the
innermost cache or scratchpad level. Validate
`l1_cache_per_unit`, `tile_scratchpad_energy_per_byte`, occupancy
thresholds.

**What we have:**
- `microbench/gemm.py` + specs at `standard_sizes.yaml` /
  `batched.yaml` — sweeps M/N/K/batch.
- `microbench/conv2d.py` + `standard_sizes.yaml` / `batched.yaml` /
  `depthwise.yaml` — standard and memory-bound conv sweeps.
- `microbench/conv_bn_resnet.py` — ResNet-shaped fused Conv+BN+ReLU.
- `fused_{linear,conv,attention}_bench.py` — fusion-pattern composites.
- `pytorch_benchmarks/`, `numpy_benchmarks/` — framework-specific
  wrappers.
- Empirical results: `gemm_sweep_full_cuda_50w.csv` and
  `conv2d_sweep_full_cuda_50w.csv` drive the GPU calibration curves.

**What is missing:**
- **Cache-line-stride sweep** that probes the L1/L2 thresholds
  explicitly (pointer-chase or strided-read benchmarks).
- **Scratchpad-residency probe** for KPU/TPU — a benchmark whose
  working set crosses the 256 KB KPU scratchpad boundary so we can
  measure the `tiling_overhead` (currently a hard-coded
  `1.0 + 0.10 × (iterations − 1)`).
- **Tile-size sweep** to validate that `_analyze_tiling`'s 80%
  efficiency heuristic matches hardware.
- **Per-precision memory traffic** attribution (currently derived by
  scaling FP32 bytes).

**Validation status:** `CALIBRATED` for GPU GEMM / Conv2D on Jetson
Orin. `THEORETICAL` for KPU tile behavior and all other SKUs.

---

### Layer 4 — On-chip L3 / data reuse

**Goal:** Measure the reuse vs. eviction cost at the chip-wide cache
or distributed L3 scratchpad boundary. Validate `l2_cache_total`,
KPU `l3_noc_energy_per_hop`, GPU L2 hit rate.

**What we have:** Nothing distinct from Layer 3. The GEMM/Conv sweeps
implicitly exercise L3 at large M/N, but no benchmark varies the
working-set ratio against L3 capacity and reports the curve.

**What is missing:**
- L3-residency sweep (working set ÷ L3 capacity = 0.25 / 0.5 / 1.0 /
  2.0 / 4.0).
- NoC-distance-aware benchmark for KPU T256 / T768 (near tile vs. far
  tile).
- GPU L2 hit-rate measurement via CUPTI / Nsight counters to validate
  the analytical 90% figure in `DataParallelEnergyModel`.
- Coherence-traffic probe on CPU (multi-core writes to shared line).

**Validation status:** `THEORETICAL` only.

---

### Layer 5 — External memory (LPDDR / GDDR / HBM / DDR)

**Goal:** Measure achieved DRAM bandwidth and pJ/byte per memory
technology. Validate `peak_bandwidth` and `energy_per_byte`, plus the
`TechnologyProfile.dram_energy_per_byte_pj` family.

**What we have:**
- `memory_bench.py` — STREAM-style copy bandwidth.
- `numpy_benchmarks/multicore_stream.py` — multicore bandwidth.
- `benchmarks/specs/memory/stream.yaml`.
- `concurrent_stream.py` — multi-stream concurrency.

**What is missing:**
- Per-technology energy attribution (need a power meter in the loop
  or rely on on-die telemetry to separate HBM vs. DDR vs. LPDDR
  Joules).
- Access-pattern sweeps: sequential vs. strided vs. random, varying
  burst length.
- Read vs. write asymmetry probe (asymmetric energy is hard-coded in
  `KPUTileEnergyModel`: `dram_read_energy` vs. `dram_write_energy`).
- Memory-controller queue-depth probe (tail latency under contention).

**Validation status:** `CALIBRATED` for bandwidth on measured SKUs,
`THEORETICAL` for energy attribution.

---

### Layer 6 — Distributed / cluster

**Goal:** Measure inter-chip, inter-socket, and pod-level communication
cost. Validate NVLink / ICI / PCIe / NUMA latency and bandwidth.

**What we have:** Nothing. No NVLink, NV-HBI, ICI (TPU), NUMA, or
PCIe benchmark. The architectural models mention these (B100 dual-die,
TPU v4 pod) but the mappers assume single-chip behavior.

**What is missing:**
- PCIe bandwidth / round-trip latency probe.
- NVLink / NV-HBI benchmark (H100 SXM, B100 dual-die).
- NUMA cross-socket latency (EPYC dual-socket, Xeon dual-socket).
- TPU v4 / v5p ICI pod-level benchmark.
- Multi-host collective-op benchmark (AllReduce, AllGather).

**Validation status:** Not modeled, not measured.

---

## 3. Cross-Cutting Observations

### 3.1 Top-down vs. bottom-up

Today's validation path is **top-down**: run a composite workload
(ResNet-18, MobileNet, ViT) on real hardware → fit an `efficiency_factor`
scalar → back-propagate into the resource model. This works end-to-end
but is **silent on mis-cancellation**: if the ALU energy is 2× high and
the cache energy is 0.5× low, the composite matmul sweep still matches.
The resulting model ships `CALIBRATED` confidence for the composite but
carries hidden errors in the individual layers.

A bottom-up path would fit each layer from an isolated probe, compose
upward, and the composite would become a **cross-check** (does the
layered model predict the composite correctly?) rather than the source
of calibration.

### 3.2 Where composite calibration succeeds today

- GPU GEMM and Conv2D on Jetson Orin AGX (15 W / 30 W / 50 W / MAXN)
  across FP32 / FP16 / INT8 — the `validation/empirical/results/*.csv`
  is the ground truth that feeds `efficiency_curves.py`.
- GPU dispatch overhead on datacenter vs. edge — `dispatch_overhead.py`
  is **the one benchmark** that cleanly isolates one model parameter
  (`kernel_launch_overhead`) at a single layer.

### 3.3 Where composite calibration is missing

- KPU (no real silicon available yet → all `THEORETICAL`).
- TPU (no first-party access → all analytical from papers).
- CPU beyond the i7-12700K tiny-MLP sweep.
- DSP, DPU, CGRA, Hailo (all flagged `⚠ ESTIMATED` in factory
  comments).

### 3.4 Confidence propagation is uneven

`EstimationConfidence` is propagated correctly through composite
reports, but individual resource-model fields don't carry confidence
annotations. If a field in `HardwareResourceModel` came from a
datasheet, a measurement, or an interpolation, the reader can't tell
without reading the comments.

---

## 4. Summary Table

| Layer | Scope | Benchmark files | Ground-truth data | Status |
|-------|-------|-----------------|-------------------|--------|
| 1 | ALU / MAC / TC | *(none)* | *(none)* | Theoretical |
| 2 | Register file / SIMD / warp / systolic | *(none)* | *(none)* | Theoretical |
| 3 | Tile / scratchpad / L1–L2 | `microbench/{gemm,conv2d,conv_bn_resnet}.py`, fused benches | `gemm_sweep_*.csv`, `conv2d_sweep_*.csv`, `mlp_sweep_*.csv` | Partial — GPU only |
| 4 | On-chip L3 / reuse | *(none)* | *(none)* | Theoretical |
| 5 | External DRAM | `memory_bench.py`, `multicore_stream.py`, `specs/memory/stream.yaml` | *(bandwidth only)* | Partial — bandwidth yes, energy no |
| 6 | Distributed / cluster | *(none)* | *(none)* | Missing |

---

## 5. Risks

1. **Model drift under mis-cancellation** — a layered model fit only at
   the composite level can look healthy while individual coefficients
   drift compensatorily. First-time use on a genuinely new workload
   exposes the error.
2. **Unverifiable comparative claims** — "KPU is 7× more energy-efficient
   than GPU for MLP" is load-bearing in vendor-facing decks but depends
   on Layer 1–2 numbers nobody has measured in isolation.
3. **Confidence over-claim** — `CALIBRATED` is reported for composite
   workloads but applied to every downstream layer by inheritance. A
   caller asking "what's your confidence on the KPU L3 NoC energy?"
   currently has no way to get a truthful answer.
4. **Silent regression surface** — a future refactor that touches
   `DomainFlowEnergyModel` coefficients has no Layer-1/2 test that can
   catch a sign error or unit mistake until the full-model sweep
   completes and only if the error is larger than the composite noise
   floor.

---

## 6. Recommendation

Build the missing layers in dependency order (Layer 1 → Layer 6) so that
each layer's benchmark can be validated against the next layer's sweeps
(bottom-up composition test). See
[`docs/plans/bottom-up-microbenchmark-plan.md`](../plans/bottom-up-microbenchmark-plan.md)
for the implementation plan.
