# Assessment: Bottom-Up Microbenchmark Coverage

**Date:** 2026-04-16
**Revised:** 2026-04-20 (9-layer framing)
**Scope:** Do the current benchmarks validate the latency and energy models from the bottom up — across the nine hardware layers defined in the [bottom-up plan](../plans/bottom-up-microbenchmark-plan.md)?

**Short answer:** Partially. About 2 of 9 layers have any measurement coverage; 7 are analytical-only. Today's stack is **top-down calibrated** (composite GEMM/Conv sweeps → fitted efficiency curves → back-propagated into resource models). A **bottom-up validation path** that probes each physical layer independently does not exist, and that is where undetected model drift hides.

This assessment uses the canonical 9-layer hierarchy:

| Layer | Hardware level                          |
|-------|-----------------------------------------|
| 1     | ALU / MAC / Tensor Core                 |
| 2     | Register file                           |
| 3     | L1 cache / scratchpad                   |
| 4     | L2 cache                                |
| 5     | L3 cache / LLC / shared                 |
| 6     | SoC data movement (on-chip fabric)      |
| 7     | External memory                         |
| 8     | Distributed memory (intra-server)       |
| 9     | Cluster interconnect (inter-server)     |

The word **layer** refers to a validation layer; cache levels are written **L1 / L2 / L3 cache** to avoid collision.

---

## 1. Current Inventory

### 1.1 Benchmark harness

```text
src/graphs/benchmarks/
├── collectors.py, runner.py, schema.py           # framework-agnostic harness
├── microbench/
│   ├── gemm.py                                   # composite (mixed Layers 1-7)
│   ├── conv2d.py                                 # composite (mixed Layers 1-7)
│   ├── conv_bn_resnet.py                         # fused composite
│   └── dispatch_overhead.py                      # cross-cutting: launch overhead
├── specs/
│   ├── gemm/{standard_sizes,batched}.yaml
│   ├── conv2d/{standard_sizes,batched,depthwise}.yaml
│   ├── memory/stream.yaml                        # Layer 7
│   └── workload/
├── pytorch_benchmarks/{matmul,memory,blas}_bench.py
├── numpy_benchmarks/{matmul,memory,blas,multicore_stream}_bench.py
├── tensorrt_benchmarks/{dla_models,dla_synthetic,nsys_profiler}.py
├── matmul_bench{,_multi}.py                      # legacy top-level
├── memory_bench.py                               # Layer 7
├── fused_{linear,conv,attention}_bench.py        # fusion-pattern composites
└── concurrent_stream.py                          # multi-stream
```

Note: GEMM and Conv2D are **mixed-layer composites** — they exercise ALU + register + L1/L2/L3 + DRAM simultaneously. Under the 9-layer framing they are workload-style benchmarks (Hierarchy B), not isolating microbenchmarks. They are listed where they currently provide partial signal but do not satisfy any layer's isolating-microbench requirement.

### 1.2 Calibration / fitting plumbing

```text
src/graphs/calibration/
├── calibrator.py, calibration_db.py, schema.py
├── efficiency_curves.py                          # fits size → efficiency
├── energy_fitter.py                              # fits energy coefficients
├── roofline_fitter.py                            # fits AI breakpoint
├── utilization_curves.py, utilization_fitter.py
├── gpu_calibration.py, gpu_clock.py, cpu_clock.py
└── power_model.py, precision_detector.py
```

The `calibration/fitters/` subpackage exists (stood up pre-pivot) and already hosts `layer1_alu_fitter.py` and `layer2_register_fitter.py`. Fitters for the remaining layers (L1/L2/L3 cache, SoC fabric, external memory, intra-server, cluster interconnect) are added as each validation phase lands.

### 1.3 Validation tests

```text
validation/
├── empirical/
│   ├── sweep_{gemm,conv2d,mlp}.py                # composite sweeps that feed top-down calibration
│   └── results/
│       ├── gemm_sweep_full_cuda_50w.csv
│       ├── mlp_sweep_full_cuda_maxn.csv
│       ├── conv2d_sweep_full_cuda_50w.csv
│       └── ...
├── energy_model/test_energy_consistency.py       # single-file consistency check
├── estimators/test_{resnet18,mobilenet,efficientnet,...}.py
└── hardware/test_{all_hardware,cross_architecture_physics,...}.py
```

The `validation/composition/` directory and per-layer `validation/empirical/results/layer<N>/` trees defined by the plan do not yet exist.

---

## 2. Coverage vs. the 9-Layer Bottom-Up Hierarchy

### Layer 1 — ALU / MAC / Tensor Core

**Goal:** Measure pJ/op and ns/op for a single ALU / MAC / tensor-core slot in isolation. Validate `PROCESS_NODE_ENERGY × CIRCUIT_TYPE_MULTIPLIER` and `ops_per_unit_per_clock × clock`.

**What we have:**
- Analytical baseline only: `resource_model.PROCESS_NODE_ENERGY` (3 nm → 28 nm) × `CIRCUIT_TYPE_MULTIPLIER` (`standard_cell` / `tensor_core` / `simd_packed` / `custom_datacenter`).
- `dispatch_overhead.py` isolates CUDA launch / event / profiler overhead — adjacent to Layer 1 but above it.

**What is missing:**
- FMA-only loops (empty-loop subtraction) for each precision.
- Per-precision throughput probes (FP64 / FP32 / TF32 / BF16 / FP16 / FP8 / INT8 / INT4).
- Physics-aware energy probe attributing measured Joules to a known op count.
- DVFS-vs-throughput probe (`clock_dvfs_probe.py`).

**Validation status:** `THEORETICAL` only.

---

### Layer 2 — Register File

**Goal:** Measure the cost of feeding operands into the ALU array — register-file read/write energy, SIMD fill, warp issue, systolic row/column load.

**What we have:**
- Modeled, not measured: `register_file_read_energy` and `register_file_write_energy` in `StoredProgramEnergyModel` and `DataParallelEnergyModel`; analogous PE-local storage in `KPUTileEnergyModel` (`l1_read/write_energy_per_byte`) and `TPUTileEnergyModel`.
- Analytical assertion in `StoredProgramEnergyModel`: *"Register file energy is comparable to ALU energy"* — asserted, not measured.

**What is missing:**
- Register read/write probe (load-use chains forcing register pressure; spill-to-L1 comparison).
- SIMD-width sweep (AVX-2 vs. AVX-512 vs. NEON dotprod vs. AMX).
- Warp issue rate / divergence-penalty probe (GPU).
- Systolic fill-and-drain benchmark isolating steady-state vs. startup on TPU / KPU.

**Validation status:** `THEORETICAL` only.

---

### Layer 3 — L1 Cache / Scratchpad

**Goal:** Measure data-movement cost and hit/miss behavior at the innermost cache or scratchpad level. Validate `l1_cache_per_unit`, scratchpad residency, per-op L1 hit rate.

**What we have:**
- Indirect signal only: GEMM/Conv composites at small sizes implicitly sit in L1, but the working set isn't varied to find the corner.

**What is missing:**
- L1 working-set sweep (0.25× / 0.5× / 1× / 2× of L1 capacity).
- Stride-walk / pointer-chase probe at L1 line size.
- Per-op L1 hit-rate measurement via performance counters (currently 95% constant in `DataParallelEnergyModel`).
- Scratchpad-residency probe for KPU/TPU (256 KB KPU tile, TPU UB slice).

**Validation status:** `THEORETICAL` only.

---

### Layer 4 — L2 Cache

**Goal:** Measure L2 capacity corner, line-fill latency, prefetcher effectiveness, per-op L2 hit rate.

**What we have:** Nothing distinct from Layer 3. GEMM/Conv composites at medium sizes implicitly use L2 but the working-set ratio isn't swept.

**What is missing:**
- L2 working-set sweep (just-above-L1 to just-below-L2).
- Line-fill latency probe (load-use chain spanning L2).
- Prefetch-effectiveness probe (sequential vs. random at fixed L2-resident working set).
- Per-op L2 hit-rate measurement (currently 90% constant).

**Validation status:** `THEORETICAL` only.

---

### Layer 5 — L3 / Last-Level / Shared Cache

**Goal:** Measure L3 / LLC capacity corner, coherence-protocol behavior, GPU LLC hit rate. *Transport cost across the on-chip fabric is attributed to Layer 6, not here.*

**What we have:** Nothing isolating. GEMM/Conv composites at large M/N implicitly exercise L3 but no benchmark varies working set against L3 capacity.

**What is missing:**
- L3 working-set sweep crossing the LLC boundary.
- Coherence-protocol probe holding sharing threads on near-neighbor cores (so fabric cost doesn't leak in).
- GPU LLC hit-rate measurement via CUPTI / Nsight counters to validate the analytical 90% in `DataParallelEnergyModel`.

**Validation status:** `THEORETICAL` only.

---

### Layer 6 — SoC Data Movement / On-Chip Fabric

**Goal:** Measure on-chip transport — per-hop latency, pJ/flit, fabric topology, bandwidth saturation, controller contention. Representative fabrics: x86 ring/mesh, AMD Infinity Fabric (CCX-internal), NVIDIA SM-to-L2 crossbar, KPU/Cerebras tile mesh, TPU reduction ring.

**What we have:** Analytical only. `KPUTileEnergyModel.l3_routing_distance_factor` and `NOC_HOP_ENERGY_PJ` exist as constants; no probe validates them.

**What is missing:**
- Hop-latency sweep (near-tile vs. far-tile on mesh; ring-position sweep on x86; SM-to-L2-slice probing on GPU).
- Per-hop energy probe with `PowerMeter` in the loop.
- Fabric-bandwidth saturation curve.
- Memory-controller contention probe (N cores → same vs. distributed controllers).

**Validation status:** `THEORETICAL` only. `SoCFabricModel` scaffolding is in place (M0 landed the dataclass and `field_provenance` machinery) but no per-SKU content or measurement backs it yet.

---

### Layer 7 — External Memory

**Goal:** Measure achieved DRAM bandwidth and pJ/byte per memory technology. Validate `peak_bandwidth`, `energy_per_byte`, and the `TechnologyProfile.dram_energy_per_byte_pj` family.

**What we have:**
- `memory_bench.py` — STREAM-style copy bandwidth.
- `numpy_benchmarks/multicore_stream.py` — multicore bandwidth.
- `benchmarks/specs/memory/stream.yaml`.
- `concurrent_stream.py` — multi-stream concurrency.

**What is missing:**
- Per-technology energy attribution (need a power meter in the loop or on-die telemetry to separate HBM vs. DDR vs. LPDDR Joules).
- Access-pattern sweeps (sequential vs. strided vs. random, varying burst length).
- Read vs. write asymmetry probe (asymmetric energy is hard-coded in `KPUTileEnergyModel`).
- Memory-controller queue-depth probe (tail latency under contention).
- `layer=EXTERNAL_MEMORY` tagging on existing benchmarks once the schema lands.

**Validation status:** `CALIBRATED` for bandwidth on measured SKUs; `THEORETICAL` for energy attribution.

---

### Layer 8 — Distributed Memory / Intra-Server Fabric

**Goal:** Measure the load/store-semantic fabric inside a single chassis or pod — PCIe, NVLink/NVSwitch, NUMA (UPI / Infinity Fabric), TPU ICI within a pod. Intra-chassis collective ops are attributed here.

**What we have:** Nothing. The architectural models reference NVLink, NV-HBI, dual-die B100, and TPU ICI but the mappers assume single-chip behavior.

**What is missing:**
- PCIe bandwidth and round-trip latency probe.
- NVLink / NVSwitch peer-to-peer benchmark (H100 SXM, B100 dual-die, full-board all-to-all on a DGX 8-GPU board).
- NUMA cross-socket latency (EPYC dual-socket, Xeon dual-socket).
- TPU v4 / v5p ICI bandwidth between chips inside one pod.
- Intra-chassis collective-op benchmark (NCCL AllReduce on 2–16 local devices).
- Per-SKU `IntraServerFabricModel` content (M0 landed the dataclass scaffolding; values are still empty).

**Validation status:** Scaffolding exists (`IntraServerFabricModel`, `field_provenance`) but no per-SKU values are populated yet.

---

### Layer 9 — Cluster Interconnect / Inter-Server Fabric

**Goal:** Measure the message-passing fabric between chassis — Ethernet (100/200/400/800 GbE), InfiniBand (NDR / XDR), RoCE, optical (TPU v7 OCS). Latency and energy are dominated by NIC, switch, and topology.

**What we have:** `ClusterInterconnectModel` dataclass scaffolding (M0) with fabric-type and topology enums; no per-SKU entries, no multi-node benchmark, and no resource-model entry yet distinguishes a 100 GbE link from an IB NDR link from a TPU OCS link.

**What is missing:**
- Point-to-point bandwidth/latency probe per fabric (`ib_send_bw`-style for IB; `netperf` for Ethernet).
- Switch-hop latency model (1 / 2 / 3 hops on a configurable topology).
- Multi-node collective-op benchmark (AllReduce/AllGather/AllToAll across 2 / 4 / 16 / 64+ nodes).
- Tail-latency under congestion probe (incast and all-to-all under background traffic).
- TPU OCS reconfiguration cost probe (if access).
- Per-SKU `ClusterInterconnectModel` content (M0 landed the dataclass itself).

**Validation status:** Scaffolding landed (M0); content and measurement still missing. Most v1 entries will remain `THEORETICAL` until multi-node access is arranged.

---

## 3. Cross-Cutting Observations

### 3.1 Top-down vs. bottom-up

Today's validation path is **top-down**: run a composite workload (ResNet-18, MobileNet, ViT) on real hardware → fit an `efficiency_factor` scalar → back-propagate into the resource model. This works end-to-end but is **silent on mis-cancellation**: if the ALU energy is 2× high and the cache energy is 0.5× low, the composite matmul sweep still matches. The resulting model ships `CALIBRATED` confidence for the composite but carries hidden errors in the individual layers.

A bottom-up path fits each layer from an isolated probe and composes upward. The composite then becomes a **cross-check** (does the layered model predict the composite correctly?) rather than the source of calibration. This is the foundation of Hierarchy A. Workloads (ResNet/ViT/etc.) belong in the separate Hierarchy B and only become meaningful once Hierarchy A is calibrated.

### 3.2 Where composite calibration succeeds today

- GPU GEMM and Conv2D on Jetson Orin AGX (15 W / 30 W / 50 W / MAXN) across FP32 / FP16 / INT8 — the `validation/empirical/results/*.csv` is the ground truth that feeds `efficiency_curves.py`. Under the 9-layer framing this is a Hierarchy B signal, not a per-layer fit.
- GPU dispatch overhead on datacenter vs. edge — `dispatch_overhead.py` is **the one benchmark** that cleanly isolates one model parameter (`kernel_launch_overhead`). It sits adjacent to Layer 1.

### 3.3 Where composite calibration is missing

- KPU (no real silicon available yet → all `THEORETICAL`).
- TPU (no first-party access → all analytical from papers).
- CPU beyond the i7-12700K tiny-MLP sweep.
- DSP, DPU, CGRA, Hailo (all flagged `WARN ESTIMATED` in factory comments).
- All multi-device / multi-node configurations (Layers 8 and 9).

### 3.4 Confidence propagation is uneven

`EstimationConfidence` is propagated correctly through composite reports. Individual resource-model fields now carry confidence annotations via `HardwareResourceModel.field_provenance`, landed with M0; subsequent milestones populate per-field entries.

### 3.5 Layer 6 was implicit, Layer 8/9 were conflated

In the previous 6-layer assessment, the on-chip fabric (Layer 6 in the new framing) was folded into "L3" and the intra-server / inter-server distinction (Layers 8 and 9) was collapsed into one "distributed/cluster" bucket. The 9-layer framing makes both splits explicit because they correspond to materially different modeling regimes (fabric topology and per-hop energy; load/store-semantic intra-chassis fabric vs. message-passing inter-chassis network).

---

## 4. Summary Table

| Layer | Scope                                   | Isolating benchmark files | Ground-truth data         | Status                                |
|-------|-----------------------------------------|---------------------------|---------------------------|---------------------------------------|
| 1     | ALU / MAC / TC                          | *(none)*                  | *(none)*                  | Theoretical                           |
| 2     | Register file                           | *(none)*                  | *(none)*                  | Theoretical                           |
| 3     | L1 cache / scratchpad                   | *(none)*                  | *(none)*                  | Theoretical                           |
| 4     | L2 cache                                | *(none)*                  | *(none)*                  | Theoretical                           |
| 5     | L3 / LLC                                | *(none)*                  | *(none)*                  | Theoretical                           |
| 6     | SoC data movement                       | *(none)*                  | *(none)*                  | Theoretical                           |
| 7     | External memory                         | `memory_bench.py`, `multicore_stream.py`, `specs/memory/stream.yaml` | bandwidth CSVs (no Joules) | Partial — bandwidth yes, energy no |
| 8     | Distributed memory (intra-server)       | *(none)*                  | *(none)*                  | Missing                               |
| 9     | Cluster interconnect (inter-server)     | *(none)*                  | *(none)*                  | Missing                               |

GEMM/Conv composites (`microbench/gemm.py`, `conv2d.py`, `conv_bn_resnet.py`, fused benches) are **not** counted in any layer row above — they are Hierarchy B workloads whose layer attribution is the point of Hierarchy B's tooling.

---

## 5. Risks

1. **Model drift under mis-cancellation** — a layered model fit only at the composite level can look healthy while individual coefficients drift compensatorily. First-time use on a genuinely new workload exposes the error.
2. **Unverifiable comparative claims** — "KPU is 7× more energy-efficient than GPU for MLP" is load-bearing in vendor-facing decks but depends on Layer 1–2 numbers nobody has measured in isolation.
3. **Confidence over-claim** — `CALIBRATED` is reported for composite workloads but applied to every downstream layer by inheritance. A caller asking "what's your confidence on the KPU L3 NoC energy?" currently has no way to get a truthful answer.
4. **Silent regression surface** — a future refactor that touches `DomainFlowEnergyModel` coefficients has no Layer-1/2 test that can catch a sign error or unit mistake until the full-model sweep completes and only if the error is larger than the composite noise floor.
5. **Layer 6 invisibility** — on-chip fabric energy is currently folded into other coefficients, so any shift in fabric topology (e.g., a ring → mesh transition between CPU generations) is silently absorbed into "efficiency" rather than attributed to its real source.
6. **Multi-node access gap** — Layers 8 and 9 will likely ship with `THEORETICAL` confidence on most SKUs because intra-server (DGX-class) and inter-server (multi-rack) hardware is rarely available to a single developer. The risk is over-claiming confidence on cluster-scale predictions; the mitigation is the explicit "no hardware" annotation in `cli/microarch_validation_report.py`.

---

## 6. Recommendation

Build the missing layers in dependency order (Layer 1 → Layer 9) so that each layer's benchmark can be validated against the next layer's microbench (bottom-up composition test). See [`docs/plans/bottom-up-microbenchmark-plan.md`](../plans/bottom-up-microbenchmark-plan.md) for the implementation plan.

Workload-style validation (ResNet-18, ViT, etc.) is explicitly out of scope for this assessment and the companion plan; it belongs in a separate Hierarchy B plan that is blocked on Hierarchy A reaching CALIBRATED on at least one SKU.
