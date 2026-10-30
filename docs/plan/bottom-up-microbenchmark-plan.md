# Plan: Bottom-Up Microbenchmark Validation Infrastructure

**Status:** Revised (9-layer framing; added SoC Data Movement and split intra-server vs. cluster interconnect)
**Original date:** 2026-04-16
**Revised:** 2026-04-20
**Companion:** [`docs/assessments/bottom-up-microbenchmark-coverage.md`](../assessments/bottom-up-microbenchmark-coverage.md)

## Framing

This plan builds the **first of two validation hierarchies** in the repo:

**Hierarchy A — micro-architectural model validation (this plan).** Each of nine hardware levels is validated in isolation by a microbenchmark family that exercises only that level's basic operations. Energy and latency models are fit per level; lower-level fits are pinned as the next level is fit. Upward composition — "given validated levels 1..N, does the model predict level N+1 behavior?" — becomes the integration test.

**Hierarchy B — workload decomposition (future plan, blocked on A).** Decomposes workloads (ResNet-18, ViT, etc.) onto the nine hardware levels and cross-checks the aggregation. Deferred to a separate plan until Hierarchy A reaches CALIBRATED on at least one SKU.

The two hierarchies must stay distinct. A workload "within 15%" on top of unvalidated micro-architectural models is false confidence.

### Naming

The word **layer** is reserved for the validation hierarchy (Layer 1..9). Cache levels are always written as **L1 / L2 / L3 cache** to avoid collision. "Phase N" refers to the development phase that builds Layer N.

### The nine validation layers

| Layer | Hardware level                          | Isolating microbench targets                                                  |
|-------|-----------------------------------------|-------------------------------------------------------------------------------|
| 1     | ALU / MAC / Tensor Core                 | FMA rate per precision; pJ/op; DVFS-vs-throughput                             |
| 2     | Register file                           | register read/write energy; SIMD width; warp issue; systolic fill             |
| 3     | L1 cache / scratchpad                   | L1 capacity corner; stride ceiling; per-op hit rate                           |
| 4     | L2 cache                                | L2 capacity corner; line-fill latency; prefetch effectiveness                 |
| 5     | L3 cache / LLC / shared                 | chip-wide capacity corner; coherence-protocol traffic; LLC hit rate           |
| 6     | SoC data movement (on-chip fabric)      | per-hop latency and pJ/flit; topology; injection rate; contention             |
| 7     | External memory                         | bandwidth + pJ/byte per technology (HBM/GDDR/DDR/LPDDR); R/W asymmetry        |
| 8     | Distributed memory (intra-server)       | PCIe / NVLink / NVSwitch / NUMA / TPU-ICI inside one chassis or pod           |
| 9     | Cluster interconnect (inter-server)     | Ethernet / InfiniBand / RoCE / optical (TPU-OCS) between chassis; topology    |

Layer 5 and Layer 6 are adjacent but orthogonal: Layer 5 answers "did we hit in the LLC and what coherence state did that imply?"; Layer 6 answers "how much did it cost to move the packet across the on-chip fabric to get there?". Coherence traffic that traverses the fabric is attributed to Layer 6 for transport cost and Layer 5 for protocol cost.

Layer 8 and Layer 9 are also distinct: Layer 8 is the **load/store-semantic** fabric inside a single server or pod (NVLink/NVSwitch on a DGX, UPI on dual-socket Xeon, TPU ICI within a pod, PCIe host↔device) — addresses are shared, coherent, or at minimum DMA-reachable without an IP stack. Layer 9 is the **message-passing** fabric between chassis (Ethernet, InfiniBand, RoCE, TPU OCS between pods) — traffic is packetized over a switched network and the latency/energy profile is dominated by NIC, switch, and topology.

## Principles

1. **One layer per benchmark family.** Each benchmark isolates a single modeling parameter (or a small set) so measurement error maps cleanly to a specific field in `HardwareResourceModel`, `ComputeFabric`, or the architectural energy model.
2. **Bottom-up composition.** Layer N is validated against its own measurements; the composed prediction of Layers 1..N then serves as a prior for Layer N+1's fit.
3. **Confidence-first.** Every benchmark emits an `EstimationConfidence` tag per coefficient it fits. A `HardwareResourceModel` field's provenance graduates to `CALIBRATED` only when the fit is backed by silicon at a known thermal point.
4. **Reproducible via YAML specs.** Each benchmark reads `benchmarks/specs/layer<N>/*.yaml`, writes CSV/JSON under `validation/empirical/results/layer<N>/`, and a fitter under `calibration/fitters/` produces a resource-model patch plus provenance annotation.
5. **Run on what we have.** Ship calibrated coverage for hardware we can physically measure (Intel i7-12700K, Jetson Orin AGX, H100 or A100 where accessible, Coral Edge TPU). KPU / TPU / DSP / CGRA stay `THEORETICAL` until silicon is available.
6. **No premature abstractions.** Start with Layers 1–2 end-to-end on one SKU before generalizing.

## Target directory layout

```text
src/graphs/benchmarks/
├── layer1_alu/                        # FMA / precision / DVFS probes
├── layer2_register/                   # register file, SIMD, warp, systolic fill
├── layer3_l1_cache/                   # private L1 capacity, stride, hit rate
├── layer4_l2_cache/                   # L2 capacity, line-fill, prefetch
├── layer5_l3_cache/                   # shared / LLC capacity, coherence, hit rate
├── layer6_soc_data_movement/          # crossbar / ring / mesh / CLOS NoC
├── layer7_external_memory/            # HBM / GDDR / DDR / LPDDR
├── layer8_distributed_memory/         # PCIe / NVLink / NVSwitch / NUMA / ICI (intra-server)
├── layer9_cluster_interconnect/       # Ethernet / IB / RoCE / optical (inter-server)
├── specs/layer{1..9}/                 # YAML specs per layer
└── runner.py, collectors.py, schema.py    # shared harness

src/graphs/calibration/fitters/
├── layer1_alu_fitter.py
├── layer2_register_fitter.py
├── layer3_l1_cache_fitter.py
├── layer4_l2_cache_fitter.py
├── layer5_l3_cache_fitter.py
├── layer6_soc_data_movement_fitter.py
├── layer7_external_memory_fitter.py
├── layer8_distributed_memory_fitter.py
└── layer9_cluster_interconnect_fitter.py

validation/empirical/results/layer{1..9}/
└── <hw>_<profile>_<precision>.csv
```

---

## Phase 0 — Foundation (1 week)

Scope: harness, result schema, and confidence annotation path before any new benchmark is written.

### Tasks
1. Extend `benchmarks/schema.py` with a `LayerTag` enum of nine entries (`ALU`, `REGISTER`, `L1_CACHE`, `L2_CACHE`, `L3_CACHE`, `SOC_DATA_MOVEMENT`, `EXTERNAL_MEMORY`, `DISTRIBUTED_MEMORY`, `CLUSTER_INTERCONNECT`); every `BenchmarkResult` carries one.
2. Add `calibration/fitters/` subpackage; move existing `energy_fitter.py`, `roofline_fitter.py`, `efficiency_curves.py` into the `fitters/` namespace with backward-compat re-exports.
3. Add `HardwareResourceModel.field_provenance: Dict[str, EstimationConfidence]` so each field's source is queryable.
4. Define a composition-test helper: `validation/composition/test_layer_composition.py` that, given Layer 1..N fits for a SKU, predicts Layer N+1 microbench behavior and diffs against that layer's measurements.
5. Power-measurement plumbing: a `PowerMeter` abstraction (Intel RAPL for x86, `tegrastats` for Jetson, NVML for discrete NVIDIA) so benchmarks request Joules, not just seconds.
6. CI hook: re-run the composition test on every PR that modifies `resource_model.py`, `architectural_energy.py`, or any `hardware/models/**/*.py`.

### Exit criteria
- `python -m graphs.benchmarks.schema --self-check` passes.
- A dummy Layer 1 benchmark emits `BenchmarkResult(layer=ALU, confidence=CALIBRATED)` and the fitter updates `field_provenance`.
- Composition test runs (trivially empty is fine) in CI.

---

## Phase 1 — Layer 1: ALU / MAC / Tensor Core (2 weeks)

### Measurement goals
- FP64 / FP32 / TF32 / BF16 / FP16 / FP8 / INT8 / INT4 ops/sec per compute unit.
- Per-op pJ for each precision on each compute fabric.
- Clock-frequency ↔ throughput relationship (validate `sm_sustained_clock_hz`, `sustained_ops_per_sec`).

### Benchmarks
- `layer1_alu/fma_rate.py` — tight FMA loop with empty-loop subtraction, unrolled so loop overhead is below 1% of ALU time. One function per precision and fabric (CPU scalar, CPU SIMD, GPU CUDA core, GPU Tensor Core, KPU tile PE simulator).
- `layer1_alu/precision_sweep.py` — sweep precision at fixed op count, record throughput and Joules via `PowerMeter`.
- `layer1_alu/clock_dvfs_probe.py` — run the FMA loop across thermal-governor settings; capture clock per trial; validate the DVFS throttle model.

### Fitter (`layer1_alu_fitter.py`)
- `ComputeFabric.ops_per_unit_per_clock[precision]`
- `ComputeFabric.energy_per_flop_fp32` (reverse lookup of measured pJ/op vs. `PROCESS_NODE_ENERGY × CIRCUIT_TYPE_MULTIPLIER`)
- Marks field provenance `CALIBRATED`.

### Hardware scope at ship
- Intel i7-12700K (AVX-2, VNNI): all precisions.
- Jetson Orin AGX GPU (Ampere SMs, Tensor Cores).
- H100 if access arranged; otherwise analytical and flagged.

### Exit criteria
- Measured pJ/op for FP32 on i7-12700K within 20% of analytical `get_base_alu_energy(7, 'standard_cell')`. If not, investigate whether register-file energy is leaking in (Layer 2 concern).
- Composition test: Layer 1 alone predicts a Layer 2 register-bound kernel's ALU-limited rate within 10%.

---

## Phase 2 — Layer 2: Register File (2 weeks)

### Measurement goals
- Register-file read/write energy per access.
- SIMD-width effective throughput vs. scalar (validate the 0.70 SIMD-efficiency constant and the 0.90 `simd_packed` multiplier).
- GPU warp issue rate and divergence penalty.
- Systolic-array fill-and-drain overhead (TPU / KPU once silicon is available).

### Benchmarks
- `layer2_register/register_rw.py` — tight load-use chains that force register pressure; compare against spill-to-L1 to isolate register-file energy.
- `layer2_register/simd_width.py` — same arithmetic at scalar / 128-bit / 256-bit / 512-bit widths, fixed precision.
- `layer2_register/warp_divergence.py` — GPU branch-divergence probe, varying divergence rate 0% to 50% in 5% steps.
- `layer2_register/systolic_fill.py` — matmul with M = N small (below array size) and K sweeping across the pipeline-depth threshold.

### Fitter (`layer2_register_fitter.py`)
- `StoredProgramEnergyModel.register_file_read/write_energy` (updates the `TechnologyProfile`)
- `DataParallelEnergyModel.warp_divergence_penalty` and `warp_divergence_rate`
- Systolic pipeline-fill overhead (feeds `TPUMapper._analyze_operation_type.pipeline_fill_overhead`)
- `CPUMapper._analyze_vectorization.vectorization_efficiency` (replacing hard-coded 0.95 / 0.80 / 0.70).

### Exit criteria
- Measured register-file energy within 30% of `TechnologyProfile.register_{read,write}_energy_pj`.
- GPU warp-divergence penalty curve fitted; the fixed 5% assumption in `DataParallelEnergyModel.warp_divergence_rate` is replaced with a per-op estimate.
- Composition test: Layers 1+2 predict a register-resident / SIMD-dominated kernel's behavior within 8%.

---

## Phase 3 — Layer 3: L1 Cache / Scratchpad (1.5 weeks)

### Measurement goals
- L1 capacity corner (working-set threshold where throughput drops).
- Stride-walk ceiling at L1 line size; associativity.
- Per-core scratchpad residency for accelerators (KPU 256 KB tile, TPU UB slice).
- Per-op L1 hit-rate model validation (currently 95% constant in `DataParallelEnergyModel`).

### Benchmarks
- `layer3_l1_cache/working_set_l1.py` — simple kernel with working set at 0.25× / 0.5× / 1× / 2× of L1 capacity; isolate the L1 corner.
- `layer3_l1_cache/stride_walk_l1.py` — pointer-chase / strided read across strides within L1 capacity, fitting line size and associativity.
- `layer3_l1_cache/l1_hit_rate.py` — performance-counter-backed (CUPTI / perf / tegrastats) measurement of L1 hit rate per op class.

### Fitter (`layer3_l1_cache_fitter.py`)
- Validates `HardwareResourceModel.l1_cache_per_unit`.
- `DataParallelEnergyModel.l1_hit_rate` per op type.
- L1 read/write energy coefficients (`TechnologyProfile.l1_{read,write}_energy_pj`).

### Exit criteria
- L1 capacity corner within 10% of vendor spec.
- Composition test: Layers 1–3 predict an L1-resident kernel's behavior within 10%.

---

## Phase 4 — Layer 4: L2 Cache (1.5 weeks)

### Measurement goals
- L2 capacity corner.
- L2→L1 line-fill latency.
- Prefetcher effectiveness (stream vs. random).
- Per-op L2 hit-rate model.

### Benchmarks
- `layer4_l2_cache/working_set_l2.py` — working-set sweep from just above L1 to just below L2.
- `layer4_l2_cache/line_fill_latency.py` — load-use chain spanning L2, measure cycles per fill.
- `layer4_l2_cache/prefetch_effectiveness.py` — sequential vs. random access at fixed L2-resident working set.

### Fitter (`layer4_l2_cache_fitter.py`)
- Validates `HardwareResourceModel.l2_cache_total` (and `l2_cache_per_unit` where applicable).
- Per-op L2 hit rate (replacing the 0.90 constant in `DataParallelEnergyModel.l2_hit_rate`).
- L2 read/write energy coefficients.

### Exit criteria
- L2 capacity corner within 10% of spec.
- Composition test: Layers 1–4 predict an L2-resident kernel's behavior within 10%.

---

## Phase 5 — Layer 5: L3 / Last-Level / Shared Cache (1.5 weeks)

### Measurement goals
- L3 / LLC capacity corner.
- Coherence-protocol behavior on multi-core CPU (true-sharing, false-sharing, padded states).
- GPU L2-as-LLC hit rate (GPUs often collapse L3 into an enlarged L2).

Note: transport cost of coherence traffic and LLC accesses across the on-chip fabric is attributed to Layer 6 — here we fit capacity, hit rate, and protocol-state transitions, not per-hop energy.

### Benchmarks
- `layer5_l3_cache/working_set_l3.py` — working-set sweep crossing the L3 boundary.
- `layer5_l3_cache/coherence_protocol.py` — multi-core CPU benchmark exercising true sharing, false sharing, and padded (no-coherence) cases on one line; isolates protocol behavior from fabric cost by keeping the sharing threads on near-neighbor cores.
- `layer5_l3_cache/gpu_llc_hitrate.py` — CUPTI / Nsight-backed GPU LLC hit rate across op types.

### Fitter (`layer5_l3_cache_fitter.py`)
- `HardwareResourceModel.l3_cache_total`.
- GPU LLC hit-rate entries in `DataParallelEnergyModel`.
- Coherence-state transition costs (protocol-only, fabric-stripped).

### Exit criteria
- LLC hit rate measured per op type, replacing the single 0.90 constant.
- Coherence-protocol cost quantified on x86 (with Layer 6 fabric cost held fixed).
- Composition test: Layers 1–5 predict an L3-resident, near-core kernel's behavior within 12%.

---

## Phase 6 — Layer 6: SoC Data Movement / On-Chip Fabric (2 weeks)

### Measurement goals
- Topology identification (crossbar / ring / 2D-mesh / CLOS) and hop-count model per SKU.
- Per-hop latency (cycles and ns).
- Per-flit energy (pJ), both quiet-fabric and loaded-fabric.
- Bandwidth ceiling of the fabric and saturation/injection-rate curve.
- Contention behavior at memory controllers and at cross-socket links on CPUs.

This layer models the **on-chip transport** that carries packets between cores, caches, memory controllers, and compute clusters. Representative fabrics: x86 ring/mesh (Intel SoC ring / Xeon mesh), AMD Infinity Fabric (CCX↔CCX on a single package), NVIDIA SM-to-L2 crossbar, KPU/Cerebras tile mesh, TPU reduction ring.

### Benchmarks
- `layer6_soc_data_movement/hop_latency.py` — near-tile vs. far-tile access latency sweep on mesh architectures; ring-position sweep on x86; SM-to-L2 slice probing on GPUs. Produces a hop-count → latency curve.
- `layer6_soc_data_movement/hop_energy.py` — same access patterns as `hop_latency.py`, with the `PowerMeter` in the loop; fits pJ/flit/hop.
- `layer6_soc_data_movement/fabric_bandwidth.py` — inject traffic from N sources to M sinks at increasing rates; measure the saturation point and the bandwidth/latency knee.
- `layer6_soc_data_movement/controller_contention.py` — N cores streaming through the same memory controller vs. distributed across controllers; isolates controller-side queueing from fabric transport.

### Fitter (`layer6_soc_data_movement_fitter.py`)
- New `SoCFabricModel` dataclass (add to `hardware/fabric_model.py`): topology enum, hop latency, pJ/flit/hop, bisection bandwidth, per-SKU controller count.
- `KPUTileEnergyModel.l3_routing_distance_factor` moves here from the prior Layer 5 framing.
- GPU SM-to-L2 crossbar coefficients feed `DataParallelEnergyModel` (new `fabric_energy_per_flit` field).
- CPU ring/mesh hop energy populates a new field on `HardwareResourceModel` (or the existing energy model, depending on the cleanest landing spot).

### Hardware scope at ship
- Intel i7-12700K: ring interconnect, near/far core probing via CPU affinity.
- Jetson Orin AGX: SM-to-L2 crossbar, memory-controller contention.
- H100 if access arranged; otherwise flag as analytical.
- Multi-socket Xeon / EPYC: cross-socket fabric cost (also feeds Layer 8's NUMA work).

### Exit criteria
- Measured per-hop energy on at least one SKU within 30% of the analytical `NOC_HOP_ENERGY_PJ` used today.
- Fabric topology model matches measurement: hop-count scaling is linear in ring position for a ring, grows with mesh dimension (roughly proportional to sqrt(node count)) for a square 2D mesh, and is constant for a full crossbar.
- Composition test: Layers 1–6 predict a kernel whose working set spans the on-chip fabric (e.g., multi-core CPU far-tile access, GPU cross-SM reduction) within 12%.

---

## Phase 7 — Layer 7: External Memory (1.5 weeks)

### Measurement goals
- Achieved bandwidth per memory technology (HBM2e, HBM3, HBM3e, GDDR6, DDR5, LPDDR5, LPDDR5X).
- pJ/byte per technology (requires power telemetry in the loop).
- Read vs. write asymmetry.
- Sequential vs. strided vs. random access cost.

### Benchmarks
- Upgrade existing `memory_bench.py` and `multicore_stream.py`: tag as `layer=EXTERNAL_MEMORY`; add power-meter integration so Joules are recorded alongside bandwidth.
- `layer7_external_memory/access_pattern.py` — sequential / strided / random sweeps at fixed working set above L3.
- `layer7_external_memory/read_write_asymmetry.py` — 100% read, 100% write, 50/50 mix.
- `layer7_external_memory/technology_matrix.py` — same benchmark across every memory technology with silicon access; output feeds a per-technology fit.

### Fitter (`layer7_external_memory_fitter.py`)
- `HardwareResourceModel.peak_bandwidth`, `energy_per_byte`.
- `TechnologyProfile.dram_{read,write}_energy_per_byte_pj`.
- Validates `KPUTileEnergyModel.dram_read_energy_per_byte` vs. `dram_write_energy_per_byte` asymmetry.

### Exit criteria
- Measured pJ/byte for DDR5 on i7-12700K within 20% of `DATACENTER_7NM_DDR5` profile.
- Composition test: Layers 1–7 predict a DRAM-bound streaming kernel (traversing the full fabric + controller) within 15%.

---

## Phase 8 — Layer 8: Distributed Memory / Intra-Server Fabric (2 weeks)

Scope: the **load/store-semantic** fabric inside a single chassis or pod — what one user described as "the tightly-coupled motherboard". Typical instances: NVLink + NVSwitch on a DGX 8-GPU board, PCIe host↔device, UPI between Xeon sockets, Infinity Fabric between EPYC CCDs, TPU ICI between chips in one pod. Addresses are shared or at minimum DMA-reachable without a network stack. Intra-chassis collective ops (e.g., NCCL AllReduce on 8 local GPUs) are attributed here.

### Measurement goals
- PCIe bandwidth and latency (host ↔ device).
- NVLink / NVSwitch / NV-HBI bandwidth: peer-to-peer within a DGX and across dual-die packages (B100).
- NUMA cross-socket latency for multi-socket CPU (UPI, Infinity Fabric).
- TPU ICI bandwidth between chips inside one pod.
- Intra-chassis collective-op cost (AllReduce, AllGather) at small / medium / large message sizes on 2–16 local devices.

### Benchmarks
- `layer8_distributed_memory/pcie_bench.py` — host ↔ device memcpy sweep.
- `layer8_distributed_memory/nvlink_bench.py` — peer-to-peer memcpy between NVLink-connected GPUs; NVSwitch all-to-all pattern on a full 8-GPU board.
- `layer8_distributed_memory/numa_bench.py` — cross-socket vs. same-socket memory access; UPI / Infinity Fabric energy and latency.
- `layer8_distributed_memory/intra_allreduce.py` — NCCL AllReduce on a single-chassis ring/tree at message sizes 4 KB → 1 GB.

### Model extensions
- `IntraServerFabricModel` dataclass in `hardware/intra_server_fabric_model.py` describing topology (bus, crossbar, full-mesh NVSwitch, PCIe tree), per-link bandwidth, link energy.
- Extend `GraphHardwareAllocation` with intra-chassis metrics (per-link bandwidth used, NVSwitch hop cost, collective-op time on N local devices).

### Exit criteria
- PCIe and NVLink fits populate a `intra_server_overhead` field on relevant GPU resource models.
- Composition test: Layers 1–8 predict an intra-chassis memcpy-and-reduce kernel (single server, 2–8 devices) within 20%.

---

## Phase 9 — Layer 9: Cluster Interconnect / Inter-Server Fabric (2 weeks)

Scope: the **message-passing** fabric between chassis — Ethernet (100/200/400/800 GbE), InfiniBand (NDR / XDR), RoCE, and optical interconnects such as the TPU v7 optical-circuit-switched (OCS) fabric between pods. Traffic is packetized over a switched network and the latency/energy profile is dominated by NIC, switch, and topology (fat-tree, dragonfly, torus, OCS reconfiguration). A Layer 9 "hop" is a switch crossing, not a NoC hop or an NVLink.

### Measurement goals
- Point-to-point bandwidth and latency per fabric (GbE at each speed grade; IB NDR/XDR; optical link).
- Tail latency under load (p50 / p99 / p99.9) — switched fabrics diverge here from fabric peak.
- Switch-hop model: latency and pJ/byte per switch crossing; topology-dependent hop-count formula.
- Multi-node collective-op cost (AllReduce, AllGather, AllToAll) at 2 / 4 / 16 / 64 / 256+ nodes.
- Optical-fabric reconfiguration cost (TPU OCS): time and energy to reshape the logical topology.

### Benchmarks
- `layer9_cluster_interconnect/pt2pt_bench.py` — two-node `ib_send_bw` / `ib_write_bw` (IB) or `netperf` (Ethernet) sweeps across message sizes 64 B → 1 GB; measure bandwidth, latency, tail.
- `layer9_cluster_interconnect/switch_hop_probe.py` — latency across 1 / 2 / 3 switch hops on a configurable topology; fit per-hop latency and energy.
- `layer9_cluster_interconnect/multi_node_allreduce.py` — NCCL / Gloo / custom MPI AllReduce across 2 / 4 / 8 / 16 / 32+ nodes; message sizes 4 KB → 1 GB.
- `layer9_cluster_interconnect/congestion_probe.py` — N-to-1 incast and all-to-all under background traffic; measure tail-latency degradation.
- `layer9_cluster_interconnect/ocs_reconfig_probe.py` — TPU OCS only: time and energy cost of a topology reconfiguration (if access is arranged).

### Fitter (`layer9_cluster_interconnect_fitter.py`)
- New `ClusterInterconnectModel` dataclass in `hardware/cluster_interconnect_model.py`: fabric type enum (`ETHERNET_100`, `ETHERNET_200`, `ETHERNET_400`, `IB_NDR`, `IB_XDR`, `OPTICAL_OCS`, ...), topology (fat-tree, dragonfly, torus, OCS), node count, per-hop latency, pJ/bit, NIC energy fixed cost.
- Populate a `cluster_overhead` field on relevant SKU resource models (analytical for SKUs without cluster access).
- Link energy attribution combines NIC energy, switch-crossing energy, and wire/optical-link energy; fitter decomposes each from dedicated benchmarks.

### Hardware scope at ship
- `~` for any SKU where multi-node access is arranged (lab IB cluster, cloud-hosted H100 cluster, TPU pod of ≥ 2 pods).
- `THEORETICAL` for everything else; analytical coefficients drawn from published datasheets (IB spec, 400G Ethernet SerDes papers, TPU v7 whitepaper).

### Exit criteria
- If any multi-node silicon access: point-to-point bandwidth within 10% of NIC spec; per-hop latency within 20% of switch datasheet.
- If no multi-node silicon: Layer 9 ships `THEORETICAL` with the scaffolding in place so future measurements plug in without code changes; the CI composition test is marked as not-applicable for Layer 9 until hardware arrives.
- Composition test (hardware-dependent): Layers 1–9 predict a multi-node AllReduce cost within 25% for message sizes above the protocol-overhead floor.

---

## Confidence graduation

After each phase completes, affected `field_provenance` entries graduate:
- Pre-measurement: `THEORETICAL` from analytical model.
- Post-Layer-N fit on one SKU: `INTERPOLATED` across the family.
- Post-Layer-N fit on the specific SKU: `CALIBRATED`.
- Composite predictions inherit the **minimum** confidence across inputs (already enforced by the aggregate rule).

## Cross-cutting work

1. **Power-meter abstraction** — land in Phase 0, expand coverage as each phase adds a new SKU.
2. **CI composition tests** — extend after each phase; PRs that modify a resource-model field fail CI if the composition test on the affected SKU regresses > 5%.
3. **Calibration database schema** — `calibration_data/<hw>/` gains a `layer<N>/` subdirectory that mirrors the benchmark hierarchy, extending the v2 layout.
4. **Documentation** — each phase closes with a `docs/designs/` update describing which resource-model fields are now `CALIBRATED` vs. `THEORETICAL`.

## Hardware coverage matrix at completion

| Layer                    | i7-12700K | Xeon SR | EPYC | Orin AGX | H100 | KPU | TPU v4 | Coral |
|--------------------------|-----------|---------|------|----------|------|-----|--------|-------|
| 1 ALU                    | +         | -       | -    | +        | ~    | sim | -      | +     |
| 2 Register               | +         | -       | -    | +        | ~    | sim | -      | ~     |
| 3 L1 cache               | +         | -       | -    | +        | ~    | sim | -      | ~     |
| 4 L2 cache               | +         | -       | -    | +        | ~    | sim | -      | -     |
| 5 L3 / LLC               | +         | -       | -    | +        | ~    | sim | -      | -     |
| 6 SoC data movement      | +         | ~       | ~    | +        | ~    | sim | -      | -     |
| 7 External memory        | +         | -       | -    | +        | ~    | -   | -      | -     |
| 8 Distributed memory     | -         | ~       | ~    | -        | ~    | -   | ~      | -     |
| 9 Cluster interconnect   | -         | ~       | ~    | -        | ~    | -   | ~      | -     |

Key: `+` shipped calibrated; `~` shipped if hardware access; `sim` cycle-accurate simulator; `-` out of scope for v1. Architectures that merge cache levels (e.g., GPUs without a distinct L3) populate the affected row with their actual on-die hierarchy. Xeon/EPYC entries show `~` on Layer 6 and Layer 8 because the multi-socket fabric straddles both layers and only becomes meaningful when a dual-socket platform is available. Layer 9 requires a multi-node facility; expect most v1 entries to remain `THEORETICAL` and graduate as cluster access is arranged.

## Ordering, effort, risks

### Ordering rationale
Layers must be built 1 -> 9 because each layer's composition test depends on earlier layers being pinned. Skipping ahead repeats today's top-down failure mode. Layer 6 (SoC fabric) sits between LLC and external memory because coherence and LLC accesses flow over the fabric, so external-memory measurements cannot be decomposed cleanly until the fabric cost is known. Layer 8 (intra-server) sits below Layer 9 (inter-server) because a multi-node AllReduce on a DGX cluster involves both intra-chassis NVLink reduction and inter-chassis Ethernet/IB exchange — the cluster fit cannot isolate the network contribution unless the intra-chassis cost is already known.

### Effort estimate
- Phase 0: ~1 week (harness + schema + CI).
- Phases 1, 2, 6, 8, 9: ~2 weeks each (10 weeks).
- Phases 3, 4, 5, 7: ~1.5 weeks each (6 weeks).
- **Total: ~17 weeks** for one developer with consistent access to at least i7-12700K and Orin AGX. Phase 9 only graduates to CALIBRATED if a multi-node facility (cloud or on-prem) is available; otherwise it ships scaffolding plus analytical fits.

### Key risks
1. **Power-meter variance.** Intel RAPL and Jetson `tegrastats` sample at >= 10 ms. Benchmarks must run long enough to drown this out; add a minimum-duration check to the harness.
2. **KPU / TPU silicon access.** Without hardware, their layers stay simulator-based; the architecture docs' `WARN ESTIMATED` annotations remain.
3. **Compiler obliteration.** ALU microbenchmarks are famously susceptible to dead-code elimination. Every Layer 1 benchmark must include a volatile-accumulator sink touched by the timing code.
4. **Timer resolution on Jetson.** Tegra kernel-launch overhead (~80 us) dominates short kernels; Layer 1 benchmarks on Jetson must bundle enough work per launch.
5. **Fabric isolation.** Layer 6 probes must hold the working set inside the LLC so external-memory traffic does not contaminate the per-hop measurement; and must control thread/core placement so hop count is known (affinity masks on CPU, SM selection on GPU). Lack of affinity control on some platforms (e.g., managed runtimes, locked-down kernels) limits coverage.
6. **Cluster access.** Layer 9 requires at minimum a 2-node setup with a known fabric (IB or 100/200 GbE), and ideally ≥ 8 nodes for collective-op scaling curves. Most developers will not have this; plan accordingly and ship Layer 9 as `THEORETICAL` with the scaffolding ready to accept measurements.
7. **Scope creep into workloads.** Temptation to add ResNet / ViT benchmarks once Layer 3 is live. **Resist.** Those belong in the workload-decomposition plan (Hierarchy B) and must wait until all nine layers have a v1 fit.

## Definition of Done

This plan is complete when, on at least one fully-covered SKU (target: Jetson Orin AGX MAXN or Intel i7-12700K):

1. Every resource-model field used by the roofline and energy models has a `field_provenance` entry. Layers 1–7 must be backed by a benchmark on the target SKU. Layers 8–9 must be backed by a benchmark **only if multi-device hardware is available**; otherwise the field provenance is `THEORETICAL` and the report annotates the layer as "no hardware" rather than "fit failed".
2. Each upward composition test (Layers 1..N -> Layer N+1 microbench) passes within its target band, conditional on the target layer's measurement being available.
3. A regression in any layer's fit triggers CI failure before it can reach `main`.
4. `cli/microarch_validation_report.py` reports CALIBRATED across Layers 1–7 for the SKU, and CALIBRATED on Layers 8–9 where multi-device hardware exists. Layers 8–9 remain `THEORETICAL` in single-device development environments; the report explicitly distinguishes "no hardware" from "fit failed".

**Workload-level end-to-end accuracy (e.g., ResNet-18 within 15%) is explicitly out of scope for this plan** and is the charter of the separate workload-decomposition plan (Hierarchy B).
