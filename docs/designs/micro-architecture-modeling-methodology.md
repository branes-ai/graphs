# Micro-Architecture Modeling Methodology

This document describes the methodology the `graphs` repository uses to
estimate latency and energy for computational graphs executing on diverse
hardware micro-architectures. It covers the analytical models, the calibration
strategy, the confidence tracking framework, and the software architecture
that implements them.

Companion documents describe the specifics for individual architecture
families:

- [`cpu-modeling.md`](./cpu-modeling.md) — x86 / ARM CPUs (Stored Program)
- [`gpu-modeling.md`](./gpu-modeling.md) — NVIDIA GPUs (SIMT Data Parallel)
- [`tpu-modeling.md`](./tpu-modeling.md) — Google TPUs (Systolic Array)
- [`kpu-modeling.md`](./kpu-modeling.md) — Stillwater KPU (Domain Flow)

---

## 1. Goals and Scope

The estimation layer must produce latency and energy estimates that are:

1. **Quantitatively defensible** — grounded in roofline analysis, datasheet
   specifications, measured calibration data, and process-physics-based
   energy models.
2. **Confidence-tracked** — every descriptor carries an
   `EstimationConfidence` (`CALIBRATED > INTERPOLATED > THEORETICAL > UNKNOWN`)
   so downstream consumers (the Embodied-AI-Architect orchestrator) know how
   much to trust each result.
3. **Architecture-aware** — the model captures why a KPU tile array has a
   fundamentally different energy signature than an NVIDIA SM, not just a
   scalar efficiency number.
4. **Composable** — the same pipeline drives DNNs, Kalman filters, PID
   controllers, and mixed perception/planning/control pipelines.

---

## 2. Analytical Foundation

### 2.1 Roofline Model for Latency

The core latency model is the roofline:

```
compute_time = ops / effective_ops_per_sec
memory_time  = bytes_transferred / peak_bandwidth
latency      = max(compute_time, memory_time) + overhead
```

- `ops` is a **precision-agnostic operation count** (FLOPs for floating
  point, INT ops for quantized, MACs × 2 when only MAC counts are known).
- `bytes_transferred` is the sum of input, weight, and output bytes that
  must cross a bandwidth boundary (DRAM, interconnect, or scratchpad).
- `effective_ops_per_sec` is derived from the hardware resource model,
  derated for DVFS, occupancy, allocated compute units, and empirical
  calibration (§4).

The **arithmetic intensity** (AI = ops / bytes) compared against the
hardware's **AI breakpoint** (peak_ops / peak_bandwidth) determines the
bottleneck classification:

- AI < breakpoint → memory-bound / bandwidth-bound
- AI > breakpoint → compute-bound
- within a factor of 1.5× → balanced

This rule lives in `HardwareMapper._calculate_latency()` and is used
uniformly across every mapper.

### 2.2 Three-Component Energy Model

Per the estimation rules:

```
total_energy = compute_energy + memory_energy + static_energy
```

- **Compute energy**: ops × energy_per_op(precision), where
  `energy_per_op` derives from process node, circuit type, and precision
  scaling (§3).
- **Memory energy**: bytes_transferred × energy_per_byte. For
  architectures with a more detailed memory hierarchy, each transfer is
  attributed to the appropriate level (L1 / scratchpad / HBM / DDR).
- **Static / idle energy**: `idle_power × latency`, where `idle_power` is
  a fraction of TDP (default 50% for nanoscale SoCs — see
  `KPUMapper.IDLE_POWER_FRACTION` and `GPUMapper.IDLE_POWER_FRACTION`).
  This captures leakage at modern process nodes and prevents the common
  underestimate that comes from counting only dynamic switching energy.

The constraint `power = total_energy / latency ≤ TDP` is a sanity check
applied at the graph level.

### 2.3 Architectural Energy Overhead

Beyond the baseline three-component model, each architecture adds a
**resource-contention-management** energy overhead captured by
`ArchitecturalEnergyModel` (see `graphs/hardware/architectural_energy.py`).
Architectures are classified into six `ArchitectureClass` values:

| Class | Representative | Overhead vs. stored-program baseline |
|-------|----------------|---------------------------------------|
| `STORED_PROGRAM` | CPU, DSP | 1.0× (baseline: fetch + decode + OoO) |
| `DATA_PARALLEL` | GPU | 2.5–3.0× (SIMT coherence + thread scheduling) |
| `SYSTOLIC_ARRAY` | Google TPU | 0.10–0.20× (fixed spatial schedule) |
| `DOMAIN_FLOW` | Stillwater KPU | 0.25–0.40× (SURE/SARE domain tracking) |
| `SPATIAL_PARTITION` | Cerebras, Hailo | 0.15–0.30× (graph-mapped mesh) |
| `ADAPTIVE_DATAPATH` | FPGA / CGRA / DPU | 0.15–0.30× + reconfiguration penalty |

Each model returns an `ArchitecturalEnergyBreakdown` with separate
MAC/FLOP/IntOp energies so mappers can route matmul work to tensor cores, 
systolic arrays, or domain flow fabrics, elementwise FLOPs to scalar or 
vector units, and integer indexing/quantization to integer ALUs.

### 2.4 Precision Scaling

Energy per operation scales sub-linearly with bit width because a fused
INT8 MAC has less switching capacitance than a full-width FP32 MAC. The
default scaling table (see `HardwareResourceModel.energy_scaling`) is:

| Precision | Energy × FP32 |
|-----------|---------------|
| FP64 | 2.00 |
| FP32 | 1.00 |
| FP16 / BF16 | 0.50 |
| FP8 (E4M3/E5M2) | 0.25 |
| FP4 | 0.125 |
| INT32 | 0.50 |
| INT16 | 0.25 |
| INT8 | 0.125 |
| INT4 | 0.0625 |

Per-fabric overrides live in each `ComputeFabric.energy_scaling` (e.g.,
KPU INT8 tiles use 0.15 because tile-internal datapaths are narrower
than the generic scaling assumes).

---

## 3. Physics-Based Energy Baseline

The fundamental energy relation is:

```
E = C × V² per switching event
```

and `energy_per_FP32_op` for a standard-cell ALU is tabulated by process
node in `resource_model.PROCESS_NODE_ENERGY` (pJ):

| Node | Energy/op | Process examples |
|------|-----------|------------------|
| 3 nm | 1.20 pJ | Intel 18A, TSMC N3 |
| 4 nm | 1.30 pJ | TSMC N4/N4P |
| 5 nm | 1.50 pJ | TSMC N5 |
| 7 nm | 1.80 pJ | TSMC N7 |
| 12 nm | 2.50 pJ | TSMC 12FFC |
| 16 nm | 2.70 pJ | TSMC 16FFC |
| 28 nm | 4.00 pJ | TSMC 28HPC+ |

These are then multiplied by a **circuit-type multiplier** that captures
layout-level efficiency differences:

| Circuit type | Multiplier | Rationale |
|--------------|------------|-----------|
| `standard_cell` | 1.00 | Baseline ALU |
| `tensor_core` | 0.85 | Fused MAC + accumulate, amortized control |
| `simd_packed` | 0.90 | Packed SIMD (AVX-512, NEON) |
| `custom_datacenter` | 2.75 | 5+ GHz custom wide-datapath, extra pipeline stages |

`get_base_alu_energy(process_node_nm, circuit_type)` produces
`energy_per_flop_fp32` from first principles. Missing nodes are linearly
interpolated. This lets us compare, e.g., a 16 nm KPU tile (2.7 pJ ×
1.0) against a 5 nm H100 tensor core (1.5 pJ × 0.85 = 1.28 pJ) on the
same physical basis rather than via unverifiable efficiency numbers.

---

## 4. Hardware Model: Multi-Layer Specification

Hardware models are specified top-down, from physical resources to
empirically measured efficiency. Each layer can override the one below
when better data is available.

```
HardwareResourceModel
├── compute_units, threads_per_unit, warps_per_unit
├── peak_bandwidth, l1_cache_per_unit, l2_cache_total, main_memory
├── compute_fabrics : [ComputeFabric]              ← §4.1
├── precision_profiles : {Precision: PrecisionProfile}
├── thermal_operating_points : {name: ThermalOperatingPoint}  ← §4.2
├── architecture_energy_model : ArchitecturalEnergyModel     ← §2.3
├── energy_per_flop_fp32, energy_per_byte, energy_scaling
├── wave_quantization, min_occupancy, max_concurrent_kernels
└── bom_cost_profile : BOMCostProfile              ← market analysis
```

### 4.1 ComputeFabric — Multi-Fabric Support

Real accelerators contain multiple compute fabrics with distinct energy
and throughput characteristics:

- GPU: CUDA cores (`standard_cell`) + Tensor Cores (`tensor_core`)
- KPU: INT8 tiles + BF16 tiles + Matrix tiles (different precision
  optimization levels)
- CPU: Scalar ALUs (`standard_cell`) + SIMD units (`simd_packed`)

Each `ComputeFabric` declares its circuit type, unit count, per-precision
`ops_per_unit_per_clock`, core frequency, and process node. Peak
throughput is derived:

```
peak_ops = num_units × ops_per_unit_per_clock[precision] × core_frequency_hz
```

Peak power is derived from energy × rate:

```
peak_power = peak_ops × energy_per_op(precision)
```

### 4.2 ThermalOperatingPoint — DVFS Modeling

Edge devices ship with multiple power modes (e.g., Jetson Orin has 15 W /
30 W / 50 W / MAXN). Each mode is a `ThermalOperatingPoint` with:

- `tdp_watts`, `cooling_solution`
- per-precision `PerformanceCharacteristics`, each containing:
  - `compute_resource` (`ComputeResource` for homogeneous,
    `KPUComputeResource` for heterogeneous tile allocations)
  - `instruction_efficiency` (compiler/ISA overhead, 0–1)
  - `memory_bottleneck_factor` (memory-system limit, 0–1)
  - `efficiency_factor` — **empirical** measured/sustained ratio
  - `measured_ops_per_sec` — direct override if available

The derivation chain is:

```
peak_ops_per_sec     = N_units × ops_per_clock × max_boost_clock_hz
sustained_ops_per_sec = N_units × ops_per_clock × sustained_clock_hz
effective_ops_per_sec = sustained × emulation_penalty × tile_util × efficiency_factor
                     or measured_ops_per_sec  (if supplied)
```

The **thermal throttle factor** is `sustained / boost`; it is not uncommon
for passive 15 W profiles to hit 0.39 (i.e., 39% of boost is
sustainable).

### 4.3 Clock Domains

`ClockDomain` holds `base`, `max_boost`, and `sustained` clocks. The
roofline uses `effective_ops_per_sec` from the thermal operating point
(which already folds in DVFS). When no thermal point is provided, the
mapper falls back to `resource_model.get_peak_ops(precision)` (legacy
datasheet peak) — this path is reserved for early-stage models and
triggers `ConfidenceLevel.THEORETICAL`.

---

## 5. Calibration Framework

The calibration layer turns theoretical peaks into confidence-backed
estimates.

### 5.1 Microbenchmarks

`src/graphs/benchmarks/` provides microbenchmarks (GEMM sweeps, memory
bandwidth, Conv2D, activation kernels) that probe a target device across
operation types and problem sizes. Results feed into
`calibration_data/<hardware>/measurements/<precision>/<model>_b<batch>.json`.

### 5.2 Size-Dependent Efficiency Curves

Small GEMMs on large GPUs are egregiously inefficient — a single H100 can
sit at < 5% utilization on a 256×256 FP32 matmul because kernel launch
overhead dominates. Calibration captures efficiency vs. problem size on
a per-operation-type basis. The GPU mapper classifies each subgraph
(`matmul`, `conv2d`, `depthwise`, `activation`, `conv2d_batchnorm`,
`unfused`) and looks up `efficiency(op_type, ops)` from the calibration
curve. Cached results are reused across all analysis runs for that
device.

### 5.3 Memory-Bound Detection

Operations with arithmetic intensity below an architecture-specific
threshold (20 FLOPs/byte for GPU depthwise convolutions) are flagged as
memory-bound, and latency is taken directly from `memory_time` rather
than the compute-side calibrated number. This avoids double-counting
when calibration curves are derived from compute-bound operations.

### 5.4 Confidence Propagation

Every descriptor (`LatencyDescriptor`, `EnergyDescriptor`,
`MemoryDescriptor`, `ConcurrencyDescriptor`) carries an
`EstimationConfidence(level, source, ...)`. The chain is:

1. `CALIBRATED` — measured against real hardware for this op type and
   size.
2. `INTERPOLATED` — interpolated between two calibrated points.
3. `THEORETICAL` — derived from the analytical roofline + datasheet
   peaks, no measurement.
4. `UNKNOWN` — model is insufficient (e.g., precision not supported).

The aggregate report takes the **minimum** confidence across the chain,
so one THEORETICAL input is enough to demote the whole estimate.

---

## 6. Mapping Pipeline

The end-to-end pipeline is orchestrated by
`estimation.unified_analyzer.UnifiedAnalyzer`:

```
PyTorch / ONNX / JAX model
         │
         ▼
  frontends.trace_and_partition()
         │  produces
         ▼
  PartitionReport  (SubgraphDescriptors with FLOPs, bytes, parallelism)
         │
         ▼
  ConcurrencyAnalyzer  (identifies execution stages)
         │
         ▼
  HardwareMapper.map_graph(fusion_report, stages, batch, precision)
         │    ├── per-subgraph: allocate compute units, occupancy, wave-quant
         │    ├── per-subgraph: roofline latency (compute vs. memory time)
         │    ├── per-subgraph: calibrated efficiency lookup (GPU)
         │    ├── per-subgraph: tiling analysis (KPU)
         │    ├── per-subgraph: baseline + architectural energy
         │    └── aggregate: stage-parallel max, sum across stages, idle
         ▼
  GraphHardwareAllocation  (latency, energy, utilization, bottlenecks)
         │
         ▼
  RooflineAnalyzer, EnergyAnalyzer, MemoryEstimator  (report formatting)
         │
         ▼
  UnifiedAnalysisResult
```

### 6.1 Stage-Level Aggregation

For each execution stage:
- latency is the **max** across subgraphs that run concurrently in that
  stage (parallel execution)
- compute units used is the **max** (only one subgraph owns the unit at
  a time)

Across stages, latencies **sum** (sequential). Total energy sums all
dynamic energies and adds idle energy over the total latency.

### 6.2 Wave Quantization and Occupancy

GPU mappers allocate SMs in groups of `wave_quantization` (typically 4)
because partial waves stall the warp scheduler. Occupancy is
`warps_required / (SMs_allocated × warps_per_SM)`. An operation with <
`min_occupancy` (default 0.25) takes a quadratic penalty — this captures
that very low occupancy hits both latency and energy.

### 6.3 Tiling and Scratchpad Constraints

KPU (and other scratchpad-based accelerators) cannot ignore memory
capacity. `KPUMapper._analyze_tiling()` produces a `TileConfiguration`
describing how many tiles/iterations are needed to process operands
through the 256 KB scratchpad, and applies a `tiling_overhead` factor
(1.0 + 0.10 × (iterations − 1)) to both compute time and bytes
transferred. See [`kpu-modeling.md`](./kpu-modeling.md) for details.

---

## 7. Software Architecture

### 7.1 Package Layout

```
src/graphs/
├── core/                     # Graph structures, confidence, descriptors
├── frontends/                # PyTorch Dynamo / torch.export tracing
├── transform/                # Partitioning, fusion, tiling strategies
├── estimation/               # Analyzers (roofline, energy, memory, concurrency)
│   ├── roofline.py           # RooflineAnalyzer + LatencyDescriptor
│   ├── energy.py             # EnergyAnalyzer + EnergyDescriptor (3-component)
│   ├── memory.py             # MemoryEstimator (peak + timeline)
│   ├── concurrency.py        # Execution stages
│   ├── architectural_modifiers.py
│   └── unified_analyzer.py   # Orchestrator
├── calibration/              # Calibration framework + JSON profiles
├── benchmarks/               # GEMM / memory / conv microbenchmarks
├── hardware/
│   ├── resource_model.py             # HardwareResourceModel + dataclasses
│   ├── architectural_energy.py       # ArchitectureClass + energy models
│   ├── mappers/                      # Per-architecture mappers
│   │   ├── cpu.py
│   │   ├── gpu.py
│   │   ├── dsp.py
│   │   ├── accelerators/{tpu,kpu,dpu,cgra,hailo}.py
│   │   └── research/dfm.py
│   └── models/                       # Per-device resource_model factories
│       ├── datacenter/
│       ├── edge/
│       ├── automotive/
│       ├── mobile/
│       ├── accelerators/
│       └── research/
└── reporting/                # Report generation (JSON, CSV, MD, text)
```

### 7.2 Class Hierarchy

```
HardwareMapper (ABC)                  # graphs/hardware/resource_model.py
├── CPUMapper                         # mappers/cpu.py
├── GPUMapper                         # mappers/gpu.py
├── DSPMapper                         # mappers/dsp.py
├── TPUMapper                         # mappers/accelerators/tpu.py
├── KPUMapper                         # mappers/accelerators/kpu.py
├── DPUMapper                         # mappers/accelerators/dpu.py
├── CGRAMapper                        # mappers/accelerators/cgra.py
├── HailoMapper                       # mappers/accelerators/hailo.py
└── DFMMapper                         # mappers/research/dfm.py    (reference)

ArchitecturalEnergyModel (ABC)        # hardware/architectural_energy.py
├── StoredProgramEnergyModel
├── DataParallelEnergyModel
├── SystolicArrayEnergyModel
├── DomainFlowEnergyModel             # KPU
├── SpatialPartitionEnergyModel
├── AdaptiveDatapathEnergyModel
└── KPUTileEnergyAdapter              # wraps heterogeneous tile model
```

### 7.3 Key Contracts

Every `HardwareMapper` implements:

```python
map_subgraph(subgraph, execution_stage, concurrent_subgraphs, precision)
  -> HardwareAllocation
map_graph(fusion_report, execution_stages, batch_size, precision)
  -> GraphHardwareAllocation
```

Internal helpers provided by the base class:

```python
_calculate_latency(ops, bytes, allocated_units, occupancy, precision)
  -> (compute_time, memory_time, bottleneck)
_calculate_energy(ops, bytes, precision)
  -> (compute_energy, memory_energy)
_calculate_energy_with_architecture(ops, bytes, precision, ctx)
  -> (compute, memory, ArchitecturalEnergyBreakdown | None)
```

These helpers enforce the invariants (latency = max, not sum; energy =
three-component; efficiency factor applied after DVFS) so that
architecture-specific mappers can focus on their unique allocation logic.

### 7.4 Registry

All mappers are registered in `hardware/mappers/__init__.py` under
category-specific dicts (`CPU_MAPPERS`, `GPU_MAPPERS`,
`ACCELERATOR_MAPPERS`, …). `get_mapper_by_name(name)` performs a
case-insensitive lookup across every category so CLI tools and the
Embodied-AI-Architect agent can address devices by plain name.

### 7.5 Entry Points

- Programmatic: `UnifiedAnalyzer().analyze_model(model, hardware, ...)`
- CLI: `cli/analyze_comprehensive.py`, `cli/analyze_batch.py`,
  `cli/compare_architectures.py`, etc.
- Agentic: `src/graphs/mcp/server.py` exposes the estimator as MCP
  tools; the orchestrator calls these rather than re-implementing
  estimation logic.
- Validation: `validation/hardware/` and `validation/estimators/`
  regression-test accuracy against measured ground truth.

---

## 8. Invariants and Conventions

From `.claude/rules/estimation.md` and `.claude/rules/hardware-mappers.md`,
enforced in review:

- Every estimator descriptor **must** include an `EstimationConfidence`.
- `latency = max(compute_time, memory_time)` for roofline — never sum.
- `energy = compute + memory + static` (three-component).
- `power = total_energy / latency ≤ TDP` is an audit check.
- Utilization ∈ [0.0, 1.0]; never negative, never > 1.
- All units documented: seconds, joules, bytes, FLOPS.
- No Unicode characters in code or output.
- Precision support comes from `precision_profiles` — never hardcode.
- Resource model parameters cite the datasheet in comments.

---

## 9. Extending the Methodology

To add a new architecture family:

1. Define its `ArchitectureClass` (or reuse one) in
   `architectural_energy.py` and implement an `ArchitecturalEnergyModel`.
2. Add device resource model factories under
   `hardware/models/<category>/<device>.py` with cited datasheet
   parameters, thermal operating points, and `ComputeFabric`s.
3. Implement `<Arch>Mapper(HardwareMapper)` under
   `hardware/mappers/[accelerators/]<arch>.py`, overriding allocation
   logic and any architecture-specific constraints (tiling, systolic
   dimensions, spatial partitioning).
4. Register factory functions in `hardware/mappers/__init__.py`.
5. Add microbenchmarks to `benchmarks/` and record calibration JSON.
6. Add an accuracy validation test in `validation/hardware/` or
   `validation/estimators/`.

Architecture-specific documents (like the KPU and GPU docs) should
describe the deviations from this baseline methodology, not restate it.
