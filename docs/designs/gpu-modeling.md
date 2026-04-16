# GPU Modeling

This document describes how the `graphs` repository models **NVIDIA GPUs**
(Volta, Turing, Ampere, Hopper, Blackwell, and Jetson Orin/Thor edge
variants) — all in the `DATA_PARALLEL` architecture class. Read
[`micro-architecture-modeling-methodology.md`](./micro-architecture-modeling-methodology.md)
first for the shared latency/energy methodology; this document only
covers GPU-specific mechanisms.

The GPU implementation lives in:

- `src/graphs/hardware/mappers/gpu.py` — `GPUMapper`, calibration
  integration, sequential/parallel mode selection
- `src/graphs/hardware/models/datacenter/` and
  `src/graphs/hardware/models/edge/` — per-device resource models
- `src/graphs/hardware/architectural_energy.py` —
  `DataParallelEnergyModel` (SIMT overhead model)
- `src/graphs/calibration/` — GPU calibration curves (size-dependent
  efficiency per operation type)

---

## 1. Architecture Summary

NVIDIA GPUs are **SIMT (Single Instruction Multiple Thread)** data-
parallel machines. Relevant modeling facts:

1. **SM hierarchy** — the chip is a grid of Streaming Multiprocessors.
   Each SM contains CUDA cores (scalar FP/INT ALUs) and Tensor Cores
   (matrix units). H100 has 132 SMs; Jetson Orin AGX has 16.
2. **Thread → warp → block → SM mapping** — 32 threads form a warp
   that executes in lockstep; warps are bundled into blocks, mapped to
   SMs. Divergent branches serialize within a warp.
3. **Multiple fabrics per SM** — CUDA cores (FP64/FP32/FP16 scalar
   FMA) and Tensor Cores (matrix FMA at BF16/FP16/FP8/FP4/INT8) have
   vastly different peak throughput and energy.
4. **Memory hierarchy** — registers → Shared Memory/L1 (unified on
   Hopper) → L2 (chip-wide, MB-scale) → HBM/GDDR. All accesses above
   registers are managed by the cache subsystem (reactive, not
   compiler-directed).
5. **Massive coherence machinery** — thousands of in-flight memory
   requests require coalescing, ordering, and atomic management. This
   is the dominant SIMT energy overhead at small batch.
6. **DVFS and thermal throttling** — datacenter parts run close to
   sustained clocks; edge parts (Jetson) throttle aggressively under
   passive cooling.
7. **Kernel launch overhead** — each CUDA kernel dispatch costs time
   and energy: ~10 µs on datacenter GPUs, ~80 µs on Jetson-class
   (Tegra software stack).

---

## 2. Device Coverage

The model covers datacenter, workstation, and edge GPUs via factory
functions in `mappers/gpu.py`:

| Family | Device | SMs | Memory | Thermal | Class |
|--------|--------|-----|--------|---------|-------|
| Volta | V100 SXM3 32 GB | 80 | 32 GB HBM2 | 300 W | Datacenter |
| Ampere | A100 SXM4 80 GB | 108 | 80 GB HBM2e | 400 W | Datacenter |
| Hopper | H100 PCIe / SXM5 80 GB | 132 | 80 GB HBM2e/3 | 350/700 W | Datacenter |
| Blackwell | B100 SXM6 192 GB | ~132 (dual die) | 192 GB HBM3e | 700 W | Datacenter |
| Jetson | Orin Nano 8 GB | 8 (Ampere) | 8 GB LPDDR5 | 7–15 W | Edge |
| Jetson | Orin NX 16 GB | 12 (Ampere) | 16 GB LPDDR5 | 10–25 W | Edge |
| Jetson | Orin AGX 64 GB | 16 (Ampere) | 64 GB LPDDR5 | 15–60 W (MAXN) | Edge |
| Jetson | Thor 128 GB | ~ (Blackwell) | 128 GB LPDDR5X | 30–130 W | Automotive |

Each factory function:

1. Calls the device's `*_resource_model()` factory (e.g.,
   `h100_sxm5_80gb_resource_model()`).
2. Attaches a `DataParallelEnergyModel(tech_profile=...)` — the
   technology profile encodes process node, memory type, and their
   per-byte energy.
3. Returns a `GPUMapper(resource_model, thermal_profile=...,
   calibration=...)`.

---

## 3. Latency Model

The GPU mapper runs in one of two modes, selected automatically per
graph.

### 3.1 Mode Selection

`GPUMapper.should_use_sequential_execution(fusion_report, batch_size)`:

- **Sequential mode** (per-kernel dispatch, 1–8 SMs each) when:
  - `batch_size < 8`, **and**
  - average FLOPs per subgraph < 200 M.
- **Parallel mode** (wave-quantized concurrent execution) otherwise.

The sequential mode is the key to accuracy for small DNNs (ResNet-18,
MobileNet, batch=1 inference): assuming full-device parallelism
overestimates throughput by 10–1000× because each kernel can't saturate
132 SMs, and kernel launch overhead dominates.

### 3.2 Sequential Mode (`compute_sequential_latency`)

For each subgraph:

1. **SM allocation from parallelism (not FLOPs):**
   ```
   output_elements = max(1, total_output_bytes / 4)      # FP32-equivalent
   warps_required  = ceil(output_elements / 32)
   sms_needed      = ceil(warps_required / warps_per_SM) # typical 48 (Ampere)
   sms_allocated   = min(sms_needed, total_SMs)
   ```
   Rationale: the warp scheduler launches one thread per output
   element. Utilization reflects hardware occupancy, not per-op
   efficiency.

2. **Per-SM throughput from microarchitecture parameters:**
   ```
   sm_flops = cuda_cores_per_sm × ops_per_clock_per_core × sm_sustained_clock_hz
   ```
   (Fallback to `peak_ops / compute_units` only when microarch fields
   are absent on older models.)

3. **Shared bandwidth (global resource, not per-SM):**
   ```
   compute_time = ops / (sm_flops × sms_allocated)
   memory_time  = bytes / peak_bandwidth
   kernel_time  = max(compute_time, memory_time)
   ```

4. **Calibration-driven efficiency (when calibration attached):**
   ```
   op_type         = _classify_operation(subgraph)   # conv2d, matmul, depthwise, …
   calibrated_eff  = calibration.get_efficiency(op_type, ops, default=0.50)
   calibrated_lat  = ops / (peak_ops × calibrated_eff)
   kernel_time     = calibrated_lat                  # replaces raw roofline
   ```
   For **depthwise** (AI < 20) the latency is taken directly from
   `memory_time` — these operations are bandwidth-bound and calibrating
   them against compute-curve samples would double-count.

5. **Kernel launch overhead:**
   - Datacenter: 10 µs per kernel.
   - Edge (Jetson/Tegra): 80 µs per kernel.
   - With calibration: `kernel_latency = max(kernel_time, overhead)`
     (calibration may already include dispatch effects for tiny ops).
   - Without calibration: `kernel_latency = kernel_time + overhead`.

6. **Total sequential latency:** sum of `kernel_latency` across all
   subgraphs. Each subgraph becomes its own execution stage.

### 3.3 Parallel Mode (`map_subgraph`)

For larger workloads, `map_subgraph` uses the classical thread → warp
→ SM cascade with **wave quantization**:

```
threads_required   = subgraph.parallelism.total_threads
warps_required     = ceil(threads_required / warp_size)         # 32
sms_ideal          = ceil(warps_required / warps_per_unit)      # e.g. 48
sms_allocated      = ceil(sms_ideal / wave_quantization) × wave_quantization
                    # typically 4-SM waves
sms_allocated      = min(sms_allocated, total_SMs)

occupancy          = min(1.0, warps_required / (sms_allocated × warps_per_unit))
utilization        = sms_allocated / total_SMs
```

Roofline then uses the base class's `_calculate_latency` with allocated
SMs and occupancy; calibration refinement is applied exactly as in
sequential mode.

### 3.4 Graph Mapping

`map_graph`:

- **Parallel mode**: within each stage, `latency = max(latency_i)` and
  `sms_used = max(sms_i)`. Across stages, latencies sum.
- **Sequential mode**: each subgraph is its own stage; sum
  `kernel_latency` directly.

The returned `GraphHardwareAllocation` contains total latency, peak
and average SM utilization, bottleneck counts, and a
`latency_correction_factor = naive / actual` (typically 5–20× for
small DNNs on datacenter GPUs).

---

## 4. Energy Model

### 4.1 Baseline Three-Component Energy

As elsewhere, `HardwareMapper._calculate_energy` produces the
compute + memory pair from:

- `compute_energy = ops × energy_per_flop_fp32 × energy_scaling[precision]`
- `memory_energy = bytes × energy_per_byte`

`GPUMapper.compute_energy_with_idle_power` adds idle energy:

- `idle_power = 0.5 × TDP` (nanoscale leakage; `IDLE_POWER_FRACTION =
  0.5`).
- `total_energy = dynamic_energy + idle_power × latency`.
- `average_power = total_energy / latency` (must not exceed TDP).

TDP comes from the active `ThermalOperatingPoint`; when absent the
model estimates `tdp ≈ 2 × dynamic_power`.

### 4.2 Architectural Overhead — `DataParallelEnergyModel`

The GPU is the most overhead-heavy class in the taxonomy
(2.5–3.0× stored-program baseline). `DataParallelEnergyModel` models
the SIMT-specific events on top of the baseline:

| Bucket | Events |
|--------|--------|
| **Instruction pipeline** | `instruction_fetch_energy`, `operand_fetch_overhead`, decode/execute, `instructions_per_op = 0.1` (one instr per 10 ops, amortized by SIMT width). |
| **SIMT coherence** | `coherence_energy_per_request` × thousands of concurrent requests; dominates energy for small batches. |
| **Thread scheduling** | `thread_scheduling_overhead` × thousands of threads. |
| **Warp divergence** | `warp_divergence_penalty` × divergent-op rate (5% default). |
| **Memory coalescing** | `memory_coalescing_overhead` × uncoalesced rate (10% default). |
| **Barriers** | `barrier_sync_energy` × `barriers_per_1000_ops` (5 by default). |
| **Compute fabric** | `cuda_core_mac_energy`, `cuda_core_flop_energy`, `int_alu_energy`, `tensor_core_mac_energy` (+ `tensor_core_utilization ≈ 0.8` for MACs that can route to tensor cores, `TENSOR_CORE_MACS_PER_OP = 64` for 4×4×4 matmul). |
| **Memory hierarchy** | `shared_memory_l1_unified_energy_per_byte` (95% hit), `l2_cache_energy_per_byte` (90% hit on L1 misses), `dram_energy_per_byte`. |
| **Register file** | `register_file_energy_per_access`. |

The model returns an `ArchitecturalEnergyBreakdown` with separate
`mac_energy`, `flop_energy`, and `intop_energy` so downstream tools
can distinguish tensor-core-bound work from CUDA-core-bound work.

All coefficients are pulled from the attached `TechnologyProfile`
(e.g., `DATACENTER_4NM_HBM3` for H100, `EDGE_8NM_LPDDR5` for Orin
Nano). This makes cross-technology studies reproducible — the same
mapper can be re-run against a synthetic 3 nm HBM4 profile.

### 4.3 Why the GPU Loses Small-Batch Energy Comparisons

At batch=1 on a 128-neuron MLP, a KPU can finish in ~µJ while a GPU
burns ~mJ for equivalent work — the 1000× gap is not compute, it's
coherence. The `DataParallelEnergyModel` attributes the overhead
cleanly:

- Coherence machinery: ~5 pJ × (thousands of concurrent requests) per
  op.
- Thread scheduling: ~1 pJ × (thousands of threads).
- Divergence + coalescing: 10–15% extra at SIMT widths < 32.

These overheads are fixed per kernel launch and dominate until the
workload is large enough to amortize them across tens of thousands of
ops per dispatch.

---

## 5. Resource Model Specification (how a GPU SKU is declared)

A `<device>_resource_model()` factory typically declares:

1. **Compute fabrics**:
   - CUDA Core fabric (`circuit_type=standard_cell`), per-SM count ×
     `ops_per_unit_per_clock` for FP64/FP32/INT32.
   - Tensor Core fabric (`circuit_type=tensor_core`), per-SM count ×
     per-precision ops (often 512 BF16, 1024 FP8, 2048 INT8 per clock
     per TC).
2. **Microarchitecture fields** on `HardwareResourceModel`:
   - `cuda_cores_per_sm` (64 for Pascal–Turing, 128 for Ampere–Hopper)
   - `ops_per_clock_per_core` (2.0 for FMA)
   - `sm_boost_clock_hz`, `sm_sustained_clock_hz`
   - `tensor_cores_per_sm`, `tensor_core_ops_per_clock`
   - `warps_per_unit` (48 for Ampere, 64 for Hopper), `warp_size = 32`
3. **Precision profiles** — one `PrecisionProfile` per supported
   precision with peak ops/sec derived from the fabric sums; default
   precision for mixed-precision models.
4. **Thermal operating points** — for edge parts (Orin 15 W / 30 W /
   MAXN) each profile has its own `ClockDomain` and per-precision
   `PerformanceCharacteristics` with empirical `efficiency_factor`s.
5. **Wave quantization** — `wave_quantization = 4` (SMs allocated in
   groups of 4).
6. **Kernel-launch overhead** — implicit via `GPUMapper._is_edge_device`
   name check; no per-device field needed yet.

The mapper factory attaches:
```python
resource_model.architecture_energy_model = DataParallelEnergyModel(
    tech_profile=DATACENTER_4NM_HBM3  # or EDGE_8NM_LPDDR5, etc.
)
```

---

## 6. Calibration

GPU calibration is the most mature in the codebase because:

- Datacenter GPUs have highly non-linear efficiency vs. problem size
  (H100 runs at < 5% on tiny GEMMs, > 70% on large ones).
- Jetson parts show similar but shifted curves (DVFS-driven).

### 6.1 Operation Classification

`_classify_operation(subgraph)` maps a fused subgraph to a calibration
key by inspecting `fusion_pattern` and node names:

| Key | Maps to |
|-----|---------|
| `depthwise` | Conv with AI < 20 (bandwidth-limited) |
| `conv2d_batchnorm` | Conv + BN fused patterns |
| `conv2d` | Standard convolution |
| `matmul` | Linear, MatMul, MM, GEMM |
| `activation` | ReLU, GELU, Softmax, Sigmoid, Tanh, SiLU, Hardswish |
| `unfused` | Add, Mul, fallback |

### 6.2 Efficiency Lookup

```python
calibrated_eff = calibration.get_efficiency(op_type, flops,
                                            default=default_efficiency)
```

The calibration backend interpolates efficiency along a measured
FLOPs-vs-efficiency curve per op type. `default_efficiency = 0.50`
when calibration is not attached.

### 6.3 Memory-Bound Bypass

When `op_type == "depthwise"`, calibration is bypassed and
`estimated_latency = memory_time`. Depthwise convs have AI ≈ 1–10 and
are nearly always bandwidth-bound; applying a compute-curve efficiency
factor would produce wildly wrong answers (calibration data is gathered
on compute-bound problems).

### 6.4 Sources

Calibration data lives under `calibration_data/<hardware>/measurements
/<precision>/` in v2.0 layout. Jetson Orin AGX has full coverage across
15 W / 30 W / 50 W / MAXN at FP32 / FP16 / INT8. Datacenter GPUs (H100,
A100) have GEMM sweeps.

---

## 7. Known Limitations (improvement backlog)

1. **Output element count from bytes** — `determine_sm_allocation()`
   currently divides `total_output_bytes` by a fixed 4 bytes/element.
   For INT8 inference this overestimates output elements by 4× and
   thus SMs needed. Replace with element counts from the subgraph's
   output tensor descriptors when available.
2. **Kernel launch overhead is a name-based check** — `_detect_edge_device`
   substrings on device name (`jetson`, `orin`, `tegra`, …). Promote to
   an explicit `HardwareResourceModel.kernel_launch_overhead_us` field.
3. **No CUDA graph amortization** — repeated small kernels in a CUDA
   graph amortize dispatch. No knob for this yet.
4. **No per-stream concurrency** — modern GPUs can dispatch multiple
   streams concurrently (MPS, MIG, CUDA streams). The mapper assumes
   one active stream.
5. **Tensor Core utilization is a constant** — `tensor_core_utilization
   = 0.80` globally in `DataParallelEnergyModel`. In reality this varies
   with tile shape (matmul N/M not divisible by 16 drops it to 0).
6. **No NVLink / PCIe modeling** — multi-GPU or host-device transfers
   are not represented.
7. **Calibration default_efficiency is a scalar** — 0.50 is fine for
   Ampere-class hardware at mid-size workloads but pessimistic for
   H100 large-GEMM, optimistic for Jetson tiny-MLP. Per-device
   defaults would close the gap.
8. **Warp divergence / coalescing rates are fixed constants** — 5% and
   10% respectively, regardless of workload. A branch-aware model
   fed from the partition report would improve fidelity on control-
   heavy graphs.

---

## 8. Validation Hooks

- `validation/hardware/test_all_hardware.py` — 10-way comparison
  including H100, A100, V100, all Jetson Orin variants.
- `validation/estimators/test_resnet18.py` — direct accuracy regression
  against measured end-to-end latency.
- `cli/calibration_coverage.py` — reports which (hardware, precision,
  op-type, size) combinations have calibration coverage.
- `cli/generate_gpu_energy_slides.py` — GPU energy comparison deck.

Any mapper change should re-run these before shipping. When changing
calibration lookup logic, ensure both sequential and parallel paths
still apply the change consistently.
