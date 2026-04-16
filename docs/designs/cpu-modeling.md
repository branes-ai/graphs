# CPU Modeling

This document describes how the `graphs` repository models **CPUs** — x86
(Intel Xeon, Core, AMD EPYC, Ryzen), ARM server (Ampere AmpereOne), and
ARM edge (Cortex-A78AE in Jetson Orin) — all in the `STORED_PROGRAM`
architecture class. Read
[`micro-architecture-modeling-methodology.md`](./micro-architecture-modeling-methodology.md)
first for the shared latency/energy methodology; this document only
covers CPU-specific mechanisms.

The CPU implementation lives in:

- `src/graphs/hardware/mappers/cpu.py` — `CPUMapper`,
  `CPUVectorization`, factory functions
- `src/graphs/hardware/models/datacenter/` — x86 and ARM server SKUs
  (`cpu_x86.py`, `cpu_arm.py`, `intel_xeon_platinum_*.py`,
  `amd_epyc_*.py`, `ampere_ampereone_*.py`, `intel_granite_rapids.py`,
  `amd_epyc_turin.py`)
- `src/graphs/hardware/models/edge/jetson_orin_agx_cpu.py` — ARM edge
- `src/graphs/hardware/architectural_energy.py` —
  `StoredProgramEnergyModel` (5-stage pipeline + 4-level memory
  hierarchy overhead)

---

## 1. Architecture Summary

CPUs are von-Neumann **stored-program** machines with modest parallelism
and deep, reactive memory hierarchies. Relevant modeling facts:

1. **Core + SIMD + (optional) matrix unit** — each core has a scalar
   pipeline plus vector units (AVX-2 = 8-wide FP32, AVX-512 = 16-wide,
   NEON = 4-wide, SVE = up to 16-wide). Modern Intel server cores add
   **AMX** (Advanced Matrix Extensions) for INT8/BF16 tile matmuls.
2. **Moderate core counts** — 10–20 consumer, 32–128 server, 192+ on
   AmpereOne and EPYC Turin.
3. **Deep cache hierarchy** — L1 per-core, L2 per-core, L3 shared
   (LLC). Cache-coherent across cores. Caches are **reactive** (vs.
   compiler-directed scratchpads on KPU/TPU) — hit rate is a function
   of the workload, not the schedule.
4. **Much lower memory bandwidth** — 75 GB/s consumer DDR5, ~300–575
   GB/s server multi-channel DDR5. Orders of magnitude below GPU
   HBM.
5. **Hybrid cores** — Intel 12th-gen+ ships Performance + Efficiency
   cores with different clocks and SIMD capabilities. Modeled as
   "effective core count" = `P + 0.6·E`.
6. **SMT (HyperThreading)** — helps latency-hiding on throughput
   workloads, but doesn't add compute throughput for well-vectorized
   code. Model uses `1 thread per core` for compute.
7. **Instruction overhead dominates at small batch** — ~2 instructions
   per FLOP, ~50 branches per 1000 ops. This is the reason CPUs pay
   the 1.0× baseline in the architectural taxonomy.

---

## 2. Device Coverage

Factory functions in `mappers/cpu.py`:

### Datacenter x86

| Device | Cores | SIMD | Peak FP32 | Peak INT8 | DRAM BW | TDP | Notes |
|--------|-------|------|-----------|-----------|---------|-----|-------|
| Intel Xeon Platinum 8490H (Sapphire Rapids) | 60 | AVX-512 + AMX | 2.78 TF | 88.7 TOPS (AMX) | 307 GB/s | 350 W | Baseline server AI |
| Intel Xeon Platinum 8592+ (Sapphire Rapids) | 64 | AVX-512 + AMX | 3.07 TF | 98.3 TOPS (AMX) | 307 GB/s | 350 W | Flagship SR |
| Intel Granite Rapids (projected) | 128 | Enhanced AMX + FP8/INT4 | 6.55 TF | 209.7 TOPS | 358 GB/s | 500 W | Next-gen tile chiplet |
| AMD EPYC 9654 (Genoa, Zen 4) | 96 | AVX-512 (double-pumped) | 1.84 TF | 7.37 TOPS | 461 GB/s | 360 W | Cloud / VM density |
| AMD EPYC 9754 (Genoa, Zen 4) | 128 | AVX-512 (DP) | 2.30 TF | 9.22 TOPS | 461 GB/s | 360 W | 128-core flagship |
| AMD EPYC Turin (Zen 5, projected) | 192 | Native AVX-512 | 3.84 TF | 15.4 TOPS | 576 GB/s | 500 W | Next-gen 3 nm |

### Datacenter ARM

| Device | Cores | SIMD | Peak FP32 | Peak INT8 | DRAM BW | TDP |
|--------|-------|------|-----------|-----------|---------|-----|
| Ampere AmpereOne A128-30X | 128 | 2×128-bit NEON/SVE | 3.69 TF | 14.75 TOPS | 333 GB/s | 210 W |
| Ampere AmpereOne A192-32X | 192 | 2×128-bit NEON/SVE | 5.5 TF | 22.1 TOPS | 333 GB/s | 283 W |

### Consumer x86

| Device | Cores | SIMD | Peak FP32 (sustained) | Notes |
|--------|-------|------|------------------------|-------|
| Intel i7-12700K (Alder Lake) | 8P+4E → 10 effective | AVX-2 + VNNI | 720 GFLOPS | Two mappers ship (tiny vs. large model tuning) |

### Edge ARM

| Device | Cores | SIMD | Peak INT8 | Power | Use |
|--------|-------|------|-----------|-------|-----|
| Cortex-A78AE in Jetson Orin AGX | 12 | NEON + dotprod | 845 GOPS @ 30 W | 30 W | Robotics host CPU |

Factory functions: `create_intel_cpu_mapper`, `create_amd_cpu_mapper`,
`create_i7_12700k_mapper`, `create_i7_12700k_large_mapper`,
`create_ampere_ampereone_{128,192}_mapper`,
`create_intel_xeon_platinum_{8490h,8592plus}_mapper`,
`create_amd_epyc_{9654,9754,turin}_mapper`,
`create_intel_granite_rapids_mapper`,
`create_jetson_orin_agx_cpu_mapper`.

---

## 3. Latency Model

### 3.1 Vectorization Analysis

`CPUMapper._analyze_vectorization(subgraph, precision)` returns a
`CPUVectorization` with:

- **Effective SIMD width** from register width and element size:
  - AVX-512 (64 B register): 16 FP32 / 32 FP16 / 64 INT8.
  - AVX-2 (32 B register): 8 FP32 / 16 FP16 / 32 INT8.
- **Vectorization efficiency** from operation type:
  - Element-wise (ReLU, sigmoid, add): **0.95** — vectorizes near-
    perfectly.
  - Matrix (Conv, Linear, MatMul): **0.80** — stride and boundary
    overhead.
  - Default / unclassified: **0.70**.
- **Special accelerators** detected from precision + SIMD width:
  - `uses_amx`: matrix op at BF16/INT8 with AVX-512 width → AMX path.
  - `uses_vnni`: matrix op at INT8 with AVX-2+ → VNNI path.

### 3.2 Core Allocation

CPU parallelism is batch-dominated. The heuristic in `map_subgraph`:

```
cores_allocated = min(batch, total_cores)
if cores_allocated < total_cores:
    extra = min(total_cores - cores_allocated, channels // 4)
    cores_allocated += extra
cores_allocated = max(1, cores_allocated)
```

Rationale: batch parallelism splits cleanly across cores; intra-op
parallelism (channel-wise) is useful only when `channels / cores ≥ 4`
(Amdahl's law — too little work per core hits synchronization
overhead).

### 3.3 Roofline With Vectorization / Accelerator Speedup

```
effective_ops = ops / (simd_width × 0.7)    # 70% SIMD efficiency
if uses_amx: effective_ops = ops / 3.0       # AMX: 2-4× matrix speedup
elif uses_vnni: effective_ops = ops / 2.0    # VNNI: ~2× INT8 speedup

compute_time, memory_time, bottleneck =
    _calculate_latency(effective_ops, bytes, cores_allocated,
                       occupancy, precision)
```

Occupancy is simply `cores_allocated / total_cores`.

### 3.4 Calibration Correction

When a `HardwareCalibration` is attached:

```
op_type        = _classify_operation(subgraph)   # matmul, conv2d, add, maxpool, unknown
calibrated_eff = calibration.get_efficiency(op_type, **extra)
compute_time  *= (default_efficiency / calibrated_eff)

theoretical_bw  = resource_model.peak_bandwidth
measured_bw     = calibration.measured_bandwidth_gbps * 1e9
memory_time    *= (theoretical_bw / measured_bw)
```

`_classify_operation` returns a `(type, extra)` tuple where `extra`
for matmul includes a rough `matrix_size` bucket (`small` < 768,
`medium` < 3072, `large` ≥ 3072). Calibration curves are keyed on
`(op_type, matrix_size)` so small-matrix inefficiency is captured
explicitly.

### 3.5 Threading Overhead

```
threading_overhead = 1.0 + (cores_allocated − 1) × 0.02
compute_time      *= threading_overhead
```

Each additional core adds 2% synchronization overhead (OpenMP/TBB
barriers, false sharing, NUMA crossing). This caps the benefit of
over-parallelizing small ops.

### 3.6 Graph Mapping

`map_graph` sums per-stage latencies (sequential across stages, max
within a stage), exactly as in the base methodology. The
`latency_correction_factor` vs. naive peak is usually 10–30× for
consumer hardware and 3–10× for AMX-accelerated server parts on
well-matched workloads.

---

## 4. Energy Model

### 4.1 Baseline Three-Component Energy

`HardwareMapper._calculate_energy()` supplies compute + memory from:

- `compute_energy = ops × energy_per_flop_fp32 × energy_scaling[precision]`
- `memory_energy = bytes × energy_per_byte`

`CPUMapper.compute_energy_with_idle_power()` adds static idle energy
at `IDLE_POWER_FRACTION = 0.5 × TDP`. Note: **idle power does not
scale with DVFS** — leakage at nanoscale is largely frequency-
independent. TDP comes from the active `ThermalOperatingPoint`; for
consumer parts there's typically a single `consumer-continuous`
profile at PL1 (e.g., 125 W for i7-12700K).

### 4.2 Architectural Overhead — `StoredProgramEnergyModel`

The CPU is the **baseline (1.0×)** in the architectural energy
taxonomy. All other classes are scaled relative to this model's
control overhead. `StoredProgramEnergyModel` decomposes the CPU
control + memory cost:

| Bucket | Events |
|--------|--------|
| **Instruction pipeline** | `instruction_fetch_energy` (I-cache read), `instruction_decode_energy`, `instruction_dispatch_energy`. ~2 instructions per FLOP. |
| **Register file** | `register_file_read_energy` (2 reads) + `register_file_write_energy` (1 write) per instruction. *Register energy ≈ ALU energy* — this is usually the surprise for CPU energy budgets. |
| **4-stage memory hierarchy** | `l1_cache_energy_per_byte` (85% hit), `l2_cache_energy_per_byte` (90% hit on L1 miss), `l3_cache_energy_per_byte` (95% hit on L2 miss), `dram_energy_per_byte` (remainder). |
| **ALU** | `alu_energy_per_op`. |
| **Branch prediction** | `branch_prediction_overhead` × `branches_per_1000_ops` (50 by default for AI workloads). |

All coefficients come from the attached `TechnologyProfile` — e.g.,
`DATACENTER_7NM_DDR5` for Xeon 8490H, `EDGE_8NM_LPDDR5` for Jetson
Orin's ARM cores. This ensures cross-family comparisons use consistent
process-node physics.

Register-file energy is worth emphasizing: for typical loads,
register reads/writes can **equal** ALU energy. This is why CPUs lose
energy comparisons to spatial architectures even before touching
DRAM.

### 4.3 Why CPUs Are Inefficient for AI (and Good for Control)

The `StoredProgramEnergyModel` breakdown makes the reasons explicit:

- **~2 instructions per FLOP** × instruction-fetch + decode + dispatch
  energies is a per-op overhead systolic/spatial arrays don't pay.
- **Register file turnover** dominates small-op energy before DRAM
  even enters the picture.
- **Reactive cache hierarchy** — L1 hit rate on streaming AI workloads
  is 85%, meaning 15% pay the L2 cost, 10% of those pay L3, 5% of
  those hit DRAM. Compiler-directed scratchpads eliminate this
  cascade.
- **Branch prediction overhead** exists even on branchy control code.

Conversely, CPUs remain the best choice for:

- Irregular control flow (Kalman updates with data-dependent branches).
- Very small workloads where kernel dispatch overhead swamps any
  accelerator speedup.
- Parts of the pipeline where memory layout is hostile to tile-based
  accelerators.

---

## 5. Resource Model Specification

A `<device>_resource_model()` factory typically declares:

1. **Clock domain** — `base`, `max_boost`, and `sustained` clocks. For
   hybrid cores the sustained "all-core" clock is used (e.g., i7-12700K
   at 4.5 GHz sustained).
2. **Compute resource / fabrics**:
   - Consumer parts: one `ComputeResource` with
     `ops_per_unit_per_clock` keyed by precision (16 FP32, 16 FP16
     emulated, 32 INT8 with VNNI for AVX-2; double those numbers for
     AVX-512).
   - Server Intel with AMX: a separate AMX `ComputeFabric` or
     precision profile where INT8/BF16 peak shoots up 10–30×.
3. **Effective-cores accounting** for hybrid:
   `effective_cores = P + 0.6·E` (E-cores ≈ 60% of P-cores on Alder
   Lake per empirical SPEC measurements).
4. **Precision profiles** — one `PrecisionProfile` per supported
   precision with sustained peak.
5. **Thermal operating points** — usually one per TDP band; consumer
   parts collapse to a single "continuous" profile at PL1.
6. **Memory hierarchy**:
   - `l1_cache_per_unit` — per-core L1 data cache (32 KB typical).
   - `l2_cache_total` — **the LLC, i.e., L3**. Working-set fit here
     drives tiling decisions; L2-per-core is ignored for the top-level
     roofline but captured inside `StoredProgramEnergyModel`.
   - `main_memory`, `peak_bandwidth`.
7. **Scheduling**:
   - `warp_size` = effective SIMD width (8 for AVX-2, 16 for AVX-512).
   - `threads_per_unit` = 2 for SMT, 1 for single-threaded cores.
   - `wave_quantization = 1` (no quantization — one thread per core).
   - `min_occupancy` = 0.4 for consumer, 0.6 for large-model server
     tuning.

Energy coefficients (for the legacy simple model):

- `energy_per_flop_fp32`: 0.5–1.5 pJ depending on process.
- `energy_per_byte`: 15–30 pJ (DRAM-heavy CPUs have high byte energy).

Factories then attach the architectural energy model:

```python
model.architecture_energy_model = StoredProgramEnergyModel(
    tech_profile=DATACENTER_7NM_DDR5   # or EDGE_8NM_LPDDR5 for Jetson
)
```

---

## 6. Workload-Class Tuning (i7-12700K twin mappers)

The consumer Alder Lake part ships **two** factory functions with
identical hardware but different calibration targets:

| Parameter | `create_i7_12700k_mapper` (tiny) | `create_i7_12700k_large_mapper` |
|-----------|----------------------------------|----------------------------------|
| Target models | MLPs batch 1–32, batch=1 inference | Transformers batch ≥ 64, ResNet-50+ |
| `efficiency_factor` (FP32) | 0.20 | 0.60 |
| `memory_bottleneck_factor` | 0.25 | 0.65 |
| `instruction_efficiency` | 0.65 | 0.80 |
| `tile_utilization` | 0.50 | 0.80 |
| `min_occupancy` | 0.40 | 0.60 |
| Expected MAPE on tiny MLPs | 10–20 % | 50–80 % (too optimistic) |
| Expected MAPE on large transformers | 25–40 % (too pessimistic) | 10–20 % |

Rationale: small workloads on consumer CPUs are dominated by
instruction-fetch and register-file overhead that doesn't amortize;
large workloads amortize that overhead and hit near-ideal SIMD
utilization. One scalar efficiency number can't fit both regimes, so
the calibration is split.

The same approach applies in principle to server parts (different
tuning for BLAS-shaped vs. irregular workloads) but hasn't been
bifurcated yet.

---

## 7. Known Limitations (improvement backlog)

1. **SIMD efficiency is a constant 0.70** — real SIMD efficiency
   varies with stride alignment, loop trip count, and reduction
   patterns. A per-op-shape model would help.
2. **AMX speedup is hardcoded 3×** — actual AMX gain depends on tile
   shape (M/N/K divisibility by 16) and precision. Should be shape-
   aware.
3. **VNNI speedup is hardcoded 2×** — real speedup is 2–4× depending
   on accumulator reuse; shape-awareness missing.
4. **Threading overhead is linear in cores** — real OpenMP overhead
   grows roughly with log(cores) for reductions and with `cores` for
   barriers. A two-term model would be more accurate.
5. **Hybrid-core modeling is a scalar reduction** — the E-core
   efficiency factor (0.6) is a single constant. Real workloads show
   larger P/E gaps on SIMD-heavy code (often 0.4) and smaller gaps on
   integer control code (0.8+).
6. **No NUMA modeling** — multi-socket EPYC / dual-Xeon systems pay
   cross-socket memory penalties that are not modeled.
7. **No AVX-512 frequency throttling** — heavy AVX-512 workloads on
   older Intel parts reduce all-core clocks by ~10–20%. Not captured.
8. **No SMT gain** — threads per core is > 1 but we only allocate one
   thread per core. SMT can help latency-hiding on memory-bound loops;
   not modeled.
9. **Cache-hit rates are fixed constants** in
   `StoredProgramEnergyModel` (85% / 90% / 95%). In reality these vary
   drastically with working-set size and access pattern.
10. **Calibration lookup for CPUs is less mature** than GPU
    calibration. Only the consumer i7 ships with a tuned default set;
    server parts carry `⚠ ESTIMATED` warnings.

---

## 8. Validation Hooks

- `validation/hardware/test_all_hardware.py` — includes server CPUs
  in the cross-platform consistency check.
- `cli/analyze_comprehensive.py --hardware intel_core_i7_12700k` —
  end-to-end sanity run.
- `calibration_data/i7_12700K/` and
  `calibration_data/intel_core_i7_12700k/` hold the tiny-MLP
  calibration sweeps that the `create_i7_12700k_mapper` tuning is
  based on.

Any mapper change should re-run `validation/` plus the CLI comparison
to check that the CPU does not unexpectedly beat the KPU or GPU on
workloads where it physically can't.
