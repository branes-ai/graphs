# KPU Modeling

This document describes how the `graphs` repository models the **Stillwater
Knowledge Processing Unit (KPU)** — a programmable distributed domain-flow
accelerator in the `DOMAIN_FLOW` architecture class, capable of direct
execution of systems of affine recurrence equations (SARE). Read
[`micro-architecture-modeling-methodology.md`](./micro-architecture-modeling-methodology.md)
first for the shared latency/energy methodology; this document only
covers KPU-specific mechanisms.

The KPU implementation lives in:

- `src/graphs/hardware/mappers/accelerators/kpu.py` — `KPUMapper`,
  `TileConfiguration`, factory functions
- `src/graphs/hardware/models/accelerators/kpu_t64.py`,
  `kpu_t256.py`, `kpu_t768.py` — per-device resource models
- `src/graphs/hardware/architectural_energy.py` —
  `DomainFlowEnergyModel`, `KPUTileEnergyModel`, `KPUTileEnergyAdapter`

---

## 1. Architecture Summary

The KPU is a **Domain Flow Architecture (DFA)**: a programmable
systolic processor with domain tracking. It is the dynamic-control
generalization of a fixed systolic array — a Google TPU is a special
case of DFA where all dynamic control has been collapsed to one
function.

Key properties relevant to modeling:

1. **Heterogeneous tile array** — silicon is allocated across several
   tile specializations (INT8-primary, BF16-primary, Matrix). All
   precisions are supported natively on all tiles, but optimization
   levels differ.
2. **EDDO memory hierarchy** — Explicit Data Distribution and
   Orchestration. Software-managed scratchpads, no tags, no coherence
   protocol, no speculative fetches. Compiler-directed data movement.
3. **Four-stage memory hierarchy** — DRAM → L3 (distributed
   scratchpad) → L2 (tile-local) → L1 (PE-local) → Fabric.
4. **Three data-movement engines** — DMA (DRAM↔L3), BlockMover
   (L3↔L2 inter-tile), Streamer (L2↔L1 intra-tile).
5. **Token-based execution** — SURE/SARE (Systems of Uniform / Affine
   Recurrence Equations) program overlay drives computation via token
   signature matching across a distributed CAM-like fabric.
6. **Hardware operator fusion** — consecutive operators can share
   intermediate data in-fabric, eliminating L2 traffic.

The KPU therefore models latency and energy at a considerably finer
grain than the baseline roofline.

---

## 2. Device Variants

Three SKUs share the same 70/20/10 tile-specialization ratio and differ
in scale / memory technology / thermal envelope:

| Device | Tiles | Mesh | Clock (sustained) | Memory | Thermal envelope | Target |
|--------|-------|------|--------------------|--------|-------------------|--------|
| KPU-T64 | 64 (44/13/7) | 8×8 | 850–950 MHz | DDR4-equiv 25.6 GB/s | 3 W / 6 W / 10 W | Battery-powered edge, drones |
| KPU-T256 | 256 (179/51/26) | 16×16 | 1.0–1.2 GHz | LPDDR5 204.8 GB/s | 15 W / 30 W / 50 W | Edge / small datacenter |
| KPU-T768 | 768 (537/154/77) | 24×32 | 1.3–1.5 GHz | HBM2 1638.4 GB/s | 30 W / 60 W / 100 W | Datacenter inference |

Tile specializations (same for every device, scaled by count):

- **INT8-primary** (69%): 16×16 PE array, 512 INT8 / 1024 INT4 / 256
  BF16 ops per tile per clock. Target: computer vision, detection.
- **BF16-primary** (20%): 16×16 PE array, 256–512 BF16 / 128 FP32 /
  512 INT8 ops per tile per clock. Target: sensor fusion, attention,
  normalization.
- **Matrix** (11%): 8×8 tensor-core PE array, 8192 INT8 / 4096 BF16
  ops per tile per clock. Target: classification heads, embeddings,
  large matmuls. Uses `circuit_type=tensor_core` (0.85× energy).

The factory functions `create_kpu_t64_mapper`, `create_kpu_t256_mapper`,
`create_kpu_t768_mapper` each return a `KPUMapper` with its resource
model and an attached `KPUTileEnergyAdapter`.

---

## 3. Latency Model

### 3.1 Tiling Analysis (the KPU-specific step)

Unlike a GPU — which has flexible per-SM memory and relies on caches —
a KPU tile has a **hard 256 KB scratchpad budget**. Every subgraph must
fit its working set (inputs + weights + outputs at the mapped
precision) within that budget, or be tiled.

`KPUMapper._analyze_tiling(subgraph, precision) -> TileConfiguration`
performs:

```
bytes_per_element  = precision bytes (FP32=4, INT8=1, INT4=0.5, …)
input_bytes        = subgraph.total_input_bytes  × bytes_per_element / 4
weight_bytes       = subgraph.total_weight_bytes × bytes_per_element / 4
output_bytes       = subgraph.total_output_bytes × bytes_per_element / 4
total_bytes        = input + weight + output

fits_in_scratchpad = (total_bytes ≤ 256 KB)

if fits:
    num_tiles_required = 1
    num_iterations     = 1
    tiling_overhead    = 1.0
else if weights alone > 80% of scratchpad:
    # pessimistic tile weights path
    bytes_per_tile     = 80% of scratchpad
    num_tiles_required = ceil(total_bytes / bytes_per_tile)
else:
    # keep weights resident, tile input/output proportionally
    remaining          = scratchpad − weight_bytes
    input_per_tile     = remaining × input/(input+output) × 0.8
    output_per_tile    = remaining × output/(input+output) × 0.8
    num_tiles_required = ceil(input_bytes / input_per_tile)

tiles_per_iteration   = min(num_tiles_required, total_tiles_on_chip)
num_iterations        = ceil(num_tiles_required / tiles_per_iteration)
tiling_overhead       = 1.0 + 0.10 × (num_iterations − 1)
```

The 0.80 efficiency and 0.10-per-iteration overhead constants are
heuristics; they are documented limitations (see §6) and are prime
candidates for improvement via EDDO-level modeling.

### 3.2 Subgraph Mapping

`map_subgraph` (`kpu.py:280`) then:

1. **Parallelism** → tile count:
   `tiles_needed = ceil(parallelism.total_threads / threads_per_tile)`,
   clamped up by `tile_config.tiles_per_iteration` and down by the
   physical tile count.
2. **Occupancy / utilization**: `tiles_allocated / num_tiles` (both
   occupancy and utilization use the same ratio since the KPU does not
   run multiple subgraphs on one tile).
3. **Roofline latency** (via `HardwareMapper._calculate_latency`):
   - `ops_with_tiling = ops × tiling_overhead` (covers extra control
     and boundary PEs that become active in tiled execution).
   - `bytes_with_tiling = bytes × num_iterations` (covers reloading
     operands per iteration).
   - `effective_ops_per_sec` from the thermal operating point (DVFS +
     `efficiency_factor` + `tile_utilization`).
4. `estimated_latency = max(compute_time, memory_time)` — roofline
   invariant.

### 3.3 Graph Mapping

`map_graph` aggregates per-stage latency as:

- Within a stage (parallel subgraphs): `latency = max(latency_i)`,
  `tiles_used = max(tiles_i)`.
- Across stages: latencies **sum**.

The result carries `total_latency`, `latency_breakdown` by stage,
`peak_utilization`, `average_utilization`, and the `naive_latency`
(pure-peak reference) for a correction factor.

### 3.4 Heterogeneous Tile Allocation (known limitation)

Currently `map_subgraph` treats all tiles as interchangeable (one pool
of `num_tiles = 64/256/768`). The heterogeneous allocation is
**declared** in `KPUComputeResource` / `TileSpecialization` and **used
for peak TOPS**, but not yet routed per-subgraph. A precision-aware
mapper would:

1. Classify the subgraph's arithmetic type (INT8 conv, BF16
   transformer, FP32 matmul).
2. Prefer tiles whose `optimization_level[precision] = 1.0`.
3. Fall back to other tile types with their optimization_level penalty.
4. Spill to remaining tiles when the primary pool is exhausted.

See §6.

---

## 4. Energy Model

The KPU energy path has two layers:

### 4.1 Baseline (three-component) energy

`HardwareMapper._calculate_energy()` produces compute + memory from:

- `compute_energy = ops × energy_per_flop_fp32 × energy_scaling[precision]`
- `memory_energy = bytes × energy_per_byte`

and `KPUMapper.compute_energy_with_idle_power()` adds idle energy:

- `idle_power  = IDLE_POWER_FRACTION × TDP` (default 50% for nanoscale
  SoCs).
- `total_energy = dynamic_energy + idle_power × latency`.

TDP is resolved from the thermal operating point (e.g., "6W-standard"
→ 6 W), with fallbacks to `"default"`, the first available profile, or
`2 × dynamic_power`.

### 4.2 Architectural overhead — KPUTileEnergyModel (8 components)

For every KPU resource model the factory attaches a
`KPUTileEnergyAdapter` wrapping a `KPUTileEnergyModel`. When
`_calculate_energy_with_architecture()` is called, the adapter produces
an 8-component `ArchitecturalEnergyBreakdown` that captures what makes
DFA unique:

| # | Component | Origin |
|---|-----------|--------|
| 1 | **4-stage memory hierarchy** — DRAM read/write + L3 + L2 + L1 | EDDO scratchpad traversal, direct addressing |
| 2 | **Data-movement engines** — DMA (DRAM↔L3), BlockMover (L3↔L2), Streamer (L2↔L1) | Explicit DMA descriptors, double-buffered |
| 3 | **Token signature matching** | Distributed CAM-like token firing |
| 4 | **SURE program loading** | Per-operator broadcast (~50 pJ); cached (~1 pJ on reuse) |
| 5 | **Distributed L3 NoC routing** | Variable mesh-hop distance × per-hop energy |
| 6 | **Operator fusion** | Hardware fusion eliminates 70% of L2 intermediate traffic (net savings) |
| 7 | **Token routing** | Per-hop signature routing through fabric |
| 8 | **PE-to-PE streaming** | Intra-tile forwarding between PEs |

Energy coefficients (pJ-scale, process-dependent) are declared per
device. Representative T64 values:

| Coefficient | Value | Coefficient | Value |
|-------------|-------|-------------|-------|
| DRAM read | 10 pJ/byte | MAC INT8 | ~0.3 pJ |
| L3 read | 2.0 pJ/byte | MAC BF16 | ~0.45 pJ |
| L2 read | 0.8 pJ/byte | MAC FP32 | ~0.9 pJ |
| L1 read | 0.3 pJ/byte | Token sig match | 0.6 pJ |
| DMA | 1.5 pJ/byte | Token routing/hop | 0.15 pJ |
| BlockMover | 0.8 pJ/byte | SURE program load | 50 pJ (broadcast) |
| Streamer | 0.3 pJ/byte | SURE cache hit | 1 pJ |

The adapter aggregates these into:

- `data_movement_overhead` = DRAM + L3 + L2 + L1 + DMA + BlockMover +
  Streamer + L3 routing
- `control_overhead` = token matching + program load + token routing +
  PE-streaming
- `compute_overhead` = fusion coordination (often negative net, once
  the 70% L2 traffic savings from fusion are counted)
- `mac_energy` / `flop_energy` / `intop_energy` broken out so the
  Embodied-AI-Architect can route work to the right fabric.

### 4.3 Why the KPU wins energy comparisons

The comparison against baseline stored-program (CPU) is:
`compute_efficiency ≈ 0.75` and `memory_efficiency ≈ 0.70` (both in
the range 0.25–0.40× of stored-program overhead). The **absolute**
energy advantage comes from:

1. No instruction fetch per op (program loaded once, reused across
   tokens).
2. No coherence machinery (EDDO single-writer, compiler-scheduled).
3. No cache tag lookups (direct-addressed scratchpads).
4. Hardware fusion retains intermediates in-fabric.
5. Local L1 (0.3 pJ/B) dominates the memory pyramid — HBM/DDR access
   is rare because tiling keeps the working set resident.

Typical modeled pJ/MAC:

- TPU (fixed systolic): ~0.8 pJ/MAC
- **KPU (Domain Flow): ~1.1 pJ/MAC**
- GPU (SIMT): ~1.5 pJ/MAC
- CPU (stored program): ~5 pJ/MAC

---

## 5. Resource Model Specification (how a KPU SKU is declared)

A `kpu_t<N>_resource_model()` factory builds, in order:

1. **Compute fabrics** — one `ComputeFabric` per tile specialization,
   each with its `num_units`, per-precision `ops_per_unit_per_clock`,
   `circuit_type` (`standard_cell` for INT8/BF16 tiles, `tensor_core`
   for Matrix tiles), `process_node_nm` (16 nm for T64, scaled for
   T256/T768), and per-precision `energy_scaling`.
2. **Clock domains** — one `ClockDomain` per thermal profile with
   `base_clock_hz`, `max_boost_clock_hz`, `sustained_clock_hz`, and a
   `dvfs_enabled` flag.
3. **Tile specializations** — `TileSpecialization` entries that
   declare `tile_type`, `num_tiles`, `array_dimensions`,
   `pe_configuration`, `ops_per_tile_per_clock`, and
   `optimization_level` per precision.
4. **`KPUComputeResource`** — aggregates the specializations; provides
   `calc_peak_ops(precision)` (sum across every tile pool whose
   `optimization_level[precision] > 0`).
5. **Thermal operating points** — one per power mode, each with a
   `KPUComputeResource` for its DVFS-throttled clock and per-precision
   `PerformanceCharacteristics` with `efficiency_factor` and
   `tile_utilization` (e.g., 0.65 / 0.93 at 6 W, INT8).
6. **`HardwareResourceModel`** wraps everything and is returned. The
   mapper factory then attaches the `KPUTileEnergyAdapter` to
   `model.architecture_energy_model`.

Empirical efficiency factors are what make the KPU model defensible
vs. vendor datasheets:

| Profile | Precision | efficiency_factor | tile_utilization | (vs. Jetson Orin Nano's ~4%) |
|---------|-----------|-------------------|-------------------|------------------------------|
| T64 @ 3 W | INT8 | 0.60 | 0.90 | — |
| T64 @ 6 W | INT8 | 0.65 | 0.93 | — |
| T64 @ 10 W | INT8 | 0.70 | 0.95 | — |

---

## 6. Known Limitations (improvement backlog)

1. **Homogeneous tile pool in `map_subgraph`** — ignores the 70/20/10
   specialization when routing a subgraph. Expand the mapper to
   consult `KPUComputeResource.get_tiles_for_precision()` and select
   the tile pool whose `optimization_level` is highest.
2. **Tiling heuristic constants** — the 80% efficiency and 10% per-
   iteration overhead are placeholders. Replace with an EDDO-level
   model that costs DMA descriptors, double-buffer depth, and
   BlockMover traffic explicitly.
3. **Fusion not propagated from the mapper** — the `KPUTileEnergyModel`
   supports `enable_fusion=True` but `KPUTileEnergyAdapter` currently
   calls it with `enable_fusion=False`. The fusion decision should
   come from the partition report (fused subgraph boundaries).
4. **GEMM-only energy** — `KPUTileEnergyAdapter` hard-codes GEMM
   dimensions (M, N, K) from `execution_context`. For non-GEMM
   operators (activation, reduction, elementwise), the adapter falls
   back to the GEMM energy with degenerate dimensions. A per-operator
   energy path is needed.
5. **No wavefront scheduling model** — `DomainFlowEnergyModel`
   mentions dynamic wavefronts and SURE/SARE overlays but the mapper
   does not yet simulate wavefront propagation across tiles.
6. **No inter-tile interconnect contention** — the mesh NoC is modeled
   energy-wise (per-hop cost) but not latency-wise.
7. **No SURE program cache warmth tracking** — every operator
   currently pays the full 50 pJ broadcast. A cache-warmth model (4
   programs per tile) would amortize load costs for repeated
   operators.

---

## 7. Validation Hooks

- `validation/hardware/test_all_hardware.py` — cross-platform
  consistency (KPU must be within expected ranges vs. GPU/CPU).
- `tools/energy_comparison_gpu_vs_kpu.py` — direct GPU vs. KPU energy
  sweep per operator shape.
- `tools/energy_sweep_array_sizes.py` — how energy advantage scales
  with problem size.
- `cli/generate_kpu_energy_slides.py` — presentation outputs for
  energy comparison decks.

Any mapper change should be re-run against these before updating the
calibration JSON or the vendor-facing comparison decks.
