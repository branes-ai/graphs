# KPU Modeling

This document describes how the `graphs` repository models the **Stillwater
Knowledge Processing Unit (KPU)** — a programmable distributed domain-flow
accelerator in the `DOMAIN_FLOW` architecture class, capable of direct
execution of systems of affine recurrence equations (SARE). Read
[`micro-architecture-modeling-methodology.md`](./micro-architecture-modeling-methodology.md)
first for the shared latency/energy methodology; this document only
covers KPU-specific mechanisms.

The KPU implementation lives in:

- **SKU data (source of truth):** `embodied-schemas`
  `data/compute_products/stillwater/*.yaml`. There are 12 KPU
  ComputeProducts: T64 / T128 / T256 on TSMC N16, GF 12FDX and TSMC N7;
  T512 on 12FDX and N7; and T768 on N7. Each references a `ProcessNodeEntry`
  for density, leakage and energy per op.
- `src/graphs/hardware/models/accelerators/kpu_yaml_loader.py` —
  `load_kpu_resource_model_from_yaml(base_id)` builds the
  `HardwareResourceModel` from a catalog SKU.
- `src/graphs/hardware/models/accelerators/kpu_t64.py`, `kpu_t128.py`,
  `kpu_t256.py`, `kpu_t768.py` — thin wrappers over the loader. They add
  only a BOM cost profile and a few documented per-SKU overrides (see
  §4.2).
- `src/graphs/hardware/mappers/accelerators/kpu.py` — `KPUMapper`,
  `TileConfiguration`, and the `create_kpu_t{64,128,256,768}_mapper`
  factories.
- `src/graphs/hardware/kpu_access.py` — `kpu_block_of` / `kpu_die_of`,
  which locate the KPU block and its die by kind. Never assume
  `dies[0].blocks[0]`.
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

The four registered SKUs share a roughly 70/20/10 tile-specialization mix.
They differ in scale, memory technology and thermal envelope. Values are
from the embodied-schemas 0.7.0 catalog: clocks and Vdd per thermal profile
(default profile in **bold**), and peak INT8 at the default clock.

| Device (catalog id) | Node | Tiles (INT8/BF16/Matrix) | Mesh | Profiles: clock @ Vdd | Memory | Peak INT8 | Die |
|---|---|---|---|---|---|---|---|
| KPU-T64 (`kpu_t64_32x32_lp5x4_16nm_tsmc_ffp`) | TSMC N16 | 64 (44/13/7) | 8x8 | 3 W: 350 MHz @ 0.636 V; **6 W: 475 @ 0.759**; 10 W: 550 @ 0.896 | LPDDR5, 64 GB/s, 8 GB | 62.3 TOPS | 50 mm^2 |
| KPU-T128 (`kpu_t128_32x32_lp5x8_16nm_tsmc_ffp`) | TSMC N16 | 128 (89/26/13) | 16x8 | 6 W: 350 @ 0.634; **12 W: 475 @ 0.756**; 18 W: 500 @ 0.886 | LPDDR5, 96 GB/s, 16 GB | 124.5 TOPS | 96 mm^2 |
| KPU-T256 (`kpu_t256_32x32_lp5x16_16nm_tsmc_ffp`) | TSMC N16 | 256 (179/51/26) | 16x16 | 15 W: 450 @ 0.623; **30 W: 600 @ 0.753**; 50 W: 700 @ 0.887 | LPDDR5, 256 GB/s, 32 GB | 314.6 TOPS | 188 mm^2 |
| KPU-T768 (`kpu_t768_16x8_hbm3x16_7nm_tsmc_hpc`) | TSMC N7 | 768 (537/154/77) | 32x24 | 30 W: 1075 @ 0.572; **60 W: 1375 @ 0.704**; 100 W: 1550 @ 0.841 | HBM3, 1638 GB/s, 64 GB | 1353.8 TOPS | 330 mm^2 |

Profile TDPs are derived by the power model from clock x Vdd x ProcessNode
energies. Vdd is tuned so the model lands on the round catalog envelopes
(embodied-schemas#85). The catalog also carries 12FDX and N7 variants of
T64 / T128 / T256, plus T512.

Tile specializations on T64 / T128 / T256 (ops per tile per clock):

- **INT8-primary** (~70%): 32x32 PE array, balanced-logic library;
  2048 INT8 / 4096 INT4 / 1024 BF16 / 1024 FP16. Target: computer vision,
  detection.
- **BF16-primary** (~20%): 32x32 PE array, balanced-logic library;
  1024 BF16 / 1024 FP16 / 512 FP32 / 2048 INT8. Target: sensor fusion,
  attention, normalization.
- **Matrix** (~10%): 32x32 PE array on the **HP-logic** library;
  2048 INT8 / 1024 BF16 / 1024 FP16. The HP library costs more energy per
  op than balanced logic (at N16: 3.5 vs 2.7 pJ per FP32 op in the
  ProcessNode table). The loader labels its fabric `circuit_type =
  standard_cell`, like every logic library; there is no `tensor_core`
  label any more.

T768 uses smaller 16x8 arrays for the INT8- and BF16-primary tiles (512
INT8 / 256 BF16 per tile per clock). Its Matrix tiles are 8x8
**weight-stationary** arrays at 128 ops/PE/clock (8192 INT8 / 4096
BF16 / 4096 FP16 per tile).

Each `create_kpu_t{64,128,256,768}_mapper` factory returns a `KPUMapper`
with its resource model, the catalog `PhysicalSpec`, and an attached
`KPUTileEnergyAdapter`.

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
| DRAM read | 10 pJ/byte | MAC INT8 | 0.10 pJ |
| L3 read | 2.0 pJ/byte | MAC BF16 | 0.16 pJ |
| L2 read | 0.8 pJ/byte | MAC FP32 | 0.30 pJ |
| L1 read | 0.3 pJ/byte | Token sig match | 0.6 pJ |
| DMA | 1.5 pJ/byte | Token routing/hop | 0.15 pJ |
| BlockMover | 0.8 pJ/byte | SURE program load | 50 pJ (broadcast) |
| Streamer | 0.3 pJ/byte | SURE cache hit | 1 pJ |

The three MAC energies are deliberate per-SKU overrides in `kpu_t64.py`.
They are more aggressive domain-flow figures than the generic N16
balanced-logic values the loader derives from the ProcessNode (0.30 /
1.4 / 2.7 pJ); a per-SKU MAC-energy schema field would move them into
embodied-schemas. The memory, engine and token coefficients come from the
loader.

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

A KPU SKU is declared as data: a `ComputeProduct` YAML in embodied-schemas,
generated or checked by `cli/generate_kpu_sku.py` from a `KPUSKUInputSpec`.
`load_kpu_resource_model_from_yaml(base_id)` turns it into a
`HardwareResourceModel`, in order:

1. **Compute fabrics** — one `ComputeFabric` per catalog tile class,
   `fabric_type = kpu_<tile_type>` (`kpu_int8_primary`,
   `kpu_bf16_primary`, `kpu_matrix`).
   - `num_units` and per-precision `ops_per_unit_per_clock` come from the
     tile class.
   - `circuit_type` comes from the tile's `pe_circuit_class`; every logic
     library maps to `standard_cell`.
   - `process_node_nm` comes from the KPU die's ProcessNode.
   - FP32 energy per op is the ProcessNode `energy_per_op_pj` for that
     library, and `energy_scaling` holds the per-precision ratios.
   - The model-level `energy_per_flop_fp32` is the balanced-logic per-MAC
     figure halved, i.e. per FLOP (#81).
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
   the tile pool whose `optimization_level` is highest. Planned as #268
   C5 (capability-aware tile pools), part of the heterogeneous-tile
   refactor (`docs/plans/kpu-heterogeneous-tile-refactor-plan.md`).
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

- `tests/hardware/test_kpu_golden.py` + `cli/kpu_golden_snapshot.py` — the
  zero-diff gate. It pins every modeled output (generator, TDP breakdown,
  silicon, floorplans, resource model, mapper, validator findings) of the
  12 catalog SKUs. Regenerate with `--update` only for a declared model
  change.
- `tests/hardware/test_kpu_yaml_loader.py`, `test_kpu_access.py`,
  `test_kpu_registry_metadata.py` — loader contract, positional-independence
  of KPU lookup, and registry metadata vs. the built models.
- `validation/hardware/test_phase4_accelerator_energy.py` — fabric process
  nodes, KPU fabric energies and the energy-vs-node ordering, all derived
  from the catalog and ProcessNode data.

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
