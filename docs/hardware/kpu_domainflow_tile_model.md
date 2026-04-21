# KPU Domain-Flow-Tile Model

**Status:** M0.5 draft (2026-04-21)
**Scope:** KPU T64, T128, T256 under the refined abstraction
**Companion:** `src/graphs/hardware/resource_model.py` (`TileSpecialization`,
`TileScheduleClass`); `src/graphs/reporting/compare_archetypes.py`.

## What the KPU is

The Stillwater KPU is a **distributed domain-flow machine** capable of
direct execution of systems of affine recurrence equations (SARE). The
`ArchitectureClass` enum already records this as `DOMAIN_FLOW`; the
matching energy model is `DomainFlowEnergyModel`. The KPU is not a
generic "dataflow" accelerator (token-based, data-driven dispatch) nor
a weight-stationary systolic array. Its distinctive characteristic is
that programs are expressed as affine-recurrence domains, statically
scheduled onto the fabric, and executed without per-operation
instruction fetch or coherence machinery.

This document describes the refined tile abstraction and the product
narrative it supports.

## Why this model exists

The pre-M0.5 KPU tile model was parameterized to reach peak-ops/s parity
with NVIDIA Tensor Cores. That framing is convenient for apples-to-
apples throughput comparisons but hides what the KPU actually is and
what it actually does well.

The KPU's competitive positioning is **energy per op** in the pipelined
steady state, not peak throughput. Compensating for the peak-throughput
gap is a design knob (larger PE arrays such as 32x32); the energy-per-
op advantage is structural.

## The abstraction in one picture

```text
   Wavefront N-1           Wavefront N              Wavefront N+1
   |---drain--|             |---steady---|            |---fill---|
              +---steady---+             +---drain---+

                 <-- one tile's pipeline -->
```

On an output-stationary KPU fabric, the **drain** of tile N-1 overlaps
with the **fill** of tile N+1 across the fabric. The workload pays
fill + drain **once** across N tiles. As N grows, effective utilization
saturates toward 1.0.

On a weight-stationary systolic array (TPU), weights must fully drain
and reload between tile boundaries. Fill + drain is paid **per tile**.
Effective utilization is a flat floor set by
`steady / (steady + fill + drain)`, independent of tile count.

This scheduling-class difference is the single most important lever in
the model, and the due-diligence chart (chart 4 of the comparison
harness) is the one picture that makes it visible.

## Data-model fields

`TileSpecialization` (in `resource_model.py`) carries these domain-
flow-aware fields in addition to the legacy `array_dimensions`,
`ops_per_tile_per_clock`, and `optimization_level`:

| Field | Purpose |
|-------|---------|
| `schedule_class` | `OUTPUT_STATIONARY` (KPU), `WEIGHT_STATIONARY` (TPU), `ROW_STATIONARY`, `SIMT_DATA_PARALLEL` (GPU Tensor Core), or `UNSPECIFIED`. Governs how utilization composes with workload tile count. |
| `pipeline_fill_cycles` | Cycles for a wavefront to propagate through the PE array; one-time per pipeline start under output-stationary scheduling. |
| `pipeline_drain_cycles` | Cycles to drain the final wavefront. |
| `pe_mac_energy_pj_steady_state` | Optional per-PE MAC energy in the steady-state regime. If None, derived from `CIRCUIT_TYPE_MULTIPLIER` and the precision-specific `energy_scaling`. |

The method
`TileSpecialization.effective_pipeline_utilization(num_tiles, steady_cycles_per_tile=128)`
returns the dimensionless pipeline utilization factor for a workload
with `num_tiles` sequential compute tiles and `steady_cycles_per_tile`
steady-state wavefront duration (representative GEMM K-dimension).

## SKU lineup (M0.5 exploration)

Tile sizes inversely scaled with tile count - smaller engines benefit
from larger per-tile PE arrays (amortize fill/drain over more
wavefront cycles); larger engines move toward smaller tiles to
preserve per-tile utilization across many concurrent tiles.

| SKU | Tiles | PE array | PE count / tile | Total PEs | Clock | Default TDP | Peak INT8 |
|-----|-------|----------|-----------------|-----------|-------|-------------|-----------|
| T64  | 64  | 24x24 | 576 | 36,864  | 900 MHz | 6 W  | ~66 TOPS |
| T128 | 128 | 24x24 | 576 | 73,728  | 1.0 GHz | 12 W | ~148 TOPS |
| T256 | 256 | 20x20 | 400 | 102,400 | 1.4 GHz | 30 W | ~230 TOPS |

**Calibration (MAC energy at 16 nm):** all SKUs use
`mac_energy_int8 = 0.10 pJ/MAC` (0.16 pJ BF16, 0.30 pJ FP32). For
reference, a 16 nm 1-bit CMOS full adder dissipates ~0.01 pJ at
nominal Vdd; an INT8 MAC requires ~8 full-adder equivalents plus
array/register overhead, so 0.08-0.15 pJ is the aggressive-but-
defensible range for optimized domain-flow silicon. First-principles
TDP feasibility can be checked with
`cli/check_tdp_feasibility.py`.

**SKU parametrization rationale:**
- T64 (24x24): smaller than T128's tiles would be physically
  attractive, but at 6 W TDP with realistic MAC energies, 32x32
  arrays across 64 tiles exceed the power envelope. 24x24 x 64
  tiles is the largest array that fits 6 W.
- T128 (24x24): sweet-spot efficiency at 12 W (12.3 TOPS/W).
- T256 (20x20): commercially must deliver more peak than T128
  despite more tiles. Dropping to 16x16 (the ideal inverse-scaling
  choice) yielded fewer total PEs than T128 and an economically
  indefensible SKU. 20x20 preserves the "smaller tiles for larger
  engines" trend while giving T256 a ~1.56x peak advantage.

These numbers use a homogeneous per-SKU PE-array size; future work will
introduce heterogeneous tile sizes on a single SoC (mix of 8x8 through
64x64 on one die). The `TileSpecialization` abstraction already supports
heterogeneity; only the SKU configuration files are homogeneous today.

## Why KPU utilization on GEMM is ~1.0

Output-stationary scheduling on a domain-flow fabric pipelines adjacent
tiles through the physical mesh with no state-change between them:

1. Tile N-1 is draining into its accumulators.
2. Tile N is in its steady phase, new rows entering, outputs accumulating.
3. Tile N+1 is filling the upstream row buffers.

All three stages run concurrently on different parts of the fabric.
The only times the pipeline is not full are the very first `fill`
cycles (at the start of the workload) and the very last `drain`
cycles (at the end). Over N tiles, the penalty is (fill + drain)
cycles amortized across N * steady useful cycles:

```
util = (N * steady) / (N * steady + fill + drain)
```

At N >= 12 and representative K = 128, util > 0.97. At N >= 32, util > 0.99.

A weight-stationary systolic array cannot compose this way. In the
naive model, each tile pays its own fill/drain. In practice (TPU v1
onward), weight double-buffering amortizes the literal fill/drain,
but the utilization floor is then dominated by **shape/tile mismatch**
against the fixed PE dimensions and by **bandwidth-bound layers**.
Published data shows this floor is 10-25% of peak on TPU v1
production workloads (Jouppi et al., ISCA 2017, Table 3) and 6-25%
on Coral Edge TPU for typical DNN inference (DeepEdgeBench 2021).
The model represents this flat floor with the formula
`steady / (steady + fill + drain)` and calibrated-effective fill/drain
values, not literal wavefront cycles.

A SIMT GPU (Tensor Core or CUDA core) is not a spatial fabric
pipeline at all - CUDA cores execute the instruction stream cycle-
by-cycle. Utilization is capped by warp divergence, warp occupancy,
and memory coherence traffic, and does not amortize with workload
tile count. This is the `SIMT_DATA_PARALLEL` class. The naive-CUDA
"one thread per output" kernel has output-stationary *software data
locality* at the thread-register level, but it runs on SIMT hardware
and does NOT enjoy the fabric-level fill/drain amortization that
defines the `OUTPUT_STATIONARY` class.

## How this maps to the product story

1. **Energy per op.** KPU per-PE steady-state MAC energy is below Tensor
   Core at matched precision because the domain-flow fabric has no
   instruction fetch, no coherence machinery, no scheduling overhead
   per operation. The fabric is pre-scheduled for the affine recurrence
   domain; PEs only compute.

2. **Peak ops/s.** Tensor Core wins on peak at matched silicon area
   due to its density. We compensate with larger PE arrays (up to
   32x32 on T64). The Tensor Core-matched peak is a design choice,
   not a structural advantage.

3. **Energy efficiency (ops/W).** The product of (1) and the pipeline
   utilization story. KPU's per-op energy advantage multiplied by
   near-unit sustained utilization yields the ops/W lead shown in
   chart 3 of the comparison harness.

4. **Utilization sensitivity.** Chart 4 makes the scheduling-class
   story visible. It is the single chart that shows why KPU scales
   differently than TPU on large workloads.

5. **Array-size scaling.** Chart 5 shows the headroom (larger arrays
   improve ops/W until fill/drain becomes negligible) and diminishing
   returns (beyond ~32x32 the advantage plateaus at representative K).

## Relationship to other Stillwater research products

- **KPU** (this document): distributed domain-flow machine for SARE
  execution. `ArchitectureClass.DOMAIN_FLOW`.
- **DFM** (Data Flow Machine): token-based data-flow research machine;
  a distinct architecture with different execution semantics. See
  `src/graphs/hardware/mappers/research/dfm.py`.

The two share the general dataflow design-space lineage but are not
the same machine. KPU positioning uses "domain flow" to make the
distinction explicit.

## Next steps

- M1-M7 will populate per-SKU Layer 1-7 content on top of this tile
  abstraction. The domain-flow-tile fields feed Layer 2 (register /
  SIMD / wavefront) and Layer 6 (SoC data movement / fabric)
  especially.
- Future: heterogeneous tile sizes within a single SoC.
- Future (post-M8): silicon measurement will graduate
  `pe_mac_energy_pj_steady_state` from `THEORETICAL` to `CALIBRATED` per
  the bottom-up microbenchmark plan.
