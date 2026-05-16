# NPU ComputeProduct schema extension -- paper exercise

Status: draft
Date: 2026-05-16
Tracking issue: graphs#187
Predecessors:
  - GPU sprint #171 (closed), design doc
    `docs/designs/gpu-compute-product-schema-extension.md`
  - CPU sprint #182 (closed), design doc
    `docs/designs/cpu-compute-product-schema-extension.md`

## Purpose

Issue #187 ("Sprint: extend ComputeProduct schema + catalog to NPUs")
proposes a 5-PR sprint to add `NPUBlock` to the `ComputeProduct`
discriminated union, mirroring sprint #171 (GPU) and #182 (CPU). This
document is **PR 1 of that sprint** -- the paper exercise: enumerate
Hailo-8's full feature set against the v3 (KPU + GPU + CPU)
ComputeProduct schema and classify every field as **covered**,
**extends an existing field**, or **needs a new field/type**.

The exercise uses **Hailo-8** as the reference SKU because:
- It's a **discrete edge NPU** -- no integration-with-SoC complications
- Its mapper is ~360 LOC, the smallest of the 3 NPU mappers
- It exposes a **single thermal profile** (no DVFS) -- minimal thermal-
  profile schema work
- It's **inference-only INT8/INT4** -- minimal precision-profile work
- It has **no external DRAM** -- simplest memory subsystem in the catalog
- Hailo publishes more area/architecture data than NVIDIA does for
  Tegra-class GPUs

Hailo-10H (transformer-capable NPU with KV cache + LPDDR4X) and Coral
Edge TPU (GF 28nm node not yet in the catalog) follow as YAML-only
additions once the schema is in place.

The doc deliberately copies the structure of the GPU + CPU paper
exercises so the diff between the three designs is easy to spot.

## Method

For each feature in the Hailo-8 resource model
(`src/graphs/hardware/models/edge/hailo8.py`, 360 LOC), this doc
records:

- The field/object as it lives in graphs today (with line ref).
- Classification: COVERED / EXTEND / NEW.
- For EXTEND: the existing schema field that grows.
- For NEW: a proposed field name + brief rationale.

Out of scope for this exercise (deferred per the sprint plan):

- Hailo-10H specifics (KV cache, LPDDR4X, transformer ops) -- separate
  follow-up YAML once schema lands.
- Coral Edge TPU specifics (GF 28nm process node, USB form factor) --
  separate follow-up YAML; needs `data/process-nodes/globalfoundries/28nm.yaml`
  as a precursor data PR.
- Integrated NPUs (Intel NPU, Qualcomm Hexagon, Apple ANE) -- v4 scope;
  needs cross-block memory link concepts.
- Datacenter AI accelerators (Groq, Cerebras, Graphcore, Tenstorrent) --
  v4+ scope; much larger surface area.
- Per-field provenance (`EstimationConfidence`) -- graphs-side feature;
  orthogonal to schema design (same call GPU + CPU sprints made).

## Reference: current v3 schema state

`embodied-schemas/src/embodied_schemas/compute_product.py`:

```python
class BlockKind(str, Enum):
    KPU = "kpu"
    GPU = "gpu"
    CPU = "cpu"

AnyBlock = Annotated[Union[KPUBlock, GPUBlock, CPUBlock], Field(discriminator="kind")]
```

Three architectures shipped:
- v1 KPU: `KPUTileSpec`, `KPUNoCSpec`, `KPUMemorySubsystem`, `KPUSiliconBin`, etc.
- v2 GPU: `GPUComputeFabric`, `GPUMemorySubsystem`, `GPUOnDieFabric`, `GPUThermalProfile`, `GPUTheoreticalPerformance`, `ClockDomain`, + 4 enums
- v3 CPU: `CoreClusterSpec`, `CPUComputeFabric`, `CPUMemorySubsystem`, `CPUOnDieFabric`, `CPUThermalProfile`, `CPUTheoreticalPerformance`, + 4 enums. First cross-block-kind type sharing (CPUThermalProfile reuses `ClockDomain` from gpu_block).

The v3 design rule established by GPU + CPU sprints: **ship per-
architecture types in this sprint, defer rename + unify to v5+** (NPU is
the 4th; rename should be the 5th sprint when 5 architectures exist).

## Feature-by-feature audit

### 1. Identity / packaging

| Field (graphs)               | Verdict  | Mapping                             |
|------------------------------|----------|-------------------------------------|
| `name="Hailo-8"`             | COVERED  | `ComputeProduct.name`               |
| `hardware_type=HardwareType.KPU` (sic!) | EXTEND | `Die.blocks[0].kind = BlockKind.NPU` (new enum value). Note: the existing model misuses `HardwareType.KPU` because there was no NPU type; v4 schema cleanup may add `HardwareType.NPU` to the graphs side or just standardize on the `BlockKind` discriminator. |
| (no vendor today)            | COVERED  | `ComputeProduct.vendor = "hailo"`   |
| (no product family today)    | COVERED  | `Market.product_family = "Hailo-8"` |
| Discrete M.2 module          | EXTEND   | `Packaging(kind=MONOLITHIC, package_type="m_dot_2")`. The existing `package_type` is free-form so this is just a convention. |

### 2. Process node

| Field (graphs)                          | Verdict | Mapping                |
|-----------------------------------------|---------|------------------------|
| `process_node_nm=16` (TSMC N16)         | COVERED | `Die.process_node_id = "tsmc_n16"`. The catalog already has `data/process-nodes/tsmc/n16.yaml` from the KPU sprint. **No new node YAML needed for Hailo-8.** Hailo-10H is also N16; Coral Edge TPU is GF 28nm and would need a new node YAML in a follow-up PR. |

### 3. Dataflow unit hierarchy (NPU-specific replacement for SM / core / tile)

NPUs don't have SMs (GPU), cores (CPU), or tiles (KPU). Hailo calls
them "dataflow units" (32 of them on Hailo-8). They have no analog in
GPU/CPU/KPU and need new fields on `NPUBlock`.

| Field (graphs)                       | Verdict | Mapping                          |
|--------------------------------------|---------|----------------------------------|
| `compute_units=32` (dataflow units)  | NEW     | `NPUBlock.num_dataflow_units: int` |
| `threads_per_unit=128` (dataflow "threads") | NEW | `NPUBlock.lanes_per_unit: int`. The name "threads" is misleading on a pure dataflow architecture; lanes is more honest. |
| `warps_per_unit=1`, `warp_size=1`    | EXTEND  | Both legacy compat fields; loader can just default them. |
| (no cluster concept on Hailo-8)      | -       | Cluster only applies to hybrid architectures (CPU). NPUs are single-cluster by construction. |

### 4. Compute fabric (single dataflow fabric)

Hailo-8 has ONE compute fabric (the dataflow architecture itself).
This is simpler than GPU (CUDA + Tensor cores) or CPU (multi-fabric
per cluster). The schema needs `NPUComputeFabric`.

| Field (graphs)                                            | Verdict | Mapping |
|-----------------------------------------------------------|---------|---------|
| `fabric_type="dataflow_architecture"`                     | NEW     | `NPUComputeFabric.dataflow_kind: NPUDataflowKind` enum (STRUCTURE_DRIVEN / SYSTOLIC / SPATIAL / WEIGHT_STATIONARY / OUTPUT_STATIONARY / INPUT_STATIONARY) |
| `circuit_type="standard_cell"`                            | EXTEND  | Reuse `CircuitClass` from `process_node` module (same as GPU/CPU) |
| `num_units=32` (= num_dataflow_units)                     | -       | Lives on `NPUBlock.num_dataflow_units`, not on the fabric |
| `ops_per_unit_per_clock: {INT8: 500, INT4: 1000}`         | NEW     | `NPUComputeFabric.ops_per_unit_per_clock: dict[str, int]` (same shape as GPU/CPU but typically only INT4/INT8 entries) |
| `core_frequency_hz=1.6e9`                                 | COVERED | Lives in the chip-level thermal profile (single profile on Hailo-8) |
| `energy_per_flop_fp32`                                    | NEW     | `NPUComputeFabric.energy_per_op_int8_pj: float` -- NPUs are INT8-dominant; FP32 baseline doesn't apply. Rename to per-INT8 energy as the primary metric. |
| `energy_scaling: {INT8: 0.125, INT4: 0.0625}`             | NEW     | `NPUComputeFabric.energy_scaling: dict[str, float]` -- scaling relative to the INT8 baseline this time. |

### 5. Memory subsystem (NPU-specific: SRAM-dominant or SRAM-only)

This is where NPUs depart the most from GPU/CPU/KPU. Hailo-8 has **no
external DRAM** -- all working memory is on-chip SRAM. Coral is similar.
Hailo-10H has 4-8 GB LPDDR4X but only for transformer model weights +
KV cache.

| Field (graphs)                          | Verdict | Mapping |
|-----------------------------------------|---------|---------|
| `peak_bandwidth=200e9` (~200 GB/s on-chip) | EXTEND | `NPUMemorySubsystem.on_chip_bandwidth_gbps: float` -- different from GPU/CPU which carry DRAM bandwidth here |
| `main_memory=0` (no external DRAM!)    | NEW     | `NPUMemorySubsystem.has_external_dram: bool = False` |
| `l1_cache_per_unit=512 KB` (per dataflow unit, on-chip SRAM) | NEW | `NPUMemorySubsystem.sram_kib_per_unit: int` -- private per-unit SRAM partition (no cache in the GPU/CPU sense; software-managed scratchpad) |
| `l2_cache_total=8 MB` (shared on-chip SRAM) | NEW | `NPUMemorySubsystem.shared_sram_kib: int` -- inter-unit shared SRAM (the "LLC" of NPU-land) |
| `l2_cache_per_unit=256 KB` (= 8 MB / 32) | -    | Derived; lives only in graphs-side legacy fields |
| `l2_topology="shared-llc"`              | COVERED | `NPUMemorySubsystem.shared_sram_layout: Literal["shared", "partitioned"]` -- shared is the dominant case |
| `l3_present=False`                      | -       | NPUs have no L3 by construction; the SRAM hierarchy stops at the shared on-chip level |
| `coherence_protocol="none"`             | EXTEND  | Same shape as GPU/CPU; NPUs default "none" since compiler-routed dataflow has no coherence |
| `memory_technology="LPDDR4 (host) + on-chip SRAM"` | EXTEND | `NPUMemorySubsystem.external_dram_type: Optional[MemoryType]` -- None for Hailo-8, LPDDR4X for Hailo-10H |
| `memory_read_energy_per_byte_pj=22.0`   | NEW     | `NPUMemorySubsystem.sram_access_energy_pj_per_byte: float` -- on-chip SRAM access; ~2 pJ/byte on Hailo-8 (the model's 22 pJ is for host LPDDR4 cold-start path) |
| `energy_per_byte=2e-12` (on-chip SRAM)  | EXTEND  | Captured by sram_access_energy_pj_per_byte above |

**Recommendation**: ``NPUMemorySubsystem`` is small (~6 fields) and
NPU-specific enough to justify its own type. Don't try to generalize
across GPU/CPU/NPU memory subsystems in v3.

### 6. On-die fabric (dataflow mesh, low-confidence)

Hailo doesn't publish NoC details. The existing model assumes an 8x4
mesh of dataflow units with `low_confidence=True` flagged.

| Field (graphs)                          | Verdict | Mapping |
|-----------------------------------------|---------|---------|
| `topology=Topology.MESH_2D`             | EXTEND  | `NPUOnDieFabric.topology: NPUNoCTopology` enum (MESH_2D / DATAFLOW_RING / SYSTOLIC / CROSSBAR) |
| `bisection_bandwidth_gbps=64.0`         | EXTEND  | Same |
| `controller_count=32`                   | EXTEND  | Renamed `unit_count` for NPU clarity |
| `flit_size_bytes=16`                    | EXTEND  | Same |
| `mesh_dimensions=(8, 4)`                | NEW     | `NPUOnDieFabric.mesh_rows / .mesh_cols` (mirrors `KPUNoCSpec`) |
| `hop_latency_ns=1.5`                    | EXTEND  | Same |
| `pj_per_flit_per_hop=1.5`               | EXTEND  | Same |
| `routing_distance_factor=1.1`           | EXTEND  | Same |
| `low_confidence=True`                   | NEW     | `NPUOnDieFabric.confidence: DataConfidence` -- reuse process_node's DataConfidence enum (THEORETICAL / INTERPOLATED / CALIBRATED) |

**Recommendation**: define `NPUOnDieFabric` separately from
`GPUOnDieFabric` / `CPUOnDieFabric` despite the field overlap -- the
topology enum differs (NPUs need DATAFLOW_RING and SYSTOLIC; GPU/CPU
don't). Field names align with GPU/CPU so v5 unification is mechanical.

### 7. Thermal profile (single operating point on Hailo-8 -- much simpler than GPU/CPU)

| Field (graphs)                          | Verdict | Mapping |
|-----------------------------------------|---------|---------|
| `name="2.5W-passive"`                   | COVERED | `NPUThermalProfile.name` |
| `tdp_watts=2.5`                         | COVERED | `NPUThermalProfile.tdp_watts` |
| `cooling_solution="passive-heatsink-small"` | EXTEND | `NPUThermalProfile.cooling_solution_id` -- references the canonical `data/cooling-solutions/` catalog. `passive_heatsink_small.yaml` exists. |
| `dvfs_enabled=False`                    | NEW     | `NPUThermalProfile.dvfs_enabled: bool = False` -- NPU default is False (one profile, no DVFS); GPU default is True |
| `clock_mhz=1600` (single value)         | EXTEND  | Reuse `ClockDomain` from gpu_block (single-value: base = boost = sustained) OR use scalar `clock_mhz` per `KPUThermalProfile` shape. Recommend scalar for NPUs with no DVFS; ClockDomain when an NPU does have DVFS. |
| per-precision efficiency (0.85 INT8, 0.80 INT4) | EXTEND | `NPUThermalProfile.efficiency_factor_by_precision: dict[str, float]` -- same shape as KPU/GPU/CPU |
| `instruction_efficiency=0.95` (dataflow scheduler quality) | NEW | `NPUThermalProfile.instruction_efficiency_by_precision: dict[str, float]` -- same as GPU |
| `memory_bottleneck_factor=0.90`         | NEW     | `NPUThermalProfile.memory_bottleneck_factor_by_precision: dict[str, float]` -- same as GPU |

### 8. NPU-specific scheduler attributes

| Field (graphs)               | Verdict | Mapping                          |
|------------------------------|---------|----------------------------------|
| `min_occupancy=0.8`          | NEW     | `NPUBlock.min_occupancy: float` -- higher default than GPU (0.3) / CPU (0.4) because dataflow compiler statically allocates |
| `max_concurrent_kernels=1`   | NEW     | `NPUBlock.max_concurrent_models: int` -- renamed: NPUs run a single compiled model, not arbitrary kernels |
| `wave_quantization=1`        | NEW     | `NPUBlock.wave_quantization: int = 1` -- like CPU, default 1 keeps shape consistent |
| (no SIMD)                    | -       | NPUs don't have SIMD efficiency; `simd_efficiency_by_op_kind` doesn't apply |

### 9. Performance roll-up

Hailo-8 ships only INT8 (26 TOPS) and INT4 (52 TOPS). No FP at all.
The schema's per-precision dict generalizes cleanly.

| Field (graphs)                                  | Verdict | Mapping |
|-------------------------------------------------|---------|---------|
| `peak_ops_per_sec` (per Precision)              | EXTEND  | `NPUTheoreticalPerformance.peak_ops_per_sec_by_precision: dict[str, float]` -- same shape as `GPUTheoreticalPerformance` and `CPUTheoreticalPerformance`. The only difference is the empty-FP set on most NPUs. |

### 10. Silicon bin (per-block transistor decomposition)

Hailo publishes more area data than NVIDIA does for Tegra GPUs, so
the silicon_bin is easier than the AGX Orin case. Per the design doc's
existing convention (silicon_bin is general-purpose, only `count_ref`
strings are architecture-flavored), Hailo-8 needs new `count_ref`
strings:

- `dataflow_unit` -> per-unit Mtx
- `sram_total_kib` -> per-KiB on-chip SRAM
- `noc_routers` -> per-router mesh fabric logic
- `host_phy` -> PCIe / M.2 host interface PHY
- `control_logic` -> dataflow scheduler + compiler-emitted instruction memory

**Verdict**: NO schema change needed -- just document the convention.
Same approach GPU + CPU sprints took.

### 11. BOM cost profile

`graphs` has the BoM data already populated for Hailo-8 ($25 die + $8
package + $4 PCB + $1 thermal + $2 other = $40 BOM, retail $160). Same
story as GPU/CPU: not in v3 schema; factory overlay until v4
`Market.bom`. NEW (optional). Defer.

## Summary table

| Category                       | Total fields | COVERED | EXTEND | NEW |
|--------------------------------|-------------:|--------:|-------:|----:|
| Identity / packaging           |       5      |    3    |   2    |  0  |
| Process node                   |       1      |    1    |   0    |  0  |
| Dataflow unit hierarchy        |       4      |    0    |   1    |  3  |
| Compute fabric                 |       7      |    1    |   2    |  4  |
| Memory subsystem               |      11      |    1    |   5    |  5  |
| On-die fabric                  |       9      |    0    |   6    |  3  |
| Thermal profile                |       8      |    2    |   3    |  3  |
| Scheduler attributes           |       4      |    0    |   1    |  3  |
| Performance roll-up            |       1      |    0    |   1    |  0  |
| Silicon bin                    |       0      |    0    |   0    |  0 (new count_ref strings only) |
| BoM cost (deferred)            |       1      |    0    |   0    |  1 (optional) |
| **Totals**                     |    **51**    | **8**   | **21** |**22**|

Of 51 fields, ~16% (8) are already covered, ~41% (21) extend an
existing schema field (highest of all 4 sprints -- NPU inherits a lot
from CPU + GPU), and ~43% (22) are net-new. The high EXTEND ratio is
the payoff from the GPU and CPU sprints' generalizable shapes.

## Recommended schema diff for the next PR

The Schema PR (PR 2 of the sprint) should land roughly the following
in embodied-schemas:

1. **`compute_product.py`**:
   - Add `BlockKind.NPU = "npu"` to the discriminator enum.
   - Update `AnyBlock = Union[KPUBlock, GPUBlock, CPUBlock, NPUBlock]`.

2. **New file `embodied_schemas/npu_block.py`** (~350 LOC, similar size
   to gpu_block / cpu_block):
   - `NPUBlock` with: `num_dataflow_units`, `lanes_per_unit`,
     `compute_fabrics: list[NPUComputeFabric]`, `memory: NPUMemorySubsystem`,
     `noc: NPUOnDieFabric`, `min_occupancy`, `max_concurrent_models`,
     `wave_quantization`, `multi_precision_alu`.
   - `NPUComputeFabric` with: `dataflow_kind: NPUDataflowKind`,
     `circuit_class`, `ops_per_unit_per_clock`, `energy_per_op_int8_pj`,
     `energy_scaling`.
   - `NPUMemorySubsystem` with: `on_chip_bandwidth_gbps`, `sram_kib_per_unit`,
     `shared_sram_kib`, `shared_sram_layout: Literal["shared","partitioned"]`,
     `has_external_dram: bool`, `external_dram_type: Optional[MemoryType]`,
     `external_dram_size_gb: Optional[float]`, `external_dram_bandwidth_gbps: Optional[float]`,
     `sram_access_energy_pj_per_byte`, `coherence_protocol`.
   - `NPUOnDieFabric` with: `topology: NPUNoCTopology`, `bisection_bandwidth_gbps`,
     `unit_count`, `flit_size_bytes`, `mesh_rows` (optional),
     `mesh_cols` (optional), `hop_latency_ns`, `pj_per_flit_per_hop`,
     `routing_distance_factor`, `confidence: DataConfidence`.
   - `NPUThermalProfile` with: `name`, `tdp_watts`, `cooling_solution_id`,
     `clock_mhz`, `dvfs_enabled`, plus the per-precision dicts
     (`efficiency_factor_by_precision`, `instruction_efficiency_by_precision`,
     `memory_bottleneck_factor_by_precision`).
   - `NPUTheoreticalPerformance` with `peak_ops_per_sec_by_precision`.
   - Enums: `NPUDataflowKind` (STRUCTURE_DRIVEN / SYSTOLIC / SPATIAL /
     WEIGHT_STATIONARY / OUTPUT_STATIONARY / INPUT_STATIONARY),
     `NPUNoCTopology` (MESH_2D / DATAFLOW_RING / SYSTOLIC / CROSSBAR).

3. **Process node**: no new YAML for Hailo-8 (TSMC N16 already in catalog
   from KPU sprint). Coral Edge TPU follow-up will need `data/process-nodes/globalfoundries/28nm.yaml`.

4. **Cooling solution**: `passive_heatsink_small.yaml` already in catalog.

## Risks called out by this exercise

1. **`HardwareType.NPU` doesn't exist on the graphs side.** The existing
   Hailo-8 model uses `HardwareType.KPU` because dataflow architectures
   were lumped with KPU when the enum was first defined. The schema can
   ignore this (the `BlockKind.NPU` discriminator is the source of truth
   in ComputeProduct), but PR 4's loader needs to decide what to set on
   the legacy `HardwareResourceModel.hardware_type` field. Recommend:
   add `HardwareType.NPU` to the graphs enum in PR 4 (graphs-side
   change, no schema impact) and set it on the loaded model.

2. **`energy_per_op` instead of `energy_per_flop_fp32`.** NPUs don't ship
   FP32; the "energy per FP32 FMA" baseline is meaningless. The schema
   needs a different baseline metric for NPU energy. Recommend INT8 as
   the baseline (the dominant precision); the loader can synthesize an
   `energy_per_flop_fp32` proxy if downstream consumers need it (=
   `energy_per_op_int8 * 8` since FP32 ≈ 8× INT8 energy as a rule of
   thumb).

3. **No-DRAM SKUs need optional DRAM fields.** Hailo-8 has zero external
   DRAM; Coral has zero; Hailo-10H has 4-8 GB. Make every DRAM field on
   `NPUMemorySubsystem` Optional with `has_external_dram` as the boolean
   gate. A model-validator can enforce consistency (if `has_external_dram`
   is True, the other fields must be populated; if False, they must be
   None). Mirrors the L3/L4 `_validate_l*_consistency` pattern from
   GPU/CPU sprints.

4. **Quantization-first precision validation.** NPUs typically don't
   ship FP. The `NPUComputeFabric.ops_per_unit_per_clock` dict should
   probably require at least one of {INT4, INT8} but tolerate missing
   FP. Validator in the model can enforce.

5. **`HardwareType.KPU` mis-use will surface in downstream tests.** Any
   test that asserts `model.hardware_type == HardwareType.NPU` would
   fail today and pass after PR 4. Worth running a grep before PR 4
   ships.

6. **Single-thermal-profile is a real shape, not a limitation.** Don't
   try to make `NPUThermalProfile` look like `GPUThermalProfile`'s
   ClockDomain just to be consistent -- Hailo-8 genuinely has one
   clock. The schema should allow both (scalar clock_mhz OR ClockDomain),
   not force the GPU shape on the NPU.

7. **Hailo-10H follow-up will need `KVCacheSpec`.** Transformer NPUs
   (Hailo-10H, future Tenstorrent / Groq inference SKUs) have first-
   class KV cache management. Out of scope for this sprint (Hailo-8
   has no KV cache); add as a follow-up extension to `NPUBlock` when
   the Hailo-10H YAML lands.

## Next step

Land this doc as PR 1 of the sprint (graphs-side). The Schema PR
(PR 2, embodied-schemas-side) implements the "Recommended schema diff"
above. Schema PR can be reviewed and merged independently of PR 3
(data PR for Hailo-8 YAML) since the new types are additive.

Estimated total sprint cost: ~2.5 days (slightly less than GPU/CPU
sprints because NPU types are smaller and Hailo publishes more area
data than NVIDIA / Intel do for their reference SKUs).
