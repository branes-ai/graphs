# GPU ComputeProduct schema extension -- paper exercise

Status: draft
Date: 2026-05-15
Tracking issue: graphs#171
Predecessor PRs: KPU v1 ComputeProduct migration (graphs PRs #156, #160-170;
                 embodied-schemas PRs #15-18)

## Purpose

Issue #171 ("Sprint: extend ComputeProduct schema + catalog to GPUs")
proposes a 5-PR sprint to add `GPUBlock` to the `ComputeProduct`
discriminated union. This document is **PR 1 of that sprint** -- the
paper exercise: enumerate Jetson AGX Orin's full feature set against
the v1 (KPU-only) ComputeProduct schema and classify every feature as
**covered**, **extends an existing field**, or **needs a new
field/type**.

This de-risks the rest of the sprint (schema PR, data PR, adapter PR,
loader migration, cleanup) by surfacing schema gaps and naming
collisions before any embodied-schemas code ships.

The exercise uses **Jetson AGX Orin 64GB** as the reference SKU because
its mapper is the most thoroughly validated GPU in the `graphs`
catalog (see `src/graphs/hardware/models/edge/jetson_orin_agx_64gb.py`).

## Method

For each feature in the Jetson AGX Orin resource model, this doc
records:

- The field / object as it lives in graphs today (with line ref).
- The classification: COVERED / EXTEND / NEW.
- For EXTEND: the existing schema field that grows.
- For NEW: a proposed field name + brief rationale.

A summary table at the end aggregates the verdicts.

Out of scope for this exercise (deferred to later in the sprint):

- DLA (Deep Learning Accelerator), PVA (Programmable Vision
  Accelerator), ARM CPU complex, ISP, video codecs -- each is a
  separate **future** block kind. v1 GPUBlock covers GPU compute only.
- Multi-GPU NVLink fabric, MIG partitioning, chiplet-GPUs
  (Blackwell B200) -- explicitly out-of-scope per issue #171.
- Per-field provenance (`EstimationConfidence`) -- graphs-side feature
  today; schema extension is orthogonal to this sprint.

## Reference: current v1 schema state

`embodied-schemas/src/embodied_schemas/compute_product.py` line 173:

```python
AnyBlock = Annotated[Union[KPUBlock], Field(discriminator="kind")]
```

`BlockKind` enum has one value: `KPU = "kpu"`. The schema docstring
calls out the future block kinds explicitly: `GPU`, `CPU`, `NPU`,
`DSP`, `MEMORY`, `IO`, `BRIDGE`, `ISP`, `VIDEO_CODEC`, `AUDIO_CODEC`,
`RADAR_DSP`, `LIDAR_PREPROC`.

KPU sub-types reused by `KPUBlock`:
`KPUTileSpec`, `KPUNoCSpec`, `KPUMemorySubsystem`, `KPUSiliconBin`,
`KPUClocks`, `KPUTheoreticalPerformance`, `KPUThermalProfile`.

Of these, `KPUClocks`, `KPUThermalProfile`, `KPUSiliconBin`, and
`KPUMemorySubsystem` are the most reusable across architectures;
`KPUTileSpec` and `KPUNoCSpec` are KPU-shaped.

## Feature-by-feature audit

### 1. Identity / packaging

| Field (graphs)               | Verdict  | Mapping                             |
|------------------------------|----------|-------------------------------------|
| `name="Jetson-Orin-AGX-64GB"`| COVERED  | `ComputeProduct.name`               |
| `hardware_type=HardwareType.GPU` | EXTEND | `Die.blocks[0].kind = BlockKind.GPU` (new enum value) |
| (no vendor today)            | COVERED  | `ComputeProduct.vendor = "nvidia"`  |
| (no product family today)    | COVERED  | `Market.product_family = "Jetson Orin"` |
| (monolithic)                 | COVERED  | `Packaging(kind=MONOLITHIC, num_dies=1)` |

### 2. Process node

| Field (graphs)                      | Verdict | Mapping                |
|-------------------------------------|---------|------------------------|
| `process_node_nm=8` on each fabric  | COVERED | `Die.process_node_id` references a `ProcessNodeEntry`. The catalog already has `data/process-nodes/samsung/8lpp.yaml` -- Samsung 8LPP (Low-Power Plus) is the marketing name for what NVIDIA Ampere ships on. **No new node YAML needed**; the AGX Orin SKU references `process_node_id: "samsung_8lpp"`. |
| Samsung 8nm (Tegra GA10B)           | COVERED | Reuses existing `samsung/8lpp.yaml` |

### 3. SM hierarchy (top-level GPU geometry)

These are the GPU-specific architectural knobs. They have no analog in
`KPUBlock` and need new fields on `GPUBlock`:

| Field (graphs)                  | Verdict | Mapping                          |
|---------------------------------|---------|----------------------------------|
| `compute_units=16` (num SMs)    | NEW     | `GPUBlock.num_sms: int`          |
| `cuda_cores_per_sm=128`         | NEW     | `GPUBlock.cuda_cores_per_sm: int` |
| `tensor_cores_per_sm=4`         | NEW     | `GPUBlock.tensor_cores_per_sm: int` |
| `threads_per_unit=64`           | NEW     | `GPUBlock.threads_per_sm: int`   |
| `warps_per_unit=2`              | NEW     | `GPUBlock.warps_per_sm: int`     |
| `warp_size=32`                  | NEW     | `GPUBlock.warp_size: int`        |

### 4. Per-fabric compute (CUDA core + Tensor core)

The model defines two `ComputeFabric` objects (CUDA cores and Tensor
cores) with per-precision ops/clock and per-fabric energy. KPUTileSpec
is the closest analog but is named for tiles, not GPU SMs.

**Recommendation: introduce a new generic `ComputeFabric` type** that
both `KPUTileSpec` and a new `GPUComputeFabric` can specialize from
(or reuse directly). Fields needed:

| Field (graphs)                                            | Verdict | Mapping |
|-----------------------------------------------------------|---------|---------|
| `fabric_type` ("cuda_core" / "tensor_core")               | NEW     | `GPUComputeFabric.fabric_type: str` |
| `circuit_type` ("standard_cell" / "tensor_core")          | EXTEND  | `KPUTileSpec.pe_circuit_class: CircuitClass` already exists; reuse |
| `num_units` (e.g., 2048 CUDA cores or 64 Tensor cores)    | NEW     | `GPUComputeFabric.num_units_per_sm` (then chip-total = num_sms * this) |
| `ops_per_unit_per_clock: dict[Precision, int]`            | EXTEND  | `KPUTileSpec.ops_per_tile_per_clock` already keys on precision string; reuse pattern |
| `core_frequency_hz`                                       | COVERED | Carried by thermal profile, not the fabric (clocks are DVFS-bound) |
| `energy_per_flop_fp32`                                    | NEW     | `GPUComputeFabric.energy_per_flop_fp32_pj: float` |
| `energy_scaling: dict[Precision, float]`                  | NEW     | `GPUComputeFabric.energy_scaling: dict[str, float]` |

**Open question**: should `KPUTileSpec` and `GPUComputeFabric` be
unified into one `ComputeFabric` type? Pro: one schema, one validator.
Con: KPUTileSpec carries `pe_array_rows / pe_array_cols / pes_per_tile`
which are mesh-shaped and don't apply to GPUs. Recommend **keeping them
separate but sharing helper sub-types** (CircuitClass, ops/clock dict).

### 5. Memory hierarchy

The GPU model has richer memory metadata than `KPUMemorySubsystem`
exposes today. Most fields belong on a generalized
`MemorySubsystem` -- which `KPUBlock` and `GPUBlock` can both reference.

| Field (graphs)                          | Verdict | Mapping |
|-----------------------------------------|---------|---------|
| `peak_bandwidth=204.8e9` (LPDDR5)       | EXTEND  | `KPUMemorySubsystem.memory_bandwidth_gbps` already covers this |
| `main_memory=64 * 1024**3` (64 GB)      | EXTEND  | `KPUMemorySubsystem.memory_size_gb` already covers this |
| `l1_cache_per_unit=128 KB` (per SM)     | EXTEND  | `KPUMemorySubsystem.l1_kib_per_pe` exists but the units are `pe`, not `sm`. **Recommend: rename `l1_kib_per_pe` -> `l1_kib_per_compute_unit`** (or add alias) |
| `l2_cache_total=4 MB`                   | EXTEND  | `KPUMemorySubsystem.l2_kib_per_tile` exists per-tile only. **Need: `l2_total_kib` chip-wide field** |
| `l1_storage_kind="cache"` (cache vs scratchpad vs unified) | NEW | `MemorySubsystem.l1_storage_kind: Literal["cache","scratchpad","unified"]` |
| `l2_topology="shared-llc"`              | NEW     | `MemorySubsystem.l2_topology: Literal["shared-llc","distributed","banked"]` |
| `l3_present=False`, `l3_cache_total=0`  | EXTEND  | `KPUMemorySubsystem.l3_kib_per_tile` exists; need `l3_present: bool` for "absent by design" semantics |
| `coherence_protocol="none"`             | NEW     | `MemorySubsystem.coherence_protocol: str` |
| `memory_technology="LPDDR5"`            | EXTEND  | `KPUMemorySubsystem.memory_type: MemoryType` already covers this (enum) |
| `memory_read_energy_per_byte_pj=15.0`   | NEW     | `MemorySubsystem.read_energy_pj_per_byte: float` |
| `memory_write_energy_per_byte_pj=18.0`  | NEW     | `MemorySubsystem.write_energy_pj_per_byte: float` |

**Recommendation**: rename `KPUMemorySubsystem` -> `MemorySubsystem`
with the following small generalizations:
- `l1_kib_per_pe` -> `l1_kib_per_compute_unit` (PE for KPU, SM for GPU)
- Add `l2_total_kib`, `l3_present`, `l3_total_kib`
- Add `l1_storage_kind`, `l2_topology`, `coherence_protocol`
- Add `read_energy_pj_per_byte`, `write_energy_pj_per_byte`
- All new fields default to None / 0 / "unknown" so existing KPU YAMLs
  continue to validate without changes

### 6. SoC fabric (on-die interconnect)

The GPU model uses an **SM-to-L2 crossbar**. `KPUNoCSpec` is shaped for
**2D mesh**. These are different topologies and different geometric
parameters.

| Field (graphs)                          | Verdict | Mapping |
|-----------------------------------------|---------|---------|
| `topology=Topology.CROSSBAR`            | NEW     | `OnDieFabric.topology: TopologyKind` (enum from `compute_product.py` -- already includes CROSSBAR-equivalent SWITCHED) |
| `bisection_bandwidth_gbps=2048.0`       | EXTEND  | `KPUNoCSpec.bisection_bandwidth_gbps` already exists |
| `controller_count=16` (= num_sms)       | NEW     | `OnDieFabric.controller_count: int` |
| `flit_size_bytes=32`                    | EXTEND  | `KPUNoCSpec.flit_bytes` already exists |
| `hop_latency_ns=2.0`                    | NEW     | `OnDieFabric.hop_latency_ns: float` |
| `pj_per_flit_per_hop=8.0`               | NEW     | `OnDieFabric.pj_per_flit_per_hop: float` |
| `routing_distance_factor=1.0`           | NEW     | `OnDieFabric.routing_distance_factor: float` |
| `mesh_rows`, `mesh_cols`                | EXTEND  | KPU-specific; on `OnDieFabric` should be optional (only populated for mesh topologies) |
| `router_circuit_class`                  | EXTEND  | Already on `KPUNoCSpec`; keep |

**Recommendation**: rename `KPUNoCSpec` -> `OnDieFabric` and make
mesh dims optional. Topology-specific fields (mesh_rows/cols vs
controller_count) become optional per-topology.

### 7. Thermal profiles

This is the most significant schema gap. KPU thermal profiles assume
**one clock** per profile; GPUs have **per-precision DVFS** with
`base_clock / boost_clock / sustained_clock` per profile.

| Field (graphs)                          | Verdict | Mapping |
|-----------------------------------------|---------|---------|
| `name="30W-active"`                     | COVERED | `KPUThermalProfile.name` |
| `tdp_watts=30.0`                        | COVERED | `KPUThermalProfile.tdp_watts` |
| `cooling_solution="active-fan"`         | EXTEND  | `KPUThermalProfile.cooling_solution_id` (need to ensure NVIDIA's "active-fan" / "passive-heatsink" cooling-solution YAMLs exist in `data/cooling_solutions/`) |
| `memory_clock_mhz=3200.0` (per profile) | NEW     | `ThermalProfile.memory_clock_mhz: float` -- thermal profiles tie GPU clock AND memory clock together |
| `clock_mhz=...` (single value)          | EXTEND  | KPUThermalProfile has one clock per profile. GPUs need `base_clock_mhz`, `boost_clock_mhz`, `sustained_clock_mhz`. **Recommend: add an optional `ClockDomain` field on ThermalProfile**, with the existing scalar `clock_mhz` retained for backward compat |
| `instruction_efficiency=0.85`           | NEW     | `ThermalProfile.instruction_efficiency_by_precision: dict[str, float]` (per precision) |
| `memory_bottleneck_factor=0.60`         | NEW     | `ThermalProfile.memory_bottleneck_factor_by_precision: dict[str, float]` |
| `efficiency_factor=0.47`                | EXTEND  | `KPUThermalProfile.efficiency_factor_by_precision` already exists -- reuse directly |
| `native_acceleration=True`              | NEW     | `ThermalProfile.native_acceleration_by_precision: dict[str, bool]` -- distinguishes "supported in hardware" from "emulated via lower-precision MAC unit" |

**Recommendation**: rename `KPUThermalProfile` -> `ThermalProfile`,
add the new optional fields above. KPU YAMLs continue to validate
because all new fields default to None.

### 8. Performance roll-up

`KPUTheoreticalPerformance` has fixed fields for INT8/BF16/FP32 and
optional INT4. GPUs need FP64, FP16 separate from BF16, and the schema
should generalize.

| Field (graphs)                                                    | Verdict | Mapping |
|-------------------------------------------------------------------|---------|---------|
| `cuda_fp32_peak`, `cuda_fp64_peak`, `combined_fp16_peak`, `combined_int8_peak` | EXTEND | Replace `KPUTheoreticalPerformance` with `TheoreticalPerformance` carrying `peak_ops_per_sec_by_precision: dict[str, float]` plus optional sparsity multiplier |

**Recommendation**: replace fixed-precision fields with a generic
`dict[Precision, float]`. KPU adapter populates the same INT8/BF16/FP32
keys; GPU adapter adds FP64/FP16/etc.

### 9. Silicon bin (per-block transistor decomposition)

`KPUSiliconBin` is general-purpose. The `count_ref` strings are KPU-flavored
(`tile.<type>`, `l3_total_kib`, `noc`, `memory`) but the `TransistorSource`
mechanism (`FIXED` / `PER_PE` / `PER_KIB` / `PER_ROUTER` /
`PER_CONTROLLER`) generalizes cleanly.

For GPUs we need new `count_ref` conventions:
- `sm.cuda_core` -> per-CUDA-core Mtx
- `sm.tensor_core` -> per-Tensor-core Mtx
- `l1_total_kib`, `l2_total_kib` -> per-KiB Mtx for unified L1 / L2
- `crossbar` -> per-port crossbar Mtx
- `lpddr5_phy.controller` -> per-LPDDR5-controller Mtx

**Verdict**: NO schema change needed -- just document the new
`count_ref` strings. The validator framework can pick them up via the
existing convention.

**Risk**: silicon_bin coefficients for GPUs are much harder to source
than for KPUs. Per-SM transistor counts and LPDDR5 PHY area need NVIDIA
disclosure or careful estimation. Issue #171 calls this out as the
top sprint risk -- pick a SKU with good public data (the AGX Orin
silicon spec doesn't disclose per-block transistor counts; will need
to estimate from die shot + GA10B die area).

### 10. GPU-specific scheduler attributes

| Field (graphs)               | Verdict | Mapping                          |
|------------------------------|---------|----------------------------------|
| `min_occupancy=0.3`          | NEW     | `GPUBlock.min_occupancy: float`  |
| `max_concurrent_kernels=8`   | NEW     | `GPUBlock.max_concurrent_kernels: int` |
| `wave_quantization=4`        | NEW     | `GPUBlock.wave_quantization: int` |

These are GPU-only and live on `GPUBlock` directly.

### 11. BOM cost profile

`graphs` has `BOMCostProfile` (silicon, package, memory, PCB, thermal,
margin -> retail_price). v1 ComputeProduct's `Market` carries
`launch_msrp_usd` only.

**Verdict**: NEW (optional). `Market.bom: Optional[BOMCostProfile]`
covers GPUs, CPUs, KPUs uniformly. Defer to a follow-up PR within the
sprint -- not a v1 GPU blocker.

### 12. Confidence / provenance

The graphs model uses `set_provenance(field_name, EstimationConfidence)`
to attach per-field provenance. The schema today has only chip-level
`confidence: DataConfidence`.

**Verdict**: orthogonal to this sprint. Per-field provenance on the
schema would mean attaching `EstimationConfidence` (or analog) to every
field via `Annotated[...]` or a sidecar dict. Worth a separate
proposal; the GPU sprint should NOT block on it.

## Summary table

| Category                       | Total fields | COVERED | EXTEND | NEW |
|--------------------------------|-------------:|--------:|-------:|----:|
| Identity / packaging           |       5      |    4    |   1    |  0  |
| Process node                   |       1      |    1    |   0    |  0  |
| SM hierarchy                   |       6      |    0    |   0    |  6  |
| Per-fabric compute             |       7      |    1    |   2    |  4  |
| Memory hierarchy               |      11      |    0    |   5    |  6  |
| SoC fabric                     |       9      |    0    |   2    |  6  |
| Thermal profiles               |       9      |    2    |   2    |  5  |
| Performance roll-up            |       1      |    0    |   1    |  0  |
| Silicon bin                    |       0      |    0    |   0    |  0 (just new count_ref strings) |
| GPU scheduler attributes       |       3      |    0    |   0    |  3  |
| BOM cost (deferred)            |       1      |    0    |   0    |  1 (optional) |
| **Totals**                     |    **53**    | **8**   | **13** |**31**|

Of 53 fields, ~15% (8) are already covered; ~25% (13) extend an
existing schema field; ~60% (31) are net-new additions. The "net-new"
count is GPU-shaped enough that a separate `GPUBlock` is well-justified
-- gluing all this onto KPUBlock would create a leaky union.

## Recommended schema diff for the next PR

The Schema PR (PR 2 of the sprint) should land roughly the following:

1. **`compute_product.py`**:
   - Add `BlockKind.GPU = "gpu"` to the discriminator enum.
   - Add `GPUBlock` with the GPU-specific fields surfaced above:
     `num_sms`, `cuda_cores_per_sm`, `tensor_cores_per_sm`,
     `threads_per_sm`, `warps_per_sm`, `warp_size`, `compute_fabrics`,
     `memory`, `noc`, `min_occupancy`, `max_concurrent_kernels`,
     `wave_quantization`, `multi_precision_alu`.
   - Update `AnyBlock = Annotated[Union[KPUBlock, GPUBlock], Field(discriminator="kind")]`.

2. **New file `embodied_schemas/gpu_block.py`**:
   - `GPUComputeFabric` (analogous to `KPUTileSpec`)
   - Reuse `CircuitClass`, generalized `MemorySubsystem`, generalized
     `OnDieFabric`, `ThermalProfile`.

3. **Generalize KPU sub-types** (rename + extend, not delete):
   - `KPUMemorySubsystem` -> `MemorySubsystem` with new optional
     fields (cache hierarchy depth, storage kinds, energy per byte).
     Keep the old name as a type alias for one release cycle.
   - `KPUNoCSpec` -> `OnDieFabric` with topology-dependent optional
     fields (mesh dims optional; controller_count, hop latency,
     pj/flit/hop added). Same alias treatment.
   - `KPUThermalProfile` -> `ThermalProfile` with optional
     `ClockDomain` and per-precision instruction_efficiency,
     memory_bottleneck_factor, native_acceleration maps.
   - `KPUTheoreticalPerformance` -> `TheoreticalPerformance` carrying
     `peak_ops_per_sec_by_precision: dict[str, float]`.

4. **Process node catalog**:
   - No new process-node YAML required for AGX Orin i this sprint.
   - Reuse existing `data/process-nodes/samsung/8lpp.yaml` via `process_node_id: "samsung_8lpp"`.

5. **Cooling solution catalog**:
   - The catalog has `passive_heatsink_large.yaml`, `active_fan.yaml`,
     `liquid_cooled.yaml`, `vapor_chamber.yaml` (underscore-separated).
     Jetson model strings use hyphens: "passive-heatsink", "active-fan",
     "active-fan-high", "active-fan-max". The data PR (PR 3) needs to:
     (a) map "passive-heatsink" -> existing `passive_heatsink_large`,
     (b) map "active-fan" -> existing `active_fan`,
     (c) add `active_fan_high.yaml` and `active_fan_max.yaml` for the
         50W and MAXN profiles (or fold into existing `active_fan` and
         encode the variant in the thermal profile itself).

## Risks called out by this exercise

1. **`KPUMemorySubsystem` rename touches every existing KPU YAML.**
   Even though we'd keep a type alias, the canonical name in new code
   moves. Risk: low (12 KPU YAMLs in `data/compute_products/stillwater/`
   today, all loaded through the same loader).

2. **`KPUThermalProfile` rename + ClockDomain addition** is the biggest
   change to a widely-used type. Mitigation: ClockDomain is **optional**
   and KPU YAMLs continue to use the scalar `clock_mhz`.

3. **Silicon_bin authoring for AGX Orin** is research-y. NVIDIA does
   not publish per-SM transistor counts. Plan: estimate from die area
   (455 mm^2 for GA10B, per public Tegra reverse-engineering) and SM
   count, then validate the silicon bin total matches `transistors_billion`
   chip-roll-up. Issue #171's "open questions" already flags this.

4. **DLA / PVA / ARM CPU cores in the same Tegra package are not
   modeled by GPUBlock.** They live in `jetson_orin_agx_cpu.py`
   separately today. v1 GPUBlock leaves them out; v2 will need
   `CPUBlock` (and potentially `DLABlock` / `VisionAccelBlock` etc.)
   to capture the full SoC. The chart we just built (PR #173) only
   plots GPU performance for the AGX Orin entry, so the v1 gap doesn't
   regress that workflow.

5. **The `compute_fabrics` field in `GPUBlock`** is structurally
   different from `KPUBlock.tiles` even though both express
   "heterogeneous compute units". The two should NOT be unified into a
   single `compute_units` field on `Die` until at least a third
   architecture (CPU? NPU?) makes the right abstraction obvious.
   Premature unification would force CUDA-core-shaped fields onto
   tile-shaped objects.

## Next step

Land this doc as PR 1 of the sprint (graphs-side). The Schema PR
(PR 2, embodied-schemas-side) implements section "Recommended schema
diff" above. Schema PR can be reviewed and merged independently of
the data PR (PR 3) since the new types are additive.
