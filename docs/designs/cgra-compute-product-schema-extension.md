# CGRA ComputeProduct schema extension -- paper exercise

Status: draft
Date: 2026-05-16
Tracking issue: graphs#196
Predecessors:
  - GPU sprint #171 (closed), design doc
    `docs/designs/gpu-compute-product-schema-extension.md`
  - CPU sprint #182 (closed), design doc
    `docs/designs/cpu-compute-product-schema-extension.md`
  - NPU sprint #187 (closed), design doc
    `docs/designs/npu-compute-product-schema-extension.md`
  - Hailo-10H + Coral migrations (closed by #192 / #194 / #195)

## Purpose

Issue #196 ("CGRA mini-sprint: add CGRABlock to ComputeProduct +
migrate Plasticine") proposes a 5-PR sprint to add `CGRABlock` to the
`ComputeProduct` discriminated union, mirroring sprints #171 (GPU),
#182 (CPU), and #187 (NPU). This document is **PR 1 of that sprint**
-- the paper exercise: enumerate Stanford Plasticine v2's full feature
set against the v4 (KPU + GPU + CPU + NPU) ComputeProduct schema and
classify every field as **covered**, **extends an existing field**,
or **needs a new field/type**.

The exercise uses **Stanford Plasticine v2** as the reference SKU
because it's the only CGRA in the catalog. Unlike the prior sprints,
there's no choice between "easy first SKU" and "hard follow-up" --
Plasticine is both. The sprint shape is condensed accordingly:

  - Same 5-PR structure (paper exercise / schema / data / loader /
    cleanup) but each PR is smaller in scope.
  - No follow-up SKU YAMLs deferred. Future CGRAs (Wave Computing,
    Cerebras WSE, SambaNova RDU) can land as pure data PRs after
    the schema is in place.

Plasticine is a hypothetical/research SKU (Stanford academic
prototype) but it exercises every CGRA architectural concept the
schema needs: spatial dataflow, reconfigurable PCUs, reconfiguration
overhead, hierarchical memory (PMU + shared L2), multi-precision
(INT8 / FP16 / emulated FP32). The hand-coded factory is ~170 LOC
(vs Hailo-8's 360, Hailo-10H's 360, Coral's 360) -- the smallest
migration target so far.

The doc deliberately copies the structure of the GPU + CPU + NPU
paper exercises so the diff between the four designs is easy to spot.

## Method

For each feature in the Plasticine v2 resource model
(`src/graphs/hardware/models/accelerators/stanford_plasticine_cgra.py`,
~170 LOC), this doc records:

- The field/object as it lives in graphs today (with line ref).
- Classification: COVERED / EXTEND / NEW.
- For EXTEND: the existing schema field that grows.
- For NEW: a proposed field name + brief rationale.

Out of scope for this exercise (deferred per the sprint plan):

- Other CGRA architectures (Wave Computing's DPU, Cerebras WSE,
  SambaNova RDU, Tenstorrent Grayskull) -- separate follow-up YAMLs
  once the schema lands; each may exercise extension fields not in
  Plasticine (e.g. larger-scale wafer-scale routing).
- Reconfiguration scheduling models (partial reconfiguration, PR-level
  dynamic remap) -- runtime concern, not a static SKU description.
- DPU sprint (Xilinx Vitis AI on AMD FPGA) -- structurally distinct
  (FPGA reconfiguration is much finer-grained than CGRA PCU reconfig);
  separate sprint.
- TPU schema extension -- separate sprint (5 SKUs justifies its own
  scope, with `tpu_v1/v3/v4/v5p` + `tpu_edge_pro` all hand-coded).
- DSP schema extension -- separate sprint (10 SKUs).
- Per-field provenance (`EstimationConfidence`) -- graphs-side
  feature; orthogonal to schema design (same call all prior sprints
  made).

## Reference: current v4 schema state

`embodied-schemas/src/embodied_schemas/compute_product.py`:

```python
class BlockKind(str, Enum):
    KPU = "kpu"
    GPU = "gpu"
    CPU = "cpu"
    NPU = "npu"

AnyBlock = Annotated[
    Union[KPUBlock, GPUBlock, CPUBlock, NPUBlock],
    Field(discriminator="kind"),
]
```

Four architectures shipped:

- v1 KPU: `KPUTileSpec`, `KPUNoCSpec`, `KPUMemorySubsystem`,
  `KPUSiliconBin`, etc.
- v2 GPU: `GPUComputeFabric`, `GPUMemorySubsystem`, `GPUOnDieFabric`,
  `GPUThermalProfile`, `GPUTheoreticalPerformance`, `ClockDomain`, +
  4 enums.
- v3 CPU: `CoreClusterSpec`, `CPUComputeFabric`, `CPUMemorySubsystem`,
  `CPUOnDieFabric`, `CPUThermalProfile`, `CPUTheoreticalPerformance`,
  + 4 enums. First cross-block-kind type sharing (`CPUThermalProfile`
  reuses `ClockDomain` from `gpu_block`).
- v4 NPU: `NPUBlock` + 5 supporting types + 3 enums. Second cross-
  block-kind reuse (`NPUOnDieFabric.confidence` uses `DataConfidence`
  from `process_node`). KVCacheSpec extension landed in
  embodied-schemas#30 for transformer-capable NPUs (Hailo-10H).

The v4 design rule established by prior sprints: **ship per-
architecture types in this sprint, defer rename + unify to v6+**
(CGRA would be the 5th; rename should be the 6th sprint when 5+
architectures exist + TPU/DPU/DSP have landed).

## Plasticine architectural primer (for reviewers not familiar)

Stanford Plasticine is a **coarse-grained reconfigurable
architecture** (CGRA). Key concepts:

- **PCU (Pattern Compute Unit)**: a programmable functional unit
  containing a small datapath (~8 MACs on Plasticine v2),
  configurable interconnect, and local scratchpad. ~32 PCUs per chip
  for v2.
- **PMU (Pattern Memory Unit)**: a per-PCU scratchpad (~64 KB) acting
  as L1 in NPU-land terms. Compiler-managed, no hardware cache.
- **Reconfiguration overhead**: the fabric is reconfigured per
  application; Plasticine v2 reports ~1000 cycles to switch between
  compiled programs. This is the **Achilles heel** of CGRAs vs
  fixed-function NPUs.
- **Spatial dataflow**: the entire program graph is mapped onto the
  PCU mesh; data flows through the fabric without instruction
  fetch/decode overhead. Architecturally similar to KPU (Stillwater)
  and NPU dataflow (Hailo) but with reconfigurability between
  applications.
- **PCU mesh NoC**: 2D mesh interconnect between PCUs + PMUs +
  external memory controllers. 40 GB/s bisection on Plasticine v2.

The architectural family includes:
- Stanford Plasticine / Plasticine-v2 (academic prototype, this sprint)
- SambaNova RDU (commercial, larger scale, post-v5 YAML candidate)
- Wave Computing DPU (defunct but well-documented in literature)
- Cerebras WSE-2 / WSE-3 (wafer-scale; arguably its own category)

## Feature-by-feature audit

### 1. Identity / packaging

| Field (graphs)                | Verdict  | Mapping                                |
|-------------------------------|----------|----------------------------------------|
| `name="CGRA-Plasticine-v2"`   | COVERED  | `ComputeProduct.name`                  |
| `hardware_type=HardwareType.CGRA` | EXTEND | `Die.blocks[0].kind = BlockKind.CGRA` (new enum value). HardwareType.CGRA already exists on the graphs side -- no enum gap to close (unlike the NPU case in #191). |
| (no vendor today)             | COVERED  | `ComputeProduct.vendor = "stanford"` (new vendor directory) |
| (no product family today)     | COVERED  | `Market.product_family = "Plasticine"` |
| Research die / no packaging   | EXTEND   | `Packaging(kind=MONOLITHIC, package_type="bare_die")`. Plasticine is academic -- no commercial packaging. |

### 2. Process node

| Field (graphs)                          | Verdict | Mapping |
|-----------------------------------------|---------|---------|
| `process_node_nm=28` (research)         | COVERED | `Die.process_node_id = "gf_28nm"`. The catalog landed `data/process-nodes/globalfoundries/28nm.yaml` in embodied-schemas#32 (Coral Edge TPU precursor). **No new node YAML needed for Plasticine** -- the GF 28nm entry is reusable. |

### 3. PCU hierarchy (CGRA-specific replacement for SM / core / tile / dataflow_unit)

CGRAs introduce the PCU + PMU abstraction. Plasticine v2 has 32 PCUs,
each with its own PMU. PCUs are reconfigurable per-application but
remain fixed during a single program execution.

| Field (graphs)                       | Verdict | Mapping                          |
|--------------------------------------|---------|----------------------------------|
| `compute_units=32` (PCUs)            | NEW     | `CGRABlock.num_pcus: int`        |
| `threads_per_unit=8` (MACs per PCU)  | NEW     | `CGRABlock.macs_per_pcu: int`    |
| `warps_per_unit=1`, `warp_size=1`    | EXTEND  | Legacy compat fields; loader defaults to 1. |
| (PMU = per-PCU scratchpad)           | NEW     | Lives on `CGRAMemorySubsystem.pmu_kib_per_pcu` -- the "Pattern Memory Unit" is conceptually L1 but compiler-managed. |
| (no cluster concept)                 | -       | Plasticine PCUs are homogeneous; no hybrid (P+E core) story. Future CGRAs may have heterogeneous PCU types. |

### 4. Compute fabric (single PCU fabric)

Plasticine v2 has ONE compute fabric (the PCU array itself). Simpler
than GPU (CUDA + Tensor cores) or CPU (multi-fabric per cluster).
Multi-fabric CGRAs (heterogeneous PCU types) would carry multiple
entries -- not common today.

| Field (graphs)                                            | Verdict | Mapping |
|-----------------------------------------------------------|---------|---------|
| `fabric_type="pcu_spatial_dataflow"`                      | NEW     | `CGRAComputeFabric.fabric_kind: CGRAFabricKind` enum (PCU_SPATIAL_DATAFLOW / SYSTOLIC_PCU / HETEROGENEOUS_PCU). Initial value: PCU_SPATIAL_DATAFLOW for Plasticine. |
| `circuit_type="standard_cell"`                            | EXTEND  | Reuse `CircuitClass` from `process_node` module (same as GPU/CPU/NPU) |
| `num_units=32` (= num_pcus)                               | -       | Lives on `CGRABlock.num_pcus`, not on the fabric. |
| `ops_per_unit_per_clock: {INT8: 320, FP16: 80}`           | NEW     | `CGRAComputeFabric.ops_per_unit_per_clock: dict[str, int]` (same shape as GPU/CPU/NPU). |
| `core_frequency_hz=1.0e9`                                 | COVERED | Lives in the chip-level thermal profile (single profile on Plasticine). |
| `energy_per_flop_fp32=get_base_alu_energy(28, 'standard_cell')` (4.0 pJ) | NEW | `CGRAComputeFabric.energy_per_op_int8_pj: float` -- CGRAs are INT8-dominant for DNN workloads; FP32 baseline still meaningful (Plasticine emulates FP32) but per-INT8 is the better contract. **Cross-block-kind reuse opportunity**: same shape as NPU's energy_per_op_int8_pj. |
| `energy_scaling: {INT8: 0.15, FP16: 0.50, FP32: 1.0}`     | NEW     | `CGRAComputeFabric.energy_scaling: dict[str, float]` -- scaling relative to the INT8 baseline (mirroring NPUComputeFabric). |

### 5. Reconfiguration overhead (CGRA-specific; no analog in CPU/GPU/KPU/NPU)

This is the **defining CGRA characteristic**. NPUs are compiled once
per model; CPUs/GPUs context-switch at thread granularity; KPUs are
fixed-function. CGRAs reconfigure the fabric per application, and the
overhead is non-trivial.

| Field (graphs)                          | Verdict | Mapping |
|-----------------------------------------|---------|---------|
| `reconfig_overhead=1000` cycles (in factory docstring, not modeled as a HardwareResourceModel field today) | NEW | `CGRABlock.reconfig_overhead_cycles: int` -- canonical full-fabric reconfiguration time in clock cycles. |
| (no partial-reconfig story today)       | NEW     | `CGRABlock.supports_partial_reconfig: bool = False` -- forward-compatibility hint for future PCU-region-level remap. Plasticine v2 is whole-fabric only. |
| (no compile-cache modeling)             | -       | Out of scope -- runtime/compiler concern. Schema only carries the architectural cost. |

**Recommendation**: model reconfig overhead as a scalar cycle count
on `CGRABlock` itself, not on the fabric. The cost is a property of
the chip's fabric-management hardware, not the compute fabric content.

### 6. Memory subsystem (PMU-dominant + small shared L2 + host memory)

CGRAs sit between NPUs (SRAM-dominant, sometimes no external DRAM)
and GPUs (LPDDR5/HBM-bound). Plasticine v2 has a per-PCU scratchpad
(PMU), a small shared L2 (2 MB), and **host memory** (DDR4 4 GB).
The host memory is accessed via the same path that Coral uses --
the chip itself has no external DRAM controllers.

| Field (graphs)                          | Verdict | Mapping |
|-----------------------------------------|---------|---------|
| `peak_bandwidth=40e9` (~40 GB/s on-chip mesh) | EXTEND | `CGRAMemorySubsystem.on_chip_bandwidth_gbps: float` (mirrors `NPUMemorySubsystem`) |
| `main_memory=4 GB` (host DDR4)          | NEW     | `CGRAMemorySubsystem.has_host_dram: bool = True` + `host_dram_size_gb: Optional[float]` + `host_dram_type: Optional[MemoryType]` + `host_dram_bandwidth_gbps: Optional[float]`. Different from `NPUMemorySubsystem.has_external_dram` because the DRAM is host-side, not chip-attached. **Naming convention TBD in schema PR -- could unify with NPU's has_external_dram if the host-vs-direct distinction is purely cosmetic; the loader-side overlay (Coral's `peak_bandwidth=4 GB/s` host-bus overlay from issue #192) is the v5 unification target either way.** |
| `l1_cache_per_unit=64 KB` (PMU per PCU) | NEW     | `CGRAMemorySubsystem.pmu_kib_per_pcu: int` -- "Pattern Memory Unit" terminology (Plasticine literature). |
| `l2_cache_total=2 MB` (shared on-chip)  | NEW     | `CGRAMemorySubsystem.shared_sram_kib: int` (mirrors `NPUMemorySubsystem.shared_sram_kib`). |
| `l2_topology="shared-llc"`              | EXTEND  | `CGRAMemorySubsystem.shared_sram_layout: Literal["shared", "partitioned"]` (mirrors NPU). |
| `l3_present=False`                      | -       | CGRAs have no L3 by construction. |
| `coherence_protocol="none"`             | EXTEND  | Same shape as GPU/CPU/NPU; CGRAs default "none" since compiler-routed dataflow has no coherence. |
| `memory_technology="DDR4"`              | EXTEND  | `CGRAMemorySubsystem.host_dram_type: Optional[MemoryType]` -- DDR4 for Plasticine; future research CGRAs may be HBM. |
| `memory_read_energy_per_byte_pj=20.0`   | NEW     | `CGRAMemorySubsystem.host_dram_access_energy_pj_per_byte: float` -- the host-DDR4 cost; on-chip PMU/L2 access is cheaper and lives on `pmu_access_energy_pj_per_byte` (NEW). |
| `energy_per_byte=12e-12` (on-chip)      | EXTEND  | Captured by `pmu_access_energy_pj_per_byte` above. |

**Recommendation**: `CGRAMemorySubsystem` is small (~10 fields) and
CGRA-specific enough to justify its own type. The host-DRAM gating
mirrors NPU's external-DRAM gating; a model_validator should enforce
consistency. Don't try to generalize across GPU/CPU/NPU/CGRA memory
subsystems in this sprint -- defer to v6 unification.

### 7. PCU mesh NoC (similar shape to NPU mesh)

| Field (graphs)                          | Verdict | Mapping |
|-----------------------------------------|---------|---------|
| `topology=Topology.MESH_2D`             | EXTEND  | `CGRAOnDieFabric.topology: CGRANoCTopology` enum (MESH_2D / TORUS_2D / CROSSBAR). Plasticine is MESH_2D; SambaNova uses torus. |
| `bisection_bandwidth_gbps=40.0`         | EXTEND  | Same |
| `controller_count=32` (= num_pcus)      | EXTEND  | Renamed `unit_count` (mirrors NPU) |
| `flit_size_bytes=16`                    | EXTEND  | Same |
| `mesh_dimensions=(4, 8)`                | NEW     | `CGRAOnDieFabric.mesh_rows / .mesh_cols` (mirrors KPU/NPU) |
| `hop_latency_ns=1.0` (estimate)         | EXTEND  | Same |
| `pj_per_flit_per_hop=1.2`               | EXTEND  | Same |
| `routing_distance_factor=1.1`           | EXTEND  | Same |
| `low_confidence=True`                   | NEW     | `CGRAOnDieFabric.confidence: DataConfidence` -- reuse `process_node.DataConfidence` (third cross-block-kind reuse after CPU/NPU). |

**Recommendation**: define `CGRAOnDieFabric` separately from
`NPUOnDieFabric` despite the field overlap -- the topology enum
differs (CGRAs need TORUS_2D for SambaNova-class; NPUs don't).
Field names align with NPU/GPU/CPU so v6 unification is mechanical.

### 8. Thermal profile (single operating point on Plasticine v2)

Plasticine v2 has a single 15W operating point with no DVFS, similar
to Hailo-8 and Coral.

| Field (graphs)                              | Verdict | Mapping |
|---------------------------------------------|---------|---------|
| `name="default"`                            | COVERED | `CGRAThermalProfile.name` |
| `tdp_watts=15.0`                            | COVERED | `CGRAThermalProfile.tdp_watts` |
| `cooling_solution="passive-air"`            | EXTEND  | `CGRAThermalProfile.cooling_solution_id` -- references `data/cooling-solutions/`. `passive_heatsink_small.yaml` exists; 15W borderline -- recommend `passive_heatsink_large` for the YAML. |
| `dvfs_enabled=False`                        | NEW     | `CGRAThermalProfile.dvfs_enabled: bool = False` -- CGRA default is False (single profile). |
| `clock_mhz=1000` (single value)             | EXTEND  | Scalar `clock_mhz` (mirrors NPU/KPU shape; not ClockDomain). |
| per-precision efficiency (0.60 INT8, ...) | EXTEND | `CGRAThermalProfile.efficiency_factor_by_precision: dict[str, float]` -- same shape as KPU/GPU/CPU/NPU |
| (no instruction efficiency metric today)    | NEW     | `CGRAThermalProfile.instruction_efficiency_by_precision: dict[str, float]` -- optional; CGRAs have minimal instruction overhead post-configuration |
| (no memory bottleneck factor today)         | NEW     | `CGRAThermalProfile.memory_bottleneck_factor_by_precision: dict[str, float]` -- same as GPU/CPU/NPU |

### 9. CGRA-specific scheduler attributes

| Field (graphs)               | Verdict | Mapping                          |
|------------------------------|---------|----------------------------------|
| `min_occupancy=0.3`          | NEW     | `CGRABlock.min_occupancy: float` -- lower default than NPU (0.8) because the reconfig overhead amortizes only on long-running kernels; PCU utilization varies more than dataflow NPUs |
| `max_concurrent_kernels=1`   | NEW     | `CGRABlock.max_concurrent_models: int = 1` -- CGRAs run a single mapped program at a time (mirrors NPU; future partial-reconfig SKUs could increase) |
| `wave_quantization=1`        | NEW     | `CGRABlock.wave_quantization: int = 1` -- like NPU/CPU, default 1 keeps shape consistent |
| (no SIMD efficiency)         | -       | CGRAs don't have SIMD lanes in the CPU sense; `simd_efficiency_by_op_kind` doesn't apply |

### 10. Performance roll-up

Plasticine v2 ships INT8 (6.14 TOPS realistic at 60% efficiency),
FP16 (1.54 TFLOPS = 1/4 INT8), and emulated FP32 (0.77 TFLOPS = 1/8
INT8). The schema's per-precision dict generalizes cleanly.

| Field (graphs)                                  | Verdict | Mapping |
|-------------------------------------------------|---------|---------|
| `peak_ops_per_sec` (per Precision)              | EXTEND  | `CGRATheoreticalPerformance.peak_ops_per_sec_by_precision: dict[str, float]` -- same shape as GPU/CPU/NPU. The signature CGRA pattern is **FP-emulated-via-INT** (Plasticine emulates FP32 by chaining INT8 ops); the schema doesn't need to model the emulation explicitly, but the relative-speedup ratios make it visible. |

### 11. Silicon bin (per-block transistor decomposition)

Plasticine is a research prototype -- published area data is sparse.
For an academic SKU, the silicon_bin transistor counts are explicitly
THEORETICAL estimates. Per the design convention (silicon_bin is
general-purpose, only `count_ref` strings are architecture-flavored),
Plasticine v2 needs new `count_ref` strings:

- `pcu_array` -> per-PCU Mtx (~5 Mtx/PCU for 8 MACs + scheduler + interconnect ports)
- `pmu_sram` -> per-PCU PMU scratchpad Mtx (~3 Mtx for 64 KB at 28nm SRAM density)
- `shared_sram` -> 2 MB shared L2 in Mtx
- `noc_routers` -> per-router mesh fabric logic (~3 Mtx each)
- `config_fabric` -> fabric reconfiguration control + bitstream storage. **CGRA-specific**: this is the cost of the programmability that distinguishes CGRAs from fixed-function NPUs.
- `host_phy` -> DDR4 PHY + PCIe / host interface
- `control_logic` -> compiler-emitted instruction memory + reconfig sequencer

**Verdict**: NO schema change needed -- just document the convention.
Same approach all prior sprints took.

Plasticine v2 die estimates (for the YAML):
- ~32 PCUs * 5 Mtx + 32 PMUs * 3 Mtx + shared SRAM ~80 Mtx + NoC ~100 Mtx +
  config_fabric ~40 Mtx + host PHY ~100 Mtx + control ~40 Mtx
  = ~520 Mtx
- At 28nm balanced_logic density (12 Mtx/mm^2 from the GF 28nm YAML),
  die area ~45 mm^2 (mostly PCU + NoC + host PHY area).
- 0.52 B transistors / 45 mm^2 = consistent with a research prototype at
  ~12 Mtx/mm^2 average.

### 12. BOM cost profile

Plasticine v2 is a research prototype -- no production BoM. The
factory has no `BOMCostProfile` today, so no overlay needed.
Optional field; defer to v6 `Market.bom`.

## Summary table

| Category                       | Total fields | COVERED | EXTEND | NEW |
|--------------------------------|-------------:|--------:|-------:|----:|
| Identity / packaging           |       5      |    3    |   2    |  0  |
| Process node                   |       1      |    1    |   0    |  0  |
| PCU hierarchy                  |       5      |    0    |   1    |  4  |
| Compute fabric                 |       7      |    1    |   2    |  4  |
| Reconfiguration overhead       |       3      |    0    |   0    |  3  |
| Memory subsystem               |      11      |    0    |   5    |  6  |
| On-die fabric                  |       9      |    0    |   6    |  3  |
| Thermal profile                |       8      |    2    |   3    |  3  |
| Scheduler attributes           |       4      |    0    |   1    |  3  |
| Performance roll-up            |       1      |    0    |   1    |  0  |
| Silicon bin                    |       0      |    0    |   0    |  0 (new count_ref strings only) |
| BoM cost                       |       0      |    0    |   0    |  0 (no BoM for research SKU) |
| **Totals**                     |    **54**    | **7**   | **21** |**26**|

Of 54 fields, ~13% (7) are already covered, ~39% (21) extend an
existing schema field (similar to NPU's 41%), and ~48% (26) are
net-new. The NEW count is higher than NPU's 22 because:

- Reconfiguration overhead is a CGRA-only concept (+3 fields)
- PMU + host-DRAM split adds memory fields not present in NPU (+3 fields)
- PCU hierarchy is structurally distinct from NPU's dataflow_unit (+3
  fields)

Cross-block-kind type reuse opportunities (third pattern, after
CPU's `ClockDomain` and NPU's `DataConfidence` reuses):

- `CGRAOnDieFabric.confidence` reuses `DataConfidence` from
  `process_node` (same as NPU)
- `CGRAMemorySubsystem.host_dram_type` reuses `MemoryType` from
  `gpu` (same as NPU's `external_dram_type`)
- `CGRAComputeFabric.circuit_class` reuses `CircuitClass` from
  `process_node` (same as GPU/CPU/NPU)

## Recommended schema diff for the next PR

The Schema PR (PR 2 of the sprint) should land roughly the following
in embodied-schemas:

1. **`compute_product.py`**:
   - Add `BlockKind.CGRA = "cgra"` to the discriminator enum.
   - Update `AnyBlock = Annotated[Union[KPUBlock, GPUBlock, CPUBlock, NPUBlock, CGRABlock], Field(discriminator="kind")]` (preserves the v2/v3/v4 discriminator metadata; a bare `Union[...]` would drop Pydantic's discriminator dispatch).

2. **New file `embodied_schemas/cgra_block.py`** (~400 LOC, similar
   in shape to `npu_block.py` but with CGRA-specific reconfig fields):
   - `CGRABlock` with: `num_pcus`, `macs_per_pcu`,
     `reconfig_overhead_cycles`, `supports_partial_reconfig: bool = False`,
     `compute_fabrics: list[CGRAComputeFabric]`,
     `memory: CGRAMemorySubsystem`, `noc: CGRAOnDieFabric`,
     `min_occupancy`, `max_concurrent_models`, `wave_quantization`,
     `multi_precision_alu`.
   - `CGRAComputeFabric` with: `fabric_kind: CGRAFabricKind`,
     `circuit_class`, `ops_per_unit_per_clock`, `energy_per_op_int8_pj`,
     `energy_scaling`.
   - `CGRAMemorySubsystem` with: `on_chip_bandwidth_gbps`,
     `pmu_kib_per_pcu`, `shared_sram_kib`,
     `shared_sram_layout: Literal["shared","partitioned"]`,
     `has_host_dram: bool`, `host_dram_type: Optional[MemoryType]`,
     `host_dram_size_gb: Optional[float]`,
     `host_dram_bandwidth_gbps: Optional[float]`,
     `pmu_access_energy_pj_per_byte`,
     `host_dram_access_energy_pj_per_byte`, `coherence_protocol`.
   - `CGRAOnDieFabric` with: `topology: CGRANoCTopology`,
     `bisection_bandwidth_gbps`, `unit_count`, `flit_size_bytes`,
     `mesh_rows` (optional), `mesh_cols` (optional), `hop_latency_ns`,
     `pj_per_flit_per_hop`, `routing_distance_factor`,
     `confidence: DataConfidence`.
   - `CGRAThermalProfile` with: `name`, `tdp_watts`, `cooling_solution_id`,
     `clock_mhz`, `dvfs_enabled`, plus the per-precision dicts
     (`efficiency_factor_by_precision`,
     `instruction_efficiency_by_precision`,
     `memory_bottleneck_factor_by_precision`).
   - `CGRATheoreticalPerformance` with `peak_ops_per_sec_by_precision`.
   - Enums: `CGRAFabricKind` (PCU_SPATIAL_DATAFLOW / SYSTOLIC_PCU /
     HETEROGENEOUS_PCU), `CGRANoCTopology` (MESH_2D / TORUS_2D /
     CROSSBAR).

3. **Validators on `CGRABlock`**:
   - `_validate_noc_unit_count_matches`: `noc.unit_count == num_pcus`
     (mirrors NPU).
   - `_validate_host_dram_consistency`: if `has_host_dram=True`, all
     host_dram_* fields must be populated (mirrors NPU's external_dram
     validator).
   - `_validate_macs_per_pcu_consistent_with_fabric`: optional sanity
     check that `compute_fabrics[0].ops_per_unit_per_clock[INT8]` is
     a plausible multiple of `macs_per_pcu` (Plasticine: 320 ops/clk /
     8 MACs = 40 ops per MAC per clock -- reflects 2 MACs/cycle *
     dual-issue * etc.; the validator should just warn-not-error).

4. **Process node**: no new YAML for Plasticine (GF 28nm SLP already
   in catalog from embodied-schemas#32 / Coral Edge TPU precursor).

5. **Cooling solution**: `passive_heatsink_large.yaml` already in
   catalog (15W borderline -- could also use small if the SKU YAML
   author prefers; either works).

## Risks called out by this exercise

1. **`HardwareType.CGRA` already exists on the graphs side.** Unlike
   the NPU case (where #191 had to add the enum value), CGRA is
   already a HardwareType. The loader can set
   `hardware_type=HardwareType.CGRA` directly with no transitional
   period. No mapper-guard changes needed.

2. **No-host-DRAM SKUs need optional host_dram_* fields.** Future
   CGRAs may bundle external DRAM directly on-chip (Cerebras WSE).
   Make every host_dram_* field on `CGRAMemorySubsystem` Optional
   with `has_host_dram` as the boolean gate. Mirrors the NPU
   `has_external_dram` pattern.

3. **Naming: `has_external_dram` (NPU) vs `has_host_dram` (CGRA).**
   The architectural distinction is real: NPU external DRAM is
   on-package or directly-attached (LPDDR4X on Hailo-10H); CGRA host
   DRAM is reached through the host bus (DDR4 on Plasticine, like
   Coral's host LPDDR4). v6 unification could pick one name. For now,
   the schema PR should use `has_host_dram` to signal the bus-mediated
   nature explicitly; the loader's peak_bandwidth overlay logic from
   issue #192 (Coral) can be reused.

4. **Reconfiguration overhead is the CGRA Achilles heel.** Downstream
   estimators (roofline, energy) need to know the reconfig cost so
   workloads with high kernel-switching frequency can be penalized
   appropriately. The schema only carries the cycle count; the cost
   modeling lives in graphs-side mappers. PR 4's loader should
   surface `reconfig_overhead_cycles` on the loaded model (new field
   on HardwareResourceModel, or attach as a `set_provenance`-style
   annotation).

5. **Reconfiguration overhead does NOT model partial reconfig.**
   `supports_partial_reconfig: bool = False` is a forward-compat
   hint. When future CGRAs (or FPGAs migrated through the same
   schema) need partial-reconfig modeling, extend the schema additively
   without breaking Plasticine's invariants.

6. **PCU multi-precision is INT-dominant but FP-capable.** Unlike
   NPU (INT-only by hardware), CGRAs can do FP16/FP32 through
   emulation. The `energy_scaling` dict must include the FP entries
   (Plasticine: FP16 at 0.50, FP32 at 1.0 of INT8 baseline) so
   downstream energy estimates correctly penalize FP workloads.

7. **Plasticine is a hypothetical SKU.** The YAML should clearly flag
   `confidence: theoretical` at the chip level (mirrors Hailo-8 / Hailo-
   10H / Coral) and all per-block confidence at theoretical. Future
   commercial CGRAs (SambaNova) would warrant a calibrated entry.

8. **PCU + PMU terminology is Plasticine-specific.** SambaNova calls
   them PCUs and "tile memories"; Cerebras uses different terminology
   entirely. The schema uses Plasticine's terms because that's the
   reference SKU; v6 unification (when SambaNova or Cerebras YAMLs
   land) can rename to vendor-neutral terms.

## Next step

Sprint sequencing:

- **PR 1 (this document)** -- docs/design, graphs-side. Lands the
  field-by-field audit + recommended schema diff. Pure docs PR; no
  code changes.
- **PR 2 -- schema, embodied-schemas-side.** Implements the
  "Recommended schema diff" above (CGRABlock + supporting types +
  enums). Additive; reviewable and mergeable independently of PR 3.
- **PR 3 -- data, embodied-schemas-side.** Authors the first Plasticine
  v2 YAML. No new process-node YAML needed (GF 28nm already landed in
  #32).
- **PR 4 -- loader, graphs-side.** Implements `cgra_yaml_loader` plus
  parity test against the hand-coded factory. Mirror of the NPU loader
  (#189) but with CGRA-specific fields. Mostly handles the new
  reconfig + PMU + host-DRAM fields; the rest reuses NPU loader patterns.
- **PR 5 -- cleanup, graphs-side.** Retires `stanford_plasticine_cgra.py`'s
  ~170-LOC body to a thin loader-wrapper + any necessary overlays.
  Closes issue #196.

The smaller scope of Plasticine vs Hailo-8 means the per-PR effort
should be ~half: schema is ~400 LOC vs 460, YAML is ~150 LOC vs 200,
loader is ~400 LOC vs 540, cleanup is ~100 LOC vs 87.
