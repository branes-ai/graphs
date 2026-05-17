# DPU ComputeProduct schema extension -- paper exercise

Status: draft
Date: 2026-05-16
Tracking issue: graphs#200
Predecessors:
  - GPU sprint #171 (closed), design doc
    `docs/designs/gpu-compute-product-schema-extension.md`
  - CPU sprint #182 (closed), design doc
    `docs/designs/cpu-compute-product-schema-extension.md`
  - NPU sprint #187 (closed), design doc
    `docs/designs/npu-compute-product-schema-extension.md`
  - CGRA sprint #196 (closed), design doc
    `docs/designs/cgra-compute-product-schema-extension.md`

## Purpose

Issue #200 ("DPU mini-sprint: add DPUBlock to ComputeProduct +
migrate Xilinx Vitis AI") proposes a 5-PR sprint to add `DPUBlock`
to the `ComputeProduct` discriminated union, mirroring sprints #171
(GPU), #182 (CPU), #187 (NPU), and #196 (CGRA). This document is
**PR 1 of that sprint** -- the paper exercise: enumerate Xilinx
Vitis AI's full feature set against the v5 (KPU + GPU + CPU + NPU +
CGRA) ComputeProduct schema and classify every field as **covered**,
**extends an existing field**, or **needs a new field/type**.

The exercise uses **Xilinx Vitis AI on Versal VE2302 (B4096 config)**
as the reference SKU because it's the only DPU in the catalog. Like
the CGRA sprint, there's no choice between "easy first" and "hard
follow-up" -- Vitis AI is both. The sprint shape is condensed
accordingly:

  - Same 5-PR structure (paper exercise / schema / data / loader /
    cleanup) but each PR is smaller in scope.
  - No follow-up SKU YAMLs deferred. Future DPUs (AMD Versal AI Core
    Premium, Intel/Altera AgileX AI) can land as pure data PRs after
    the schema is in place.

The hand-coded factory is ~160 LOC (similar to Plasticine's 170;
smallest migration targets in the catalog). The doc deliberately
copies the structure of the GPU + CPU + NPU + CGRA paper exercises
so the diff between the five designs is easy to spot.

## Method

For each feature in the Vitis AI resource model
(`src/graphs/hardware/models/accelerators/xilinx_vitis_ai_dpu.py`,
~160 LOC), this doc records:

- The field/object as it lives in graphs today (with line ref).
- Classification: COVERED / EXTEND / NEW.
- For EXTEND: the existing schema field that grows.
- For NEW: a proposed field name + brief rationale.

Out of scope for this exercise (deferred per the sprint plan):

- Other DPU architectures (AMD Versal AI Core / Premium series with
  AIE-ML v2, Intel/Altera AgileX AI) -- separate follow-up YAMLs once
  the schema lands; each may exercise extension fields not in Vitis
  AI B4096 (e.g. AIE-ML v2's different tile layout).
- FPGA bitstream / partial reconfiguration scheduling models --
  runtime concern, not a static SKU description.
- TPU schema extension -- separate sprint (5 SKUs justifies its own
  scope, with `tpu_v1/v3/v4/v5p` + `tpu_edge_pro` all hand-coded).
- DSP schema extension -- separate sprint (10 SKUs).
- Per-field provenance (`EstimationConfidence`) -- graphs-side
  feature; orthogonal to schema design (same call all prior sprints
  made).

## Reference: current v5 schema state

`embodied-schemas/src/embodied_schemas/compute_product.py`:

```python
class BlockKind(str, Enum):
    KPU = "kpu"
    GPU = "gpu"
    CPU = "cpu"
    NPU = "npu"
    CGRA = "cgra"

AnyBlock = Annotated[
    Union[KPUBlock, GPUBlock, CPUBlock, NPUBlock, CGRABlock],
    Field(discriminator="kind"),
]
```

Five architectures shipped:

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
  from `process_node`). `KVCacheSpec` extension landed in
  embodied-schemas#30 for transformer-capable NPUs.
- v5 CGRA: `CGRABlock` + 5 supporting types + 2 enums. Third cross-
  block-kind reuse pattern (DataConfidence, MemoryType, CircuitClass
  all reused). New fields: `reconfig_overhead_cycles`,
  `has_host_dram`, PCU + PMU hierarchy.

The design rule established by prior sprints: **ship per-architecture
types in this sprint, defer rename + unify to v7+** (DPU is the 6th;
rename should be the 7th sprint when 6+ architectures exist + TPU/DSP
have landed).

## DPU architectural primer (for reviewers not familiar)

Xilinx Vitis AI DPU (Deep Processing Unit) is an **FPGA-based AI
accelerator** -- the soft IP that turns an AMD/Xilinx Versal FPGA
into a neural-network inference engine. Key concepts:

- **AIE tile (AI Engine)** -- a programmable hardened compute tile
  on the Versal die, containing a small datapath with INT8 MACs +
  configurable interconnect + local scratchpad. ~64 tiles in the
  B4096 config (one of several published configs: B512/B1024/B2304/B4096).
- **AIE-ML v1** -- the variant on Versal VE2302; subsequent
  generations are AIE-ML v2 (Versal AI Core / Premium) and AIE-HD
  (high-density, datacenter target). Each ships different MAC widths
  and precision support.
- **Static FPGA reconfiguration** -- the DPU bitstream is baked into
  the FPGA at deployment time. Unlike CGRAs (which do runtime PCU
  remap), DPUs are statically configured per application. The
  reconfig cost is deployment-only (~seconds for full bitstream
  load); not a runtime concern.
- **Multi-precision** -- INT8 (128 ops/tile/clock, native), FP16
  (32 ops/tile/clock, native via AIE-ML), FP32 (emulated, 1/8 INT8 rate).
- **Versal SoC memory hierarchy** -- per-tile scratchpad (~64 KB) +
  shared L2 (~4 MB) + on-package LPDDR4X / DDR4 (4-16 GB depending
  on Versal SKU). VE2302 is the edge-class part with 8 GB DDR4.

The architectural family includes:
- Xilinx Vitis AI on Versal VE2302 (this sprint -- B4096 config)
- AMD Versal AI Core / Premium (post-v6 YAMLs; AIE-ML v2)
- Intel/Altera AgileX AI Series (post-v6 YAMLs; different IP)

## Feature-by-feature audit

### 1. Identity / packaging

| Field (graphs)                | Verdict  | Mapping                                |
|-------------------------------|----------|----------------------------------------|
| `name="DPU-Vitis-AI-B4096"`   | COVERED  | `ComputeProduct.name`                  |
| `hardware_type=HardwareType.DPU` | EXTEND | `Die.blocks[0].kind = BlockKind.DPU` (new enum value). `HardwareType.DPU` already exists on the graphs side -- no enum gap to close (like CGRA in #196, unlike NPU's #191). |
| (no vendor today)             | COVERED  | `ComputeProduct.vendor = "xilinx"` (new vendor directory) |
| (no product family today)     | COVERED  | `Market.product_family = "Vitis AI"`   |
| Versal VE2302 SoC packaging   | EXTEND   | `Packaging(kind=MONOLITHIC, package_type="bga")`. Versal ships as a packaged SoC. |

### 2. Process node

| Field (graphs)                          | Verdict | Mapping |
|-----------------------------------------|---------|---------|
| `process_node_nm=16` (Versal VE2302)    | COVERED | `Die.process_node_id = "tsmc_n16"`. The catalog already has `data/process-nodes/tsmc/n16.yaml` from the KPU sprint. **No new node YAML needed.** |

### 3. AIE tile hierarchy (DPU-specific replacement for SM / core / tile / dataflow_unit / PCU)

DPUs introduce the AIE tile abstraction. Vitis AI B4096 has 64 AIE
tiles, each with 64 MACs (= 4096 MACs total). AIE tiles are
hardened (not LUT-based) but the surrounding fabric is FPGA-soft.

| Field (graphs)                       | Verdict | Mapping                          |
|--------------------------------------|---------|----------------------------------|
| `compute_units=64` (AIE tiles)       | NEW     | `DPUBlock.num_aie_tiles: int`    |
| `threads_per_unit=64` (MACs/tile)    | NEW     | `DPUBlock.macs_per_tile: int`    |
| `warps_per_unit=8`, `warp_size=8` (vector lanes per tile) | NEW | `DPUBlock.simd_lanes_per_tile: int = 1`. AIE-ML tiles have vector-width datapaths; the SIMD count varies per AIE generation. Default 1 keeps the field shape consistent for non-SIMD architectures. |
| (no cluster concept)                 | -       | B4096 tiles are homogeneous. AIE-ML v2 (future SKU) may have heterogeneous tile types -- defer to v7 unification. |

### 4. Compute fabric (single AIE-ML fabric on Vitis AI B4096)

Vitis AI B4096 has ONE compute fabric (the AIE-ML tile array).
Heterogeneous-tile DPUs (Versal AI Core Premium with mixed AIE/DSP58
tiles) would carry multiple entries -- defer to v7+.

| Field (graphs)                                            | Verdict | Mapping |
|-----------------------------------------------------------|---------|---------|
| `fabric_type="aie_ml_tile"`                               | NEW     | `DPUComputeFabric.fabric_kind: DPUFabricKind` enum (AIE_ML_V1 / AIE_ML_V2 / AIE_HD / SOFT_LUT_DPU). Initial value: AIE_ML_V1 for Vitis AI B4096. |
| `circuit_type="standard_cell"`                            | EXTEND  | Reuse `CircuitClass` from `process_node` module (same as GPU/CPU/NPU/CGRA). DPU note: AIE tiles are hardened standard cells; surrounding FPGA fabric is LUT-based -- the schema can grow to model this distinction if needed. |
| `num_units=64` (= num_aie_tiles)                          | -       | Lives on `DPUBlock.num_aie_tiles`, not on the fabric. |
| `ops_per_unit_per_clock: {INT8: 128, FP16: 32}`           | NEW     | `DPUComputeFabric.ops_per_unit_per_clock: dict[str, int]` (same shape as GPU/CPU/NPU/CGRA). |
| `core_frequency_hz=1.25e9`                                | COVERED | Lives in the chip-level thermal profile (single profile on Vitis AI B4096). |
| `energy_per_flop_fp32=get_base_alu_energy(16, 'standard_cell')` (2.7 pJ) | NEW | `DPUComputeFabric.energy_per_op_int8_pj: float`. DPUs are INT8-dominant for DNN workloads (FP16 is native but secondary). |
| `energy_scaling: {INT8: 0.15, FP16: 0.50, FP32: 1.0}`     | NEW     | `DPUComputeFabric.energy_scaling: dict[str, float]` -- scaling relative to the INT8 baseline (mirroring NPU/CGRA fabrics). |
| (FPGA fabric overhead vs ASIC equivalent)                 | NEW     | `DPUComputeFabric.fpga_fabric_overhead_factor: float = 1.0`. Models the ~20-30% energy penalty FPGA AIE tiles pay vs equivalent ASIC implementations (LUT-based interconnect, programmable routing). Default 1.0 for non-FPGA-overhead-modeled SKUs. |

### 5. Reconfiguration model (DPU-specific; differs from CGRA)

Unlike CGRAs which reconfigure PCUs at runtime (~1000 cycles),
DPUs are statically configured at deployment via FPGA bitstream
load. The cost is deployment-only (~seconds), not a runtime concern.
Captured as a forward-compat flag rather than a runtime cost field.

| Field (graphs)                          | Verdict | Mapping |
|-----------------------------------------|---------|---------|
| (no reconfig modeling today)            | NEW     | `DPUBlock.is_statically_reconfigurable: bool = True` -- always True for FPGA-based DPUs; flag exists for future hard-DPU SKUs that ship as fixed-function silicon. |
| (no bitstream-load cost today)          | NEW     | `DPUBlock.bitstream_load_time_ms: Optional[float]` -- typical bitstream load time (seconds-range). Optional because the legacy doesn't carry it; Vitis AI VE2302 is ~1-3 seconds for the full bitstream. |
| (no partial-reconfig story today)       | -       | Out of scope for v6; FPGA partial reconfig is a runtime concern. |

**Recommendation**: model FPGA reconfig as a deployment cost on
`DPUBlock`, not as a runtime cycle count (like CGRA's
`reconfig_overhead_cycles`). The semantics are fundamentally
different -- DPU reconfig is rare and large; CGRA reconfig is
frequent and small.

### 6. Memory subsystem (scratchpad-dominant + shared L2 + chip-attached DDR)

DPUs sit between CGRAs (host-bus DRAM) and NPUs (chip-attached LPDDR).
Vitis AI VE2302 has per-tile scratchpad (64 KB), shared L2 (4 MB),
and chip-attached DDR4 (8 GB) on the Versal SoC package. The DDR4
controller is on-die; not host-bus mediated.

| Field (graphs)                          | Verdict | Mapping |
|-----------------------------------------|---------|---------|
| `peak_bandwidth=50e9` (~50 GB/s DDR4)   | EXTEND  | `DPUMemorySubsystem.on_chip_bandwidth_gbps: float` (the AIE tile array fabric) AND `DPUMemorySubsystem.external_dram_bandwidth_gbps`. The legacy's 50 GB/s is the DDR4 number; for DPU roofline, DDR4 is the bottleneck (matches NPU's external_dram convention, not CGRA's on-chip-only). |
| `main_memory=8 GB` (chip-attached DDR4) | NEW     | `DPUMemorySubsystem.has_external_dram: bool = True` + `external_dram_size_gb: Optional[float]`. Uses NPU-style `has_external_dram` (chip-attached) rather than CGRA-style `has_host_dram` (bus-mediated). |
| `l1_cache_per_unit=64 KB` (per-tile scratchpad) | NEW | `DPUMemorySubsystem.scratchpad_kib_per_tile: int` -- AIE tile data memory. |
| `l2_cache_total=4 MB` (shared on-chip)  | NEW     | `DPUMemorySubsystem.shared_sram_kib: int` (mirrors NPU/CGRA). |
| `l2_topology="shared-llc"`              | EXTEND  | `DPUMemorySubsystem.shared_sram_layout: Literal["shared", "partitioned"]`. |
| `l3_present=False`                      | -       | DPUs have no L3 by construction. |
| `coherence_protocol="none"`             | EXTEND  | Same shape as GPU/CPU/NPU/CGRA; DPUs default "none" (compiler-routed dataflow). |
| `memory_technology="DDR4"`              | EXTEND  | `DPUMemorySubsystem.external_dram_type: Optional[MemoryType]` -- DDR4 for VE2302; LPDDR5 for some future Versal SKUs. |
| `energy_per_byte=15e-12` (DDR4)         | NEW     | `DPUMemorySubsystem.scratchpad_access_energy_pj_per_byte: float` (on-chip) AND `external_dram_access_energy_pj_per_byte: float` (DDR4). |

**Recommendation**: define `DPUMemorySubsystem` separately from
`NPUMemorySubsystem` / `CGRAMemorySubsystem` despite the field
overlap -- the field names differ (AIE-tile-specific terminology)
and the `has_external_dram` semantics here are NPU-aligned (chip-
attached) rather than CGRA-aligned (host bus). v7 unification can
collapse these.

### 7. AIE tile fabric NoC (similar shape to KPU mesh)

| Field (graphs)                          | Verdict | Mapping |
|-----------------------------------------|---------|---------|
| (not modeled today)                     | NEW     | `DPUOnDieFabric.topology: DPUNoCTopology` enum (AIE_MESH / CROSSBAR). The AIE tile array uses a streaming-mesh interconnect distinct from packet-based NoCs. |
| `bisection_bandwidth_gbps` (estimate)   | EXTEND  | `DPUOnDieFabric.bisection_bandwidth_gbps: float` |
| `unit_count=64` (= num_aie_tiles)       | EXTEND  | `DPUOnDieFabric.unit_count: int` (mirrors NPU/CGRA) |
| `flit_size_bytes` (32 for AIE streams)  | EXTEND  | `DPUOnDieFabric.flit_size_bytes: int` |
| `mesh_dimensions=(8, 8)` (estimate)     | NEW     | `DPUOnDieFabric.mesh_rows / .mesh_cols` (mirrors KPU/NPU/CGRA) |
| `hop_latency_ns` (estimate)             | EXTEND  | `DPUOnDieFabric.hop_latency_ns: float` |
| `pj_per_flit_per_hop` (FPGA fabric)     | EXTEND  | `DPUOnDieFabric.pj_per_flit_per_hop: float` |
| `low_confidence=True`                   | NEW     | `DPUOnDieFabric.confidence: DataConfidence` -- reuse `process_node.DataConfidence` (fourth cross-block-kind reuse after CPU/NPU/CGRA). Xilinx doesn't publish per-hop AIE fabric details. |

**Recommendation**: define `DPUOnDieFabric` separately from
`NPUOnDieFabric` / `CGRAOnDieFabric` -- AIE mesh has streaming
semantics distinct from packet-based NoCs. Field names align with
NPU/CGRA so v7 unification is mechanical.

### 8. Thermal profile (single 20W operating point on Vitis AI B4096)

VE2302 ships with active-fan cooling (15-20W envelope; edge-class).

| Field (graphs)                              | Verdict | Mapping |
|---------------------------------------------|---------|---------|
| `name="default"`                            | COVERED | `DPUThermalProfile.name` |
| `tdp_watts=20.0`                            | COVERED | `DPUThermalProfile.tdp_watts` |
| `cooling_solution="active-fan"`             | EXTEND  | `DPUThermalProfile.cooling_solution_id` -- references `data/cooling-solutions/`. `active_fan.yaml` exists. |
| `dvfs_enabled=False`                        | NEW     | `DPUThermalProfile.dvfs_enabled: bool = False` -- DPU default is False. |
| `clock_mhz=1250` (single value)             | EXTEND  | Scalar `clock_mhz` (mirrors NPU/CGRA shape; not ClockDomain). |
| per-precision efficiency (0.75 INT8)        | EXTEND  | `DPUThermalProfile.efficiency_factor_by_precision: dict[str, float]` -- same shape as KPU/GPU/CPU/NPU/CGRA. |
| (no instruction efficiency metric today)    | NEW     | `DPUThermalProfile.instruction_efficiency_by_precision: dict[str, float]` -- optional. |
| (no memory bottleneck factor today)         | NEW     | `DPUThermalProfile.memory_bottleneck_factor_by_precision: dict[str, float]` -- same as GPU/CPU/NPU/CGRA. |

### 9. DPU-specific scheduler attributes

| Field (graphs)               | Verdict | Mapping                          |
|------------------------------|---------|----------------------------------|
| `min_occupancy=0.3`          | NEW     | `DPUBlock.min_occupancy: float = 0.3` -- similar to CGRA default; FPGA fabric overhead means tile utilization varies. |
| `max_concurrent_kernels=4`   | NEW     | `DPUBlock.max_concurrent_models: int = 4` -- DPUs can run multiple compiled models simultaneously by partitioning AIE tiles (different from NPU/CGRA which are single-model). |
| `wave_quantization=2`        | NEW     | `DPUBlock.wave_quantization: int = 2` -- AIE tiles allocated in pairs (per Xilinx documentation). |
| (no SIMD efficiency on DPUs) | -       | DPUs don't have CPU-style SIMD efficiency; the per-tile SIMD count is fixed. |

### 10. Performance roll-up

Vitis AI B4096 ships INT8 (7.68 TOPS realistic at 75% efficiency),
FP16 (1.92 TFLOPS native), and emulated FP32 (0.96 TFLOPS).

| Field (graphs)                                  | Verdict | Mapping |
|-------------------------------------------------|---------|---------|
| `peak_ops_per_sec` (per Precision)              | EXTEND  | `DPUTheoreticalPerformance.peak_ops_per_sec_by_precision: dict[str, float]` -- same shape as GPU/CPU/NPU/CGRA. DPU's distinctive pattern is **native-FP16 + emulated-FP32** (similar to CGRA but FP16 is hardware native here). |

### 11. Silicon bin (per-block transistor decomposition)

Versal VE2302 is a complex SoC -- AIE tile array + FPGA programmable
logic + PCIe/DDR PHYs + ARM Cortex-A72 control complex + NoC.
Published transistor count is ~3.7 B for the full SoC. The DPU
silicon_bin should cover just the AIE-ML + fabric-IP fraction
attributable to the DPU function:

- `aie_tile_array` -> 64 hardened AIE tiles
- `tile_scratchpad_sram` -> per-tile scratchpad SRAM
- `shared_sram` -> 4 MiB shared L2
- `aie_noc` -> AIE tile fabric interconnect
- `fpga_fabric_overhead` -> LUT-based glue logic for DPU bitstream
- `ddr4_phy` -> DDR4 controller + PHY
- `pcie_host_phy` -> PCIe Gen4 host interface
- `arm_control_complex` -> Cortex-A72 control + caches (the "platform"
  component of Versal, modeled as a sibling block in v7 unification)

**Verdict**: NO schema change needed -- just document the convention.
Same approach all prior sprints took. The full Versal SoC could be
modeled as multiple blocks per die in a future PR (DPU + CPU complex
+ FPGA fabric); for v6 we model only the DPU function.

### 12. BOM cost profile

`graphs` has no BoM data populated for Vitis AI today. The Versal
VE2302 SoM/SoC is in the $300-500 retail range; could add as overlay
later. Defer to v7 `Market.bom`.

## Summary table

| Category                       | Total fields | COVERED | EXTEND | NEW |
|--------------------------------|-------------:|--------:|-------:|----:|
| Identity / packaging           |       5      |    3    |   2    |  0  |
| Process node                   |       1      |    1    |   0    |  0  |
| AIE tile hierarchy             |       4      |    0    |   1    |  3  |
| Compute fabric                 |       8      |    1    |   2    |  5  |
| Reconfiguration model          |       3      |    0    |   0    |  2 (+1 deferred) |
| Memory subsystem               |      11      |    0    |   5    |  6  |
| On-die fabric                  |       9      |    0    |   6    |  3  |
| Thermal profile                |       8      |    2    |   3    |  3  |
| Scheduler attributes           |       4      |    0    |   1    |  3  |
| Performance roll-up            |       1      |    0    |   1    |  0  |
| Silicon bin                    |       0      |    0    |   0    |  0 (new count_ref strings only) |
| BoM cost                       |       0      |    0    |   0    |  0 (defer overlay) |
| **Totals**                     |    **54**    | **7**   | **21** |**25** (+1 deferred) |

Of 54 fields, ~13% (7) are already covered, ~39% (21) extend an
existing schema field (similar to NPU's 41% and CGRA's 39%), and
~46% (25) are net-new. The NEW count sits between NPU (22) and CGRA
(26):

- Fewer NEW than CGRA (no PCU + PMU terminology rebrand;
  reconfiguration is simpler / static)
- More NEW than NPU (FPGA fabric overhead + bitstream load cost +
  AIE-tile naming layer)

Cross-block-kind type reuse opportunities (fourth pattern, after
CPU's `ClockDomain`, NPU's `DataConfidence`, and CGRA's three-way
reuse):

- `DPUOnDieFabric.confidence` reuses `DataConfidence` from
  `process_node` (same as NPU/CGRA)
- `DPUMemorySubsystem.external_dram_type` reuses `MemoryType` from
  `gpu` (same as NPU)
- `DPUComputeFabric.circuit_class` reuses `CircuitClass` from
  `process_node` (same as GPU/CPU/NPU/CGRA)

By this point, the cross-block-kind reuse pattern has accumulated
enough evidence (5 architectures, 3+ shared primitives each) to make
the v7 vendor-neutral `compute_block_common` module a clear win.
DPU is the LAST sprint before that unification becomes overdue.

## Recommended schema diff for the next PR

The Schema PR (PR 2 of the sprint) should land roughly the following
in embodied-schemas:

1. **`compute_product.py`**:
   - Add `BlockKind.DPU = "dpu"` to the discriminator enum.
   - Update `AnyBlock = Annotated[Union[KPUBlock, GPUBlock, CPUBlock, NPUBlock, CGRABlock, DPUBlock], Field(discriminator="kind")]` (preserves the v2-v5 discriminator metadata).

2. **New file `embodied_schemas/dpu_block.py`** (~420 LOC, similar in
   shape to `cgra_block.py`):
   - `DPUBlock` with: `num_aie_tiles`, `macs_per_tile`,
     `simd_lanes_per_tile`, `is_statically_reconfigurable: bool = True`,
     `bitstream_load_time_ms: Optional[float]`,
     `compute_fabrics: list[DPUComputeFabric]`,
     `memory: DPUMemorySubsystem`, `noc: DPUOnDieFabric`,
     `min_occupancy`, `max_concurrent_models`, `wave_quantization`,
     `multi_precision_alu`.
   - `DPUComputeFabric` with: `fabric_kind: DPUFabricKind`,
     `circuit_class`, `ops_per_unit_per_clock`, `energy_per_op_int8_pj`,
     `energy_scaling`, `fpga_fabric_overhead_factor: float = 1.0`.
   - `DPUMemorySubsystem` with: `on_chip_bandwidth_gbps`,
     `scratchpad_kib_per_tile`, `shared_sram_kib`,
     `shared_sram_layout: Literal["shared","partitioned"]`,
     `has_external_dram: bool`,
     `external_dram_type: Optional[MemoryType]`,
     `external_dram_size_gb: Optional[float]`,
     `external_dram_bandwidth_gbps: Optional[float]`,
     `scratchpad_access_energy_pj_per_byte`,
     `external_dram_access_energy_pj_per_byte`, `coherence_protocol`.
   - `DPUOnDieFabric` with: `topology: DPUNoCTopology`,
     `bisection_bandwidth_gbps`, `unit_count`, `flit_size_bytes`,
     `mesh_rows` (optional), `mesh_cols` (optional),
     `hop_latency_ns`, `pj_per_flit_per_hop`,
     `routing_distance_factor`, `confidence: DataConfidence`.
   - `DPUThermalProfile` with: `name`, `tdp_watts`,
     `cooling_solution_id`, `clock_mhz`, `dvfs_enabled`, plus the
     per-precision dicts.
   - `DPUTheoreticalPerformance` with `peak_ops_per_sec_by_precision`.
   - Enums: `DPUFabricKind` (AIE_ML_V1 / AIE_ML_V2 / AIE_HD /
     SOFT_LUT_DPU), `DPUNoCTopology` (AIE_MESH / CROSSBAR).

3. **Validators on `DPUBlock`**:
   - `_validate_noc_unit_count_matches`: `noc.unit_count == num_aie_tiles`
     (mirrors NPU/CGRA).
   - `_validate_external_dram_consistency`: if `has_external_dram=True`,
     all `external_dram_*` fields must be populated (mirrors NPU
     `external_dram` validator).
   - `_validate_macs_per_tile_consistent_with_fabric`: optional sanity
     check that `compute_fabrics[0].ops_per_unit_per_clock[INT8]` is
     a plausible multiple of `macs_per_tile * 2` (Vitis AI: 128 ops/
     clk = 64 MACs * 2 ops -- the ratio).

4. **Process node**: no new YAML needed (TSMC N16 already in catalog
   from KPU sprint).

5. **Cooling solution**: `active_fan.yaml` already in catalog.

## Risks called out by this exercise

1. **`HardwareType.DPU` already exists on the graphs side.** Unlike
   the NPU case (where #191 had to add the enum value), DPU is
   already a HardwareType. The loader can set
   `hardware_type=HardwareType.DPU` directly with no transitional
   period.

2. **DDR4 is chip-attached, not host-bus.** Unlike CGRA's
   Plasticine (which reaches DDR4 via host PCIe), Versal VE2302 has
   on-die DDR4 controllers. Use NPU-style `has_external_dram`
   (chip-attached) rather than CGRA-style `has_host_dram` (host-bus
   mediated). v7 unification can resolve the naming.

3. **FPGA fabric overhead is real but hard to quantify.** Xilinx
   doesn't publish the exact ASIC-vs-FPGA penalty for AIE tiles
   (most of the AIE datapath is hardened, but routing + glue logic
   is LUT-based). The `fpga_fabric_overhead_factor` defaults to 1.0
   for SKUs that don't quantify it; Vitis AI YAML can set ~1.2-1.3
   to model the published energy penalty.

4. **Bitstream load time is deployment-only, not runtime.**
   `bitstream_load_time_ms` is captured for completeness but should
   not feed into per-inference latency models. The legacy doesn't
   carry it; future YAMLs can populate it (~1-3 seconds for VE2302).

5. **AIE-ML v1 vs v2 vs HD have different MAC widths and precisions.**
   The schema's `DPUFabricKind` enum is the discriminator. Vitis AI
   B4096 is AIE_ML_V1; AMD Versal AI Core Premium would be AIE_ML_V2
   (different ops/clock numbers); future datacenter DPUs are AIE_HD.
   The YAML's `ops_per_unit_per_clock` dict captures the actual
   numbers; the enum is metadata for grouping / reporting.

6. **`HardwareType.DPU` may conflict with v6 unification.** The
   architectural taxonomy on graphs (TPU/NPU/DPU/CGRA/KPU) was
   defined when these were thought of as fundamentally different
   classes. Post-v6 the schema-side `BlockKind` may be more
   important than the graphs `HardwareType`; the v7 unification
   sprint should decide whether to keep the legacy enum, deprecate
   it, or map BlockKind <-> HardwareType automatically.

7. **Wave quantization = 2 (tiles allocated in pairs).** This is a
   Xilinx-specific detail (AIE tiles ship and configure as pairs).
   Future AIE-HD SKUs may use different quantization; keep the
   `wave_quantization` field flexible.

## Next step

Sprint sequencing:

- **PR 1 (this document)** -- docs/design, graphs-side. Lands the
  field-by-field audit + recommended schema diff. Pure docs PR; no
  code changes.
- **PR 2 -- schema, embodied-schemas-side.** Implements the
  "Recommended schema diff" above (DPUBlock + supporting types +
  enums). Additive; reviewable and mergeable independently of PR 3.
- **PR 3 -- data, embodied-schemas-side.** Authors the first Vitis
  AI B4096 YAML. No new process-node YAML needed (TSMC N16 already
  in catalog).
- **PR 4 -- loader, graphs-side.** Implements `dpu_yaml_loader` plus
  parity test against the hand-coded factory. Mirror of the CGRA
  loader (#198) but with DPU-specific fields. Mostly handles the new
  AIE tile + FPGA overhead + chip-attached DRAM fields; the rest
  reuses CGRA loader patterns.
- **PR 5 -- cleanup, graphs-side.** Retires `xilinx_vitis_ai_dpu.py`'s
  ~160-LOC body to a thin loader-wrapper + any necessary overlays.
  Closes issue #200.

The scope is comparable to the CGRA sprint (Vitis AI is ~160 LOC vs
Plasticine's ~170): schema ~420 LOC vs 440, YAML ~190 LOC vs 200,
loader ~440 LOC vs 455, cleanup ~30 LOC vs 25.

After this sprint closes, only TPU (5 SKUs) and DSP (10 SKUs) remain
as schema gaps; the catalog will be at 20/42 SKUs migrated (~48%).
