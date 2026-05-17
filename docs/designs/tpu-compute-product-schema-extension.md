# TPU ComputeProduct schema extension -- paper exercise

Status: draft
Date: 2026-05-17
Tracking issue: graphs#204
Predecessors:
  - GPU sprint #171 (closed)
  - CPU sprint #182 (closed)
  - NPU sprint #187 (closed)
  - CGRA sprint #196 (closed)
  - DPU sprint #200 (closed)

## Purpose

Issue #204 ("TPU mini-sprint: add TPUBlock to ComputeProduct +
migrate TPU v4") proposes a 5-PR sprint to add `TPUBlock` to the
`ComputeProduct` discriminated union, mirroring sprints #171 (GPU),
#182 (CPU), #187 (NPU), #196 (CGRA), and #200 (DPU). This document
is **PR 1 of that sprint** -- the paper exercise: enumerate Google
TPU v4's full feature set against the v6 (KPU + GPU + CPU + NPU +
CGRA + DPU) ComputeProduct schema and classify every field as
**covered**, **extends an existing field**, or **needs a new
field/type**.

The exercise uses **TPU v4** as the reference SKU because:

  - **Well-documented** -- Google's ISCA / HotChips papers cover
    architecture in detail
  - **Modern process + memory** -- 7nm + HBM2e exercises the
    canonical datacenter TPU shape
  - **Mid-complexity** -- 176 LOC factory; the schema needs to
    accommodate all 5 SKUs but only v4 ships in this sprint
  - **2-MXU layout** is the most common (v3/v4/v5p share it)
  - **TSMC N7 process node already in catalog** from prior sprints

Other 4 TPU SKUs (v1, v3, v5p, edge_pro) follow as **pure data PRs**
once this sprint's schema lands. Their requirements informed the
schema design (see the "Multi-SKU coverage" section below).

The doc deliberately copies the structure of the prior 5 paper
exercises so the diff between the six designs is easy to spot.

## Method

For each feature in the TPU v4 resource model
(`src/graphs/hardware/models/datacenter/tpu_v4.py`, 176 LOC), this
doc records:

- The field/object as it lives in graphs today (with line ref).
- Classification: COVERED / EXTEND / NEW.
- For EXTEND: the existing schema field that grows.
- For NEW: a proposed field name + brief rationale.

Out of scope for this exercise (deferred per the sprint plan):

- **TPU v4 pod topology** -- 4096-chip 3D torus interconnect. v8+
  multi-chip / system-level scope. The schema can model the
  single-chip ICI port surface; pod-level coverage comes later.
- Other TPU SKUs (v1, v3, v5p, edge_pro) -- separate follow-up YAMLs
  once the schema lands.
- Coral Edge TPU re-migration from NPU schema to TPU schema -- the
  Coral overlay (`HardwareType.TPU` set via factory) was a v6
  workaround; v8+ unification may resolve.
- DSP schema extension -- separate sprint (10 SKUs).
- Per-field provenance (`EstimationConfidence`) -- graphs-side
  feature; orthogonal to schema design.

## Reference: current v6 schema state

`embodied-schemas/src/embodied_schemas/compute_product.py`:

```python
class BlockKind(str, Enum):
    KPU = "kpu"
    GPU = "gpu"
    CPU = "cpu"
    NPU = "npu"
    CGRA = "cgra"
    DPU = "dpu"

AnyBlock = Annotated[
    Union[KPUBlock, GPUBlock, CPUBlock, NPUBlock, CGRABlock, DPUBlock],
    Field(discriminator="kind"),
]
```

Six architectures shipped. By this point the cross-block-kind reuse
pattern is well-established:
  - `ClockDomain` shared (gpu_block <- cpu_block)
  - `DataConfidence` shared (process_node <- cgra_block, dpu_block, npu_block)
  - `MemoryType` shared (gpu <- npu_block, cgra_block, dpu_block)
  - `CircuitClass` shared (process_node <- gpu/cpu/npu/cgra/dpu)

The v8 vendor-neutral `compute_block_common` module is overdue but
not blocking; this TPU sprint follows the established per-architecture-
types convention. v8 sprint can collapse all 7 architectures' shared
primitives in one pass.

## TPU architectural primer (for reviewers not familiar)

Google TPU is a **datacenter-class systolic AI accelerator**.
Key concepts:

- **MXU (Matrix Multiplier Unit)** -- the canonical TPU systolic
  array. v1 had a single 256x256 MXU; v3/v4/v5p have 2x (128x128).
  Edge TPU Pro has 1x (128x128). Each MXU MACs at clock frequency;
  per-MXU peak = rows * cols * 2 ops * clock.
- **Unified Buffer (UB)** -- large on-chip SRAM (32 MiB on v4) that
  serves dual roles: weight + activation staging, and inter-MXU
  scratchpad. Replaces conventional L1/L2 distinction; TPU's "L2"
  is collapsed into the UB.
- **Accumulator** -- per-MXU 32-bit accumulator (2 MiB on v4) sized
  to hold the output of one full systolic pass at the roofline knee.
- **Weight FIFO** -- per-MXU weight prefetch buffer (small; tens of
  KiB) that decouples DRAM bandwidth from MXU throughput.
- **HBM / HBM2 / HBM3** -- High-Bandwidth Memory stacks. TPU v1 used
  DDR3 (an early design choice that bottlenecked it); v3+ shifted to
  HBM. Bandwidth scales 30 GB/s (v1 DDR3) → 1.2 TB/s (v4 HBM2e) →
  ~2 TB/s+ (v5p HBM3).
- **ICI (Inter-Chip Interconnect)** -- TPU-specific multi-chip fabric
  used to build pods. TPU v4 pod = 4096 chips in 3D torus. Single-chip
  view: each chip has ICI ports + per-port bandwidth. Pod topology is
  deferred to v8+.
- **Tile energy model** -- TPU community uses a fine-grained energy
  decomposition (weight FIFO + accumulator + UB read/write + MAC) for
  architectural analysis. Lives in graphs as `TPUTileEnergyModel`.

The architectural family includes:
- TPU v1 (Norm Jouppi et al., ISCA 2017 reference)
- TPU v2, v3, v4 (BF16 + HBM, scaling)
- TPU v5p / Trillium (newest; 4nm; HBM3)
- Coral Edge TPU (different IP family; lives in NPU schema for now)
- Hypothetical "TPU Edge Pro" (graphs-only hypothetical extension)

## Multi-SKU coverage (schema must support all 5)

The schema design must handle these variations across the 5 SKUs
this sprint targets / will target:

| Aspect | tpu_v1 | tpu_v3 | **tpu_v4** | tpu_v5p | tpu_edge_pro |
|---|---|---|---|---|---|
| Process | 28nm | 16nm | **7nm** | 4nm | 7nm |
| MXU count | 1 | 2 | **2** | 2+ | 1 |
| MXU dim | 256x256 | 128x128 | **128x128** | 128x128 | 128x128 |
| Clock | 700 MHz | 940 MHz | **1.05 GHz** | 1.1 GHz | 850 MHz |
| TDP | 75W | 200W | **350W** | 400W | 15/30/45W (3 profiles) |
| Memory | DDR3 | HBM | **HBM2e** | HBM3 | LPDDR4X |
| Form factor | datacenter | datacenter | **datacenter** | datacenter | edge SoM |
| DVFS | none | none | **none** | none | yes (3 profiles!) |
| Cooling | active-fan | active-liquid | **active-liquid** | active-liquid | passive |

**Design implications**:
1. ``num_mxus`` ranges 1-2 (forward-compat for 4+ on future SKUs)
2. ``mxu_dim_rows`` / ``mxu_dim_cols`` vary 128 or 256 (could grow)
3. ``MemoryType`` enum must include DDR3 + HBM + HBM2 + HBM3 + LPDDR4X
   (all already in the existing `MemoryType` enum from `gpu` module --
   no new memory types needed)
4. Thermal profile shape needs to accommodate single-profile (most)
   AND multi-profile DVFS (tpu_edge_pro) -- the GPU sprint's
   `ClockDomain` reuse handles this elegantly

## Feature-by-feature audit

### 1. Identity / packaging

| Field (graphs)                | Verdict  | Mapping                                |
|-------------------------------|----------|----------------------------------------|
| `name="TPU-v4"`               | COVERED  | `ComputeProduct.name`                  |
| `hardware_type=HardwareType.TPU` | EXTEND | `Die.blocks[0].kind = BlockKind.TPU` (new enum value). `HardwareType.TPU` already exists -- no graphs-side enum gap. |
| (no vendor today)             | COVERED  | `ComputeProduct.vendor = "google"` (already exists from Coral; no new vendor directory) |
| (no product family today)     | COVERED  | `Market.product_family = "TPU"`        |
| Datacenter packaging          | EXTEND   | `Packaging(kind=MONOLITHIC, package_type="datacenter_oam")`. Open Accelerator Module form factor used by datacenter TPUs. |

### 2. Process node

| Field (graphs)                          | Verdict | Mapping |
|-----------------------------------------|---------|---------|
| `process_node_nm=7` (TSMC N7)           | COVERED | `Die.process_node_id = "tsmc_n7"` -- already in catalog from KPU sprint. **No new node YAML needed for v4.** v1 follow-up will need TSMC 28nm node (different from existing GF 28nm). |

### 3. MXU hierarchy (TPU-specific replacement for SM / core / tile / dataflow_unit / PCU / AIE tile)

TPUs introduce the MXU abstraction. v4 has 2 MXUs, each 128x128.
This is fundamentally different from prior architectures: an MXU is
a **single large systolic block** rather than a grid of small units.

| Field (graphs)                       | Verdict | Mapping                          |
|--------------------------------------|---------|----------------------------------|
| `compute_units=2` (MXUs)             | NEW     | `TPUBlock.num_mxus: int`         |
| `threads_per_unit=128*128=16384` (MACs/MXU) | NEW | (Derived from `mxu_dim_rows * mxu_dim_cols`) |
| `warps_per_unit=128`, `warp_size=128` | NEW | `TPUBlock.mxu_dim_rows: int`, `TPUBlock.mxu_dim_cols: int` -- the systolic array dimensions. TPU v1 had 256x256; v3+ uses 128x128. |
| (no cluster concept)                 | -       | TPU MXUs are homogeneous within a chip. Heterogeneous-MXU SKUs would be a v8+ extension. |

### 4. Compute fabric (single systolic fabric on most TPUs)

TPU v4 has ONE compute fabric (the MXU array). Even though there
are 2 MXUs, they share the same fabric type. Future TPUs with
heterogeneous compute (e.g. AIE-style mixed precision tiles) would
carry multiple fabrics; not common today.

| Field (graphs)                                            | Verdict | Mapping |
|-----------------------------------------------------------|---------|---------|
| `fabric_type="systolic_array"`                            | NEW     | `TPUComputeFabric.fabric_kind: TPUFabricKind` enum (TPU_V1_STYLE / TPU_V2_PLUS / TPU_HD). Initial value: TPU_V2_PLUS for v4 (and v3/v5p/edge_pro). |
| `circuit_type="standard_cell"`                            | EXTEND  | Reuse `CircuitClass` from `process_node` (5th cross-block-kind reuse). |
| `num_units=2*128*128`                                     | -       | Derived from `num_mxus * mxu_dim_rows * mxu_dim_cols`. The schema carries the dimensions; the loader derives the legacy `num_units`. |
| `ops_per_unit_per_clock: {BF16: 2, INT8: 2}`              | NEW     | `TPUComputeFabric.ops_per_unit_per_clock: dict[str, int]` (same shape as GPU/CPU/NPU/CGRA/DPU). Per-MAC ops; the schema's "unit" is the MAC, not the MXU. |
| `core_frequency_hz=1.05e9`                                | COVERED | Lives in the chip-level thermal profile. |
| `energy_per_flop_fp32=get_base_alu_energy(7, 'standard_cell')` (1.8 pJ) | NEW | `TPUComputeFabric.energy_per_op_bf16_pj: float` -- TPUs are BF16-dominant for training (unlike NPU's INT8-dominant inference). |
| `energy_scaling: {FP32: 1.0, BF16: 0.5, INT8: 0.125}`     | NEW     | `TPUComputeFabric.energy_scaling: dict[str, float]` -- scaling relative to the BF16 baseline (TPU-specific; mirrors NPU's INT8 baseline). |

### 5. Tile energy model (TPU-specific architectural decomposition)

TPU community uses a fine-grained energy decomposition unique to the
systolic-array + weight-FIFO + accumulator + UB architecture. The
graphs side already has ``TPUTileEnergyModel`` for this; the schema
needs to capture its key dimensions so the loader can reconstruct
the model.

| Field (graphs)                          | Verdict | Mapping |
|-----------------------------------------|---------|---------|
| `array_width=128, array_height=128`     | EXTEND  | Lives on `TPUBlock.mxu_dim_rows / .mxu_dim_cols` (covered in section 3). |
| `num_arrays=2`                          | EXTEND  | Lives on `TPUBlock.num_mxus`. |
| `weight_tile_size=32 KiB`               | NEW     | `TPUBlock.weight_tile_size_kib: int` -- per-MXU weight prefetch buffer size. |
| `weight_fifo_depth=2`                   | NEW     | `TPUBlock.weight_fifo_depth: int` -- number of tiles buffered (decouples DRAM BW from MXU throughput). |
| `pipeline_fill_cycles=128`              | NEW     | `TPUBlock.pipeline_fill_cycles: int` -- systolic pipeline fill latency (typically = MXU dim). |
| `accumulator_size=2 MiB per MXU`        | NEW     | `TPUBlock.accumulator_size_kib_per_mxu: int` |
| `accumulator_width=128`                 | EXTEND  | Lives on `TPUBlock.mxu_dim_cols` (same as systolic array width by construction). |
| `unified_buffer_size=32 MiB`            | NEW     | `TPUBlock.unified_buffer_size_kib: int` -- the canonical TPU UB. Replaces conventional L1/L2 distinction. |
| per-byte energies (weight memory, FIFO, UB, accumulator, MAC) | NEW | `TPUBlock.energy_*` dict OR scalar fields. **Recommendation**: ship as a `TPUTileEnergyCoefficients` sub-type with the canonical 9 coefficients (mac_energy, weight_memory_energy_per_byte, weight_fifo_energy_per_byte, unified_buffer_read/write_energy_per_byte, accumulator_read/write_energy_per_element, weight_shift_in_energy_per_element, activation_stream_energy_per_element). |

**Recommendation**: define `TPUTileEnergyCoefficients` as a separate
sub-type so the loader can construct the `TPUTileEnergyModel`
cleanly. The legacy graphs `TPUTileEnergyModel` has more fields than
the schema needs (e.g. derived values); the schema captures only the
raw coefficients.

### 6. Memory subsystem (Unified Buffer + HBM)

This is where TPUs depart most from prior architectures. The
Unified Buffer (UB) IS the on-chip memory hierarchy -- there's no
distinct L1/L2 in the GPU/CPU sense. HBM is the off-chip memory tier.

| Field (graphs)                          | Verdict | Mapping |
|-----------------------------------------|---------|---------|
| `peak_bandwidth=1.2e12` (HBM2e)         | EXTEND  | `TPUMemorySubsystem.external_dram_bandwidth_gbps: float = 1200`. NPU-style "DRAM tier" (TPUs are HBM-bound for training). |
| `main_memory=32 GB` (HBM2e)             | EXTEND  | `TPUMemorySubsystem.has_external_dram: bool = True` + `external_dram_size_gb: float`. |
| `l1_cache_per_unit=16 MB per MXU`       | EXTEND  | Captured by `TPUBlock.unified_buffer_size_kib` (the UB is per-chip, not per-MXU; the legacy split was an artifact). Derived value: `unified_buffer_size_kib // num_mxus`. |
| `l2_cache_total=32 MB`                  | EXTEND  | Same -- the UB IS the L2 (TPUs collapse L1+L2 into the UB). The legacy split was redundant. |
| `l3_present=False`                      | -       | TPUs have no L3. |
| `coherence_protocol="none"`             | EXTEND  | TPU dataflow is compiler-routed; no coherence. |
| `memory_technology="HBM2e"`             | EXTEND  | `TPUMemorySubsystem.external_dram_type: MemoryType` -- reuse `MemoryType` enum (includes HBM/HBM2/HBM3/DDR3/LPDDR4X -- covers all 5 SKUs). |
| `energy_per_byte=10e-12` (HBM2e)        | NEW     | `TPUMemorySubsystem.external_dram_access_energy_pj_per_byte: float` |
| (no on-chip BW field today)             | NEW     | `TPUMemorySubsystem.on_chip_bandwidth_gbps: float` -- UB-to-MXU streaming BW (typically >>HBM BW). |
| (no UB-specific energy today)           | NEW     | `TPUMemorySubsystem.unified_buffer_access_energy_pj_per_byte: float` -- ~0.5 pJ/B for on-chip SRAM. |

**Recommendation**: define `TPUMemorySubsystem` with the UB as the
primary on-chip tier (no L1/L2 split). The `has_external_dram` flag
gates the HBM fields (NPU-style). This is a structurally cleaner
representation than the legacy's awkward L1-per-MXU split.

### 7. Inter-Chip Interconnect (ICI) -- single-chip surface

TPUs build pods via the ICI fabric -- TPU v4 pod = 4096 chips in a
3D torus. **Pod-level topology is v8+ scope.** This sprint captures
only the single-chip ICI surface: how many ports, what bandwidth
per port.

| Field (graphs)                          | Verdict | Mapping |
|-----------------------------------------|---------|---------|
| (no ICI modeling today)                 | NEW     | `TPUBlock.ici_port_count: int` -- e.g., 6 for v4 (3D torus = 6 neighbors) |
| (no ICI BW today)                       | NEW     | `TPUBlock.ici_bandwidth_per_port_gbps: float` -- per-port BW; v4 ~400 GB/s |
| (no ICI topology today)                 | NEW     | `TPUBlock.ici_topology_hint: Optional[str]` -- free-form ("3d_torus_2x2x2", "2d_torus_4x4") for v8+ pod modeling; optional. |

### 8. On-die fabric (UB-to-MXU streaming)

| Field (graphs)                          | Verdict | Mapping |
|-----------------------------------------|---------|---------|
| (no on-die fabric today)                | NEW     | `TPUOnDieFabric` with topology (always `CROSSBAR` for single-MXU; `MULTI_CROSSBAR` for 2+ MXUs sharing UB), bandwidth, latency. Simpler than NPU/CGRA/DPU NoCs because the UB-to-MXU connection is direct. |

### 9. Thermal profile

TPU v4 has a single 350W operating point (no DVFS). tpu_edge_pro
has 3 (15W/30W/45W) -- the schema needs both.

| Field (graphs)                              | Verdict | Mapping |
|---------------------------------------------|---------|---------|
| `name="default"`                            | COVERED | `TPUThermalProfile.name` |
| `tdp_watts=350.0`                           | COVERED | `TPUThermalProfile.tdp_watts` |
| `cooling_solution="active-liquid"`          | EXTEND  | `TPUThermalProfile.cooling_solution_id` -- references `data/cooling-solutions/`. `liquid_cooled.yaml` exists. |
| `dvfs_enabled=False`                        | NEW     | `TPUThermalProfile.dvfs_enabled: bool = False` |
| `clock_mhz=1050` (single value)             | EXTEND  | Scalar `clock_mhz` (mirrors NPU/CGRA/DPU shape; not ClockDomain). For tpu_edge_pro's multi-profile case, each profile has its own scalar clock. |
| per-precision efficiency                    | EXTEND  | `TPUThermalProfile.efficiency_factor_by_precision: dict[str, float]` |
| (no instruction efficiency today)           | NEW     | `TPUThermalProfile.instruction_efficiency_by_precision: dict[str, float]` |
| (no memory bottleneck factor today)         | NEW     | `TPUThermalProfile.memory_bottleneck_factor_by_precision: dict[str, float]` |

### 10. TPU-specific scheduler attributes

| Field (graphs)               | Verdict | Mapping                          |
|------------------------------|---------|----------------------------------|
| `min_occupancy=0.5`          | NEW     | `TPUBlock.min_occupancy: float = 0.5` -- TPUs need high systolic utilization to be efficient; higher default than NPU/CGRA/DPU. |
| `max_concurrent_kernels=1`   | NEW     | `TPUBlock.max_concurrent_models: int = 1` -- TPUs typically run one large model/batch at a time. |
| `wave_quantization=1`        | NEW     | `TPUBlock.wave_quantization: int = 1` |
| (no XLA program structure)   | -       | XLA HLO program structure is a runtime/compiler concern; not in the schema. |

### 11. Performance roll-up

TPU v4: BF16 275 TFLOPS, INT8 550 TOPS (2x BF16), FP32 137.5 TFLOPS
(1/2 BF16, emulated). The schema's per-precision dict generalizes.

| Field (graphs)                                  | Verdict | Mapping |
|-------------------------------------------------|---------|---------|
| `peak_ops_per_sec` (per Precision)              | EXTEND  | `TPUTheoreticalPerformance.peak_ops_per_sec_by_precision: dict[str, float]` -- same shape as GPU/CPU/NPU/CGRA/DPU. TPU's distinctive pattern is **BF16-dominant + INT8 at 2x BF16** (training-first design). |

### 12. Silicon bin (per-block transistor decomposition)

Google publishes overall die size + transistor counts; per-block
breakdowns are not published. Estimates for v4 (~7nm, ~600 mm²,
~50B transistors):

- `mxu_array` -- 2 MXUs of 128x128 each
- `accumulator_sram` -- per-MXU accumulator (2 MiB)
- `unified_buffer_sram` -- 32 MiB shared UB
- `weight_fifo_sram` -- small per-MXU FIFO
- `hbm_phy` -- HBM2e controller + PHY (4 stacks)
- `ici_phy` -- ICI port PHYs (6 ports for 3D torus)
- `control_logic` -- XLA dispatch + scheduler

**Verdict**: NO schema change needed -- just document the convention.

### 13. BOM cost profile

TPU v4 is not commercially sold (Google Cloud rental at ~$8/chip-hour).
No BOM data populated in graphs today. Defer to v8 `Market.bom`.

## Summary table

| Category                       | Total fields | COVERED | EXTEND | NEW |
|--------------------------------|-------------:|--------:|-------:|----:|
| Identity / packaging           |       5      |    3    |   2    |  0  |
| Process node                   |       1      |    1    |   0    |  0  |
| MXU hierarchy                  |       4      |    0    |   1    |  3  |
| Compute fabric                 |       7      |    1    |   2    |  4  |
| Tile energy model              |       8      |    0    |   2    |  6  |
| Memory subsystem               |       9      |    0    |   6    |  3  |
| ICI surface                    |       3      |    0    |   0    |  3  |
| On-die fabric                  |       1      |    0    |   0    |  1  |
| Thermal profile                |       7      |    2    |   2    |  3  |
| Scheduler attributes           |       3      |    0    |   1    |  2  |
| Performance roll-up            |       1      |    0    |   1    |  0  |
| Silicon bin                    |       0      |    0    |   0    |  0 (new count_ref strings only) |
| BoM cost                       |       0      |    0    |   0    |  0 (defer overlay) |
| **Totals**                     |    **49**    | **7**   | **17** |**25**|

Of 49 fields, ~14% (7) are already covered, ~35% (17) extend an
existing schema field, and ~51% (25) are net-new. The NEW count is
the highest of any sprint (NPU=22, CGRA=26, DPU=25, **TPU=25**),
driven by three TPU-unique concept clusters:

  - **Tile energy model** (6 fields) -- TPU-specific systolic
    energy decomposition (FIFO + accumulator + UB + MAC)
  - **ICI surface** (3 fields) -- multi-chip interconnect ports
  - **Unified Buffer concept** -- collapses L1/L2 split that prior
    architectures preserve

Cross-block-kind type reuse (fifth pattern, after CPU/NPU/CGRA/DPU):
- `TPUMemorySubsystem.external_dram_type` reuses `MemoryType`
- `TPUOnDieFabric.confidence` reuses `DataConfidence`
- `TPUComputeFabric.circuit_class` reuses `CircuitClass`

**By this point the v8 vendor-neutral compute_block_common
unification is so obviously overdue that the recommendation here is:
ship TPUBlock following the existing per-architecture-types convention,
then do v8 unification immediately after as the next sprint (collapsing
all 7 architectures' shared primitives in one pass).**

## Recommended schema diff for the next PR

The Schema PR (PR 2 of the sprint) should land roughly the following
in embodied-schemas:

1. **`compute_product.py`**:
   - Add `BlockKind.TPU = "tpu"` to the discriminator enum.
   - Update `AnyBlock = Annotated[Union[..., TPUBlock], Field(discriminator="kind")]`

2. **New file `embodied_schemas/tpu_block.py`** (~500 LOC):
   - `TPUBlock` with: `num_mxus`, `mxu_dim_rows`, `mxu_dim_cols`,
     `weight_tile_size_kib`, `weight_fifo_depth`, `pipeline_fill_cycles`,
     `accumulator_size_kib_per_mxu`, `unified_buffer_size_kib`,
     `ici_port_count`, `ici_bandwidth_per_port_gbps`,
     `ici_topology_hint: Optional[str]`,
     `compute_fabrics: list[TPUComputeFabric]`,
     `tile_energy_coefficients: TPUTileEnergyCoefficients`,
     `memory: TPUMemorySubsystem`, `noc: TPUOnDieFabric`,
     `min_occupancy`, `max_concurrent_models`, `wave_quantization`,
     `multi_precision_alu`.
   - `TPUComputeFabric` with: `fabric_kind: TPUFabricKind`,
     `circuit_class`, `ops_per_unit_per_clock`,
     `energy_per_op_bf16_pj`, `energy_scaling`.
   - `TPUTileEnergyCoefficients` with: 9 canonical energy fields
     (mac_energy, weight_memory_energy_per_byte, weight_fifo_energy_per_byte,
     unified_buffer_read/write_energy_per_byte,
     accumulator_read/write_energy_per_element,
     weight_shift_in_energy_per_element,
     activation_stream_energy_per_element) -- all in pJ.
   - `TPUMemorySubsystem` with: `on_chip_bandwidth_gbps`,
     `unified_buffer_access_energy_pj_per_byte`,
     `has_external_dram: bool`, `external_dram_type: Optional[MemoryType]`,
     `external_dram_size_gb: Optional[float]`,
     `external_dram_bandwidth_gbps: Optional[float]`,
     `external_dram_access_energy_pj_per_byte`,
     `coherence_protocol`.
   - `TPUOnDieFabric` with: `topology: TPUNoCTopology`,
     `bisection_bandwidth_gbps`, `unit_count` (= num_mxus),
     `flit_size_bytes`, `hop_latency_ns`, `pj_per_flit_per_hop`,
     `routing_distance_factor`, `confidence: DataConfidence`.
   - `TPUThermalProfile` with: `name`, `tdp_watts`,
     `cooling_solution_id`, `clock_mhz`, `dvfs_enabled`, per-
     precision dicts.
   - `TPUTheoreticalPerformance` with `peak_ops_per_sec_by_precision`.
   - Enums: `TPUFabricKind` (TPU_V1_STYLE / TPU_V2_PLUS / TPU_HD),
     `TPUNoCTopology` (CROSSBAR / MULTI_CROSSBAR).

3. **Validators on `TPUBlock`**:
   - `_validate_noc_unit_count_matches`: `noc.unit_count == num_mxus`
     (mirrors NPU/CGRA/DPU pattern).
   - `_validate_external_dram_consistency`: NPU-style.
   - `_validate_mxu_dimensions`: `mxu_dim_rows == mxu_dim_cols` for
     v1 (square) and v2+ (canonical 128x128 or 256x256).
   - `_validate_ici_ports`: if `ici_topology_hint` set, `ici_port_count`
     must be consistent (e.g., 6 for 3d_torus).

4. **Process node**: no new YAML needed for v4 (TSMC N7 already in
   catalog).

5. **Cooling solution**: `liquid_cooled.yaml` already in catalog.

## Risks called out by this exercise

1. **`HardwareType.TPU` already exists on the graphs side.** Unlike
   the NPU case (where #191 had to add the enum value), TPU is
   already a HardwareType (Coral uses it via factory overlay, as
   noted in #195). The loader sets it directly with no transitional
   period.

2. **Coral Edge TPU schema confusion.** Coral currently uses
   `NPUBlock` (per the Coral YAML in embodied-schemas#33) but the
   graphs factory overlays `hardware_type=HardwareType.TPU` so
   TPUMapper accepts it. After TPUBlock lands, Coral COULD migrate to
   TPUBlock, but that's a v8+ concern. The current overlay continues
   to work; the parity test for the Coral cleanup (#199) documented
   the mismatch as a v7 reconciliation item.

3. **TPU v4 pod topology is out of scope.** The single-chip schema
   captures ICI port count + per-port bandwidth, but the full pod
   3D-torus topology requires multi-chip / system-level modeling.
   Pod-level support is v8+.

4. **TPU v1 needs TSMC 28nm, which is NOT in catalog.** GF 28nm SLP
   landed in embodied-schemas#32 (for Coral), but TPU v1 was TSMC
   28nm (a different node). The v1 follow-up data PR will need to
   add `data/process-nodes/tsmc/n28.yaml` first.

5. **TPU Edge Pro is hypothetical.** The legacy `tpu_edge_pro.py`
   (~408 LOC) describes a hypothetical 30W edge TPU; not a real
   shipping product. The YAML should clearly flag `is_hypothetical:
   true` (or use the engineering_sample lifecycle from Plasticine's
   precedent).

6. **TPU Edge Pro has 3 thermal profiles.** The schema's
   `TPUThermalProfile` supports per-profile clock + TDP, mirroring
   the GPU sprint's ClockDomain reuse. Verify the multi-profile case
   end-to-end in the v4 sprint so the edge_pro follow-up YAML lands
   smoothly.

7. **The tile energy coefficient set is TPU-specific.** No other
   architecture in the catalog has weight FIFOs + accumulator
   per-element energies. The `TPUTileEnergyCoefficients` sub-type
   is intentionally architecture-specific; v8 unification will not
   try to generalize it.

## Next step

Sprint sequencing:

- **PR 1 (this document)** -- docs/design, graphs-side. Lands the
  field-by-field audit + recommended schema diff. Pure docs PR.
- **PR 2 -- schema, embodied-schemas-side.** Implements the
  "Recommended schema diff" above (TPUBlock + supporting types).
- **PR 3 -- data, embodied-schemas-side.** Authors the first TPU v4
  YAML at `data/compute_products/google/tpu_v4.yaml` (joins Coral in
  the existing `google/` vendor directory).
- **PR 4 -- loader, graphs-side.** Implements `tpu_yaml_loader` plus
  parity test against the hand-coded factory.
- **PR 5 -- cleanup, graphs-side.** Retires `tpu_v4.py`'s ~176-LOC
  body to a thin loader-wrapper + any necessary overlays. Closes
  issue #204.

The scope is comparable to the DPU sprint despite TPU's higher
complexity (49 fields vs DPU's 54 -- TPU has fewer total fields
because the Unified Buffer concept collapses what would otherwise
be separate L1+L2 fields). Schema ~500 LOC, YAML ~250 LOC, loader
~500 LOC, cleanup ~30 LOC.

After this sprint closes:
- 4 follow-up YAMLs (v1, v3, v5p, edge_pro) land as pure data PRs
  with their corresponding graphs cleanups (~8 PRs total, file
  separately)
- v8 vendor-neutral `compute_block_common` unification becomes the
  next major sprint (7 architectures × consistent reuse pattern =
  clear evidence)
- Only DSP remains as a schema gap (10 SKUs; separate sprint)
