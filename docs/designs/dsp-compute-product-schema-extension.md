# DSP ComputeProduct schema extension -- paper exercise

Status: draft
Date: 2026-05-17
Tracking issue: graphs#211
Predecessors:
  - GPU sprint #171 (closed)
  - CPU sprint #182 (closed)
  - NPU sprint #187 (closed)
  - CGRA sprint #196 (closed)
  - DPU sprint #200 (closed)
  - TPU sprint #204 (closed)
  - v8 unification #208 + follow-up #210 (closed)

## Purpose

Issue `#211` ("DSP mini-sprint: add DSPBlock to ComputeProduct +
migrate Cadence Vision Q8") proposes a 5-PR sprint to add `DSPBlock`
to the `ComputeProduct` discriminated union. This is the **final**
category schema gap; after it closes, every hand-coded factory in
the catalog has a corresponding YAML-backed thin-loader path. This
document is **PR 1 of that sprint** -- the paper exercise: enumerate
Cadence Tensilica Vision Q8's full feature set against the v8
(KPU + GPU + CPU + NPU + CGRA + DPU + TPU + compute_block_common)
ComputeProduct schema and classify every field as **covered**,
**extends an existing field**, or **needs a new field/type**.

The exercise uses **Cadence Tensilica Vision Q8** as the reference
SKU because:

  - **Pure IP core** -- no SoC fabric / co-processor entanglement;
    the cleanest case to establish the schema before complex
    automotive SoCs land
  - **Single fabric** -- 1024-bit SIMD vector DSP only (no paired
    NPU / tensor co-processor); the schema for multi-fabric DSPs
    extends this baseline cleanly
  - **Smallest hand-coded factory** -- 222 LOC; comparable to the
    DPU sprint's reference SKU complexity
  - **Well-documented** -- Cadence Tensilica Vision Q8 Product
    Brief (2021) + Tensilica Vision DSP Family specifications
  - **Single thermal profile** (1W) -- multi-profile DVFS validated
    in follow-up SKUs (Qualcomm SA8775P at 20/30/45W)
  - **TSMC N16 process node already in catalog** from prior sprints

The other 9 DSP SKUs (CEVA NeuPro NPM11, Synopsys ARC EV7x, the
5 TI TDA4 / Qualcomm SA8775P automotive SoCs, the 2 Qualcomm edge
SoCs) follow as **pure data PRs** once this sprint's schema lands.
Their requirements informed the schema design (see the "Multi-SKU
coverage" section below).

The doc deliberately copies the structure of the prior 6 paper
exercises so the diff between the seven designs is easy to spot.
**One material difference from prior sprints**: DSPBlock is the
first block kind authored **after** v8 unification, so it uses
`compute_block_common`'s `TheoreticalPerformance` / `MemoryType` /
`CircuitClass` / `ClockDomain` / `DataConfidence` from day 1
rather than redefining them.

## Method

For each feature in the Cadence Vision Q8 resource model
(`src/graphs/hardware/models/ip_cores/cadence_vision_q8.py`,
222 LOC), this doc records:

- The field/object as it lives in graphs today (with line ref).
- Classification: COVERED / EXTEND / NEW.
- For EXTEND: the existing schema field that grows.
- For NEW: a proposed field name + brief rationale.

Out of scope for this exercise (deferred per the sprint plan):

- **Other 9 DSP SKUs** (CEVA NPM11, Synopsys EV7x, TI TDA4{AL,VH,
  VL,VM}, Qualcomm SA8775P, QRB5165, QCS6490) -- separate follow-up
  YAMLs once the schema lands.
- **Multi-fabric scheduling models** (which fabric runs which op,
  fabric-to-fabric handoff costs) -- runtime / mapper concern, not
  a schema concern.
- **SoC-level integration validation** (which DSPs share which
  fabric NoC with adjacent NPUs / CPUs in the same SoC) -- v9
  chiplet / system-level scope.
- **v9 NEAR_UNIFIABLE patterns** (`ThermalProfile`, `OnDieFabric`
  cross-block-kind unification) -- separate concern; orthogonal to
  this sprint.
- **`has_external_dram` vs `has_host_dram` naming reconciliation**
  -- v9 cleanup item.
- Per-field provenance (`EstimationConfidence`) -- graphs-side
  feature; orthogonal to schema design.

## Reference: current v7 schema state

`embodied-schemas/src/embodied_schemas/compute_product.py`:

```python
class BlockKind(str, Enum):
    KPU = "kpu"
    GPU = "gpu"
    CPU = "cpu"
    NPU = "npu"
    CGRA = "cgra"
    DPU = "dpu"
    TPU = "tpu"
    # DSP = "dsp"  <-- this sprint adds

AnyBlock = Annotated[
    Union[KPUBlock, GPUBlock, CPUBlock, NPUBlock,
          CGRABlock, DPUBlock, TPUBlock],
    Field(discriminator="kind"),
]
```

`embodied-schemas/src/embodied_schemas/compute_block_common.py`
(landed in v8 unification, `#208` + `#210`):

```python
# Re-exports
CircuitClass, DataConfidence  # from process_node
MemoryType                    # from gpu
ClockDomain                   # from gpu_block

# Unified type
class TheoreticalPerformance(BaseModel):
    peak_ops_per_sec_by_precision: dict[str, float]
    sparse_peak_ops_per_sec_by_precision: dict[str, float] | None
```

DSPBlock uses these from day 1; no `DSPTheoreticalPerformance`
alias is needed (though one can be defined for symmetry with the
6 other modern block kinds).

## DSP architectural primer (for reviewers not familiar)

A Digital Signal Processor (DSP) is a programmable vector / VLIW
processor optimized for streaming numerical workloads. Modern AI-
era DSPs share a small set of architectural features that distinguish
them from CPUs (lower control overhead, wider SIMD) and from NPUs
(more programmable, less specialized to a single dataflow):

  - **Wide SIMD** -- 256-bit (older), 512-bit (mainstream), 1024-bit
    (Cadence Vision Q8, Qualcomm HVX), 2048-bit (high-end CEVA).
    The SIMD lane count drives `ops_per_unit_per_clock`.
  - **VLIW issue width** -- 2 / 4 / 6 / 8 slots per cycle. Multiple
    SIMD operations + scalar control + load/store in one instruction.
    Not all DSPs are VLIW; some are pure SIMD (Cadence Vision Q8 is
    primarily SIMD-only).
  - **Multi-fabric is the rule, not the exception.** AI-era DSPs
    typically pair a vector core (HVX, C7x, CEVA-XM6) with a tensor
    co-processor (HMX, MMA, NeuPro tensor units). The vector fabric
    handles activations / pooling / pre+post-processing; the tensor
    fabric handles conv / matmul. The two share memory through a
    common SoC NoC. This is the **single largest distinction from
    TPU schema design** (which assumed a single systolic fabric).
  - **Mixed integer + float precision.** Most DSPs ship native
    INT8 / INT16 + FP16 / FP32 (some FP64 in older signal-processing
    designs). Newer ones add INT4 / FP8.
  - **Tight local memory hierarchy.** L1 D-cache (16-64 KB per core)
    + small dedicated SRAM "scratchpad" + an SoC-level shared L2
    or last-level cache. No HBM. Bandwidth comes from the host
    SoC's DDR / LPDDR controller.
  - **Lower power than NPU / GPU at peer TOPS.** 1-2W for vision
    DSP IP cores; 10-45W for automotive SoCs (split across CPU +
    GPU + DSP + NPU + ISP). DSP-only TDP fraction is hard to
    quote; the catalog uses chip-level TDP and lets the mapper
    apportion.

The "DSP" category is **less homogeneous than TPU**: it spans
licensable IP cores quoting typical-integration bandwidth (Cadence,
CEVA, Synopsys) and shipping SoCs with measured external bandwidth
(Qualcomm SA8775P, TI TDA4). The schema must distinguish these
two deployment kinds cleanly without forcing a polymorphism that
mappers have to handle.

## Multi-SKU coverage (schema must support all 10)

The schema landed in PR 2 of this sprint must accommodate all 10
DSPs in the catalog, even though only Cadence Vision Q8 ships in
the reference YAML (PR 3):

| SKU | Category | Process | Fabric(s) | Clock | TDP modes | External memory |
|---|---|---|---|---|---|---|
| **cadence_vision_q8** | **IP** | **N16** | **SIMD 32x** | **0.6-1.2 GHz** | **1W** | **40 GB/s typical** |
| ceva_neupro_npm11 | IP | N16 | tensor 64 + vector 64 | 1.0 GHz | 2W | 50 GB/s typical |
| synopsys_arc_ev7x | IP | N16 | (vector + DNN engine) | 1.0 GHz | (TBD) | (TBD) typical |
| qualcomm_sa8775p | automotive SoC | N5 | HVX 4 + HMX 2 | 2.0 GHz | 20 / 30 / 45W (3) | 90 GB/s LPDDR5 |
| ti_tda4vm | automotive SoC | N16 | C7x DSP 8 + MMAv1 1 | 1.0 GHz | 10 / 20W (2) | 60 GB/s LPDDR4x |
| ti_tda4al | automotive SoC | N16 | C7x DSP (subset) | 1.0 GHz | (TBD) | (TBD) |
| ti_tda4vh | automotive SoC | N16 | C7x DSP (subset) | 1.0 GHz | (TBD) | (TBD) |
| ti_tda4vl | automotive SoC | N16 | C7x DSP (subset) | 1.0 GHz | (TBD) | (TBD) |
| qrb5165 | edge SoC | N7 | HVX | 1.5 GHz | (TBD) | LPDDR5 |
| qualcomm_qcs6490 | edge SoC | N6 | HVX | 1.5 GHz | (TBD) | LPDDR5 |

Key requirements this matrix imposes on `DSPBlock`:

1. **`compute_fabrics: list[DSPComputeFabric]`** must accommodate
   1 (Cadence) through 2 (HVX+HMX, C7x+MMA, CEVA tensor+vector).
   Not just an optional second fabric -- a first-class list.
2. **Multi-profile thermal** (3 profiles for SA8775P, 2 for TDA4VM,
   1 for Cadence) -- same pattern as GPU / TPU multi-profile DVFS.
3. **Process node spans N5 / N6 / N7 / N16** -- already in catalog.
4. **External memory is sometimes "typical integration"** (Cadence,
   CEVA, Synopsys) and sometimes "measured platform" (SoCs). The
   `external_dram_bandwidth_gbps` field needs a sibling
   `external_dram_bandwidth_kind: typical | measured` discriminator
   so consumers can decide whether to trust it absolutely.
5. **`deployment_kind: standalone_ip | soc_integrated`** -- new
   field that mappers + cost models use to decide which energy
   coefficients apply (IP core energy is the DSP alone; SoC
   energy is the DSP fraction of total SoC TDP).
6. **Memory hierarchy** -- L1 per-unit (16-64 KB), L2 shared
   (256 KB - 8 MB), main memory bandwidth. Cadence Vision Q8
   has 32 KB L1 / 1 MB L2 / 4 GB main; SA8775P has measured DDR
   bandwidth. Cover both shapes with optional sub-fields.
7. **VLIW issue width** (4 / 6 / 8) is informative but not
   load-bearing for performance modeling (which works off
   `ops_per_unit_per_clock`); include as documentation field.

## Feature-by-feature audit

### 1. Identity / packaging

Graphs (`cadence_vision_q8.py`):
```python
name="Cadence-Tensilica-Vision-Q8"
hardware_type=HardwareType.DSP
```

Classification: **COVERED** by existing `ComputeProduct` envelope
fields (`product_id`, `vendor`, `display_name`, `lifecycle`). The
`hardware_type` field is graphs-side; the schema doesn't carry
it (the block kind discriminator carries it). `HardwareType.DSP`
already exists in the graphs enum (no `#191`-style addition
needed; the graphs side has had this since the first DSP
factory landed).

### 2. Process node

Graphs (`cadence_vision_q8.py`, line 88):
```python
process_node_nm=16,              # 16nm (typical for vision DSP IP)
```

Classification: **COVERED**. TSMC N16 already in catalog
(`data/process-nodes/tsmc/n16.yaml`); the YAML references via
`process_node_id: "tsmc.n16"`. For N5 / N6 / N7 follow-ups,
existing catalog entries cover.

### 3. Multi-fabric architecture (DSP-specific shape)

Graphs (`cadence_vision_q8.py`, lines 77-96):
```python
simd_fabric = ComputeFabric(
    fabric_type="vision_q8_simd",
    circuit_type="simd_packed",
    num_units=32,
    ops_per_unit_per_clock={INT8: 119, INT16: 119, FP32: 4, FP16: 8},
    core_frequency_hz=1.0e9,
    process_node_nm=16,
    energy_per_flop_fp32=...,
    energy_scaling={...},
)
```

Classification: **NEW**: `DSPComputeFabric` sub-type with explicit
**DSP fabric kind** discriminator. Unlike TPU's single systolic
fabric, DSPs canonically have 1 or 2 fabrics with different
**roles**: vector (general SIMD) vs tensor (matrix accelerator).
The schema models this with:

```python
class DSPFabricKind(str, Enum):
    VECTOR_SIMD       = "vector_simd"        # Cadence, CEVA vector, HVX
    TENSOR_MATRIX     = "tensor_matrix"      # MMA, HMX, NeuPro tensor
    VLIW_SCALAR       = "vliw_scalar"        # C7x scalar slots
    HYBRID            = "hybrid"             # rare; explicit
```

Rationale: every DSP in the catalog falls into 1 or 2 of these.
The discriminator lets mappers route ops correctly (matmul -> tensor,
conv activation -> vector) without parsing fabric_type strings.

### 4. Compute fabric list (always plural, can be length 1)

Classification: **NEW**: `compute_fabrics: list[DSPComputeFabric]`.

**Justification for list vs optional pair**:
  - Cadence Vision Q8: 1 fabric (SIMD only).
  - CEVA NeuPro NPM11: 2 fabrics (vector + tensor).
  - Qualcomm SA8775P: 2 fabrics (HVX vector + HMX tensor).
  - TI TDA4VM: 2 fabrics (C7x DSP + MMAv1 tensor).
  - Future 3-fabric DSPs (vector + tensor + scalar VLIW): list
    handles cleanly.

Validator: `len(compute_fabrics) >= 1`. No upper bound (lets
hypothetical 3-fabric SKUs land without schema change).

### 5. Deployment kind (IP vs SoC-integrated)

No direct field in graphs today; the distinction is encoded by
file location (`ip_cores/` vs `automotive/` vs `edge/`).

Classification: **NEW**: `deployment_kind: DSPDeploymentKind`.

```python
class DSPDeploymentKind(str, Enum):
    STANDALONE_IP   = "standalone_ip"   # Cadence, CEVA, Synopsys
    SOC_INTEGRATED  = "soc_integrated"  # all 7 SoC SKUs
```

Consumers (mappers, cost models, energy accounting) need this
discriminator to decide:
  - whether the external DRAM bandwidth is `typical` or `measured`
  - whether to apportion SoC TDP across CPU/GPU/DSP/NPU
  - whether the DSP can run as a standalone benchmark target

### 6. Memory subsystem

Graphs (`cadence_vision_q8.py`, lines 198-201):
```python
peak_bandwidth=40e9,             # 40 GB/s (typical SoC integration)
l1_cache_per_unit=32 * 1024,     # 32 KB per unit
l2_cache_total=1 * 1024 * 1024,  # 1 MB shared cache
main_memory=4 * 1024**3,         # Up to 4 GB
```

Classification: **NEW**: `DSPMemorySubsystem`.

Field set:
  - `l1_size_bytes_per_unit: int`              # 32 KB for Q8
  - `l2_size_bytes_total: int | None`          # 1 MB for Q8
  - `l2_bandwidth_gbps: float | None`          # on-chip L2 BW
  - `has_external_dram: bool`                  # always True for SoC; True for typical-int IP
  - `external_dram_type: MemoryType | None`    # LPDDR4x / LPDDR5 / DDR4
  - `external_dram_bandwidth_gbps: float | None`
  - `external_dram_bandwidth_kind: Literal["typical", "measured"]`
  - `external_dram_size_gb: float | None`      # typical max for IP cores; spec'd for SoCs
  - `external_dram_access_energy_pj_per_byte: float`
  - `coherence_protocol: CoherenceProtocol | None`

Reuses `MemoryType` from `compute_block_common` (v8). New value
`LPDDR4X` may be needed (verify against existing enum members).

### 7. Energy coefficients (per-fabric + chip-level)

Graphs (`cadence_vision_q8.py`, lines 89, 204-211):
```python
energy_per_flop_fp32=get_base_alu_energy(16, 'simd_packed'),  # 2.43 pJ
energy_per_byte=12e-12,           # 12 pJ/byte
energy_scaling={INT8: 0.15, INT16: 0.15, FP32: 1.0, FP16: 0.50}
```

Classification: **EXTEND** `DSPComputeFabric` to include the
per-fabric energy_per_op + per-precision scaling already standardized
by NPU/CGRA/DPU/TPU. **No DSP-specific tile energy decomposition**
-- DSPs don't have weight FIFOs / unified buffers like TPUs.

Chip-level `energy_per_byte` lives on `DSPMemorySubsystem`
(`external_dram_access_energy_pj_per_byte`).

### 8. Clock domain (DVFS)

Graphs (`cadence_vision_q8.py`, lines 101-106):
```python
ClockDomain(
    base_clock_hz=600e6,
    max_boost_clock_hz=1.2e9,
    sustained_clock_hz=1.0e9,
    dvfs_enabled=True,
)
```

Classification: **COVERED**. `ClockDomain` re-exported from
`compute_block_common` (v8). DSPBlock embeds 1 ClockDomain at the
chip level + optionally per-thermal-profile clocks (mirrors TPU
multi-profile pattern for SA8775P).

### 9. Thermal profile(s)

Graphs (`cadence_vision_q8.py`, lines 120-153):
```python
thermal_1w = ThermalOperatingPoint(
    name="1W-vision",
    tdp_watts=1.0,
    cooling_solution="passive-mobile",
    performance_specs={...},
)
thermal_operating_points={"1W": thermal_1w}
default_thermal_profile="1W"
```

Classification: **EXTEND** existing per-block-kind `ThermalProfile`
pattern: `DSPThermalProfile` with:
  - `name: str`
  - `tdp_watts: float`
  - `cooling_solution_id: str`
  - `clock_hz: int | None` (overrides chip-level ClockDomain.sustained)
  - `dvfs_enabled: bool`

Cardinality: 1 for Cadence Q8 / CEVA / Synopsys; 2 for TDA4VM;
3 for SA8775P. `thermal_profiles: list[DSPThermalProfile]` with
`default_thermal_profile_name: str` (matches the SA8775P / GPU
multi-profile pattern).

v9 deferral: `ThermalProfile` as a vendor-neutral type in
`compute_block_common`. Today each block kind defines its own;
the field shapes are 90% identical and a v9 unification will
collapse.

### 10. DSP-specific scheduler attributes

Graphs (`cadence_vision_q8.py`, lines 213-215):
```python
min_occupancy=0.70,
max_concurrent_kernels=4,
wave_quantization=4,
```

Classification: **COVERED** by existing per-block-kind scheduler
attribute pattern. Add to `DSPBlock`:
  - `min_occupancy: float`         (default 0.70 for DSPs)
  - `max_concurrent_kernels: int`  (DSPs typically run 1-4 kernels)
  - `wave_quantization: int`       (SIMD lane group; informs roofline)
  - `vliw_issue_width: int | None` (informational; default None)

### 11. Performance roll-up

Graphs (`cadence_vision_q8.py`, lines 172-194):
```python
precision_profiles={
    INT8: PrecisionProfile(peak_ops_per_sec=3.8e12, ...),
    INT16: PrecisionProfile(peak_ops_per_sec=3.8e12, ...),
    FP32: PrecisionProfile(peak_ops_per_sec=129e9, ...),
}
default_precision=Precision.INT8
```

Classification: **COVERED** by v8 unified `TheoreticalPerformance`
from `compute_block_common`. DSPBlock embeds:
  - `theoretical_performance: TheoreticalPerformance`
  - `default_precision: str`

No `DSPTheoreticalPerformance` alias is strictly needed -- DSPBlock
can reference `TheoreticalPerformance` directly. **First block kind
to do so**, simplifying the design surface.

(For symmetry with the other 6 modern block kinds, the schema may
add `DSPTheoreticalPerformance = TheoreticalPerformance` alias --
zero-cost backward-compat hook for any future caller; decide in
PR 2.)

### 12. Silicon bin (per-block transistor decomposition)

Graphs: not modeled at the block level today. Other block kinds
(NPU, CGRA, DPU, TPU) carry an optional `silicon_bin` for area
budget exercises.

Classification: **EXTEND** with optional `silicon_bin: SiliconBin
| None` field mirroring TPU/DPU pattern. Default None for DSP
SKUs (IP cores rarely publish transistor counts; SoCs publish
total die but rarely DSP fraction).

### 13. BOM cost profile

Graphs (`cadence_vision_q8.py`): not encoded. (Most DSP factories
omit BOM.)

Classification: **EXTEND** with optional `bom_cost_profile:
BOMCostProfile | None`. Default None. SoC SKUs may populate
chip-level cost; IP cores typically don't (licensing model differs).

## Summary table

| Feature | Class | Field / type |
|---|---|---|
| Identity | COVERED | `ComputeProduct` envelope |
| Process node | COVERED | catalog refs (`tsmc.n5` / `n6` / `n7` / `n16`) |
| Multi-fabric | NEW | `compute_fabrics: list[DSPComputeFabric]`, `DSPFabricKind` enum |
| Fabric (per-entry) | NEW | `DSPComputeFabric` (kind, circuit_class, ops, energy) |
| Deployment kind | NEW | `deployment_kind: DSPDeploymentKind` |
| Memory subsystem | NEW | `DSPMemorySubsystem` (L1/L2/external DRAM, bandwidth-kind) |
| Energy (per fabric) | EXTEND | per-fabric in `DSPComputeFabric` |
| Energy (DRAM) | EXTEND | on `DSPMemorySubsystem` |
| Clock domain | COVERED | `ClockDomain` from `compute_block_common` |
| Thermal profile(s) | EXTEND | `DSPThermalProfile` list + default |
| Scheduler attrs | COVERED | scalar fields on `DSPBlock` (min_occupancy, max_concurrent, wave_quant) |
| VLIW issue width | NEW | `vliw_issue_width: int | None` (informational) |
| Performance roll-up | COVERED | `TheoreticalPerformance` from `compute_block_common` |
| Silicon bin | EXTEND | optional `SiliconBin` |
| BOM | EXTEND | optional `BOMCostProfile` |

NEW count: 6 (`compute_fabrics` list, `DSPComputeFabric`,
`DSPFabricKind`, `DSPDeploymentKind`, `DSPMemorySubsystem`,
`vliw_issue_width`).
EXTEND count: 4 (per-fabric energy, DRAM energy, thermal list,
silicon_bin, BOM).
COVERED count: 5 (identity, process_node, ClockDomain, scheduler
attrs, `TheoreticalPerformance`).

This is the **lowest NEW count yet** (TPU had 9 NEW, DPU had 8,
CGRA had 7) because v8 unification already shipped the cross-
block-kind primitives DSPBlock needs.

## Recommended schema diff for the next PR

The Schema PR (PR 2 of the sprint) should land roughly the following
in embodied-schemas:

1. **`compute_product.py`**:
   - Add `BlockKind.DSP = "dsp"` to the discriminator enum.
   - Update `AnyBlock = Annotated[Union[..., DSPBlock], Field(discriminator="kind")]`

2. **New file `embodied_schemas/dsp_block.py`** (~400 LOC):
   - `DSPBlock` with: `compute_fabrics: list[DSPComputeFabric]`,
     `deployment_kind: DSPDeploymentKind`,
     `memory: DSPMemorySubsystem`,
     `clock_domain: ClockDomain`,
     `thermal_profiles: list[DSPThermalProfile]`,
     `default_thermal_profile_name: str`,
     `theoretical_performance: TheoreticalPerformance`,
     `default_precision: str`,
     `min_occupancy: float`, `max_concurrent_kernels: int`,
     `wave_quantization: int`,
     `vliw_issue_width: int | None`,
     `silicon_bin: SiliconBin | None`,
     `bom_cost_profile: BOMCostProfile | None`,
     `multi_precision_alu: bool`.
   - `DSPComputeFabric` with: `fabric_kind: DSPFabricKind`,
     `circuit_class: CircuitClass`,
     `num_units: int`,
     `ops_per_unit_per_clock: dict[str, int]`,
     `energy_per_op_fp32_pj: float`,
     `energy_scaling: dict[str, float]`.
   - `DSPMemorySubsystem` with: `l1_size_bytes_per_unit: int`,
     `l2_size_bytes_total: int | None`,
     `l2_bandwidth_gbps: float | None`,
     `has_external_dram: bool`,
     `external_dram_type: MemoryType | None`,
     `external_dram_bandwidth_gbps: float | None`,
     `external_dram_bandwidth_kind: Literal["typical", "measured"]`,
     `external_dram_size_gb: float | None`,
     `external_dram_access_energy_pj_per_byte: float`,
     `coherence_protocol: CoherenceProtocol | None`.
   - `DSPThermalProfile` with: `name`, `tdp_watts`,
     `cooling_solution_id`, `clock_hz: int | None`, `dvfs_enabled`.
   - Enums: `DSPFabricKind` (VECTOR_SIMD / TENSOR_MATRIX /
     VLIW_SCALAR / HYBRID), `DSPDeploymentKind` (STANDALONE_IP /
     SOC_INTEGRATED).
   - Imports from `compute_block_common`: `TheoreticalPerformance`,
     `MemoryType`, `CircuitClass`, `ClockDomain`, `DataConfidence`.

3. **Validators on `DSPBlock`**:
   - `_validate_at_least_one_fabric`: `len(compute_fabrics) >= 1`.
   - `_validate_thermal_profile_default`: default name must be in
     the profiles list.
   - `_validate_external_dram_consistency`: if `has_external_dram`,
     then `external_dram_type` + `external_dram_bandwidth_gbps`
     must be set (mirrors NPU/DPU/TPU pattern).
   - `_validate_external_dram_kind_required`: if `has_external_dram`,
     then `external_dram_bandwidth_kind` must be set explicitly
     (no implicit defaults -- this is a load-bearing distinction
     between IP-core typical-integration and SoC-measured numbers).
   - `_validate_ops_per_unit_positive`: extend existing
     `_validate_int_precision_required` pattern (rejects negative
     OR zero ops_per_unit_per_clock per the DPU PR `#36`
     CodeRabbit fix).

4. **Process node**: no new YAML needed for Cadence Q8 (TSMC N16
   already in catalog).

5. **Cooling solution**: `passive-mobile.yaml` -- verify present;
   if not, add (probably needed -- DSP IP cores are the first
   1W passive-mobile workload class to land).

## Risks called out by this exercise

1. **`HardwareType.DSP` already exists on the graphs side.** Like
   TPU's `#191` analog, no new HardwareType enum value is needed.
   The loader sets it directly with no transitional period.

2. **`external_dram_bandwidth_kind` is load-bearing.** IP cores
   (Cadence, CEVA, Synopsys) all quote "typical SoC integration"
   bandwidth -- the actual bandwidth depends on the system
   integrator's choice of DDR controller. SoC SKUs (SA8775P, TDA4VM)
   quote measured datasheet bandwidth. Mixing these as if they
   were equivalent leads to false-precision in roofline analysis.
   The explicit `typical | measured` discriminator forces consumers
   to acknowledge the distinction.

3. **Multi-fabric is the rule for all DSPs except Cadence Q8.**
   The reference SKU is **atypical** in being single-fabric. The
   schema must be tested early against a 2-fabric SKU (recommend
   filing the CEVA NeuPro NPM11 YAML as the first follow-up
   data PR after this sprint closes -- it exercises the multi-
   fabric path without the deployment-kind / multi-profile
   complications of the SoC SKUs).

4. **VLIW issue width is informational only.** Several DSPs are
   VLIW (TI C7x is 8-wide VLIW); the issue width matters for
   compiler scheduling but not for the analytical roofline. Schema
   includes it as `int | None` documentation; mappers ignore it.

5. **Energy apportionment for SoC-integrated DSPs is unsolved.**
   When a SoC has CPU + GPU + DSP + NPU sharing a single TDP, the
   DSP's energy fraction depends on the workload mix. This sprint
   uses chip-level TDP on `DSPThermalProfile`; mapper-side
   apportionment is a separate concern. Documented here as a v9
   item.

6. **`compute_block_common` reuse is the default pattern.** This
   is the first block kind authored after v8 unification. Future
   block kinds (if any -- AI accelerator taxonomy is converging)
   should follow the same pattern: define block-specific types
   only for genuinely new shapes; reuse for primitives.

7. **No DSP-specific tile energy decomposition.** Unlike TPU
   (`TPUTileEnergyCoefficients` with 9 canonical fields), DSPs
   don't have a centralized tile energy story. Per-fabric
   energy_per_op + per-precision scaling is sufficient; the
   schema deliberately stops there.

8. **Reference SKU is single-thermal-profile.** Multi-profile
   DVFS (SA8775P at 20/30/45W) is not exercised by the reference
   SKU's YAML. The schema supports it (mirroring GPU / TPU
   patterns), but the first end-to-end multi-profile test lands
   when SA8775P's follow-up YAML lands. Recommend prioritizing
   SA8775P or TDA4VM as the second SKU after Cadence Vision Q8.

## Next step

Sprint sequencing:

- **PR 1 (this document)** -- docs/design, graphs-side. Lands the
  field-by-field audit + recommended schema diff. Pure docs PR.
- **PR 2 -- schema, embodied-schemas-side.** Implements the
  "Recommended schema diff" above (DSPBlock + supporting types).
- **PR 3 -- data, embodied-schemas-side.** Authors the first DSP
  YAML at `data/compute_products/cadence/cadence_vision_q8.yaml`
  (new `cadence/` vendor directory).
- **PR 4 -- loader, graphs-side.** Implements `dsp_yaml_loader`
  plus parity test against the hand-coded factory.
- **PR 5 -- cleanup, graphs-side.** Retires `cadence_vision_q8.py`'s
  ~222-LOC body to a thin loader-wrapper + any necessary overlays.
  Closes issue `#211`.

The scope is **smaller than the DPU or TPU sprints** because v8
unification already shipped the cross-block-kind primitives
DSPBlock needs (6 NEW fields vs 8-9 in DPU/TPU). Schema ~400 LOC,
YAML ~150 LOC, loader ~400 LOC, cleanup ~30 LOC.

After this sprint closes:
- 9 follow-up YAMLs (CEVA, Synopsys, 5 TI TDA4, 2 Qualcomm edge,
  SA8775P) land as pure data PRs with their corresponding graphs
  cleanups (~18 PRs total, file separately, grouped by vendor
  for review batching)
- Every category schema gap is closed. The catalog reaches
  schema-complete state for v7+.
- v9 follow-ups (`ThermalProfile` / `OnDieFabric` unification,
  `has_external_dram` / `has_host_dram` reconciliation) become
  the next sprint candidates.
