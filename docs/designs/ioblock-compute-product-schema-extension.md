# IOBlock ComputeProduct schema extension -- paper exercise

Status: draft
Date: 2026-05-23
Tracking issue: graphs#245
Predecessors:
  - GPU sprint #171 (closed)
  - CPU sprint #182 (closed)
  - NPU sprint #187 (closed)
  - CGRA sprint #196 (closed)
  - DPU sprint #200 (closed)
  - TPU sprint #204 (closed)
  - DSP sprint #211 (closed)
  - v8 unification #208 + follow-up #210 (closed)
  - v9-v12 lockdown arc (closed; see 2026-05-18 changelog)
  - PhysicalSpec backfill campaign #130 / #234 / #241 (closed at 91%)

## Purpose

Issue `#245` proposes the IOBlock mini-sprint -- the **9th block kind**
in the `ComputeProduct` discriminated union, following the 8 category
block kinds (KPU/GPU/CPU/NPU/CGRA/DPU/TPU/DSP) that close every
compute fabric the catalog ships today. IOBlock is the next
architecturally orthogonal block kind: it represents non-compute
silicon that nonetheless occupies real die area (memory controllers,
PCIe controllers, coherence fabric, security processors,
power-management controllers).

The v9 `BlockKind` docstring explicitly anticipates this work:

> Future block kinds (`MEMORY`, `IO`, `BRIDGE`, ...) come in
> subsequent PRs as their catalogs are added.

This document is **PR 1 of that sprint** -- the paper exercise:
enumerate AMD Genoa IOD's full feature set against the v12 schema
and classify every field as **covered**, **extends an existing
field**, or **needs a new field/type**.

The exercise uses **AMD Genoa IOD** as the reference because:

  - **Best-documented IOD in the public domain** -- AMD HotChips
    2022 presentation, AnandTech Genoa launch coverage, WikiChip
    Zen 4 die-shot analyses all converge on the same per-block
    decomposition.
  - **Reused unchanged across two SKUs** in the catalog (EPYC 9654
    Genoa + EPYC 9754 Bergamo) -- good "stable starting point" for
    schema design; v2 chiplet products with different IODs follow
    as pure data PRs.
  - **Separate physical die** with its own `process_node_id` (TSMC
    N6 vs the CCDs' N5) -- exercises the per-die process node
    distinction that the chiplet schema was designed for.
  - **Single die role** -- no overlap with compute fabrics; the
    cleanest case for establishing the schema before more complex
    "mixed" dies (chips with iGPU+IO on the same die) land.

Out of scope for v1 of IOBlock (deferred):

- **Other IODs** -- Turin IOD (EPYC 9965), Intel client SoC
  IO-integrated tiles (SPR/EMR/GNR don't have a separate IOD --
  see "Intel tile architectures" section). Each follows as a pure
  data PR once the schema lands.
- **`MEMORY` block kind** -- separating HBM stacks into their own
  dies with `die_role=MEMORY` carrying a `MemoryBlock`. Next future
  block kind per the docstring. Distinct from this sprint because
  HBM stacks are physically separate dies bonded via TSV, not
  silicon-on-the-IOD.
- **`BRIDGE` block kind** -- EMIB / silicon-interposer modeling.
  Next future block kind. Distinct from IOBlock because bridges
  are passive routing, not active functional silicon.
- **iGPU on the IOD** -- AMD APUs (e.g., Phoenix) and Intel client
  CPUs put iGPU silicon on the IO die. The schema already supports
  multi-block dies (`Die.blocks: list[AnyBlock]` with `min_length=1`
  but no upper bound); a future SKU could add a `GPUBlock` to the
  IOD's `blocks[]` alongside the new IOBlock. v1 of IOBlock does
  not require this; documented for future expansion.

## Method

For each feature on the AMD Genoa IOD (per AMD HotChips 2022 slides
+ AnandTech Genoa launch deep-dive + WikiChip Zen 4 IOD page), this
doc records:

- The field/object as it would live in `IOBlock` (with rationale).
- Classification: COVERED / EXTEND / NEW.
- For COVERED: which existing schema type already represents it
  (often inherited from `compute_block_common` or reused from
  `CPUBlock.memory`).
- For EXTEND: the existing field that grows.
- For NEW: a proposed field name + sub-type + brief rationale.

The exercise then projects onto the **8 chiplet CPU SKUs in the
catalog** that the IOBlock schema must support (the 3 AMD EPYCs +
the 3 Intel Xeons, with Xeon analysis showing they may not need
IOBlock at all).

## Reference: current v12 schema state

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
    DSP = "dsp"
    # IO = "io"  <-- this sprint adds

class DieRole(str, Enum):
    COMPUTE = "compute"
    MEMORY = "memory"   # for future MemoryBlock
    IO = "io"           # ALREADY EXISTS; this sprint enables a Block to fill it
    BRIDGE = "bridge"
    MIXED = "mixed"

AnyBlock = Annotated[
    Union[KPUBlock, GPUBlock, CPUBlock, NPUBlock,
          CGRABlock, DPUBlock, TPUBlock, DSPBlock],
    Field(discriminator="kind"),
]
```

`embodied-schemas/src/embodied_schemas/compute_block_common.py`
(landed in v8 unification, expanded in v9-v12):

```python
# Re-exports
CircuitClass, DataConfidence  # from process_node
MemoryType                    # from gpu (DDR3/DDR4/DDR5, HBM2/2e/3, LPDDR4/5, GDDR6, etc.)
ClockDomain                   # from gpu_block
TheoreticalPerformance        # unified peak_ops_per_sec roll-up
ThermalProfile (v9 alias)     # cross-kind thermal-operating-point shape
OnDieFabric (v10 base)        # inheritance base for *NoCFabric subclasses
DramAttachment (v11/v12)      # chip_attached / host_attached / hbm_stack
```

IOBlock uses these from day 1.

## IOD architectural primer (for reviewers not familiar)

An "IOD" (I/O die / IO Hub die) is a non-compute chiplet that
aggregates the SoC's memory and I/O interfaces, plus the coherence
fabric that ties the compute chiplets together. In AMD's chiplet
philosophy, the IOD physically separates two concerns from the
compute CCDs:

  - **Process-node mismatch**: the bleeding-edge process gains
    matter for logic-dense compute (CCDs on N5), but memory PHYs,
    PCIe SerDes, and analog-heavy IO scale poorly with node
    shrinks. Keeping IO on an older node (Genoa IOD on TSMC N6) is
    economically rational -- the IO silicon doesn't lose much area
    from being on N6 vs N5, but the wafer cost is meaningfully
    lower.
  - **Yield / cost**: a defect in an IO PHY can kill an entire
    monolithic die. Separating the IOD into its own chip lets the
    CCDs be yield-binned independently. AMD trades package-level
    EMIB-style interconnect cost for die-level yield improvements.

A datacenter-class IOD's headline silicon decomposition:

  - **Memory controllers + PHY** (N channels of DDR5 / LPDDR5 /
    HBMx) -- the largest single area consumer on most IODs
  - **PCIe controllers + SerDes** (X lanes of Gen4 / Gen5 / Gen6)
  - **Inter-socket coherence links** (G-link for AMD, UPI for
    Intel, NVLink-C2C for NVIDIA-AMD coherent platforms)
  - **Intra-package coherence fabric** (Infinity Fabric routers
    for AMD, mesh-extension for Intel SPR-style tiles)
  - **Security processor** (AMD PSP, Intel SGX/TXT engine, ARM
    TrustZone-equivalent)
  - **Power management controller** (PMC / SMU -- handles DVFS,
    boost, thermal throttling)
  - **Clock generation** (PLLs and clock trees feeding all CCDs)
  - **Boot ROM / firmware** (microcode + BIOS attach point)
  - **(Optional)** Integrated GPU silicon, USB/SATA PHYs, ethernet
    MACs -- client/embedded variants only; not on datacenter IODs.

Most of this fits the existing `Die.silicon_bin` block taxonomy
(`memory_controller`, `pcie_phy`, `infinity_fabric`, etc.) -- but
the IOD as a *Block* needs to expose roll-up properties: total
memory bandwidth, total PCIe lane count, coherence-fabric topology,
inter-socket scale-out.

## Multi-SKU coverage (schema must support all 8 chiplet SKUs)

### Group A: separate IOD die (4 SKUs; IOBlock target)

These SKUs have a physical IOD die joined to CCDs via package-level
interconnect. They're the primary IOBlock target.

| SKU | IOD | Process | Memory | PCIe | Inter-socket |
|---|---|---|---|---|---|
| EPYC 9654 (Genoa) | Genoa IOD | TSMC N6 | 12-ch DDR5-4800 | 128 PCIe Gen5 | 4x G-link |
| EPYC 9754 (Bergamo) | Genoa IOD (reused) | TSMC N6 | 12-ch DDR5-4800 | 128 PCIe Gen5 | 4x G-link |
| EPYC 9965 (Turin Dense) | Turin IOD | TSMC N6 | 12-ch DDR5-6000 | 128 PCIe Gen5 | 4x G-link |
| (future) AMD APUs | client IOD with iGPU | varies | 2-ch DDR5/LPDDR5 | 16-24 PCIe | none |

### Group B: tile-internal IO (3 SKUs; IOBlock probably NOT used)

Intel Xeon Sapphire / Emerald / Granite Rapids tile architectures
integrate memory controllers + PCIe + UPI directly into each
compute tile. There is no separate IOD die. The current `CPUBlock`
already captures memory + on-die fabric for these SKUs; adding
IOBlock would either:

  - **(B1)** Stay as today: single `CPUBlock` per tile with IO
    fields embedded (current modeling).
  - **(B2)** Split IO into a sibling block on the same tile die
    (multi-block-per-die; schema supports it, but produces
    minimal fidelity gain since the IO is physically intermixed
    with the cores).
  - **(B3)** Wait for the v8 multi-block-per-die schema work
    (separate sprint per `DECISION-2026-05-21-001`) to clarify
    whether tile-internal IO belongs in a block at all, or just
    in `Die.silicon_bin`.

This paper exercise **recommends (B1)** for Xeon SKUs -- their
current single-virtual-die representation remains accurate at the
chip level since their tiles aren't physically separate from
compute. The IOBlock schema is added for AMD-style separate-IOD
products; Intel tile SKUs are documented in `BlockKind` as
"out of scope for IOBlock; tile-internal IO stays in CPUBlock".

### Group C: future expansion (not in this sprint)

- AMD client APUs (Phoenix, Strix Point) -- IOD + iGPU on the same
  die. v1 IOBlock + future GPU-on-IOD multi-block representation.
- NVIDIA Grace Hopper -- the C2C-coherent ARM CPU side has a die
  that arguably contains IOBlock-shaped silicon.
- Custom-fabric server SoCs (Tenstorrent's chiplet platform, etc.)

## Feature-by-feature audit

What follows is the audit against the **AMD Genoa IOD** as
reference. Source: AMD HotChips 2022 + Anandtech Genoa launch +
WikiChip Zen 4 IOD page.

### 1. Block identity

| Field | Classification | Notes |
|---|---|---|
| `kind: Literal[BlockKind.IO]` | NEW (enum value) | Discriminator. Matches pattern of all 8 existing block kinds. |

### 2. Memory subsystem

The IOD owns the physical memory controllers + PHY. Today's
`CPUBlock.memory: CPUMemorySubsystem` carries these fields:
`memory_type`, `memory_size_gb`, `memory_bus_bits`,
`memory_bandwidth_gbps`, `memory_controllers`, `l3_present`,
`l3_total_kib`, ... etc.

| Field | Classification | Notes |
|---|---|---|
| `memory_type: MemoryType` | COVERED | Reuse from `compute_block_common`. DDR5 / HBM3 / LPDDR5X all already supported. |
| `memory_channels: int` | EXTEND | `CPUBlock.memory.memory_controllers` -- could rename for IO-block-level cohesion, or copy under a clearer name. Recommendation: reuse `memory_controllers` to keep cross-block-kind invariants. |
| `memory_bus_bits: int` | COVERED | Existing `CPUMemorySubsystem.memory_bus_bits`. |
| `memory_bandwidth_gbps: float` | COVERED | Reuse. |
| `max_memory_size_gb: float` | COVERED | Reuse `memory_size_gb` semantic. |
| `ecc_supported: bool` | NEW | Datacenter IODs all support ECC; client IODs often don't. Not on `CPUMemorySubsystem` today (assumed-true for server CPUs). Worth surfacing as an explicit field. |
| `memory_controller_energy_pj_per_byte: float` | NEW | Per-byte access energy at the IMC level (distinct from PHY-only); useful for per-tile vs per-IMC roofline modeling. Defaultable. |

**Recommended sub-type**: `IOMemorySubsystem` (mirrors
`CPUMemorySubsystem` and `TPUMemorySubsystem` shapes -- one
sub-type per block-kind owning memory). Likely 90% identical fields
to `CPUMemorySubsystem` minus the cache hierarchy fields (which the
IOD doesn't own).

**Open question**: does the `l3_total_kib` field belong on
IOBlock or stay on CPUBlock? For AMD Zen 4 CCDs, L3 lives **on the
CCDs**, not on the IOD. So `l3_*` fields stay on CPUBlock. The IOD
owns DRAM access, not L3.

### 3. PCIe / inter-socket / coherence-fabric interfaces

The IOD's other major area consumer is the PCIe SerDes + Inter-
Socket coherence links + the on-package coherence fabric routers.

| Field | Classification | Notes |
|---|---|---|
| `pcie_lanes: int` | NEW | 128 for EPYC; 80 for current Xeon (if Xeon used IOBlock, which it doesn't per Group B). Today this is `platform.pcie_lanes` in the legacy v1 GPU YAML schema, not on any *Block. NEW for IOBlock. |
| `pcie_generation: PCIeGen` | NEW | Enum exists in `gpu.py::PCIeGen` (pcie_3.0 / 4.0 / 5.0 / 6.0). Re-export to `compute_block_common` so IOBlock can use it without circular import. |
| `cxl_supported: bool` | NEW | Most current IODs support CXL 1.1+ via PCIe-physical layer. |
| `cxl_version: str \| None` | NEW | Free-form string ("1.1", "2.0", "3.0"); CXL spec versioning is volatile. |
| `inter_socket_links: list[InterSocketLink]` | NEW | One entry per link kind (G-link for AMD, UPI for Intel). Per-link bandwidth + count. New sub-type `InterSocketLink` -- see "Inter-socket link sub-type" below. |
| `coherence_fabric: IOOnDieFabric` | NEW | Routes traffic between IO endpoints and CCDs. Inherits from `OnDieFabric` base (v10). Subclass adds a typed `IOFabricTopology` enum. |
| `coherence_fabric.bisection_bandwidth_gbps: float` | COVERED via OnDieFabric | |
| `coherence_fabric.topology: IOFabricTopology` | NEW | Enum: `INFINITY_FABRIC` / `MESH_EXTENSION` / `EMIB_HUB` / `CXL_FABRIC` |

**Recommended sub-type**: `IOInterconnect` (or roll PCIe + CXL +
inter-socket into a flat `IOBlock` field set; the design exercise
will decide based on how the AMD Genoa IOD's silicon_bin
decomposes).

### 4. Security + management silicon

| Field | Classification | Notes |
|---|---|---|
| `security_processor_kind: str` | NEW | Free-form ("AMD PSP", "Intel TXT/SGX", "ARM TrustZone-CSE"). Most IODs ship one. |
| `security_processor_die_area_mm2: float \| None` | NEW | Optional; few datasheets publish this. |
| `power_management_controller: bool` | NEW | "Does the IOD own the PMC / SMU?" Most datacenter IODs do; client SoCs may put it on the CPU die instead. |
| `boot_rom_present: bool` | NEW | Most IODs have a boot ROM for firmware fetch. |

**Open question**: should these be modeled as discrete fields, or
collapsed into a `mgmt_silicon` sub-object? Recommendation: discrete
fields for simplicity in v1; consolidate to a sub-object if v2+
products need to express richer security topology (multi-tenant
attestation, secure-boot chains, etc.).

### 5. Process / die-level fields (already on Die, not Block)

These live on the `Die` not the `IOBlock` (per the chiplet schema's
existing separation):

| Field | Classification | Notes |
|---|---|---|
| `process_node_id` | COVERED on Die | Genoa IOD: `tsmc_n6`. Already supported by v12 Die schema. |
| `die_size_mm2` | COVERED on Die | ~397 mm^2 estimate. |
| `transistors_billion` | COVERED on Die | ~12 B estimate. |
| `die_id` | COVERED on Die | E.g., `genoa_iod`. |
| `die_role: DieRole.IO` | COVERED on Die | DieRole.IO already exists in v1; just needs IOBlock to populate `dies[].blocks[]` finally. |

### 6. Theoretical-performance roll-up

| Field | Classification | Notes |
|---|---|---|
| `performance: TheoreticalPerformance` | COVERED via compute_block_common | But the IOBlock doesn't have meaningful peak ops/sec -- it's not a compute fabric. Could populate as `{"memory_bandwidth_gbps": <value>}` or just leave as `peak_ops_per_sec_by_precision={}` (empty dict). Recommendation: make `TheoreticalPerformance` optional on IOBlock (override `Field(...)` to `Field(None)`), or skip the field entirely on IOBlock. |

This is the only field where IOBlock genuinely diverges from the
other 8 block kinds.

**Decision needed**: Override `performance` to optional on IOBlock?
Or define IOBlock without a `performance` field at all?

Recommendation: **no `performance` field on IOBlock**. Rationale:
the IOD's "performance" is captured by memory bandwidth + PCIe
aggregate bandwidth + inter-socket aggregate bandwidth + coherence
fabric bisection bandwidth -- all of which are already discrete
typed fields on the proposed IOBlock. Forcing them into the
generic `peak_ops_per_sec_by_precision` dict loses type safety.

### 7. Energy + power roll-ups

| Field | Classification | Notes |
|---|---|---|
| `idle_power_watts: float \| None` | NEW | IODs have non-trivial idle power (memory PHY refresh, PSP heartbeat, etc.). ~30-50W for datacenter IODs per published AMD figures. |
| `pcie_aggregate_energy_pj_per_byte: float \| None` | NEW | Optional; sum of PCIe controller + SerDes per-byte energy. |
| `inter_socket_aggregate_energy_pj_per_byte: float \| None` | NEW | Optional. |

These are deferrable to v2 of IOBlock if datasheets don't publish
the numbers. Most v1 IODs will leave these `None`.

## Summary table

| Category | COVERED | EXTEND | NEW |
|---|---|---|---|
| Block identity | 0 | 0 | 1 (`BlockKind.IO`) |
| Memory subsystem | ~4 | ~0 | ~2 (`ecc_supported`, `memory_controller_energy_*`) |
| PCIe / inter-socket / fabric | 0 | 0 | ~7 (`pcie_*`, `cxl_*`, `inter_socket_links`, `coherence_fabric`) |
| Security + mgmt | 0 | 0 | 4 (`security_processor_*`, `pmc`, `boot_rom`) |
| Process / die fields | 5 | 0 | 0 (all live on Die, not Block) |
| Performance roll-up | 0 | 0 | 0 (IOBlock has no `performance` field; see §6) |
| Energy + power | 0 | 0 | 3 (deferrable to v2) |
| **Total** | **9** | **0** | **~17** |

Compares to the prior 8 block kinds:

| Block kind | NEW fields when added |
|---|---|
| KPU (v1) | ~25 (the schema baseline) |
| GPU (v2) | ~12 |
| CPU (v3) | ~10 |
| NPU (v4) | ~9 |
| CGRA (v5) | ~11 |
| DPU (v6) | ~13 |
| TPU (v7) | ~15 |
| DSP (v9) | ~10 |
| **IO (this sprint, v13)** | **~17** |

IOBlock's NEW-field count is on the high end because PCIe / CXL /
inter-socket / coherence-fabric data hasn't been modeled anywhere
in the schema yet. The other 8 block kinds all assumed "host
provides the IO surface"; IOBlock is the first to model that surface
explicitly.

## Recommended schema diff for PR 2

`embodied-schemas/src/embodied_schemas/`:

### `compute_product.py`

```python
class BlockKind(str, Enum):
    KPU = "kpu"
    GPU = "gpu"
    CPU = "cpu"
    NPU = "npu"
    CGRA = "cgra"
    DPU = "dpu"
    TPU = "tpu"
    DSP = "dsp"
    IO = "io"   # NEW (v13)

AnyBlock = Annotated[
    Union[KPUBlock, GPUBlock, CPUBlock, NPUBlock,
          CGRABlock, DPUBlock, TPUBlock, DSPBlock,
          IOBlock],   # NEW
    Field(discriminator="kind"),
]

# DieRole.IO is unchanged (already in v1).
```

### `io_block.py` (NEW module)

```python
from typing import Literal
from pydantic import BaseModel, Field
from embodied_schemas.gpu import MemoryType, PCIeGen
from embodied_schemas.compute_block_common import OnDieFabric


class IOFabricTopology(str, Enum):
    INFINITY_FABRIC = "infinity_fabric"   # AMD
    MESH_EXTENSION = "mesh_extension"     # Intel SPR-like (tile internal)
    EMIB_HUB = "emib_hub"                 # Intel chiplet bridge
    CXL_FABRIC = "cxl_fabric"             # CXL-native fabric (future)


class IOOnDieFabric(OnDieFabric):
    topology: IOFabricTopology = Field(...)


class IOMemorySubsystem(BaseModel):
    memory_type: MemoryType = Field(...)
    memory_size_gb: float = Field(..., gt=0)
    memory_bus_bits: int = Field(..., gt=0)
    memory_bandwidth_gbps: float = Field(..., gt=0)
    memory_controllers: int = Field(..., gt=0)
    ecc_supported: bool = Field(True)
    memory_controller_energy_pj_per_byte: float | None = Field(None, ge=0)


class InterSocketLink(BaseModel):
    name: str = Field(..., description="e.g., 'AMD G-link', 'Intel UPI 2.0'")
    bandwidth_per_link_gbps: float = Field(..., gt=0)
    link_count: int = Field(..., gt=0)


class IOBlock(BaseModel):
    kind: Literal["io"] = Field("io")

    memory: IOMemorySubsystem
    coherence_fabric: IOOnDieFabric

    pcie_lanes: int = Field(..., gt=0)
    pcie_generation: PCIeGen
    cxl_supported: bool = Field(False)
    cxl_version: str | None = Field(None)

    inter_socket_links: list[InterSocketLink] = Field(default_factory=list)

    security_processor_kind: str | None = Field(None)
    security_processor_die_area_mm2: float | None = Field(None, ge=0)
    power_management_controller: bool = Field(False)
    boot_rom_present: bool = Field(False)

    # Energy roll-ups (optional in v1)
    idle_power_watts: float | None = Field(None, ge=0)
    pcie_aggregate_energy_pj_per_byte: float | None = Field(None, ge=0)
    inter_socket_aggregate_energy_pj_per_byte: float | None = Field(None, ge=0)
```

### `compute_block_common.py`

```python
# v13 addition: re-export PCIeGen so IOBlock doesn't have to import
# from embodied_schemas.gpu directly (which would create a circular
# dependency once IOBlock is added to AnyBlock).
from embodied_schemas.gpu import PCIeGen
```

## Risks called out by this exercise

### Risk 1: IOBlock duplicates fields from CPUBlock.memory

`IOMemorySubsystem` and `CPUMemorySubsystem` will share ~5 fields
(`memory_type`, `memory_size_gb`, `memory_bus_bits`,
`memory_bandwidth_gbps`, `memory_controllers`). Two paths:

  - **(R1.a)** Keep them separate; accept the duplication. The
    classes have semantically distinct ownership: CPUBlock.memory
    is "what the CPU's MMU sees"; IOMemorySubsystem is "what the
    IOD's controllers manage". v8 unification didn't collapse
    {KPU,CPU,TPU,NPU,DSP,CGRA,DPU,GPU}MemorySubsystem; staying
    consistent here is reasonable.
  - **(R1.b)** Promote a shared `MemorySubsystemBase` in
    `compute_block_common` and inherit. The v10 `OnDieFabric`
    precedent shows the schema supports inheritance for cross-kind
    invariants. Worth doing if v13+ adds MemoryBlock (which would
    be a 10th-block-kind sprint sharing the same memory fields).

Recommendation: **(R1.a)** for v13 IOBlock; revisit if MemoryBlock
materializes in v14.

### Risk 2: Re-authoring 8 chiplet YAMLs is a substantial data PR effort

The chiplet CPU YAMLs from sprints #62 + #68 (3 EPYCs + 3 Xeons)
all use the single-virtual-die approximation today. Re-authoring
them to proper multi-die layout requires:

  - Adding the IOD as a second `Die` entry per SKU
  - Moving the IOD's `silicon_bin` entries from the compute die to
    the IOD
  - Populating the new `IOBlock` fields per SKU
  - Updating each SKU's `Die.interconnects[]` to capture die-to-die
    links (e.g., IFOP for AMD)
  - Bumping parity tests to expect multi-die layout

For Intel Xeon SKUs (Group B), per the design exercise the
recommendation is **no change** -- they stay single-die since they
don't have a separate IOD. The IOBlock isn't applicable.

Net effort: **~3 SKU re-authoring PRs** (EPYC 9654 reference, then
9754 + 9965 as follow-ups). Same pattern as sprint #62 -- one PR
per SKU.

### Risk 3: PCIeGen enum re-export creates a small circular-import risk

`embodied_schemas.gpu` imports `MemoryType` which is now re-exported
via `compute_block_common`. Adding `PCIeGen` to the re-export list
should be similarly safe (both are leaf enum types with no Pydantic
dependencies), but the test suite should specifically exercise this
re-export path to confirm no module-load-order issues.

### Risk 4: `Die.interconnects[]` is empty on every existing SKU

Today, every chiplet SKU's `Die.interconnects[]` is `[]` because
the single-virtual-die approximation collapsed the cross-die
fabric. Proper multi-die layout needs populated `interconnects[]`
entries (e.g., `IFOP` per CCD<->IOD link for AMD). The existing
`Interconnect` Pydantic class supports this (level=DIE_TO_DIE,
topology=POINT_TO_POINT/HUB_AND_SPOKE, per-link bandwidth +
latency + energy). No schema change needed; data-side only.

## Next step

PR 2 of this sprint: implement the `IOBlock` schema in
`embodied-schemas` per the §"Recommended schema diff" above.
Include conformance tests for:

- `BlockKind.IO` discriminator dispatch through `AnyBlock`
- `IOMemorySubsystem` validation (positive bandwidth, positive
  channel count, ECC default)
- `IOBlock` field roll-ups (per-link bandwidth aggregation, PCIe
  lane count > 0)
- Pydantic round-trip serialization
- v8 unification stability: KPU/GPU/CPU/NPU/CGRA/DPU/TPU/DSP YAMLs
  still validate against `AnyBlock` (additive guarantee)
- `Die(die_role=DieRole.IO, blocks=[IOBlock(...)])` validates
  end-to-end

PR 3 (data, embodied-schemas): re-author EPYC 9654 to multi-die
layout (1 compute die with CPUBlock + 1 IO die with IOBlock,
joined by `Die.interconnects[]` IFOP entries).

PR 4-5 (data, embodied-schemas): re-author EPYC 9754 (reuses Genoa
IOD; mostly inherits PR 3's IOD entry) and EPYC 9965 (new Turin IOD).

PR 6 (graphs): downstream PhysicalSpec loader verification. The
loader sums `dies[].die_size_mm2` and `dies[].transistors_billion`
across the chiplets, so the headline PhysicalSpec values stay
correct. Add the per-die process node distinction to the loader
output (Genoa IOD on N6 now visible alongside CCDs on N5).

Sprint exit: `DECISION-2026-05-21-001` Question A is amended to
note the IOBlock work landed, and the chiplet approximation is
retired.
