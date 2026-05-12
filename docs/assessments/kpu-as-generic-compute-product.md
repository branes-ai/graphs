# Assessment: Is the KPU now fully modeled as a generic ComputeProduct?

**Date:** 2026-05-12
**Context:** The architectural goal of the `graphs` + `embodied-schemas` stack
is a universal `ComputeProduct` model on which we can evaluate latency, energy,
and throughput of compute graphs across CPUs, GPUs, TPUs, DSPs, KPUs, and
potentially FPGA/CGRAs -- a universal compute fabric specialized by a netlist
overlay representing the resource graph. The recent sprint (PRs
[embodied-schemas#14](https://github.com/branes-ai/embodied-schemas/pull/14),
[graphs#153](https://github.com/branes-ai/graphs/pull/153),
[graphs#155](https://github.com/branes-ai/graphs/pull/155)) targeted this for
the KPU, with the intent to apply what we learned to migrate the GPU and CPU
models. This assessment reports whether that goal has been reached for the KPU
and provides a class diagram a human can validate against the codebase.

---

## TL;DR

**No, the KPU is NOT yet a generic ComputeProduct.** The KPU has become the
**most-developed exemplar architecture-specific schema** in the catalog, but
the unified `ComputeProduct` schema -- proposed in
`embodied-schemas/docs/rfcs/0001-compute-product-unification.md` -- is still
**Draft status, not implemented**.

There is no class named `ComputeProduct` in either repo today. The term appears
only in:

- `embodied-schemas/src/embodied_schemas/cooling_solution.py:7` -- docstring
  mention of "ComputeSolution"
- `embodied-schemas/src/embodied_schemas/kpu.py:450` -- docstring "ComputeSolution
  = ProcessNode + CoolingSolution + KPUEntry"
- `graphs/src/graphs/hardware/physical_spec.py:14` -- "When the
  embodied-schemas ComputeProduct unification (RFC 0001) lands, PhysicalSpec
  will become a thin loader that reads from a ComputeProduct YAML"
- `graphs/src/graphs/hardware/sku_validators/framework.py:153` -- "A
  ComputeSolution is the blend of ProcessNode + CoolingSolution + SKU"
- `graphs/cli/show_compute_product.py` -- a CLI tool that composes the per-layer
  inspectors into one assembled view (created in this sprint)

The KPU does have all the structural ingredients RFC 0001 calls for, and so it
is the right TEMPLATE to promote when the unification work begins.

---

## Class diagram (current state, ASCII)

Three layers. Drawn for the KPU but with the cross-architecture comparison column.

```
================================================================================
LAYER 1: embodied-schemas data layer (Pydantic catalog)
================================================================================

   +----------------------+        +-----------------------+
   | ProcessNodeEntry     |<---ref-| KPUEntry              |
   |  (foundry, node_nm,  | by id  |  - id, name, vendor   |
   |   densities,         |        |  - process_node_id    |---+
   |   leakage,           |        |  - die: KPUDieSpec    |   |
   |   energy_per_op_pj,  |        |  - kpu_architecture:  |   |
   |   sram_access_pj,    |        |      KPUArchitecture  |   |
   |   dram_io_pj,        |        |  - silicon_bin:       |   |
   |   noc_pj_per_flit)   |        |      KPUSiliconBin    |   |
   +----------------------+        |  - clocks: KPUClocks  |   |
            ^                      |  - performance:       |   |
            |                      |     KPUTheoretical    |   |
            |                      |       Performance     |   |
            |                      |  - power: KPUPowerSpec|   |
            |                      |  - market: KPUMarket  |   |
            |                      +-----------------------+   |
            |                            |                     |
            |                            | composes            |
            |                            v                     |
            |    +---------------------------------------+     |
            |    | KPUArchitecture                       |     |
            |    |  - total_tiles, multi_precision_alu   |     |
            |    |  - tiles: list[KPUTileSpec]           |     |
            |    |  - noc: KPUNoCSpec                    |     |
            |    |  - memory: KPUMemorySubsystem         |     |
            |    +---------------------------------------+     |
            |                                                  |
            |    +---------------------------------------+     |
            |    | KPUSiliconBin                         |     |
            |    |  - blocks: list[SiliconBinBlock]      |     |
            |    |     - circuit_class: CircuitClass     |-----+ used by
            |    |     - transistor_source:              |       silicon_math
            |    |        TransistorSource               |       to expand
            |    |        (kind: per_pe|per_kib|         |       per-block
            |    |               per_router|             |       counts using
            |    |               per_controller|fixed)   |       node densities
            |    +---------------------------------------+
            |
            |    +---------------------------------------+
            |    | KPUPowerSpec                          |
            |    |  - tdp_watts (DERIVED from profiles)  |
            |    |  - max/min_power_watts                |
            |    |  - idle_power_watts (= leakage)       |
            |    |  - default_thermal_profile (name)     |
            |    |  - thermal_profiles: list[            |
            |    |       KPUThermalProfile]              |
            |    |       - clock_mhz                     |
            |    |       - vdd_v          <- Orin DVFS   |
            |    |       - cooling_solution_id ----------+----> CoolingSolutionEntry
            |    |       - activity_factor               |     (max_total_w,
            |    |       - efficiency_factor_by_prec     |      max_w_per_mm2,
            |    |       - tile_utilization_by_prec     |      mechanism, etc.)
            |    +---------------------------------------+

       Vendor parallel (today) -- IS NOT shaped like KPU:
       +-----------+      +-----------+      +-----------+
       | GPUEntry  |      | CPUEntry  |      | NPUEntry  |
       |  - die    |      |  (no die) |      |  (no die) |
       |  - compute|      |  - cores  |      |  - compute|
       |  - clocks |      |  - clocks |      |  - memory |
       |  - memory |      |  - cache  |      |  - power  |
       |  - perf   |      |  - memory |      | (no proc  |
       |  - power  |      |  - power  |      |  node ref,|
       |  - market |      |  - market |      |  no       |
       |           |      |           |      |  silicon  |
       |  (no proc |      |  (process |      |  bin, no  |
       |  node ref,|      |   node    |      |  thermal  |
       |  no sili- |      |   ENUM    |      |  profiles |
       |  con bin, |      |   only,   |      |  with     |
       |  no       |      |   no ref  |      |  vdd_v)   |
       |  thermal  |      |   to PNE) |      +-----------+
       |  profiles |      +-----------+
       |  with     |
       |  vdd_v)   |
       +-----------+

================================================================================
LAYER 2: graphs schema-to-model bridge (loader + generator + validator + power)
================================================================================

   KPUEntry  (catalog)
     |
     | input_spec_from_kpu_entry()
     v
   KPUSKUInputSpec  (architect-authoring shape; drops generator-derived fields)
     |
     | generate_kpu_sku()              kpu_power_model.compute_thermal_profile_tdp_w()
     | rolls up die area, transistors,    5-term roll-up:
     | performance, tdp_watts (per          PE compute, L2 SRAM, L3 SRAM,
     | profile, V^2*f) using                NoC traversal, DRAM PHY, leakage
     |    silicon_math (resolve_block_      with V^2*f scaling on dynamic terms
     |    area, total_chip_leakage_w)
     v
   KPUEntry  (regenerated; round-trips within rounding)
     |
     | sku_validators.run_all()
     | (registry of pluggable checks across categories:
     |  area / electrical / energy / thermal / reliability /
     |  geometry / consistency / cross_ref)
     v
   list[Finding]  (severity: ERROR | WARNING | INFO)

   For GPU/CPU/NPU today:  none of generator / power_model / validators / silicon_bin exists.

================================================================================
LAYER 3: graphs runtime hardware model (mapper consumes; estimators read)
================================================================================

   +--------------------------------+
   | HardwareResourceModel          |  (one shape per HW; loaded from YAML)
   |  - hardware_type: HardwareType |
   |  - compute_units (SMs/cores/   |
   |     tiles/arrays)              |
   |  - peak_bandwidth, caches      |
   |  - energy_per_flop, _per_byte  |
   |  - precision_profiles: dict    |
   |  - thermal_operating_points:   |
   |     dict[str, ThermalOperat-   |
   |     ingPoint]                  |
   |  - compute_fabrics:            |
   |     list[ComputeFabric] |None  |
   |  - physical_spec: PhysicalSpec |  (interim spine -- chip-level facts)
   +--------------------------------+
        |
        | category-specific compute resource (ONE OF):
        v
   +--------------------+   +---------------------+   +----------------------+
   | KPUComputeResource |   | (GPU: SM-allocation |   | (CPU: core/SIMD      |
   |  - total_tiles     |   |  via ComputeFabric +|   |  alloc via Compute   |
   |  - tile_special-   |   |  PrecisionProfile)  |   |  Fabric)             |
   |    izations:       |   +---------------------+   +----------------------+
   |    list[TileSpec-  |
   |       ialization]  |
   +--------------------+

   +-----------------------+
   | HardwareMapper (ABC)  |
   |  - sub-classed per    |
   |    architecture       |
   |  - holds a Hardware-  |
   |    ResourceModel      |
   |  - exposes:           |
   |     allocate(graph)   |
   |     -> GraphHardware- |
   |        Allocation     |
   +-----------------------+
        ^
        | inherits
        +----------------- KPUMapper, GPUMapper, CPUMapper, etc.
                           (registered in mappers/__init__.py)
```

---

## Capability comparison: KPU vs GPU/CPU/NPU vs RFC 0001 ComputeProduct

| Capability | KPU today | GPU today | CPU today | NPU today | RFC 0001 spec |
|---|---|---|---|---|---|
| Top-level Entry class | `KPUEntry` | `GPUEntry` | `CPUEntry` | `NPUEntry` | `ComputeProduct` (single) |
| ProcessNode reference | yes, by id | inline `DieSpec` only | enum value only | none | yes, shared field |
| Silicon bin (per-block transistor accounting) | yes | none | none | none | optional, uniform shape |
| Per-profile DVFS (vdd_v + cooling ref) | yes | none (single TDP) | none (single TDP) | none | yes, via `power.modes` |
| Derived TDP (V^2*f power model) | yes | none | none | none | yes, planned |
| Validator framework | yes, pluggable registry | none | none | none | yes, planned |
| Generator with input/output split | yes | none | none | none | yes, planned |
| Floorplanner | yes (Stage 8) | none | none | none | (deferred per RFC 0001) |
| Naming convention with foundry/library tags | yes (this sprint) | legacy | legacy | legacy | yes, implied |
| Discriminated `blocks` list (heterogeneous SoC) | tile-class list (KPU-only flavor) | none | none | none | **central design element of RFC 0001** -- KPU's `tiles` does not generalize |
| Hierarchy via `contains` (board -> chips -> dies) | none | none | none | none | **central design element of RFC 0001** -- not yet present anywhere |

---

## Honest claim a human can validate

**True:** Within its own category, the KPU is now a complete model with:

1. ProcessNode reference (full per-foundry per-library catalog system)
2. Silicon decomposition (`silicon_bin` per-block transistor accounting)
3. Derived TDP from `(clock, Vdd, ProcessNode energies, WorkloadAssumption)`
   via the new `kpu_power_model` (5-term roll-up with V^2*f scaling)
4. Per-profile DVFS (clock, vdd_v, cooling_solution_id, activity_factor)
5. Cooling-solution references with cross-checked thermal envelopes
6. Pluggable validator framework (electrical / area / energy / thermal /
   reliability / geometry / consistency)
7. Generator with input/output split and round-trip-safe regeneration
8. Floorplanner with two views (circuit-class + architectural-role) and
   geometric validators
9. A comparison-protocol-friendly naming convention
   (`kpu_t<count>_<rows>x<cols>_<mem><ch>_<value>nm_<foundry>_<library>`)
   query-able via `cli/list_kpus.py --foundry / --node-nm / --library`

The CPU/GPU/NPU schemas are at the parallel-but-shallower level they were
before this sprint. None of them reference ProcessNode, none have a silicon
bin, none have per-profile DVFS, none have a validator framework or generator
or power model.

**False:** The KPU is NOT yet "modeled as a generic ComputeProduct."

There is no `ComputeProduct` class. RFC 0001 captures the unification plan
(single spine + `blocks: list[Block]` discriminated union + recursive
`contains` for hierarchy) but is still **Draft**. Two features central to
RFC 0001 are not present anywhere in the codebase:

- The discriminated `blocks` list. The KPU's `KPUArchitecture.tiles` is
  shaped specifically for spatial mesh accelerators -- it can not directly
  serve as the unified `Block` type. RFC 0001 calls for `kind: kpu | gpu |
  cpu | npu | ...` discriminator with per-kind fields.
- Recursive `contains: list[ProductRef]`. There is no way today to express
  "this DGX H100 board contains 8x H100 + dual Sapphire Rapids + NVSwitch"
  as a single product hierarchy.

---

## Voltage and clock domain caveat (single-rail single-clock breaks DVFS modelling)

The KPU's `KPUThermalProfile` carries a single `vdd_v` and a single `clock_mhz`
that apply chip-wide. That captures monolithic edge KPUs reasonably -- one
voltage rail, one clock domain, every tile on the same DVFS schedule. It
breaks for the products where aggressive DVFS is the central performance /
power lever:

| Product | Independent voltage / clock domains |
|---|---|
| NVIDIA Hopper (H100) | compute clock (~1980 MHz boost), memory clock (HBM3 5.2 GHz), fabric/uncore clock, I/O clock -- all DVFS independently |
| AMD Zen 4 | per-core voltage / clock via Curve Optimizer (each of 16 cores); CCD vs IOD on separate rails |
| Apple M3 | GPU performance-state rail, ANE rail, P-cluster rail, E-cluster rail, fabric rail -- all independent |
| Intel hybrid (Alder Lake+) | P-core ratio domain, E-core ratio domain, ring/uncore domain |
| Jetson Orin AGX | per-rail VDD on Carmel CPU cluster, GPU, DLA0/DLA1, PVA, NVENC, NVDEC -- nvpmodel changes 6+ rails |
| FD-SOI (12FDX, GF 22FDX) | body-bias domains in addition to Vdd; ULL retention domain at near-threshold |
| 3D-stacked V-Cache | TSV-coupled but on its own voltage rail (Vdd_cache != Vdd_core) |

**Aggressive DVFS policies that today's model can't express:**

- Race-to-sleep on the memory rail while compute holds at boost
- Droop-compensation guard-bands per rail
- Per-block sleep states (NVENC powered-off while GPU active)
- Asymmetric DVFS during heterogeneous compute (P-cores at boost, E-cores at idle)
- Body-bias on FD-SOI in addition to Vdd
- AVFS / per-instance adaptive voltage (each SM picks its own Vdd based on
  ring-oscillator measurements; common on Hopper / Blackwell)

**What needs to change in the schema:**

```
Die
  - voltage_rails: list[VoltageRail]      # NEW per-die rails
      VoltageRail:
        - rail_id (e.g., "vdd_compute", "vdd_l2", "vdd_phy",
                  "vdd_uncore", "vdd_p_cluster", "vdd_e_cluster")
        - nominal_v
        - operating_range_v: (min_v, max_v)
        - powers_blocks: list[block_id]   # which blocks share this rail
        - body_bias_supported: bool       # FD-SOI rails have FBB/RBB
        - body_bias_range_mv: (min, max) | None

  - clock_domains: list[ClockDomain]      # NEW per-die clock domains
      ClockDomain:
        - domain_id (e.g., "compute_clk", "mem_clk", "fabric_clk",
                    "io_clk", "p_core_clk", "e_core_clk")
        - base_mhz, max_mhz
        - drives_blocks: list[block_id]   # which blocks share this clock
        - source_rail: rail_id            # voltage rail this domain runs on
                                          # (clock and voltage usually but
                                          # not always co-vary)

ThermalProfile (per-product, per operating point):
  - vdd_by_rail: dict[rail_id, float]     # multi-rail Vdd per profile
  - body_bias_by_rail: dict[rail_id, int] | None    # mV, FD-SOI only
  - clock_by_domain: dict[domain_id, float]         # multi-clock per profile
  - per_die_tdp_w: dict[die_id, float]
  - cooling_solution_id
  - active_blocks: list[block_id] | None  # NEW -- which blocks are powered
                                          # in this profile (for sleep states)
```

The current single `vdd_v` and single `clock_mhz` on `KPUThermalProfile`
become a special case where every block is on a single rail / domain. The
power model already scales by `(vdd / nominal_vdd)^2`; with multi-rail it
applies the rail-specific Vdd per block.

Body-bias on FD-SOI rails is what makes the GF 12FDX SKUs interesting --
FBB raises Fmax at the same Vdd, RBB crashes leakage. The current schema
captures `body_bias_supported` and `back_bias_range_mv` on `ProcessNodeEntry`
but doesn't express the per-rail bias choice in a thermal profile. That's a
hole even today for the 12FDX sweep SKUs.

## Floorsweeping / harvested SKU caveat (separate ComputeProducts, linked by parentage)

Real silicon ships in multiple SKUs with different sections enabled to
recover yield. The same physical die family often spans 3-6 catalog SKUs:

| Silicon family | Shipping SKUs |
|---|---|
| NVIDIA GH100 (144 SMs designed) | H100 SXM5 (132 SMs / 80 GB HBM3 / 6-stack), H100 PCIe (114 SMs / 80 GB / 5-stack), H100 NVL (132 SMs / 188 GB / paired-card), H200 SXM5 (132 SMs / 141 GB HBM3e), H800 (export-control restricted), CMP HX (compute-only mining variant) |
| NVIDIA GB200 | B100, B200, B200 NVL, GB200 NVL72 (board-level) |
| AMD Zen 4 CCD | Ryzen 7950X (16C, 2 CCDs), 7900X (12C, harvested), 7800X3D (8C + V-Cache), 7700X (8C, no V-Cache), 7600X (6C, harvested CCD) |
| Apple M3 family | M3 (8 GPU cores), M3 Pro (14-18 GPU cores, harvested die), M3 Max (30-40 GPU cores, separate die actually), M3 Ultra (2x M3 Max bonded) |
| Intel Raptor Lake | i9-13900K (24C), i7-13700K (16C, harvested), i5-13600K (14C, more harvesting) |

**Two modelling approaches and the one to pick:**

| Option | Pro | Con |
|---|---|---|
| (A) Each SKU is its own `ComputeProduct` | Simple queries (no resolve-enables pass); matches how vendors publish specs; each SKU's price/availability/binning is a market fact not derived; the catalog is queryable directly | More YAML files (one per SKU); shared metadata duplicated across the family unless deduplicated by reference |
| (B) One `ComputeProduct` per silicon family + enable/disable flags + per-variant overrides | Less data duplication; the silicon family is a first-class concept; yield / cost model is direct | Every consumer needs an "apply variant" pass before reading config; mapper / estimator paths get conditional on variant; pricing / availability becomes per-variant which awkwardly nests under a single product |

**Recommendation: option (A)** -- each shipping SKU is its own ComputeProduct,
with an optional `harvested_from: ProductRef` field linking to the
full-silicon parent. As-shipped config is the source of truth for performance
modelling; the parent reference is metadata for yield / cost modelling and
for "which SKUs come from the same die?" queries.

```
ComputeProduct
  - harvested_from: ProductRef | None     # NEW -- optional parent (full silicon)
                                          # h100_sxm5_80gb.harvested_from =
                                          #   gh100_full_silicon
  - dies: list[Die]                       # AS-SHIPPED -- 5 HBM stacks on
                                          # H100 PCIe, 6 on the parent

  Die.blocks reflects AS-SHIPPED config:
    KPUBlock:
      - num_tiles: 132                    # AS-SHIPPED enabled count
      (no num_tiles_max -- look up the parent product if you want
       the full-silicon count)

  market:
    - binning_class: enum                 # NEW -- reference (full silicon) |
                                          #        primary | salvage | export_restricted
    - sibling_skus: list[ProductRef] | None    # OPTIONAL siblings from same parent
                                                #   (could be derived from
                                                #    reverse-traversing harvested_from)
```

Worked example -- the GH100 silicon family:

```
ComputeProduct(id="nvidia_gh100_full_silicon")
  binning_class: REFERENCE
  dies: [Die(blocks: [GPUBlock(sms=144, ...)]),
         Die(blocks: [MemoryBlock(kind=hbm3, capacity_gb=16)]) x 6]
                                          # 6-stack HBM = 96 GB max
  market: { is_available: false }         # not a shipping product

ComputeProduct(id="nvidia_h100_sxm5_80gb")
  harvested_from: { id: "nvidia_gh100_full_silicon" }
  binning_class: PRIMARY
  dies: [Die(blocks: [GPUBlock(sms=132, ...)]),    # 12 SMs disabled
         Die(blocks: [MemoryBlock(kind=hbm3, capacity_gb=16)]) x 5]
                                          # 5-stack = 80 GB; 6th stack
                                          # disabled or absent
  market: { is_available: true, msrp_usd: 30000 }

ComputeProduct(id="nvidia_h100_pcie_80gb")
  harvested_from: { id: "nvidia_gh100_full_silicon" }
  binning_class: SALVAGE                  # heavier harvesting + lower TDP
  dies: [Die(blocks: [GPUBlock(sms=114, ...)]),    # 30 SMs disabled
         Die(blocks: [MemoryBlock(kind=hbm3, capacity_gb=16)]) x 5]
  market: { is_available: true, msrp_usd: 25000 }

ComputeProduct(id="nvidia_h800_sxm5")
  harvested_from: { id: "nvidia_gh100_full_silicon" }
  binning_class: EXPORT_RESTRICTED
  dies: [Die(blocks: [GPUBlock(sms=132, ...,
                               nvlink_gbps_capped=400)])  # vendor-disabled
                                                          # FP64 + bandwidth
                                                          # for export rules
         Die(blocks: [MemoryBlock(kind=hbm3, capacity_gb=16)]) x 5]
  market: { is_available: true, available_in_regions: [china] }
```

Why this shape works:

- **Performance / power modelling** reads the as-shipped `dies + blocks`
  directly; no resolve-enables logic needed in the mapper / estimator hot
  path. This matters a lot -- `graphs.estimation.unified_analyzer` and the
  mappers shouldn't need to know about harvesting at all.
- **Yield / cost modelling** traverses the `harvested_from` chain and reads
  the parent's `silicon_bin` (full silicon area is a physical property of
  the die, not what's enabled) to compute defect-density-driven yield
  equations.
- **"Which SKUs share silicon?"** is a graph traversal (group by
  `harvested_from.id`).
- **Vendor publishes new harvest** = add one new ComputeProduct YAML;
  parent and other siblings are unaffected.
- **Per-SKU pricing, availability, binning class** are first-class fields
  on each SKU, not nested overrides.

What lives where:

| Property | Lives on | Why |
|---|---|---|
| `silicon_bin` | parent (full silicon) | Physical property of the die; doesn't change with harvesting |
| `dies[].die_size_mm2` | parent | Physical property of the die |
| `dies[].process_node_id` | parent | Physical property of the die |
| `dies[].blocks` | each shipping SKU | As-shipped enabled count; **differs per SKU** |
| `dies[].voltage_rails` | parent | Physical wiring -- can't be added per-SKU |
| `power.thermal_profiles` | each shipping SKU | Per-SKU TDP envelope (often differs: H100 SXM5 700W vs PCIe 350W) |
| `market.binning_class` | each shipping SKU | Marketing fact |
| `harvested_from` | each shipping SKU | Pointer up |

The KPU is monolithic AND has no harvested variants today, so today's
single-SKU-per-silicon model works. But the unified ComputeProduct needs
this concept to handle the GPU and CPU SKUs already in the catalog
(`nvidia_h100_sxm5_80gb` and `intel_core_i7_12700k` both have harvested
siblings in the wild). Adding `harvested_from` is cheap, and adding it now
prevents the catalog from growing into a soup of independent SKUs that
"happen to look similar" -- the parentage relationship would be
recoverable only by humans inspecting names.

## Memory caveat (memory dies are first-class, not a chip-level field)

The KPU's `KPUMemorySubsystem` packs memory as a single homogeneous chip-level
field (one memory type, one bandwidth, on-die caches as per-tile/per-PE
constants). That works for a monolithic edge KPU where all memory is either
in `silicon_bin` (L1/L2/L3 SRAM as part of the compute die) or off-package
LPDDR. It breaks for the entire class of products where memory is **its own
die or stack of dies on-package**:

| Product | Memory dies on-package |
|---|---|
| AMD Ryzen 7800X3D / EPYC 9684X | 1 CCD + 1 V-Cache SRAM die (3D-stacked via TSVs) |
| AMD MI300A | 6 GPU chiplets + 3 CPU chiplets + IOD + **8 HBM3 stacks** |
| AMD MI300X | 8 GPU chiplets + IOD + **8 HBM3 stacks (192 GB)** |
| NVIDIA H100 SXM5 | 1 GPU die + **5 HBM3 stacks (80 GB)** |
| NVIDIA B100 / B200 | 2 GPU dies + **8 HBM3e stacks (192 GB)** |
| Grace-Hopper | Grace CPU + Hopper GPU + **6 HBM3 stacks** + 480 GB LPDDR5X on Grace |
| Apple M3 Max | 1 SoC die + **8x LPDDR5 dies on-package** (PoP) |
| Jetson Orin AGX | 1 SoC die + **64 GB LPDDR5 on-module** |
| Samsung HBM-PIM | HBM stacks with **per-stack compute** -- memory IS compute |

Each memory die has properties of its own that don't fit a flat `memory:`
field on the parent ComputeProduct:

- **Process node** -- HBM uses DRAM-friendly older nodes (10-12 nm class)
  different from the compute die's logic node. V-Cache is on the same node
  as the CCD it stacks onto. LPDDR PoP dies are on yet another node. Each
  needs its own `process_node_id`.
- **Capacity, bandwidth, latency, energy-per-byte** -- per stack /
  per die.
- **Interconnect to compute die(s)** -- HBM via CoWoS / EMIB interposer
  (~1024-bit per stack), V-Cache via TSV (through-silicon vias), LPDDR on-
  package via PoP traces, GDDR via PCB. Each has its own bandwidth,
  latency, and energy-per-byte. The same `Interconnect` first-class entity
  proposed for inter-compute-die links covers this.
- **Power budget** -- HBM stacks are 5-15 W each; V-Cache is ~2-3 W.
  Aggregating to a chip-level TDP loses this.
- **Internal hierarchy** -- an HBM stack is itself 4-12 DRAM dies +
  a base logic die, connected via TSVs. From the ComputeProduct
  perspective the stack acts as one logical Die, but the per-stack metadata
  needs `dies_in_stack` and `stack_height_um` to support thermal /
  reliability analysis.

There are three categories of memory to model, and they live in different
parts of the schema:

| Memory category | Examples | Where it lives |
|---|---|---|
| On-die SRAM (caches) | L1/L2/L3 inside compute silicon | Compute Die's `silicon_bin` (already correct -- KPU model does this for L2/L3) |
| On-package memory dies | HBM stacks, V-Cache, PoP LPDDR | First-class entries in `dies: list[Die]` with `MemoryBlock` blocks; `Interconnect`s describe how they connect to compute dies |
| Off-package memory | Socketed DIMMs, GDDR on PCB, CXL modules | Modelled at board / system level via `contains: list[ProductRef]` |

The "on-package vs off-package" boundary is fuzzy (Jetson on-module LPDDR
behaves as on-package; AM5 socketed DDR5 is off-package) -- the rule is
"is the memory die in the same physical package boundary that the product
ships as a unit?"

**My earlier migration sketch had `memory:` as a flat chip-level field**
under ComputeProduct -- that repeats the same monolithic-only flaw the
chiplet caveat below identifies for `process_node_id`. The shape needs to
move memory dies into `dies: list[Die]` and reduce the chip-level
`memory:` to off-package summary information only.

## Chiplet caveat (the KPU template does NOT generalize as-is)

The KPU is monolithic: one die, one process node, one `silicon_bin`. So
`process_node_id` and `silicon_bin` are top-level fields on `KPUEntry` and
that works correctly. But promoting this shape to a universal
`ComputeProduct` would break the products RFC 0001 explicitly cites as
motivation:

| Product | Different process nodes per die |
|---|---|
| AMD MI300A | CDNA3 GPU chiplets on N5 + IOD on N6 |
| Intel Meteor Lake | Compute tile (Intel 4) + GPU tile (TSMC N5) + SoC tile (TSMC N6) + IOE tile (TSMC N6) |
| AMD Zen 4 (Ryzen 7000) | CCDs on TSMC N5 + IOD on TSMC N6 |
| AMD Zen 2/3 (EPYC Rome/Milan) | CCDs on TSMC N7 + IOD on GF 12LP |
| NVIDIA Grace-Hopper | Grace on TSMC N4P + Hopper on TSMC N4 (different stepping) |
| Apple M-series Ultra | Two M-Max dies bonded via UltraFusion |

The implications generalize beyond `process_node_id`:

- **`silicon_bin` must be per-die** -- each die has its own circuit-class
  area decomposition, leakage profile, and densities.
- **`die_size_mm2` and `transistors_billion`** are correctly defined as
  sums-across-dies in `KPUDieSpec` today, but for a chiplet ComputeProduct
  the per-die values are also needed (for per-die area validation, per-die
  thermal modelling).
- **Inter-die interconnect is a first-class entity** that doesn't exist in
  the schema today: NV-HBI on B100, Infinity Fabric on MI300, UltraFusion
  on M-Ultra, NVLink-C2C on Grace-Hopper, EMIB on Sapphire Rapids HBM.
  Bandwidth, latency, energy-per-byte, and topology of inter-die links
  are all needed for accurate fabric modelling on these products.
- **Per-die thermal/power allocation** matters when one die in the package
  is the hot spot (e.g., MI300A's GPU chiplets dominate the thermal
  envelope vs the IOD). Today's `KPUThermalProfile` is chip-wide; for
  chiplets we'd want to either keep chip-wide TDP and add per-die thermal
  contribution, or move to per-die thermal profiles with a package-level
  cooling envelope.

**RFC 0001's own schema sketch has the same flaw** (`physical.process_node_nm`
at the top level). The unification plan should be revised to put these
fields in a per-die structure before implementation, otherwise the new
schema bakes in a chiplet bug at the spine.

## Suggested next concrete migration step (revised for chiplet support)

The right shape splits chip-level facts (packaging, market, top-level power
envelope, on-package memory) from per-die facts (process node, silicon
bin, area, transistors, the architectural blocks it carries):

```
ComputeProduct
  - id, name, vendor
  - packaging: { kind, num_dies, package_type, ... }
  - dies: list[Die]                       # NEW -- per-die structure
      Die:
        - die_id (e.g., "ccd0", "iod", "compute_tile_p",
                  "hbm3_stack_0", "v_cache_die")
        - die_role: enum                   # COMPUTE | MEMORY | IO | BRIDGE | MIXED
        - process_node_id                  # PER-DIE (HBM uses different node)
        - die_size_mm2, transistors_billion
        - silicon_bin                      # PER-DIE area decomposition
        - blocks: list[Block]              # discriminated union per RFC 0001
                                           # KPUBlock | GPUBlock | CPUBlock |
                                           # NPUBlock | DSPBlock | MemoryBlock |
                                           # IOBlock | BridgeBlock | ...
        - clocks (per-die clock domain)
        - dies_in_stack: int = 1           # for HBM stacks (4-12 DRAM dies +
                                           #   1 base die); 1 for non-stacked
        - stack_height_um: float | None    # for thermal modelling

  Block (discriminated union, kind: ...)
    KPUBlock:    tiles + NoC + memory subsystem (today's KPUArchitecture)
    GPUBlock:    SMs, tensor cores, RT cores, etc.
    CPUBlock:    cores (P/E split), caches, vector ext
    NPUBlock:    NPU type, dataflow class, peak TOPS
    DSPBlock:    VLIW + vector + tensor units
    MemoryBlock:                            # NEW -- on-package memory dies
      - kind: hbm2 | hbm3 | hbm3e | hbm4 |
              sram_3d_stacked (V-Cache) | lpddr5 | lpddr5x |
              ddr5 | gddr6 | gddr6x | gddr7 | hbm_pim
      - capacity_gb
      - bandwidth_gbps                      # per stack/die
      - bus_width_bits                      # 1024 for HBM, etc.
      - data_rate_mbps                      # 6400 for HBM3, etc.
      - latency_ns                          # tCAS-equivalent
      - energy_pj_per_byte                  # PHY-side + array access
      - num_channels                        # per die/stack
      - has_compute: bool = False           # True for HBM-PIM
    IOBlock:    PCIe complex, NVLink controller, NIC
    BridgeBlock: NVSwitch, IOD coherence fabric

  - interconnects: list[Interconnect]      # NEW -- ALL inter-die links
      Interconnect:
        - kind: nvlink_c2c | nv_hbi | infinity_fabric | ultra_fusion |
                emib | tsv_3d_stack | cowos | hbm_phy | nvswitch |
                pcie | lpddr_pop | gddr_pcb
        - bandwidth_gbps, latency_ns, energy_pj_per_byte
        - connects: [die_id, die_id]       # which dies it connects
        - lanes / channels (kind-specific)

  - memory_summary: { ... } | None         # OPTIONAL roll-up for queries
                                           # e.g., total_on_package_gb,
                                           # peak_aggregate_bandwidth_gbps;
                                           # derived from dies + interconnects,
                                           # not a source of truth

  - power:                                 # chip-level envelope
      tdp_watts (sum across dies), thermal_profiles: list[ThermalProfile]
      ThermalProfile:
        - cooling_solution_id              # package-level cooling
        - per_die_tdp_w: dict[die_id, float]    # per-die power allocation
                                                # (HBM stack ~5-15W each;
                                                #  V-Cache ~2-3W;
                                                #  compute die remainder)
        - clock_mhz, vdd_v can vary per-die (need per-die clock+Vdd here too)

  - market
  - contains: list[ProductRef]             # board/system-level: { id, count }
                                           # also where socketed DIMMs / CXL
                                           # modules / GDDR on PCB go
```

Worked example -- NVIDIA H100 SXM5 (1 GPU die + 5 HBM3 stacks):

```
ComputeProduct(id="nvidia_h100_sxm5_80gb")
  dies:
    - Die(die_id="gh100", die_role=COMPUTE,
          process_node_id="tsmc_n4",
          die_size_mm2=814, transistors_billion=80,
          blocks: [GPUBlock(sms=132, tensor_cores=528, ...)])
    - Die(die_id="hbm3_stack_0", die_role=MEMORY,
          process_node_id="sk_hynix_1a_dram",
          die_size_mm2=92, transistors_billion=24,
          dies_in_stack=12,                    # 12-Hi HBM3
          blocks: [MemoryBlock(kind=hbm3, capacity_gb=16,
                               bandwidth_gbps=819, bus_width_bits=1024,
                               num_channels=16)])
    - Die(die_id="hbm3_stack_1", ...)         # 4 more HBM3 stacks
    - ...
  interconnects:
    - Interconnect(kind=hbm_phy, bandwidth_gbps=819,
                   connects=["gh100", "hbm3_stack_0"])
    - ...                                      # 4 more HBM PHY links
  power:
    thermal_profiles:
      - per_die_tdp_w: {gh100: 600, hbm3_stack_0: 12, hbm3_stack_1: 12, ...}
        cooling_solution_id: "datacenter_dtc_700w"
```

Worked example -- AMD Ryzen 7800X3D (1 CCD + 1 V-Cache + 1 IOD):

```
ComputeProduct(id="amd_ryzen_7800x3d")
  dies:
    - Die(die_id="ccd0", die_role=COMPUTE,
          process_node_id="tsmc_n5",
          blocks: [CPUBlock(cores=8, isa=x86_64, vector_ext=[avx2, avx512])])
    - Die(die_id="v_cache_0", die_role=MEMORY,
          process_node_id="tsmc_n7",            # different node from CCD
          blocks: [MemoryBlock(kind=sram_3d_stacked, capacity_gb=0.064,
                               bandwidth_gbps=2000, latency_ns=4,
                               energy_pj_per_byte=0.3)])
    - Die(die_id="iod", die_role=IO,
          process_node_id="tsmc_n6",            # third node
          blocks: [IOBlock(pcie_lanes=24, ...),
                   MemoryBlock(kind=ddr5, capacity_gb=0,  # DRAM is off-pkg
                               num_channels=2)])           # MC for 2-ch DDR5
  interconnects:
    - Interconnect(kind=tsv_3d_stack, bandwidth_gbps=2000,
                   connects=["ccd0", "v_cache_0"])         # V-Cache to CCD
    - Interconnect(kind=infinity_fabric, bandwidth_gbps=64,
                   connects=["ccd0", "iod"])               # CCD to IOD
  contains: [{id: "ddr5_5200_dimm_16gb", count: 2}]        # off-package DRAM
```

Migration steps:

1. Define the new `ComputeProduct` + `Die` + `Block` (discriminated union) +
   `Interconnect` + `ThermalProfile` Pydantic classes in
   `embodied_schemas/compute_product.py`. Per-die `process_node_id` and
   `silicon_bin` from day one.
2. Move `KPUArchitecture` (tiles + NoC + memory subsystem) into a
   `KPUBlock(Block)` that lives inside a `Die`. For the KPU's monolithic
   case this means one `Die` containing one `KPUBlock` -- the existing
   data passes through unchanged in shape, just nested one level deeper.
3. Add `GPUBlock`, `CPUBlock`, `NPUBlock`, `DSPBlock` shapes (initially
   carrying just what the existing `GPUEntry.compute` / `CPUEntry.cores` /
   `NPUEntry.compute` already hold). Each lives inside a `Die` so a
   monolithic GPU is one Die with one GPUBlock; a chiplet AMD MI300A
   becomes multiple Dies on different process nodes with different blocks.
4. Add `Interconnect` populated for chiplet products; left empty for
   monolithic.
5. Add `contains: list[ProductRef]` for board / system level products.
6. Migrate one monolithic SKU per category as proof (existing `KPUEntry`
   loaders write into the new schema, then read back). Then migrate two
   chiplet products end-to-end to validate the per-die scoping:
   - **AMD MI300A** -- canonical stress test for compute chiplets on
     mixed nodes + 8 HBM3 stacks + IOD; exercises every block kind in
     one product.
   - **AMD Ryzen 7800X3D** -- canonical stress test for 3D-stacked
     SRAM memory die (V-Cache); validates the `tsv_3d_stack`
     interconnect kind and the `MemoryBlock(kind=sram_3d_stacked)`
     shape against a real product where the SRAM die is on a different
     process node from the compute die it stacks onto.
7. Promote the validator/generator/power-model frameworks from KPU-specific
   (`graphs.hardware.kpu_*`) to ComputeProduct-aware:
   - Validators iterate over dies + blocks, dispatch by `block.kind`.
   - Power model rolls up per-die contributions (each die has its own
     ProcessNode -> energies, each die has its own voltage rails and
     clock domains) and adds inter-die interconnect power.
   - Generator becomes per-die area/transistor roll-up + chip-level
     aggregation; per-rail Vdd, per-domain clock fed into the power model
     as a vector instead of two scalars.
8. Keep the legacy `GPUEntry` / `CPUEntry` / `NPUEntry` / `KPUEntry`
   schemas as deprecation shims for one or two releases; loaders return
   `ComputeProduct` instances regardless of source format.

The RFC 0001 sketch should be amended to reflect per-die scoping (process
node, silicon bin, area), per-die voltage rails and clock domains,
first-class memory dies (`MemoryBlock` in the discriminated union), and
the harvested-SKU pattern (separate ComputeProducts linked by
`harvested_from`) before any implementation begins.

## Thermal and cooling caveat (per-die limits, stack coupling, throttle policy, time-domain, TIM, temp grade, scope)

Today's thermal model has `KPUThermalProfile.cooling_solution_id` per profile,
`CoolingSolutionEntry` carrying `max_total_w` / `max_power_density_w_per_mm2`
/ `mechanism`, and a chip-level `idle_power_watts` derived from the leakage
model. The cross-check panel in `show_compute_product.py` compares per-profile
TDP against the cooling envelope and flags violations.

That works for a monolithic edge KPU with a single cooling solution covering
the package. It misses everything that arises once dies have different
thermal characteristics, or once cooling becomes a real-time problem rather
than an envelope check.

### 1. Per-die junction temperature limits

Different dies have different `Tj_max` and different sensitivity to it:

| Die kind | Typical Tj_max (commercial) | Behaviour at limit |
|---|---:|---|
| Logic (HP/balanced) | 95-105 C | DVFS throttle; reliability degrades exponentially above limit |
| HBM3 / HBM3e | 85-95 C | Refresh rate doubles at ~85 C, full throttle ~95 C |
| LPDDR5 / LPDDR5X | 95 C | Refresh rate ramps; controller-managed |
| GDDR6 / GDDR6X | 100-110 C | Self-refresh rate, then hard-stop at junction |
| 3D V-Cache (SRAM) | 95 C | Limits the CCD's max Vdd (X3D parts have lower boost) |
| FD-SOI ULL retention | 125 C | Designed for always-on; near-threshold |
| Automotive grade | 125 C | AEC-Q100 Grade 2 |
| Industrial grade | 105 C | Higher than commercial, lower than automotive |

The chip-level `idle_power_watts` (= total leakage) and the cross-check's
chip-level TDP miss that an HBM stack at 14W in a hot corner of the
package is a thermal violation independent of the chip-level TDP being
within the cooling envelope.

### 2. 3D-stack thermal coupling

This is the canonical example everyone gets wrong: AMD's X3D parts have
*lower* max boost than their non-X3D siblings (7800X3D boost 5.0 GHz vs
7700X 5.4 GHz). The reason is purely thermal -- the V-Cache SRAM die
stacked on top of the CCD blocks heat escape, raising the effective
junction-to-case thermal resistance of the CCD.

```
            ambient
              |
              v
  +------------------------+
  |     heatspreader       |   <- TIM_2 (CCD top to IHS)
  +------------------------+
            |
            v
  +------------------------+
  |   V-Cache SRAM die     |   <- adds R_thermal_above to CCD
  +------------------------+   <- TSV interconnect
  +------------------------+
  |        CCD compute     |   <- hotspot here
  +------------------------+
            |
            v
       substrate / IOD
```

The schema needs to express vertical stacking AND its thermal cost:

```
Die
  - thermal: DieThermal             # NEW
      DieThermal:
        - tj_max_c                  # junction temp limit
        - rjc_c_per_w               # junction-to-case (no cooling)
        - peak_power_density_w_per_mm2
        - peak_power_density_window_s    # time the hotspot density
                                          # is sustainable
        - thermal_couples_above: [die_id] | None
                                    # which dies sit on top and add
                                    # thermal resistance
        - couples_above_extra_c_per_w: float
                                    # how much each stacked die above
                                    # adds to effective Rjc

  - dies_in_stack                   # already proposed for HBM
  - stack_height_um                 # already proposed
```

The same applies to HBM stacks (each DRAM die in the stack adds thermal
resistance for the dies below it; the bottom die runs hottest because
it's furthest from the heatspreader).

### 3. Throttle policy is per-die, not chip-wide

When the cooling envelope is exceeded the chip throttles, but *what* it
does differs by die:

| Die kind | Throttle response |
|---|---|
| Compute logic | DVFS: drop clock first, then Vdd, then both. Modern parts use AVFS to find the lowest stable point per-instance. |
| HBM | Refresh-rate ramp (no Vdd/clock change at the DRAM die; controller manages); full stop at limit |
| Memory PHY (compute side) | Drop bandwidth (lane width, data rate) before stopping |
| Fixed-function (NVENC, ISP) | Hard-stop (frame drop) -- can't DVFS without breaking the IP |
| HBM-PIM | Drop compute first (preserve memory function), then memory stops too |

```
Die
  - thermal: DieThermal
      ...
      - throttle_policy: ThrottlePolicy   # NEW
          ThrottlePolicy:
            - kind: dvfs | refresh_ramp | bandwidth_step |
                    hard_stop | tile_powergate
            - throttle_steps: list[ThrottleStep]    # what reductions
                                                    # are available
            - host_signal: bool   # does the die signal the host
                                  # before throttling? (NVML / RAPL etc.)
```

The current `efficiency_factor_by_precision` and
`tile_utilization_by_precision` on `KPUThermalProfile` partially capture
"what fraction of peak does the chip realize at this profile" -- that's
a coarse proxy for throttle behaviour. The new `throttle_policy` says
*what mechanism* causes the realized fraction to differ from peak.

### 4. Time-domain thermal: boost vs sustained

`tdp_watts` is a single number. Real chips have:

- **Peak power** (instantaneous, 100us scale): C * V^2 * f at full activity
- **Boost power** (seconds scale, while die warms): higher than TDP, time-bounded
- **TDP** (sustained, minutes scale): the cooling solution's continuous capability
- **Idle power** (no workload, leakage + clocks running)
- **Sleep power** (gated, retention only)

Cooling solutions also have time constants (a heatsink with high thermal
mass tolerates short bursts; a thin one doesn't). Workloads have duty
cycles (LLM inference is bursty; training is sustained).

```
ThermalProfile
  - tdp_watts                        # sustained envelope (today's value)
  - peak_power_w                     # NEW -- instantaneous capability
  - boost_power_w                    # NEW -- short-window (seconds) capability
  - boost_window_s                   # how long boost is sustainable
  - sleep_power_w                    # NEW -- with clocks gated, retention only
  - per_die_tdp_w                    # already proposed for chiplet
  - per_die_peak_w                   # NEW -- per-die instantaneous

CoolingSolutionEntry
  - max_total_w                      # sustained (today's value)
  - max_burst_w                      # NEW -- short-window capacity
  - burst_window_s                   # NEW -- how long burst is sustainable
  - thermal_capacitance_j_per_c      # NEW -- heat the cooling can absorb
                                     # before T_case rises 1 C; controls
                                     # how long bursts last
```

NVIDIA's GPU Boost is exactly this -- the chip runs above TDP until junction
temperature rises into the throttle band, then steps down. Without
peak/boost/sustained separation in the schema, the catalog can't represent
this behaviour faithfully.

### 5. Thermal interface material (TIM) is first-class

Two cooling solutions with identical `max_total_w` but different TIM
deliver wildly different `max_w_per_mm2` at the die surface:

| TIM kind | Typical R_cs (C/W) | Notes |
|---|---:|---|
| Standard paste | 0.10 - 0.30 | Default; ages over 2-5 years (pump-out) |
| Premium paste (Conductonaut, etc.) | 0.05 - 0.15 | Better but pump-out worse |
| Solder TIM (indium) | 0.02 - 0.05 | NVIDIA Hopper, AMD high-end; doesn't pump out |
| Liquid metal | 0.02 - 0.04 | Best paste; conductive; risk of spillover |
| Direct die (delidded) | 0 (TIM eliminated) | DIY only; bypasses IHS |
| Phase-change | 0.05 - 0.15 | Industrial; thermal pad alternative |

```
CoolingSolutionEntry
  - tim_kind: TIMKind                # NEW enum
  - r_cs_c_per_w: float              # NEW -- case-to-sink resistance
                                     # (depends on TIM choice)
  - tim_pump_out_years: float | None # NEW -- expected service life
                                     # before TIM degradation; None for
                                     # solder TIM (no pump-out)
```

The current `max_power_density_w_per_mm2` is implicitly TIM-dependent.
Splitting TIM out lets the cooling solution catalog cover one heatsink
SKU shipped with multiple TIM options.

### 6. Temperature grade is per-SKU

Same silicon ships in commercial / industrial / automotive grades with
different ambient and Tj_max guarantees. Often the automotive part is
the same wafer as the commercial part, screened harder for high-temp
operation -- making it a **harvested binning** variant (links back to
the harvested-SKU caveat above).

```
ComputeProduct
  - temp_grade: enum                 # NEW
      commercial   |   # 0..70 C ambient, Tj_max 95-105 C
      extended     |   # -40..85 C
      industrial   |   # -40..85 C, Tj_max 105-125 C
      automotive_grade2 |   # AEC-Q100 G2: -40..105 C
      automotive_grade1 |   # AEC-Q100 G1: -40..125 C
      military     |   # -55..125 C
  - ambient_c_range: (min, max)      # NEW -- supported ambient
```

The KPU catalog already implicitly has this -- the GF 12FDX SKUs are
positioned for industrial / automotive (12FDX has a 125 C Tj_max
characterized in the datasheet) but the schema doesn't express it.

### 7. Cooling scope -- who does this cooling cover?

`cooling_solution_id` on every `KPUThermalProfile` implicitly assumes
"one cooling solution covers the whole package." That's mostly true but
breaks at two extremes:

- **Sub-package**: rare but real -- some HBM stacks have additional
  per-stack heatspreaders or separate liquid loops; some board-level
  designs put a small fan over the HBM area independent of the main heatsink.
- **Super-package**: DGX systems have one chassis cooling that covers
  8 H100s + 2 Sapphire Rapids + NVSwitch + ConnectX -- the cooling
  solution applies at the system level, not per-product.

```
CoolingSolutionEntry
  - scope: CoolingScope              # NEW enum
      per_die       |   # rare: applies to one die in the package
      per_package   |   # common: covers all dies in one package
      per_node      |   # 1U/2U server-level cooling
      per_rack      |   # rack-level direct-to-chip liquid
      per_facility  |   # immersion / heat-recovery
  - applies_to_dies: list[die_id] | None
                                     # used when scope=per_die
                                     # (which die in the package)

ThermalProfile
  - cooling_solution_ids: list[str]  # NEW -- a profile can reference
                                     # multiple coolers (e.g., per-die
                                     # cold plate + chassis fan).
                                     # Single-id is the degenerate case.
```

For a chiplet with HBM stacks the cross-check then becomes per-die: each
die's `peak_power_w` and `peak_power_density_w_per_mm2` are checked
against the cooling solution(s) that cover it.

## Interconnect topology caveat (graphs, not flat fields; switches as first-class)

Interconnect is the most under-developed axis in the current model. The
KPU catches one slice (intra-die `KPUNoCSpec` -- mesh_rows / mesh_cols /
flit_bytes / router_circuit_class / bisection_bandwidth_gbps), and the
chiplet caveat above proposed a flat `Interconnect` list for inter-die
links. Both are point-solutions. The general problem is bigger.

### What today's model captures vs what's needed

| Level | Today | Needed |
|---|---|---|
| Intra-die NoC | `KPUNoCSpec` (one mesh, single topology string) | First-class with multiple NoCs per die, routing kind, VCs, per-router latency / energy |
| Inter-die within package | proposed `Interconnect` list (chiplet caveat) | Topology-aware: hub-and-spoke (AMD IOD), point-to-point (Grace-Hopper), 3D-stack (V-Cache), mesh (multiple GPU chiplets) |
| Inter-package within board / node | **nothing** | PCIe (P2P or via PEX switch), NVLink (P2P or via NVSwitch), xGMI / Infinity Fabric (AMD socket-to-socket), Intel UPI, CXL fabric |
| Inter-node within rack | **nothing** | InfiniBand (HDR / NDR / XDR), Ethernet (RoCE), NVLink Switch fabric, Slingshot, Tofu |
| Inter-rack / cluster | **nothing** | Spine-leaf datacenter networks; partially in scope |
| Storage / IO | **nothing** | NVMe, SAS, IB-storage |
| Coherence domain | **nothing** (implicit) | Which dies / blocks share a coherent address space, vs DMA-only |

The proposed `contains: list[ProductRef]` field can express "DGX H100 has
8x H100 + 4x NVSwitch + 2x Sapphire Rapids + 8x ConnectX-7" as a
membership list, but it does not express **how** those 14 components are
wired together -- which is exactly what a fabric-aware estimator needs to
compute all-reduce latency, model congestion, or evaluate placement.

### Topology is a graph, not a flat field

Real interconnect fabrics have:

- **Nodes**: dies, ports, lanes, switches, NICs
- **Edges**: links with bandwidth, latency, energy/byte, lane count
- **Topology patterns**: how edges form (point_to_point, ring, mesh_2d,
  mesh_3d, torus, fat_tree, dragonfly, all_to_all, hub_and_spoke,
  switched, hierarchical, custom)
- **Routing policy**: deterministic XY, deterministic dimension-order,
  adaptive, oblivious, multi-path
- **Quality-of-service**: virtual channels, traffic classes, priority
- **Coherence overlay**: which links carry coherent traffic, which are DMA-only
- **Asymmetry**: PCIe full-duplex but P2P read vs write to peer memory
  has different realized BW; HBM read vs write at different power; NVLink
  uplink vs downlink in some asymmetric topologies
- **Oversubscription**: switched fabrics often oversubscribe (e.g., a
  fat tree with 1.5:1 from spine to leaf)

Collapsing this into a single `bandwidth_gbps` per `Interconnect` loses
the structure that drives realistic perf/power modelling.

### Real-product examples the current model can't represent

| Product | What today's model misses |
|---|---|
| DGX H100 | 8 GPUs + 4 NVSwitches forming a 3-stage NVLink fabric; cross-section BW = 7.2 TB/s but only at certain communication patterns; PCIe topology to dual SPR + 8 ConnectX-7 NICs is separate |
| AMD MI300X 8-GPU node | 8x MI300X + Infinity Fabric in a fully-connected mesh (every GPU to every other GPU at 128 GB/s per peer); cross-section depends on direction |
| NVIDIA GH200 NVL32 | 32x Grace-Hopper SuperChips connected by NVLink Switch System; 4x NVL32 form an "exascale" cluster |
| Cerebras WSE | Single die with a 2D mesh of 850K cores -- the NoC IS the chip; mesh has fault-tolerant adaptive routing, can map dead sites |
| Google TPU v5p pod | 8960 TPUs in a 3D torus; topology is the architecture; collective operations are torus-aware |
| Tesla Dojo | Hierarchical: D1 chip mesh of cores; tile of 25 D1s mesh; tray of tiles; cabinet of trays. Each level has its own topology + BW |
| Grace-Hopper | Coherent NVLink-C2C between Grace CPU and Hopper GPU (one product); separate non-coherent NVLink to other GH chips (DGX-class system) |
| AMD EPYC dual socket | xGMI between sockets; UMA vs NUMA configuration; coherent across sockets |

### Schema additions

```
Interconnect (extended from prior caveats)
  - id                                   # unique within scope
  - level: InterconnectLevel             # NEW enum
      noc_intra_die       |   # within one die
      die_to_die          |   # within one package (NV-HBI, EMIB, TSV)
      package_to_package  |   # within one board / node (PCIe, NVLink)
      node_to_node        |   # within one rack (IB, Ethernet, NVL Switch)
      rack_to_rack        |   # cross-rack
      storage             |   # NVMe / SAS / IB-storage
  - link_kind: enum                      # PHY-level identification
      nvlink5 | nvlink4 | nvlink_c2c | nv_hbi |
      infinity_fabric | xgmi | upi |
      pcie5 | pcie6 | cxl3 |
      hbm_phy | tsv_3d_stack | cowos | emib | lpddr_pop | gddr_pcb |
      ib_ndr | ib_xdr | roce_400g | nvswitch | slingshot | tofu_d |
      ethernet_400g | ethernet_800g
  - topology: TopologyKind               # NEW enum
      point_to_point | ring | mesh_2d | mesh_3d | torus_2d | torus_3d |
      fat_tree | dragonfly | hypercube | all_to_all | hub_and_spoke |
      switched | hierarchical | custom
  - dimensions: dict[str, int] | None    # NEW -- topology-specific
                                         # mesh: {rows, cols}
                                         # torus_3d: {x, y, z}
                                         # fat_tree: {radix, levels}
  - per_link_bandwidth_gbps              # one direction
  - per_link_bandwidth_full_duplex_gbps  # bidirectional capacity
  - per_link_latency_ns                  # zero-load latency per hop
  - per_link_energy_pj_per_byte          # PHY + protocol overhead
  - num_links                            # total in this topology
  - aggregate_bisection_bandwidth_gbps   # derived; cached for queries
  - oversubscription_ratio: float = 1.0  # for switched / fat-tree
                                         # (1.5:1, 3:1, etc.)
  - coherent: bool                       # cache-coherent or DMA-only
  - routing: RoutingKind                 # NEW enum
      deterministic_xy | deterministic_dim_order |
      adaptive_minimal | adaptive_nonminimal | oblivious_valiant
  - virtual_channels: int = 1
  - asymmetric_bandwidth: bool = False
  - asymmetric_send_gbps / asymmetric_receive_gbps: float | None
  - endpoints: list[EndpointRef]         # what this connects
      EndpointRef:
        - kind: die | port | block | product_ref | switch
        - id
        - port_index: int | None         # for multi-port endpoints
```

```
Switch                                   # NEW first-class entity
  - id
  - kind: nvswitch3 | ib_switch_ndr | ethernet_switch_400g |
          pcie_pex_switch | xgmi_switch
  - port_count
  - per_port_bandwidth_gbps
  - aggregate_bandwidth_gbps             # all ports
  - latency_ns_per_hop                   # cut-through or store-forward
  - power_w
  - is_blocking: bool                    # blocking vs non-blocking
  - process_node_id                      # switches are silicon too
  - die: DieRef                          # often packaged on its own die
```

```
CoherenceDomain                          # NEW overlay structure
  - id
  - members: list[EndpointRef]           # which dies / blocks share
                                         # a coherent address space
  - protocol: MOESI | MESIF | MOESI_F | DASH | custom
  - max_latency_ns                       # worst-case coherence latency
                                         # (matters for false sharing model)
```

```
ComputeProduct (system-level, kind: system)
  - contains: list[ProductRef]           # already proposed
      ProductRef:
        - id
        - count
        - position: dict | None          # NEW -- where in the system
                                         # (rack_unit, socket, bay, ...)
  - interconnects: list[Interconnect]    # promoted to system level
                                         # endpoints can be ProductRef
                                         # (e.g., GPU0 to NVSwitch3)
  - switches: list[Switch]               # NEW -- first-class
  - coherence_domains: list[CoherenceDomain]    # NEW
  - topology_summary: dict               # OPTIONAL roll-up for queries
                                         # e.g., bisection BW at each
                                         # level, hop count matrix
```

### Worked example -- DGX H100 8-GPU node

```
ComputeProduct(id="nvidia_dgx_h100", kind=system)
  contains: [
    {id: "nvidia_h100_sxm5_80gb",   count: 8, position: {bay: 0..7}},
    {id: "nvidia_nvswitch3",        count: 4, position: {bay: switch0..3}},
    {id: "intel_xeon_8480c_spr",    count: 2, position: {socket: 0, 1}},
    {id: "nvidia_connectx7_400g",   count: 8, position: {nic: 0..7}},
  ]
  switches: [
    Switch(id="nvs0", kind=nvswitch3, port_count=64,
           per_port_bandwidth_gbps=900, aggregate_bandwidth_gbps=57600,
           latency_ns_per_hop=300, power_w=70),
    ...                              # 3 more NVSwitch3
  ]
  interconnects: [
    # NVLink fabric: each GPU to all 4 NVSwitches
    Interconnect(level=node_to_node, link_kind=nvlink4, topology=fat_tree,
                 dimensions={radix: 18, levels: 2},
                 per_link_bandwidth_gbps=900,
                 per_link_bandwidth_full_duplex_gbps=1800,
                 per_link_latency_ns=300,
                 per_link_energy_pj_per_byte=1.5,
                 num_links=32,                   # 8 GPUs * 4 switches
                 aggregate_bisection_bandwidth_gbps=7200,
                 oversubscription_ratio=1.0,
                 coherent=False,                  # NVLink is DMA, not coherent
                 routing=adaptive_minimal,
                 virtual_channels=4,
                 endpoints=[
                   {kind=product_ref, id="nvidia_h100_sxm5_80gb", port_index=0..17},
                   {kind=switch, id="nvs0..nvs3"},
                 ]),

    # PCIe: each GPU to its companion NIC + to a CPU socket
    Interconnect(level=package_to_package, link_kind=pcie5, topology=switched,
                 per_link_bandwidth_gbps=64,           # PCIe 5.0 x16
                 per_link_latency_ns=500,
                 num_links=8,
                 coherent=False,
                 endpoints=[
                   {kind=product_ref, id="nvidia_h100_sxm5_80gb"},
                   {kind=product_ref, id="nvidia_connectx7_400g"},
                 ]),

    # UPI: socket-to-socket on the dual-Xeon host
    Interconnect(level=package_to_package, link_kind=upi, topology=point_to_point,
                 per_link_bandwidth_gbps=180,
                 coherent=True,                  # UPI is coherent
                 num_links=4,
                 endpoints=[
                   {kind=product_ref, id="intel_xeon_8480c_spr", port_index=0..3},
                 ]),
  ]
  coherence_domains: [
    # Two NUMA domains, one per CPU socket
    CoherenceDomain(id="numa0", protocol=MESIF,
                    members=[{kind=product_ref, id=cpu0}]),
    CoherenceDomain(id="numa1", protocol=MESIF,
                    members=[{kind=product_ref, id=cpu1}]),
    # GPUs are NOT in the CPU coherence domain (PCIe DMA)
  ]
```

### Worked example -- intra-die KPU NoC (replaces today's `KPUNoCSpec`)

```
Die(die_id="kpu_compute")
  blocks: [
    KPUBlock(...)
  ]
  interconnects: [                       # NEW -- per-die NoCs in here
    Interconnect(id="compute_noc",
                 level=noc_intra_die,
                 link_kind=custom,        # KPU-specific NoC
                 topology=mesh_2d,
                 dimensions={rows: 16, cols: 16},
                 per_link_bandwidth_gbps=128,    # per-flit-direction
                 per_link_latency_ns=1,
                 per_link_energy_pj_per_byte=1.0,
                 num_links=480,                  # 16*15*2 horiz + 15*16*2 vert
                 aggregate_bisection_bandwidth_gbps=2048,
                 routing=deterministic_xy,
                 virtual_channels=2,
                 endpoints=[
                   {kind=block, id=tile_0_0..15_15},   # all tiles
                 ]),

    # If the chip has separate NoCs (compute, coherence, DMA), add them too:
    Interconnect(id="dma_noc", level=noc_intra_die, ...),
  ]
```

The KPU's monolithic single-NoC case becomes one element in the per-die
`interconnects` list. Multi-NoC designs (Cerebras WSE has multiple NoCs;
Tesla Dojo has hierarchical NoCs; Apple has separate compute / coherence
fabrics) become multiple elements.

### Why this shape matters for the estimator path

The whole point of the ComputeProduct unification is that mappers and
estimators in `graphs.estimation` can reason uniformly. For interconnect
this means:

- **Hop count matrix**: estimator wants "how many hops between resource
  A and resource B?" -- a graph BFS over the interconnect list with
  switches as relay nodes
- **Cross-section bandwidth**: "what's the BW between any two halves of
  this fabric?" -- min-cut over the bandwidth-weighted graph
- **Collective operation cost**: all-reduce, all-gather, broadcast costs
  depend on the topology pattern and oversubscription
- **Coherence cost**: "is this access coherent or does it require
  explicit DMA?" -- coherence domain membership lookup
- **Power roll-up**: "what's the dynamic power of the interconnect at
  this workload's BW utilization?" -- per-link energy * byte_rate * util
- **Asymmetric placement**: "which GPU pair has highest BW for this
  collective?" -- topology-aware placement

None of these queries work over a flat `bandwidth_gbps` field. They all
require the interconnect to be a graph with a stable schema for the
nodes and edges.

### KPU today vs the unified model

The KPU is a single-die monolithic part with one NoC, no inter-die or
inter-package interconnect. Migration is structural with no data loss:

- Move `kpu_architecture.noc: KPUNoCSpec` into
  `Die.interconnects: list[Interconnect]` as one element
  (level=noc_intra_die)
- Translate `topology: "mesh_2d"` to enum + `dimensions: {rows, cols}`
- Translate `bisection_bandwidth_gbps` to `aggregate_bisection_bandwidth_gbps`
- Add `routing: deterministic_xy` and `virtual_channels: 2` (best-guess
  defaults; architect refines if needed)
- The KPU has no switches, no inter-die / inter-package fabric, no
  coherence domain -- those lists stay empty

## Software / SDK caveat (realized perf is SDK-gated, not just silicon-gated)

Today's catalog has near-zero coverage of software stack. `NPUEntry` carries
a `SoftwareSpec` (basic SDK name + version + supported frameworks);
`KPUEntry` / `GPUEntry` / `CPUEntry` are silent. But the realized performance
on a chip is gated by the SDK and framework backends as much as by the
hardware -- same H100 silicon delivers very different throughput under naive
CUDA vs cuBLAS vs cuDNN vs TensorRT vs FlashAttention vs cutlass.

Without modelling this, the estimator can't honestly answer **"what will
this workload achieve on this product?"** -- only **"what can this hardware
theoretically do?"** The roofline / energy / latency models in
`graphs.estimation` need to know which SDK is assumed, because:

- Different SDKs have different operator coverage (custom kernels, fused
  attention, sparse kernels)
- Different SDKs have different driver overhead per launch
- Different SDKs ship different kernel libraries with different efficiency
- Frameworks have native vs translated backends (PyTorch on Apple MLX is
  native; PyTorch on AMD via ZLUDA-style translation is not)

### Real-product examples the model can't represent

| Product | SDK ecosystem | Realized vs theoretical |
|---|---|---|
| NVIDIA H100 | CUDA + cuBLAS + cuDNN + TensorRT-LLM + cutlass + Triton + FlashAttention | TensorRT-LLM achieves 2-3x naive PyTorch on LLM inference |
| AMD MI300X | ROCm + MIOpen + RCCL + hipBLASLt + Composable Kernel | ~70% of CUDA-equivalent for many workloads in 2026; gap shrinks every release |
| Apple M3 Max | Metal + MLX + Core ML + Accelerate + MPS for PyTorch | MLX is faster than PyTorch-MPS by 2-3x on transformer inference |
| Intel Xeon SPR | oneAPI + oneDNN + MKL + OpenVINO + IPEX | OpenVINO INT8 achieves 4-8x over FP32 |
| Hailo-8 | HailoRT + Hailo Dataflow Compiler | Vendor-only path; PyTorch model -> Hailo binary; generic dataflow compilation (op graph -> token-firing schedule on Hailo's tile fabric) |
| Stillwater KPU | Stillwater Domain Flow Compiler (proprietary) | Single SDK; no PyTorch backend yet. **Compilation is fundamentally different from generic dataflow**: SURE / SARE programs (systems of recurrence equations) are space-time mapped onto the 2D PE mesh via polyhedral scheduling. Kernel reuse with Cerebras / Hailo / Dojo toolchains is essentially zero -- they share neither IR, scheduler, nor cost model. |
| Tesla Dojo | Dojo compiler + Dojo SDK | Vendor-only; generic dataflow over hierarchical mesh of D1 chips |
| Cerebras WSE | Cerebras SDK + CGCS | Vendor-only; generic dataflow over single-die 850K-core mesh with fault-tolerant routing |

### Schema additions

```
ComputeProduct
  - software: SoftwareSupport            # NEW
      SoftwareSupport:
        - programming_models: list[ProgrammingModel]    # NEW enum
            simt          |   # CUDA, ROCm
            spmd          |   # TPU XLA, OpenCL
            dataflow      |   # Cerebras WSE, Tesla Dojo, Hailo --
                              # token-firing through an op graph; firing
                              # rule is "all inputs available"; structure
                              # can be irregular and runtime-dependent
            domain_flow   |   # Stillwater KPU -- DISTRIBUTED dataflow
                              # executing systems of recurrence equations
                              # (SURE / SARE) over a regular iteration
                              # domain; computation is statically scheduled
                              # by polyhedral space-time mapping onto the
                              # 2D PE mesh. NOT the same as generic
                              # dataflow -- separate compiler discipline.
            vliw          |   # Hexagon DSP, Tensilica
            x86_simd      |   # CPU AVX
            arm_neon_sve  |
            riscv_vector  |
            metal         |   # Apple
            custom

        - sdks: list[SDKSupport]
            SDKSupport:
              - id (e.g., "cuda_12_4", "rocm_6_2", "metal_3_2")
              - kind: cuda | rocm | oneapi | metal | mlx | hailort |
                      tensorrt | xla | iree | mlir | dojo |
                      hailo_dataflow_compiler |
                      stillwater_domain_flow_compiler |
                      custom
                                          # NOTE: domain-flow compiler
                                          # (Stillwater KPU, polyhedral
                                          # SURE/SARE -> mesh) is a
                                          # SEPARATE kind from generic
                                          # dataflow compilers (Hailo,
                                          # Cerebras, Dojo) -- different
                                          # IR, scheduler, cost model
              - vendor: str
              - version
              - lts: bool
              - eol_date: str | None
              - supports_features: list[str]    # tensor_cores, sparse,
                                                # fp8, transformer_engine, ...

        - framework_backends: list[FrameworkBackend]
            FrameworkBackend:
              - framework: pytorch | tensorflow | tflite | jax |
                           onnxruntime | tensorrt_llm | vllm | mlx |
                           sglang | llama_cpp
              - backend_id: str           # "torch.cuda", "torch.mps",
                                          # "ort_trt_ep", "vllm_cuda",
                                          # "jax_xla_pjrt"
              - tier: TierKind            # NEW enum
                  native        |   # vendor first-party (CUDA on NV)
                  vendor_supported | # vendor-supported but not first-party
                                     # (PyTorch on AMD ROCm)
                  community    |    # community port (PyTorch on Apple MLX)
                  translated   |    # translation layer (CUDA-on-ROCm via
                                    # ZLUDA, etc.) -- usually slower
                  experimental
              - performance_relative_to_native: float | None
                                          # 1.0 = same as native; 0.7 =
                                          # 30% slower; useful for
                                          # estimator dispatch

        - inference_runtimes: list[str]   # vLLM, TensorRT-LLM, llama.cpp,
                                          # MLX-LM, sglang, etc.

        - operator_coverage_ref: str | None
                                          # URL or catalog ref to a
                                          # detailed (SDK x op x precision)
                                          # support matrix; lives outside
                                          # the ComputeProduct YAML

        - native_quantization: list[str]  # int8 | int4 | fp8_e4m3 |
                                          # fp8_e5m2 | fp4 | nf4 |
                                          # awq | gptq | gguf
        - native_sparsity: list[str]      # 2:4 | 4:8 | block_sparse | none

  - driver_firmware: DriverFirmwareInfo | None    # NEW (deployment-time)
      - min_driver_version
      - min_firmware_version
      - tested_versions: list[str]
      - determinism_supported: bool       # bit-exact reproducibility
                                          # mode available?
```

For the KPU specifically: today's `multi_precision_alu` field on
`KPUArchitecture` says what the HARDWARE supports; the new `software.sdks`
+ `software.native_quantization` say what the SOFTWARE actually exposes
to user code -- usually a subset.

### Why dataflow vs domain_flow is a load-bearing distinction (not a marketing-tier label)

The `programming_models` enum splits **dataflow** (Cerebras WSE, Tesla
Dojo, Hailo) from **domain_flow** (Stillwater KPU). They sound similar
but the catalog needs to keep them separate because the compilation
pipelines, IRs, schedulers, and cost models are entirely different --
kernel reuse across the boundary is essentially zero.

| Dimension | Generic dataflow (Cerebras / Hailo / Dojo) | Domain flow (Stillwater KPU) |
|---|---|---|
| Program shape | Arbitrary directed graph of operators | System of recurrence equations (SURE / SARE) over an iteration domain |
| Firing rule | Token-driven: an op fires when all inputs are available; can be data-dependent | Statically scheduled: every PE knows what to compute and when via a polyhedral space-time map |
| Scheduling discipline | Operator placement + token routing; runtime-adaptive | Polyhedral space-time scheduling; affine transformations on iteration spaces |
| Compiler IR | Op graph IR (typically an MLIR dialect or proprietary graph IR) | Polyhedral IR (Z-polyhedra, affine schedules, space-time mappings) |
| Cost model | Op latency + token routing congestion | Wavefront timing + PE utilization on the regular mesh |
| Handles irregular control flow | Yes (data-dependent token firing) | No -- requires affine bounds; non-affine code falls back to a host |
| Best fit | Heterogeneous op graphs; vision pipelines with branching | Dense linear algebra; convolutions; structured tensor ops where the iteration domain is a polyhedron |
| Catalog implication | A `Hailo` SDK kernel can plausibly be ported to `Cerebras` with shared IR / scheduler concepts | A `KPU` kernel cannot be ported to a generic dataflow target without redesigning both the IR and the scheduling discipline |

Estimator-path implication: latency / energy models for a generic-dataflow
target predict on operator latency + token routing; for a domain-flow
target they predict on wavefront timing + PE utilization on the iteration
polytope. These are different roofline shapes; the estimator dispatch
needs to know which it's looking at, which is what the
`programming_models` enum is for.

Where it lives: software is a property of the **product** (sometimes the
**product family**), not the silicon. Same H100 silicon underpins every
H100 SXM5 / PCIe / NVL / H200 / H800 SKU; they share the SDK story but
differ on driver feature flags (export-restricted SKUs disable certain
SDKs / runtimes). So `SoftwareSupport` lives on `ComputeProduct`, with
the harvested-SKU links carrying the vendor-disabled feature deltas.

## Sensors caveat (telemetry on the chip + input sensors the chip is wired to)

"Sensors" in this codebase has two meanings, both relevant to the
ComputeProduct goal:

1. **On-chip telemetry sensors** -- what the silicon exposes for
   measurement (per-rail power, per-die thermal, per-link BW counters,
   activity monitors). Drives validation (compare model predictions
   against measured values), online DVFS feedback, and reliability
   monitoring.

2. **Embodied-AI input sensors** -- what the product is *connected to*
   (cameras, lidar, radar, IMU, audio). Drives workload definition for
   the autonomous-vehicle / robotics / smart-city use cases the
   embodied-schemas catalog actually targets. The existing
   `embodied_schemas/sensors.py` defines `SensorEntry`; ComputeProduct
   needs to express which sensors it's compatible with and how many
   it can handle concurrently.

Both are missing today.

### 1. On-chip telemetry sensors

What modern chips expose:

| Vendor | Telemetry interface | What's exposed |
|---|---|---|
| NVIDIA | NVML / DCGM / nvidia-smi | Per-GPU power, per-rail power (limited), GPU temp, memory temp, clock, Vdd (some), SM utilization, memory BW, PCIe BW |
| AMD GPU | rocm-smi / amd-smi | Per-GPU power, edge/junction/HBM temp, clock, Vdd, GPU/MEM utilization, fan |
| AMD CPU | AMD u-Profile / RAPL | Per-package + per-core energy, frequency, temperature |
| Intel CPU | RAPL / powercap / Intel PMU | Per-package + per-core + DRAM energy, frequency, temperature, MSR-level perf counters |
| Apple | powermetrics / IOReport | Per-cluster power (P/E cores, GPU, ANE), thermal, ANE utilization (limited) |
| Jetson | tegrastats / jetson_clocks | Per-rail power (CPU/GPU/SOC/DDR/SYS), per-rail freq, thermal zones |
| Hailo | HailoRT API | Per-chip power, thermal |
| KPU | (vendor-specific) | TBD |

Schema additions:

```
ComputeProduct
  - telemetry: TelemetrySupport          # NEW
      TelemetrySupport:
        - interface: TelemetryInterface  # NEW enum
            nvml | dcgm | rocm_smi | amd_smi | rapl | powercap |
            ipmi | redfish | tegrastats | powermetrics | ioreport |
            vendor_proprietary
        - power_per_die: bool            # per-die power readings?
        - power_per_rail: bool           # per-voltage-rail readings?
        - power_per_block: bool          # per-block utilization-derived?
        - power_resolution_w: float      # smallest meaningful reading
        - power_sample_period_us: float  # min interval between reads

        - thermal_per_die_sensors: dict[die_id, int]    # count per die
        - thermal_resolution_c: float
        - thermal_sample_period_us: float

        - bandwidth_counters: list[str]  # which links are monitored
                                         # ("hbm_phy_0..4", "nvlink_0..17",
                                         #  "pcie_root", ...)

        - activity_counters: list[str]   # ("sm_busy", "tensor_core_active",
                                         #  "tile_active", "dma_busy", ...)

        - reliability_monitors: list[str]    # NBTI, HCI, EM degradation
                                             # monitors; rare but real
                                             # on automotive parts

        - access: TelemetryAccess        # NEW enum
            user_unprivileged   |
            user_privileged     |
            kernel_only         |
            firmware_only       |
            offline_only        # only via PMIC / BMC
```

This matters for **validation**: today's `kpu_power_model` produces
estimates; once silicon ships, the validation framework wants to compare
against per-die measured power. The catalog needs to say *what's
measurable* on each product so validation harnesses can be wired up.

### 2. Embodied-AI input sensors

The existing `SensorEntry` in `embodied_schemas/sensors.py` covers
cameras, lidar, radar, IMU, depth sensors, microphones. ComputeProducts
in the embodied-AI catalog (Jetson Orin, KPU edge SKUs, Hailo-8) target
specific sensor configurations -- and the connection isn't expressed.

Schema additions:

```
ComputeProduct
  - sensor_io: SensorIOSupport | None    # NEW; None for datacenter parts
      SensorIOSupport:
        - interfaces: list[SensorInterface]
            SensorInterface:
              - kind: mipi_csi2 | mipi_csi2_v3 | gmsl2 | gmsl3 |
                      fpd_link4 | hispi | sublvds | parallel |
                      lvds | usb3 | gige_vision | ethernet_avb |
                      can | canfd | flexray | i2c | spi
              - lanes / channels
              - bandwidth_gbps
              - port_count                # how many of this interface

        - max_concurrent_sensors: dict[str, int]
                                          # NEW -- per sensor kind
                                          # {"camera_4k": 8, "lidar": 2,
                                          #  "radar_77ghz": 6, "imu": 1}

        - preprocessing_blocks: list[BlockRef]
                                          # NEW -- which on-die blocks
                                          # do the sensor preprocessing
                                          # (ISP, video decoder, JPEG
                                          # decoder, audio codec, etc.)
                                          # references blocks in dies[]

        - sensor_to_compute_path:
                                          # NEW -- pipeline metadata
            - latency_us                  # sensor-frame to compute-ready
            - bandwidth_gbps_aggregate    # all sensors at full rate
            - dma_path: enum              # direct_to_l3 | through_dram |
                                          #   through_l2

        - compatible_sensor_refs: list[ProductRef]
                                          # KNOWN-GOOD sensor SKUs
                                          # (links to sensors.py entries)
        - target_sensor_workloads: list[str]
                                          # ("8x_4k_camera_object_detection",
                                          #  "32x_lidar_segmentation",
                                          #  "4x_camera_+_4_radar_avp", ...)
```

For the KPU SKUs in our catalog: the t64/t128/t256 SKUs are positioned
for embodied AI (drones, robots, edge servers). The catalog YAML
*describes* this in the `notes:` field as free text -- the reader has to
infer what camera/lidar configurations the chip can actually handle.
Promoting to structured `sensor_io` lets the LLM orchestrator answer
questions like "give me all SKUs that support 8x 4K cameras" or "give me
all SKUs targeted at automotive ADAS".

### Where preprocessing blocks fit

Sensor preprocessing accelerators (ISPs, video decoders, JPEG decoders,
audio codecs) are blocks on the die in heterogeneous SoCs:

| Product | Sensor preprocessing blocks |
|---|---|
| Jetson Orin AGX | 2x ISP, NVENC (4-stream 4K), NVDEC (12-stream 4K), VIC, OFA, PVA |
| Apple M3 | Image Signal Processor (ISP), ProRes encoder, hardware H.264/HEVC/AV1 |
| Qualcomm Snapdragon | Spectra ISP, Adreno GPU, Hexagon DSP for audio |
| Hailo-15 | Image signal processor + neural processing combined |
| KPU edge SKU | (today: none; could add ISP block for camera-pipeline use cases) |

These are real blocks with their own area, power, and silicon decomposition
-- they fit naturally into the per-die `blocks: list[Block]` discriminated
union proposed earlier:

```
Block (extended discriminator)
  ...
  ISPBlock:    image signal processor (camera input pipeline)
  VideoCodecBlock: NVENC / NVDEC / VPU / hardware H.264/265/AV1
  AudioCodecBlock: MFCC, DSP for voice processing
  RadarDSPBlock:  77 GHz radar baseband / FFT
  LidarPreprocBlock: point-cloud accumulation + filtering
```

This closes the loop: a Jetson Orin AGX `ComputeProduct` has compute
blocks (CPU + GPU + DLA + PVA) AND preprocessing blocks (ISP + NVENC +
NVDEC + VIC) AND sensor I/O interfaces (MIPI CSI-2 v3 ports + GMSL2 +
CAN bus) AND a target sensor workload list -- a full description of
what the product can actually be deployed to do.

## Summary -- the "anything that can vary across dies / SKUs / rails" rule

The pattern across all four caveats: **anything that can vary independently
across dies, harvested variants, voltage rails, or clock domains needs its
own first-class structure, not a flat top-level field.**

| Property | Varies independently across | Needs |
|---|---|---|
| Process node | Dies (CCD on N5, IOD on N6, V-Cache on N7) | `Die.process_node_id` |
| Silicon decomposition | Dies (each die has its own area / transistor budget) | `Die.silicon_bin` |
| Memory type / capacity / BW | Dies (HBM stacks / V-Cache / PoP LPDDR each carry their own) | `MemoryBlock` in `Die.blocks`; `Interconnect` for the link to compute dies |
| Voltage | Rails (compute / memory / fabric / I/O / per-cluster / per-IP) | `Die.voltage_rails`; `ThermalProfile.vdd_by_rail` |
| Clock | Domains (compute / memory / fabric / I/O / per-cluster) | `Die.clock_domains`; `ThermalProfile.clock_by_domain` |
| Body bias (FD-SOI) | Rails (per-rail FBB/RBB) | `VoltageRail.body_bias_supported`; `ThermalProfile.body_bias_by_rail` |
| Active-block selection | Profiles (sleep states; per-block power-gating) | `ThermalProfile.active_blocks` |
| Enabled block count | Harvested variants (132 SMs vs 144) | Each shipping SKU is a separate `ComputeProduct` with its own `Die.blocks`; optional `harvested_from` links to parent |
| TDP | Variants (H100 SXM5 700W vs PCIe 350W from same silicon) | Per-SKU `ThermalProfile`; `per_die_tdp_w` for chiplet allocation |
| Pricing / availability / binning | Variants (each harvested SKU has its own market position) | Per-SKU `market` |
| Off-package memory | Boards / systems (DIMMs, GDDR PCB, CXL modules) | `contains: list[ProductRef]` at board level |
| Junction temperature limit | Dies (HBM 85 C vs compute 105 C vs ULL 125 C) | `Die.thermal.tj_max_c` |
| Hotspot density limit | Dies (PE compute area vs SRAM array vs PHY) | `Die.thermal.peak_power_density_w_per_mm2` |
| Throttle policy (response to hotspot) | Dies (compute does DVFS, HBM does refresh ramp, fixed-function hard-stops) | `Die.thermal.throttle_policy` |
| Vertical thermal coupling | Stacked dies (V-Cache above CCD lowers boost) | `Die.thermal.thermal_couples_above` + per-pair Rja modifier |
| Peak vs sustained power | Time domain (boost for seconds, TDP for sustained) | `ThermalProfile.{peak,boost,sustained}_power_w` + `boost_window_s` |
| Sleep / retention power | Power state (clocks gated, retention only) | `ThermalProfile.sleep_power_w` |
| Thermal interface material (TIM) | Cooling SKU variants (paste vs solder vs liquid metal) | `CoolingSolutionEntry.tim_kind` + `r_cs_c_per_w` |
| Temperature grade | SKU variants (commercial vs industrial vs automotive vs military) | `ComputeProduct.temp_grade` + `ambient_c_range` |
| Cooling scope | Cooling SKUs (per-die / per-package / per-node / per-rack) | `CoolingSolutionEntry.scope` + `applies_to_dies`; `ThermalProfile.cooling_solution_ids: list` |
| Interconnect topology | Levels (intra-die NoC / die-to-die / package-to-package / node-to-node / rack-to-rack) | `Interconnect.level` + `topology` + `dimensions`; switches as first-class entities; endpoints as graph nodes (not flat fields) |
| Interconnect routing & QoS | Per-fabric (deterministic XY vs adaptive vs oblivious; VCs; oversubscription) | `Interconnect.routing` + `virtual_channels` + `oversubscription_ratio` |
| Asymmetric link bandwidth | Per-link (PCIe P2P read vs write; HBM read vs write; NVLink uplink vs downlink) | `Interconnect.asymmetric_send_gbps` / `_receive_gbps` |
| Cache-coherence overlay | Selected dies / blocks / sockets (Grace-Hopper coherent NVLink-C2C, EPYC dual-socket UPI/xGMI, GPU-to-GPU NOT coherent) | `CoherenceDomain` first-class with members + protocol |
| Switch / fabric components | Systems (NVSwitch, IB switch, PCIe PEX, xGMI switch -- silicon in their own right) | `Switch` first-class entity at system level (not a `Block` -- it's package-scoped) |
| System-level composition | Boards, nodes, racks (DGX H100 = 8 GPUs + 4 NVSwitches + 2 CPUs + 8 NICs wired together) | `ComputeProduct(kind=system)` with `contains` + system-level `interconnects` + `switches` + `coherence_domains` |
| Programming model | Architecture (SIMT vs SPMD vs **dataflow** vs **domain_flow** vs VLIW vs SIMD vs Metal -- domain_flow is the KPU-specific SURE/SARE polyhedral execution model, NOT the same as generic dataflow) | `SoftwareSupport.programming_models: list[ProgrammingModel]` |
| SDK / toolchain availability | Vendors per product (CUDA / ROCm / oneAPI / Metal / MLX / HailoRT / TensorRT / Stillwater domain-flow compiler / Hailo dataflow compiler / ...) | `SoftwareSupport.sdks: list[SDKSupport]` with kind / version / lts / eol |
| Framework backend tier | Per (framework, product) pair (PyTorch native on NVIDIA, vendor-supported on AMD, community on Apple MLX, translated via ZLUDA, ...) | `SoftwareSupport.framework_backends: list[FrameworkBackend]` with tier + `performance_relative_to_native` |
| Native quantization / sparsity | Per SDK (int8 / int4 / fp8_e4m3 / fp8_e5m2 / nf4 / awq / gptq; 2:4 sparse vs block-sparse) | `SoftwareSupport.native_quantization` + `native_sparsity` lists |
| Driver / firmware versions | Deployment-time (NVIDIA driver minor versions move perf 5-15% on big workloads) | `ComputeProduct.driver_firmware: DriverFirmwareInfo` with min versions + tested set + determinism flag |
| On-chip telemetry sensors | Per product (NVML on NVIDIA, RAPL on Intel, rocm-smi on AMD, tegrastats on Jetson, ...) | `ComputeProduct.telemetry: TelemetrySupport` with interface + per-die / per-rail flags + sample period + access tier |
| Embodied-AI input sensor support | Per edge product (Jetson Orin handles 8x 4K cameras + lidar + radar; KPU edge SKU TBD) | `ComputeProduct.sensor_io: SensorIOSupport` with interfaces + max_concurrent + preprocessing_blocks + compatible_sensor_refs |
| Sensor preprocessing accelerators | Per die (ISP, NVENC, NVDEC, VIC, audio codec on heterogeneous SoCs) | `ISPBlock` / `VideoCodecBlock` / `AudioCodecBlock` / etc. extending the `Block` discriminator on `Die.blocks` |

The flat-field anti-pattern is the same in every case: collapse N
independent things into one top-level value, lose information, and break
the moment a real product needs the discrimination. Putting them in a
list / dict structure with stable ids preserves the per-thing facts and
lets consumers traverse, filter, or roll up as they need.

The KPU's monolithic single-die single-rail single-clock single-SKU
shape is a degenerate case of this richer structure -- one element in
each list. Migrating the KPU to the new shape costs one extra nesting
level and zero data loss.

---

## Source references (for the curious)

| Class / file | Location | Purpose |
|---|---|---|
| `KPUEntry` | `embodied-schemas/src/embodied_schemas/kpu.py:445` | Top-level KPU SKU schema |
| `KPUArchitecture` | `embodied-schemas/src/embodied_schemas/kpu.py:140` | Tile mix + NoC + memory |
| `KPUSiliconBin` | `embodied-schemas/src/embodied_schemas/kpu.py:266` | Per-block transistor accounting |
| `KPUThermalProfile` | `embodied-schemas/src/embodied_schemas/kpu.py:315` | Per-(clock, Vdd, cooling) operating point |
| `ProcessNodeEntry` | `embodied-schemas/src/embodied_schemas/process_node.py:129` | Per-foundry per-library densities + energies |
| `CoolingSolutionEntry` | `embodied-schemas/src/embodied_schemas/cooling_solution.py` | Thermal envelope per cooling mechanism |
| `GPUEntry` | `embodied-schemas/src/embodied_schemas/gpu.py:491` | Parallel category schema (no PNE ref, no silicon bin) |
| `CPUEntry` | `embodied-schemas/src/embodied_schemas/cpu.py:387` | Parallel category schema (process node enum only) |
| `NPUEntry` | `embodied-schemas/src/embodied_schemas/npu.py:184` | Parallel category schema (no PNE ref, no silicon bin) |
| `KPUSKUInputSpec` | `graphs/src/graphs/hardware/kpu_sku_input.py` | Architect-authoring shape (drops generator-derived fields) |
| `generate_kpu_sku()` | `graphs/src/graphs/hardware/kpu_sku_generator.py` | Generator that rolls up die / perf / TDP from spec |
| `compute_thermal_profile_tdp_w()` | `graphs/src/graphs/hardware/kpu_power_model.py` | 5-term V^2*f power model |
| `sku_validators` | `graphs/src/graphs/hardware/sku_validators/` | Pluggable validator registry |
| `derive_kpu_floorplan()` / `derive_kpu_architectural_floorplan()` | `graphs/src/graphs/hardware/silicon_floorplan.py` | Stage 8 floorplanner |
| `PhysicalSpec` | `graphs/src/graphs/hardware/physical_spec.py:24` | Interim chip-level spine (will become ComputeProduct loader per RFC 0001) |
| `HardwareResourceModel` | `graphs/src/graphs/hardware/resource_model.py:906` | Runtime model consumed by mappers/estimators |
| `KPUComputeResource` | `graphs/src/graphs/hardware/resource_model.py:587` | KPU-specific compute resource (heterogeneous tile allocation) |
| `HardwareMapper` (ABC) | `graphs/src/graphs/hardware/resource_model.py:1718` | Base class for per-architecture mappers |
| RFC 0001 | `embodied-schemas/docs/rfcs/0001-compute-product-unification.md` | The unification plan (Draft) |
| Prior assessment | `graphs/docs/assessments/compute-product-unification.md` | Earlier framing (2026-05-08) |
