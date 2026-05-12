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
