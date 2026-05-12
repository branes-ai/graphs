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
        - die_id (e.g., "ccd0", "iod", "compute_tile_p", "compute_tile_e")
        - process_node_id                  # PER-DIE
        - die_size_mm2, transistors_billion
        - silicon_bin                      # PER-DIE area decomposition
        - blocks: list[Block]              # discriminated union per RFC 0001
                                           # KPUBlock | GPUBlock | CPUBlock | NPUBlock | DSPBlock | ...
        - clocks (per-die clock domain)
  - interconnects: list[Interconnect]      # NEW -- inter-die links
      Interconnect:
        - kind (NVLink-C2C | NV-HBI | InfinityFabric | UltraFusion | EMIB | NVSwitch | PCIe | ...)
        - bandwidth_gbps, latency_ns, energy_pj_per_byte
        - connects: [die_id, die_id]       # which dies it connects
  - memory:                                # on-package, shared across dies
      memory_type, on_package_gb, bandwidth_gbps, ...
  - power:                                 # chip-level envelope
      tdp_watts (sum or peak), thermal_profiles: list[ThermalProfile]
      ThermalProfile:
        - cooling_solution_id              # package-level cooling
        - per_die_tdp_w: dict[die_id, float]    # per-die power allocation
        - clock_mhz, vdd_v can vary per-die (need per-die clock+Vdd here too)
  - market
  - contains: list[ProductRef]             # board/system-level: { id, count }
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
   loaders write into the new schema, then read back). Then migrate one
   chiplet product (AMD MI300A is the canonical stress test) end-to-end
   to validate the per-die scoping.
7. Promote the validator/generator/power-model frameworks from KPU-specific
   (`graphs.hardware.kpu_*`) to ComputeProduct-aware:
   - Validators iterate over dies + blocks, dispatch by `block.kind`.
   - Power model rolls up per-die contributions (each die has its own
     ProcessNode -> energies) and adds inter-die interconnect power.
   - Generator becomes per-die area/transistor roll-up + chip-level
     aggregation.
8. Keep the legacy `GPUEntry` / `CPUEntry` / `NPUEntry` / `KPUEntry`
   schemas as deprecation shims for one or two releases; loaders return
   `ComputeProduct` instances regardless of source format.

The RFC 0001 sketch should be amended to reflect this per-die scoping
before any implementation begins.

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
