# KPU SKU Generator + Process-Node Database — Implementation Plan

Status: design complete, Phase 0 in progress (2026-05-09)
Owner: Theo Omtzigt (Stillwater) + Claude Code

## Problem statement

The repo's CLI tools (`cli/list_hardware_resources.py`, `cli/list_hardware_mappers.py`)
must produce honest comparisons across CPU / GPU / TPU / KPU SKUs in terms of die
size, transistor count, TDP, and process node. Today, KPU rows render `N/A` because
KPU mappers don't attach a `PhysicalSpec`. There is no canonical KPU YAML in the
`embodied-schemas` data catalog, no generator to derive die area / transistor budget
/ TDP from architectural knobs, and no validator to catch implausible SKU
configurations (e.g., "1 PetaOP @ 25 W" — a TDP-vs-throughput violation that is
trivial to introduce by hand-editing numbers).

The plan addresses three coupled gaps:

1. **Data**: KPU SKU YAMLs alongside the GPU / CPU / NPU catalog, and a
   per-process-node, per-circuit-class density / energy database.
2. **Tooling**: a generator that derives roll-up numbers from architectural
   knobs, a registry of validators that catch product-risk violations, and
   inspection CLIs that make the database self-documenting.
3. **Methodology**: ComputeSolution = ProcessNode + CoolingSolution + SKU.
   Validators run against the blend, not the SKU alone.

## Data hierarchy

```
embodied-schemas/data/
  process-nodes/<foundry>/<node>.yaml    # silicon fabrication (NEW)
  cooling-solutions/<id>.yaml            # thermal removal (NEW)
  kpus/<vendor>/<id>.yaml                # KPU SKUs (NEW; peer of gpus/, npus/)
  gpus/, cpus/, npus/, chips/            # existing
```

A KPU SKU references a ProcessNode by id and a default CoolingSolution per
thermal profile. Validators operate on the `(ProcessNode, CoolingSolution, SKU)`
triple — the **ComputeSolution**.

## Schemas (Pydantic BaseModel, peer of existing `gpu.py` / `cpu.py` / `npu.py`)

### `embodied_schemas/process_node.py`

```python
class TransistorTopology(str, Enum):
    BULK_PLANAR = "bulk_planar"
    FINFET = "finfet"
    FD_SOI = "fd_soi"
    GAA = "gaa"

class CircuitClass(str, Enum):
    HP_LOGIC = "hp_logic"
    BALANCED_LOGIC = "balanced_logic"
    LP_LOGIC = "lp_logic"
    ULL_LOGIC = "ull_logic"
    SRAM_HD = "sram_hd"
    SRAM_HC = "sram_hc"
    SRAM_HP = "sram_hp"
    ANALOG = "analog"
    IO = "io"
    MIXED = "mixed"

class LibraryDensity(BaseModel):
    circuit_class: CircuitClass
    mtx_per_mm2: float
    library_name: str | None = None
    confidence: ConfidenceLevel  # THEORETICAL | INTERPOLATED | CALIBRATED
    source: str
    notes: str | None = None

class ProcessNodeEntry(BaseModel):
    id: str                              # e.g. "tsmc_n16"
    foundry: Foundry                     # reuses existing enum
    node_name: str                       # "N16"
    node_nm: int                         # 16
    transistor_topology: TransistorTopology
    body_bias_supported: bool = False
    back_bias_range_mv: tuple[int, int] | None = None
    vt_options: list[str] = []
    nominal_vdd_v: float
    densities: dict[CircuitClass, LibraryDensity]
    energy_per_op_pj: dict[str, float]   # key = "circuit_class:precision"
    leakage_w_per_mm2: dict[CircuitClass, float]
    em_j_max_by_temp_c: dict[int, float] # A/cm^2 by junction temp
    routing_metal_width_um: dict[CircuitClass, float] = {}
    m0_pitch_nm: int | None = None
    m1_pitch_nm: int | None = None
    cooling_compatible: list[str] = []   # advisory cooling-solution ids
    source: str
    confidence: ConfidenceLevel
    last_updated: str
```

### `embodied_schemas/cooling_solution.py`

```python
class CoolingType(str, Enum):
    PASSIVE_FANLESS = "passive_fanless"
    PASSIVE_HEATSINK_SMALL = "passive_heatsink_small"
    PASSIVE_HEATSINK_LARGE = "passive_heatsink_large"
    ACTIVE_FAN = "active_fan"
    VAPOR_CHAMBER = "vapor_chamber"
    LIQUID_COOLED = "liquid_cooled"
    DATACENTER_DTC = "datacenter_direct_to_chip"
    IMMERSION = "immersion"

class CoolingSolutionEntry(BaseModel):
    id: str                              # e.g. "passive_heatsink_large_15w"
    name: str
    cooling_type: CoolingType
    max_power_density_w_per_mm2: float   # ceiling for thermal hotspot validator
    max_total_w: float                   # whole-package envelope
    ambient_c_max: float                 # operating envelope
    junction_c_max: float                # Tjmax — feeds EM validator
    form_factor_constraints: list[str] = []  # e.g. ["height<=10mm", "fanless"]
    weight_g: float | None = None
    cost_usd: float | None = None
    source: str
    confidence: ConfidenceLevel
```

### KPU SKU YAML (under `data/kpus/stillwater/`)

Reuses the GPU/NPU YAML block layout (`die`, `clocks`, `memory`, `performance`,
`power`, `market`) so the existing `physical_spec_loader` works unchanged. Adds:

- `process_node_id: tsmc_n16` — references a ProcessNode entry.
- `silicon_bin.blocks[]` — per-block decomposition with `circuit_class` and
  `transistor_source` (derived from architectural knobs or fixed Mtx).
- `kpu_architecture` — total_tiles, tile_mix, pe_array_per_tile,
  ops_per_pe_per_clock, l1/l2/l3 sizes, NoC topology, memory_controllers,
  multi_precision_alu list, clock_mhz_by_profile.
- `thermal_profiles[].cooling_solution_id` — references a CoolingSolution.

## Generator (`graphs/src/graphs/hardware/kpu_sku_generator.py`)

Pure function `generate_kpu_sku(input_spec, process_node) -> dict`. Per-block
math:

```
for block in silicon_bin.blocks:
    transistors_block  = resolve(block.transistor_source)
    density            = process_node.density_for(block.circuit_class).mtx_per_mm2
    area_block_mm2     = transistors_block * 1000 / density
    leakage_block_w    = process_node.leakage_w_per_mm2[class] * area_block_mm2

die.transistors_billion = sum(transistors_block) / 1000
die.die_size_mm2        = sum(area_block_mm2)
power.tdp_watts         = dynamic_from_fabrics + sum(leakage_block_w)
```

Records provenance — which ProcessNode, at what confidence — in the output YAML.
A SKU built against PDK-derived data flows CALIBRATED confidence; one built
against public estimates flows THEORETICAL.

CLI: `cli/generate_kpu_sku.py --input <spec> --node <process_node_id> --output <yaml>`

## Validator framework (`graphs/src/graphs/hardware/sku_validators/`)

Pluggable registry. New product-risk checks become new classes; nothing else
changes.

```python
class ValidatorCategory(Enum):
    INTERNAL = "internal"
    ELECTRICAL = "electrical"
    AREA = "area"
    ENERGY = "energy"
    THERMAL = "thermal"
    RELIABILITY = "reliability"
    GEOMETRY = "geometry"      # for floorplan validators (Stage 8)

class Severity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"

@dataclass
class Finding:
    validator: str
    category: ValidatorCategory
    severity: Severity
    block: str | None
    message: str
    citation: str | None = None

class SKUValidator(Protocol):
    name: str
    category: ValidatorCategory
    def check(self, sku, ctx: ValidatorContext) -> list[Finding]: ...

class ValidatorContext:
    process_node: ProcessNodeEntry
    cooling_solution: CoolingSolutionEntry
    sku: dict
```

### Initial validator set

| Validator | Category | Notes |
|---|---|---|
| `tile_mix_consistency` | INTERNAL | `Σtile_mix == total_tiles`; perf == Σ(tiles × PEs × ops × clock) |
| `bandwidth_math` | ELECTRICAL | reuses existing `validate_physical_spec` |
| `power_profile_monotonicity` | ELECTRICAL | clock@high > clock@low |
| `composite_density_envelope` | AREA | composite density ∈ node's [min,max] across libraries |
| `block_library_validity` | AREA | every block's `circuit_class` ∈ `process_node.densities` |
| `area_self_consistency` | AREA | die_size_mm2 == Σ(block areas) ≤ 5 % |
| `tops_per_watt_envelope` | ENERGY | int8_tops/tdp ≤ ceiling(node, topology) |
| `thermal_hotspot` | THERMAL | per-block W/mm² ≤ cooling.max_power_density |
| `electro_migration` | RELIABILITY | per-block J ≤ node.em_j_max(Tj) |

CLI: `cli/validate_sku.py --sku <id> [--category <c>] [--severity error]`,
non-zero exit on ERROR.

### Future validators (Stage 8)

- `dvfs_feasibility` (THERMAL)
- `memory_bandwidth_headroom` (ELECTRICAL)
- `process_yield_risk` (RELIABILITY)
- `floorplan_pitch_match` (GEOMETRY) — KPU checkerboard
- `aspect_ratio_bounds` (GEOMETRY)
- `noc_routability` (GEOMETRY)
- `io_ring_placement` (GEOMETRY)

## Inspection CLIs (Phase 0c)

| Tool | Output |
|---|---|
| `cli/list_process_nodes.py` | Table: id, foundry, node, topology, peak density, source, confidence |
| `cli/show_process_node.py <id>` | Per-(class) density / energy / leakage / EM J_max |
| `cli/list_circuit_classes.py` | Enum + descriptions; cross-tab nodes × classes |
| `cli/list_cooling_solutions.py` | Table: id, type, max W/mm², max W, Tj_max |
| `cli/show_cooling_solution.py <id>` | Full spec + form-factor constraints |

All support `--output {text,json,csv,md}`.

## PDK ingestion (Phase 7)

`cli/import_pdk.py --pdk <path> --out <node>.yaml` reads PDK characterization
files, emits ProcessNode YAML with `confidence: CALIBRATED` and
`source: "PDK <foundry> <node> rev <X>"`. PDK data is often confidential;
support `PROCESS_NODE_DATA_DIR` env var (parallel to `EMBODIED_SCHEMAS_DATA_DIR`)
so confidential YAMLs live outside the public repo. Higher-confidence entries
win when both exist.

Zero consumer-code changes — generator, validator, resource models all key on
`(process_node_id, circuit_class)` lookups.

## Initial process-node dataset

Hand-authored from public estimates (Wikichip, ISSCC, foundry presentations).
Each entry carries `confidence: THEORETICAL` and a citation.

| File | Topology | Why |
|---|---|---|
| `tsmc/n16.yaml` | finfet | KPU-T64/T256 today |
| `tsmc/n12.yaml` | finfet | T256/T768 mid-tier |
| `tsmc/n7.yaml` | finfet | KPU-T768 datacenter |
| `tsmc/n5.yaml` | finfet | Future KPU SKUs; Hopper context |
| `samsung/8lpp.yaml` | finfet | Orin (already used by embodied-schemas GPU YAMLs) |
| `globalfoundries/12lp.yaml` | finfet | GF mainstream FinFET |
| `globalfoundries/12lp_plus.yaml` | finfet | GF performance-tuned 12LP |
| `globalfoundries/12fdx.yaml` | fd_soi | FD-SOI; body bias; ULL leakage |

The FD-SOI entry exercises code paths the FinFET entries don't (body-bias range,
different leakage model, lower peak density but wider DVFS range).

## Initial cooling-solution dataset

| File | Type | Max W/mm² | Max W | Notes |
|---|---|---|---|---|
| `passive_fanless.yaml` | passive_fanless | 0.3 | 6 | Drone, robot, fanless edge |
| `passive_heatsink_small.yaml` | passive_heatsink_small | 0.5 | 15 | Edge module |
| `passive_heatsink_large.yaml` | passive_heatsink_large | 1.0 | 30 | Industrial edge |
| `active_fan.yaml` | active_fan | 2.5 | 100 | Standard PCIe card |
| `vapor_chamber.yaml` | vapor_chamber | 4.0 | 250 | High-end consumer |
| `liquid_cooled.yaml` | liquid_cooled | 8.0 | 500 | Workstation / server |
| `datacenter_dtc.yaml` | datacenter_direct_to_chip | 12.0 | 1000 | Hyperscaler |

## Phasing

| # | Phase | Status |
|---|---|---|
| 0 | Cross-repo schema slots: `kpus/`, `process-nodes/`, `cooling-solutions/` directories in embodied-schemas | in progress |
| 0a | `ProcessNodeEntry` / `CircuitClass` / `TransistorTopology` schema (Pydantic) | in progress |
| 0a | `CoolingSolutionEntry` schema (Pydantic) | in progress |
| 0b | Hand-authored 8 ProcessNode YAMLs + 7 CoolingSolution YAMLs | in progress |
| 0c | Inspection CLIs (5 tools) | pending |
| 1 | KPU SKU YAML schema (silicon_bin + process_node_id + cooling refs) | pending |
| 2a | Validator framework (registry, Finding, base classes) | pending |
| 2b | Initial validators (consistency, density, library, energy) | pending |
| 2c | Thermal hotspot validator | pending |
| 2d | EM validator | pending |
| 3 | Generator (`generate_kpu_sku`) | pending |
| 4a | KPU resource-model loader (`load_kpu_resource_model_from_yaml`) | done (2026-05-09) |
| 4b | Collapse 4 hand-coded factories into thin wrappers around the loader | pending |
| 5 | Mapper wiring (4 × 2 lines in `accelerators/kpu.py`) | pending |
| 6 | CI gate: every SKU YAML validated | pending |
| 7 | PDK ingestion + `PROCESS_NODE_DATA_DIR` override | pending |
| 8 | **Stage 8 (later)**: DVFS feasibility, memory headroom, yield risk, **floorplanner + viz + geometric validators (KPU checkerboard pitch-match)** | pending |

## File inventory

### embodied-schemas (cross-repo)

- `src/embodied_schemas/process_node.py` — schemas
- `src/embodied_schemas/cooling_solution.py` — schemas
- `src/embodied_schemas/kpu.py` — KPU entry schema
- `src/embodied_schemas/loaders.py` — register new types
- `src/embodied_schemas/data/process-nodes/{tsmc,samsung,globalfoundries}/*.yaml`
- `src/embodied_schemas/data/cooling-solutions/*.yaml`
- `src/embodied_schemas/data/kpus/stillwater/stillwater_kpu_t{64,128,256,768}.yaml`

### graphs (this repo)

- `src/graphs/hardware/physical_spec_loader.py` — extend categories with `kpus`
- `src/graphs/hardware/kpu_sku_generator.py` — new
- `src/graphs/hardware/sku_validators/` — new package: registry + 9 validators
- `src/graphs/hardware/models/accelerators/kpu_yaml_loader.py` — new; replaces inline factories
- `src/graphs/hardware/models/accelerators/kpu_t{64,128,256,768}.py` — collapsed to thin wrappers
- `src/graphs/hardware/mappers/accelerators/kpu.py` — add `load_physical_spec_or_none()` calls
- `cli/list_process_nodes.py`, `cli/show_process_node.py`, `cli/list_circuit_classes.py`,
  `cli/list_cooling_solutions.py`, `cli/show_cooling_solution.py` — new
- `cli/generate_kpu_sku.py`, `cli/validate_sku.py`, `cli/import_pdk.py` — new
- `tests/hardware/test_kpu_sku_yamls.py` — CI gate

## Phase 4b migration plan -- collapsing the hand-coded factories

Phase 4a landed `load_kpu_resource_model_from_yaml(base_id)` plus 42
unit tests confirming every SKU loads cleanly and the resulting
`HardwareResourceModel` is consumable by `KPUMapper`. The four
hand-coded factories in `src/graphs/hardware/models/accelerators/
kpu_t{64,128,256,768}.py` (~2200 LOC total) are still in place and
still drive the `create_kpu_t*_mapper()` factories. Phase 4b swaps
them over.

### Known YAML-vs-factory deltas

The hand-coded factories carry historical inconsistencies the YAML
fixed during Phase 1; merging them naively would silently change
downstream test expectations. The two material ones:

| Field | Factory | YAML / loader | Notes |
|---|---|---|---|
| `ComputeFabric.ops_per_unit_per_clock[INT8]` (T64 INT8 fabric) | 512 | 2048 | Factory uses pre-M0.5 16x16 array math; YAML uses M0.5 32x32. Loader follows YAML (matches `tile_specialization.ops_per_tile_per_clock` already in the same file). |
| `ThermalOperatingPoint` per-precision `efficiency_factor` | hand-tuned 0.65--0.85 | flat 0.70 placeholder | Loader v1 doesn't have measured efficiency data per profile. |

Other deltas exist for `tile_energy_model`, `bom_cost_profile`,
`soc_fabric`, and the dozen `set_provenance(...)` calls that the
factories add after constructing the base model -- the YAML doesn't
carry any of that, so Phase 4b's factory wrappers will keep that
augmentation hand-coded.

### Migration steps (Phase 4b)

1. **Snapshot expected outputs**: capture every value the existing
   `tests/hardware/test_kpu_*.py` assertions check against, per SKU.
   This is the contract the loader must satisfy.
2. **Reconcile the INT8 ops/clock delta**: either (a) update the YAML
   to the factory's pre-M0.5 numbers (regression), (b) update the
   tests to the M0.5 numbers (sharper truth), or (c) keep both paths
   and mark the factory deprecated. Recommended: (b), since the M0.5
   numbers are what the validator framework already enforces.
3. **Per-profile efficiency factors**: extend `KPUEntry.power.thermal_profiles`
   with optional `efficiency_factor_by_precision: dict[str, float]`
   so the YAML carries calibration data the loader can read. Start
   with the existing factory-hand-coded values copied over.
4. **Add `tile_energy_model` to the loader**: the energy model
   parameters (`mac_energy_int8`, `dram_read_energy_per_byte`, etc.)
   can be derived from `process_node.energy_per_op_pj` plus the
   existing `_MEM_PHY_PJ_PER_BYTE_BY_TYPE` table in the silicon_math
   module. Add a `_build_tile_energy_model(sku, node)` helper.
5. **Add a hand-coded augmentation hook**: each `kpu_t*.py` file
   becomes ~30 lines that call `load_kpu_resource_model_from_yaml()`
   then add `bom_cost_profile`, `soc_fabric`, and `set_provenance`
   calls. Total drops from ~2200 LOC to ~120 LOC across the four
   files.
6. **Remove the redundant tile_specialization construction in the
   factories**: post-loader, `KPUComputeResource` is built once by
   the loader; the factories' duplicate construction goes away.
7. **Run the full test suite**: 51 KPU tests + 42 loader tests + the
   broader validation suite. Expect a small number of test updates
   per (2) above.

### Risk

Replacing the factories at once is a 51-test reconciliation (the
hand-coded values are the test expectations). The incremental path
above keeps each step's blast radius small. Preserved-as-is: the
loader doesn't change anything until the factories actually get
swapped over.

## Open items deferred to Stage 8

- Floorplanner utility: takes `silicon_bin` + topology hint, emits 2D placement.
- Visualization tool: SVG/PNG with annotated dimensions, pitches, IO ring.
- Geometric constraint validators: pitch-match (KPU checkerboard), aspect ratios,
  NoC routability, IO placement.
- Whether cooling-solution -> ceiling table moves into a per-cooling Pydantic
  field (currently planned to live there from day one) vs validator local
  default. Resolved: lives in CoolingSolutionEntry.
- DVFS feasibility validator (does throttling actually let advertised TOPS land?)
- Memory bandwidth headroom validator (roofline knee).
- Process yield risk (D0 × area).
