# KPU Heterogeneous Tile Refactor -- Requirements and Implementation Plan

Status: DRAFT for review
Date: 2026-09-14
Tracking issue: graphs#268 (sprint: v14 "KPU heterogeneous tiles")
Related:
  - `docs/plans/soc-microarchitecture-study-plan.md` (the SoC studies consume these tile classes as engines)
  - `docs/designs/kpu-cluster-organization-for-dvfs-and-floorsweeping.md` (cluster/quadrant DVFS; unimplemented, folded in here)
  - `docs/assessments/kpu-as-generic-compute-product.md` (voltage rails, clock domains, interconnect graph; ISP as a separate block kind)
  - `docs/designs/kpu-sku-and-process-node-plan.md` (silicon_bin, ProcessNode, validators, floorplan)

## 1. Motivation

Today's KPU is a checkerboard of *uniform programmable PE-array tiles*: the
INT8-primary, BF16-primary and Matrix classes. They differ only in library,
per-PE area coefficient and ops/clock. The architecture has moved on:

- **Specialization wins by orders of magnitude.** An ISP, a pure systolic
  matmul array, or a full VIO pipeline in one custom datapath (Navion: MIT,
  JSSC 2019, stereo VIO in ~2 mW at 65 nm) is orders of magnitude more
  energy-efficient on its function than a programmable PE fabric. The win has
  two sources:
  - a datapath with no programmability overhead;
  - a whole *compute segment* encapsulated, so intermediate data never leaves
    the tile.

  Embodied autonomy is full of such segments: SGM stereo, radar FFT/CFAR,
  feature tracking, VIO, TSDF/ESDF, the MPC/CBF solver cores.
- **The programmable PE fabric is itself parameterizable.** It is a
  nearest-neighbor, systolic-style domain-flow fabric. Each PE carries a
  *custom-allocated datapath*: an INT8xINT8->INT32 MAC, an LNS MAC, or custom
  operators such as lerp. A *network overlay* emulates communication patterns
  that are not nearest-neighbor.
- **The tile abstraction and the compute/memory checkerboard already exist.**
  They should become the uniform organizing grid for all of these tile types.
  We should not bolt accelerators onto the side of the KPU.

The spec cannot express any of this today. This plan turns those ideas into
requirements, proposes a schema, and sequences a migration that leaves the 12
catalog SKUs byte-identical.

## 2. Current state (as-built constraints)

The full inventory is from the consumer sweep on 2026-09-14. These are the
constraints that shape the plan:

| # | Constraint in today's code | Where | Consequence |
|---|---|---|---|
| C1 | A tile *is* a PE array: `pe_array_rows * pe_array_cols` drives area (PER_PE), L1, floorplan pitch, `threads_per_unit`, `pes_per_tile` and `--pe-array` | `kpu.py:52-90`, `silicon_math.py:145-158`, `silicon_floorplan.py:305-326`, `kpu_yaml_loader.py:327,695` | A fixed-function tile has no PE array, so every one of these breaks |
| C2 | `ops_per_tile_per_clock` is keyed by precision string only | `kpu.py:61-68`, `kpu_power_model.py:127-135` | Cannot express `mac:lns16`, `lerp:fp16`, or "frames/s" |
| C3 | Memory is uniform: `l1_kib_per_pe`, `l2/l3_kib_per_tile x total_tiles`; the checkerboard pairs every compute tile 1:1 with an identical L3 tile | `silicon_math.py:75-89`, `silicon_floorplan.py:974-1023` | Cannot give ISP line buffers, Navion's on-chip state, or systolic weight/accumulator buffers |
| C4 | One chip clock and one Vdd per thermal profile | `KPUThermalProfile`, `kpu_power_model.py:187` | Fixed-function tiles run at sensor rate and are gated otherwise; that cannot be expressed. (The runtime `TileSpecialization.clock_domain` exists but the loader fills one shared domain.) |
| C5 | One mesh NoC. No overlays, express links or stream links. `Die.interconnects` is always empty for KPUs. | `KPUNoCSpec` | The PE-level overlay and tile-to-tile streaming (ISP -> VIO) have no representation |
| C6 | Non-`pe_*` silicon_bin blocks get **0 W dynamic** (block-name heuristics) | `silicon_math.py:326-370` | A fixed-function block would be modeled as free |
| C7 | `KPUTheoreticalPerformance` has hard-coded int8/bf16/fp32/int4 fields | `kpu.py:304-312` | No place for systolic-only or fixed-function capability |
| C8 | `tops_per_watt_envelope` caps INT8 TOPS/W by node | `energy.py:84-100` | Correctly specialized tiles would be flagged as implausible |
| C9 | `KPUArchitecture` and `KPUBlock` duplicate every field and are copied by hand in 6 places | `kpu.py:140`, `compute_product.py:162`, `kpu_sku_generator.py:105,295,418`, `kpu_power_model.py:327`, `loaders.py:390` | Every schema change would have to be made in 6 places |
| C10 | `blocks[0]` / `_kpu_block` assumptions appear in 6 or more places | silicon_math, floorplan, consistency validator, loader, show_kpu, list_kpus | Also blocks multi-block SoC dies (SoC plan) |
| C11 | Integrity gaps: `mesh_rows*mesh_cols == total_tiles` is never checked (the floorplan pads or truncates silently), and L1 (4 KiB/PE) has **no area block** | floorplan `:470-481`, all 12 SKUs | Must be fixed deliberately, because the L1 area feeds leakage and therefore TDP |
| C12 | Stale artifacts | `kpu-test-contract-snapshot.md` (T256 still 20x20), `kpu-modeling.md` (16x16), `validation/hardware/test_phase4_accelerator_energy.py` (expects fabric names the loader no longer emits) | Fix before the refactor so the baseline is trustworthy |

Pinned invariants that must survive:

- **The embodied-schemas TDP roundtrip**: `test_roundtrip_power_default_tdp_is_derived`
  asserts stored TDP == derived TDP within 0.5 W. Any deliberate model change
  follows the established protocol: re-tune `vdd_v` to keep the round TDP.
- **About 200 KPU-touching tests.** The heaviest are the floorplan tests (29, x12
  SKUs), the loader tests (28), the domain-flow tile tests (20) and the
  generator tests (18).

Note: the earlier assessment (`kpu-as-generic-compute-product.md:1594-1625`)
proposed ISP and VideoCodec as *separate `Die.blocks` kinds*. This plan puts
specialized tiles **inside** the KPU checkerboard, which departs from that
proposal. Section 5.6 reconciles the two placements.

## 3. Requirements

### Functional requirements (from the architecture direction)

| ID | Requirement |
|---|---|
| R1 | **Tile taxonomy.** A KPU tile class has a `tile_kind`: `pe_fabric` (programmable domain-flow), `systolic` (fixed-schedule GEMM/conv array), `fixed_function` (encapsulated compute segment). Reserve `scalar` (a control/irregular-code core) and `io_bridge` (a sensor-ingress tile) as future kinds. |
| R2 | **PE datapath customization.** A `pe_fabric` tile declares its per-PE datapath: functional units (op, operand format, accumulate format, lanes), registers, and domain-flow token/coordinate match. Ops/clock, area and energy are **derived** from the datapath. |
| R3 | **Number formats beyond today's precision strings**: LNS, fp8 variants, fixed-point, and room for posits. The legacy strings (`int8`, `bf16`, ...) remain valid formats. |
| R4 | **Custom operators**: lerp, min-plus / compare-select, abs-diff (SAD), popcount-xor (Hamming), CORDIC, rsqrt, and exp via LUT. Each has an ops-counting rule so it compares with workload op counts. |
| R5 | **Network overlays.** At PE level inside a tile: broadcast buses, express links of span k, reduction trees, transpose, butterfly, segmented buses. At tile level: express channels, plus **stream links** that chain fixed-function tiles (ISP -> VIO) without a round trip through L3 or DRAM. Overlays are *statically configured* (the schedule lives in the topology). They are not packet routers with tables. |
| R6 | **Fixed-function contract.** Declares a function id, input/output contract, configuration limits, numeric formats, throughput in *function units* (pixels, frames, features, chirps) per clock, energy per unit at a reference node, local memory, own silicon, I/O bytes per unit, software-equivalent op count (for reporting only), and provenance. |
| R7 | **Checkerboard organization.** All tile kinds live in the same compute/memory checkerboard. Multi-site footprints (a 2x2 VIO tile) may absorb the memory cells inside the footprint. Placement affinities: next to the IO edge (an ISP near MIPI), next to a memory controller, adjacent to a stream-link partner. Pitch matching still holds. |
| R8 | **Per-tile-class memory**: L1/L2/line buffers/weight buffers per class. L3 stays in the memory cells. |
| R9 | **Per-tile-class clock/voltage and power gating** through power domains. The unimplemented cluster/quadrant proposal becomes the default partition of the `pe_fabric` sites; each fixed-function class gets its own domain. |
| R10 | **Energy models per tile kind**, reflecting the specialization ladder: fixed_function << systolic < pe_fabric (per useful op on the tile's intended function), with every rung traceable to ProcessNode data or a cited anchor. |

### Derived requirements (from the as-built constraints)

| ID | Requirement |
|---|---|
| R11 | **Zero-diff migration.** All 12 catalog SKUs produce byte-identical die area, transistors, TDP per profile, peak performance and floorplan through every PR, unless a PR declares a deliberate change with a Vdd re-tune. |
| R12 | **Stable tile-class ids** (`tile_class_id`), separate from the human label. `count_ref: tile.<label>` keeps resolving through an alias. |
| R13 | **Honest roll-ups.** Peak performance is reported by tile kind. Headline TOPS counts only `pe_fabric` + `systolic`. Fixed-function capability is reported in function units/s, and its "equivalent TOPS" appears only in a clearly separate field. |
| R14 | **Validators are tile-kind aware**: split the TOPS/W envelope by kind, and add cell accounting, pitch fit per footprint, datapath consistency, overlay/stream-link consistency, power-domain coverage, and a no-double-count rule for silicon. |
| R15 | **Process-node retargeting for specialized tiles.** Datapath energy is expressed as a ratio to an anchor op in `energy_per_op_pj` at the same node. Fixed-function energy is published at a reference node and scaled by the node's logic and SRAM energy ratios. Area comes from the per-circuit-class transistor budget. |
| R16 | **Reusable tile-class library.** Tile classes are defined once and referenced by SKUs. The SoC study plan reuses the fixed-function core definitions for standalone (out-of-mesh) blocks. |
| R17 | **Mapper/analyzer capability dispatch.** DNN subgraphs go to `systolic` / `pe_fabric` by op and format coverage. Pipeline stages and segments go to `fixed_function` tiles by `function_id`. A stage's precision floor is checked against the tile's numeric formats; violations produce findings, with an explicit waiver field. |
| R18 | **Architectural fidelity.** Keep the PE-fabric model as a mesh of datapaths with token match. Throughput basis is PEs x clock. Do not invent routers or schedule ROMs inside a tile. |

## 4. Motivating tile set: mapping to the autonomy pipeline

Candidate tile classes the 7-tier workload justifies. This is a hypothesis to
test in Phase E, not a claim.

| Pipeline stage (PDF section 2) | Candidate tile | Why |
|---|---|---|
| Stereo rectify + SGM (239 GOP/s, INT8+FP32) | `ff_stereo_sgm` or `pe_minplus` (abs-diff + min-plus datapath) | Cost-volume DP; the largest T1 term |
| Radar FMCW (FFT, CFAR) | `pe_fabric` + butterfly overlay, or `ff_radar_rd_cfar` | FFT needs non-nearest-neighbor exchange, which is exactly what the overlay is for |
| Visual front-end + windowed BA (+ IMU preintegration) | `ff_vio` (Navion-class segment) | Whole-segment encapsulation. Test the precision floor (PDF Class C) explicitly. |
| TSDF integration, ESDF propagation | `pe_lerp` (trilinear/SDF lerp) and `pe_minplus` (brushfire wavefront) | SURE-expressible wavefront; poor locality on CPUs |
| Open-vocabulary detection, SDF encoder, DRL policy | `systolic_int8` + `pe_fabric` fp16 for Class B layers | Dense GEMM/conv goes on the systolic array; norms and softmax go on the fabric |
| VLM prefill/decode | `systolic_int8` / `pe_lns16` | Decode is weight-streaming and bandwidth-bound; LNS density helps capacity |
| Nonlinear MPC, CBF QP | `pe_fabric` fp32 FMA | Small dense linear algebra (Cholesky/KKT) maps to the systolic-style fabric |
| Camera ISP (SoC plan extension stage) | `ff_isp_raw2yuv` | Classic fixed function; sits next to the MIPI edge |

## 5. Proposed schema (embodied-schemas)

### 5.1 Tile union

`KPUArchitecture.tiles: list[AnyKPUTile]`, discriminated on `tile_kind`. A
`mode="before"` validator injects `tile_kind: pe_fabric` and
`tile_class_id: slug(tile_type)` when either is absent, so all 12 YAMLs load
unchanged.

```
KPUTileBase                      (common to every kind)
  tile_kind: TileKind            pe_fabric | systolic | fixed_function | (reserved: scalar, io_bridge)
  tile_class_id: str             stable id; count_ref / domain / placement refer to it
  label: str                     = legacy tile_type (kept as alias)
  num_tiles: int
  footprint: {rows: 1, cols: 1, absorbs_memory_cells: false}
  local_memory: [LocalMemory]    {level: l1|l2|line_buffer|weight_buffer|accumulator, kib, per: pe|tile, circuit_class}
  silicon: [SiliconBinBlock]     optional per-tile transistor budget (see 5.5)
  power_domain_id: str | None
  placement: {affinity: any|io_edge|memory_edge|center, adjacent_to: [tile_class_id]}
  tile_class_ref: str | None     provenance: library id this class was resolved from
  confidence, source

PEFabricTile(KPUTileBase)        tile_kind = pe_fabric   (all 12 legacy SKUs)
  pe_array_rows, pe_array_cols
  datapath: PEDatapath | None    None -> legacy path using explicit ops_per_tile_per_clock
  ops_per_tile_per_clock: dict   optional when datapath is set (then derived; the validator checks agreement)
  pe_circuit_class               kept; defaults from datapath.circuit_class
  interconnect: FabricInterconnect | None
  schedule_class, pipeline_fill_cycles, pipeline_drain_cycles

SystolicTile(KPUTileBase)        tile_kind = systolic
  array_rows, array_cols, dataflow: weight_stationary | output_stationary
  mac: FunctionalUnit            one fixed MAC datapath
  supported_kernels: [gemm, conv2d, ...]
  fill/drain derived from array dims

FixedFunctionTile(KPUTileBase)   tile_kind = fixed_function
  core: FunctionCore             (separate module; also usable as a standalone Die block, see 5.6)
```

### 5.2 PE datapath, number formats, operators

```
NumberFormat (registry, string key)  int4 int8 int16 int32 | fp8_e4m3 fp8_e5m2 fp16 bf16 fp32 fp64
                                     | lns8 lns16 (log number system) | fixed<m>.<n> | (reserved: posit<n>)
OpKind                               mac fma add mul | min_plus cmp_select abs_diff | lerp
                                     | popcount_xor cordic rsqrt exp_lut
FunctionalUnit
  op: OpKind, operand_format, accumulate_format, lanes
  ops_per_invocation                  from the counting convention (section 7, D3), e.g. mac = 2, lerp = 3
  energy: EnergyRef                   {anchor: "<circuit_class>:<format>", ratio}   (scales with node)
                                      | {pj: X, ref_node: <id>}                     (cited absolute)
  mtx                                 transistors per unit instance
PEDatapath
  datapath_id, circuit_class
  functional_units: [FunctionalUnit]
  operand_regs, accumulator_bits
  token_match: true                   domain-flow coordinate/token match (a real structure, R18)
```

Derived capability: `ops_per_tile_per_clock["<op>:<format>"] = rows * cols *
sum(lanes * ops_per_invocation)`. It is also *projected* onto legacy precision
keys (`mac:int8` -> `int8`) so every existing consumer keeps working.

### 5.3 Overlays

```
FabricInterconnect              (intra-tile, PE level)
  base: nearest_neighbor_4 | nearest_neighbor_8, link_bits
  overlays: [FabricOverlay]
FabricOverlay
  kind: row_broadcast | col_broadcast | express | reduction_tree | transpose | butterfly | segmented_bus
  span (express), width_bits, instances_per: row | col | tile
  configuration: static_per_schedule        (no routing tables; R5, R18)
  circuit_class, mtx_per_instance, energy_pj_per_bit_mm (or a node-relative EnergyRef)

KPUNoCSpec (extended; eventually subclasses compute_block_common.OnDieFabric like NPU/CGRA/TPU)
  overlays: [NoCOverlay]
NoCOverlay
  kind: express_channel | stream_link | multicast_tree
  endpoints: [tile_class_id] (stream_link: ordered producer -> consumer)
  width_bytes, span, energy, mtx
```

### 5.4 Checkerboard, domains, operating points

```
CheckerboardSpec                 (new; replaces the implicit mesh_rows x mesh_cols == total_tiles)
  compute_sites: {rows, cols}    compute sites; memory cells pair 1:1 as today
  memory_cell: {l3_kib, circuit_class}          (alias of memory.l3_kib_per_tile)
  placement: auto | explicit
  placement_map: [[tile_class_id | "."]] | None (explicit layout; "." = spare/whitespace)
  spare_sites: int = 0

PowerDomain                      (generic; in compute_block_common per the cluster doc's genericity requirement)
  domain_id, kind: cluster | tile_class | uncore
  members: tile_class_ids | site ranges
  rail_id, clock_domain_id, gateable: bool

KPUThermalProfile (extended, additive)
  domain_operating_points: {domain_id: {clock_mhz, vdd_v, gated: bool, activity}}   default = chip clock/vdd
  tdp_scenario: {tile_class_id: activity}       concurrency assumption used to derive TDP (D6)
```

Site accounting invariant: `sum(num_tiles * footprint.rows * footprint.cols) +
spare_sites == compute_sites.rows * compute_sites.cols`. This replaces today's
silent pad/truncate (C11).

### 5.5 Silicon accounting

- **Tiles may carry their own silicon.** This covers datapath per PE, local SRAM
  per KiB, overlays per instance, and fixed-function blocks. The chip-level
  `silicon_bin` keeps the uncore: L3 memory cells, NoC, PHYs, IO, control.
- **New `TransistorSource` kinds:** `PER_TILE` (count_ref `tile_class:<id>`) and
  `PER_OVERLAY`. The legacy `tile.<label>` form resolves through the label
  alias.
  *As built (C1):* not added. Tile-carried silicon (`silicon_math.carried_silicon`)
  counts per-tile and per-overlay transistors directly from the tile, overlay
  and core specs, with no schema release. `count_ref` `tile.<ref>` accepts a
  `tile_class_id` as well as the legacy label. Both kinds can still be added
  if a chip-level block ever needs to scale per tile or per overlay.
- **No double counting.** A tile class may be counted either in its own
  `silicon` or in chip-level `per_pe` blocks, never both. A validator enforces
  this.
- **Dynamic power dispatches on `tile_kind`**, not on block-name prefixes (C6).

### 5.6 `FunctionCore`: one definition, two placements

```
FunctionCore                     (new module function_core.py)
  function_id                    e.g. isp.raw_to_yuv, vio.stereo_inertial, stereo.sgm, radar.rd_cfar
  contract: {inputs, outputs, config_limits}
  numeric_formats: [NumberFormat]
  throughput: {unit: pixel|frame|feature|point|chirp, units_per_clock | cycles_per_unit, fmax_mhz_ref}
  energy: {pj_per_unit, ref_node, logic_fraction, sram_fraction}
  ops_equivalent_per_unit: {format: ops}   reporting only (R13)
  io_bytes_per_unit: {in, out}
  local_memory, silicon
  provenance: {source, ref_node, confidence}
```

- **In the mesh:** `FixedFunctionTile.core`.
- **Out of the mesh:** a future `FixedFunctionBlock` in `Die.blocks` that wraps
  the same `FunctionCore`. That is the SoC plan's ISP/codec path and the
  assessment's separate-block proposal.

Both placements share the definition, the energy scaling and the provenance.
Placement is an allocation decision the study tool can sweep: in-mesh gains
stream links and checkerboard power domains; out-of-mesh avoids consuming
compute sites.

### 5.7 Performance roll-up

- **Move to the generic `TheoreticalPerformance.peak_ops_per_sec_by_precision`**
  (compute_block_common). Add `by_tile_kind` and
  `fixed_function_throughput: {function_id: units_per_s}`.
- **Keep the legacy KPU fields** (`int8_tops`, `bf16_tflops`, ...). They are
  derived from `pe_fabric` + `systolic` only.
- **Fixed-function "equivalent TOPS" never enters the legacy fields.**

### 5.8 Tile-class library

- **Location:** `embodied-schemas/data/kpu-tile-classes/<id>.yaml`, with a
  loader `load_kpu_tile_classes()`.
- **Private overlay:** an optional `KPU_TILE_DATA_DIR` env var, following the
  `PROCESS_NODE_DATA_DIR` precedent, for Stillwater-internal RTL-characterized
  datapaths (CALIBRATED, possibly confidential).
- **SKUs are self-contained.** They store fully resolved tile classes plus
  `tile_class_ref`, so validators never need the library.
- **Reference nodes:** add `tsmc_n65` (THEORETICAL) as the scaling anchor for
  65 nm literature such as Navion.

## 6. graphs-side changes

| Area | Change |
|---|---|
| `kpu_sku_input.py` / generator | Tiles may be `{use: <library id>, num_tiles, overrides}`. Performance roll-up by kind. `--pe-array CLASS=RxC` (scoped to `pe_fabric`; the bare form stays as today for legacy). New `--tile-mix CLASS=N,...`. Emits the resolved tiles. |
| `silicon_math.py` | Per-tile silicon, `PER_TILE` / `PER_OVERLAY`, per-class local memory, dispatch on `tile_kind`, `kpu_block_of(cp)` lookup instead of `blocks[0]` |
| `kpu_power_model.py` | Per-kind energy. **pe_fabric**: datapath ops x EnergyRef, plus the existing per-PE structures (token match, regs, forward, clock), plus overlay traffic (*deferred in C2*: no workload overlay-traffic model yet). **systolic**: MAC plus buffers. **fixed_function**: units/s x pj_per_unit scaled to the node. Per-domain V/f, gating, `tdp_scenario`. The legacy path is unchanged. |
| Validators | Split `tops_per_watt_envelope`: programmable kinds keep the node ceiling; a new `fixed_function_energy_plausibility` checks energy per unit against a cited anchor band. New: `checkerboard_site_accounting`, `tile_footprint_pitch_fit`, `datapath_energy_resolution` (ops consistency is enforced by the schema), `overlay_consistency`, `stream_link_adjacency`, `stream_link_bandwidth`, `power_domain_coverage`, `silicon_no_double_count`, and the DVFS doc's cluster validators `cluster_rail_and_clock` and `cluster_geometry_consistent`. *Deferred:* harvesting, process-variation-bin and rail-registry reference validators, which need schema fields (per-cluster disable, variation bins, `Die.voltage_rails`) |
| `silicon_floorplan.py` | `TileRole` gains FIXED_FUNCTION and SYSTOLIC. The placer is explicit map, or greedy: multi-site footprints first, then affinities, then stream-link adjacency, then fill. Absorbed memory cells; per-class whitespace; what-if views per class; power-domain overlay rendering. |
| `kpu_yaml_loader.py` / resource model | One `ComputeFabric` per tile class keyed by `tile_class_id`. Precision profiles come only from programmable kinds. Per-class `TileSpecialization.clock_domain` comes from power domains (the runtime field already exists). A new optional `HardwareResourceModel.fixed_function_units`. Per-class `KPUTileEnergyModel`, keeping the dominant-class accessor for legacy tests. |
| KPU mapper | Precision- and capability-aware tile pools, as `kpu-modeling.md:160-173` already proposes. GEMM/conv prefer `systolic`. |
| Reporting | Stop treating `tile_specializations[0]` as representative (`compare_archetypes.py`, `native_op_energy.py`, `layer2_register.py`). Do **not** rename `TileSpecialization`, because about 13 model files import it. |
| CLIs | `show_kpu` / `list_kpus` / `validate_sku` / `show_floorplan` become kind-aware. `show_kpu --tile-classes` lists the library. |
| SoC plan hook | The SoC analyzer's engine list includes each KPU tile class. Workload `segments` (named stage sets) can be absorbed by a `fixed_function` tile whose `function_id` matches; absorbed intermediate traffic is removed. Precision-floor findings come from R17. |

## 7. Decisions for review

| # | Decision | Recommendation |
|---|---|---|
| D1 | Specialized engines as KPU tile classes (in the mesh) or as sibling `Die.blocks` | **Both, through one `FunctionCore` definition** (5.6). Default to in-mesh for KPU SKUs, per the architecture direction. |
| D2 | Where the tile-class library lives | embodied-schemas `data/kpu-tile-classes/`, plus the private overlay env var. SKUs store resolved copies. |
| D3 | Ops-counting convention for custom ops and fixed-function equivalents | Count software-equivalent scalar ops, matching the workload op counts (mac 2, fma 2, lerp 3, min_plus 2, abs_diff 2). Equivalents are reported separately and never in headline TOPS. |
| D4 | Checkerboard with multi-site tiles | Uniform site pitch. Footprints are integer site rectangles that may absorb their memory cells. Variable-pitch rows are rejected (they break the pitch-matching guarantee). |
| D5 | Power-domain granularity | Generic `PowerDomain`. `pe_fabric` sites default to the 4x4 cluster partition from the DVFS doc; each fixed-function class gets its own gateable domain. |
| D6 | TDP definition with heterogeneous tiles | A declared `tdp_scenario` (concurrency plus activity) per thermal profile. The default reproduces today's formula exactly for legacy SKUs. |
| D7 | Source of LNS / custom-datapath energy | THEORETICAL ratios to the anchor op now. Stillwater RTL synthesis numbers, when available, go in the private overlay as CALIBRATED. |
| D8 | Migrate t768 Matrix (8x8, weight-stationary, 128 ops/PE) to `systolic` | Not in this sprint (it would break zero-diff). New SKUs use `systolic`. |
| D9 | SKU id convention for heterogeneous KPUs (today `kpu_t<N>_<R>x<C>_...` bakes in one PE array) | `kpu_h<sites>_<mixtag>_<mem>_<node>`, for example `kpu_h64_auto1_lp5x8_7nm_tsmc_hpc`. Legacy ids unchanged. |
| D10 | Reserve a `scalar` tile kind (a small core for Class C irregular work: kNN trees, hashing, graph search) | Reserve the enum value now; no model this sprint |

## 8. Implementation plan (PR sequence)

Gate on every PR: the **golden snapshot** (A0) for all 12 catalog SKUs is
byte-identical, unless the PR is labeled `model-change` and carries the Vdd
re-tune data PR in embodied-schemas. "ES" = embodied-schemas; "G" = graphs.

**Phase A: baseline and de-duplication** (no behavior change)
- A0 (G): **DONE (2026-09-14).**
  - Pieces: `src/graphs/hardware/kpu_golden.py`, `cli/kpu_golden_snapshot.py`,
    `tests/hardware/test_kpu_golden.py` (59 tests) and
    `tests/cli/test_kpu_golden_snapshot.py` (8 tests). The 12 goldens are in
    `tests/hardware/golden/kpu/` (golden schema v2).
  - Per SKU, it snapshots:
    - the catalog input;
    - the generator round-trip, including the TDP breakdown per profile;
    - transistors, area, leakage, and dynamic power at every supported precision, per silicon_bin block;
    - both floorplans, including every placed block;
    - the PhysicalSpec;
    - the full resource model;
    - KPUMapper results on 4 synthetic subgraphs at every precision;
    - all validator findings.
  - Comparison is structural, with a relative float tolerance of 1e-9.
  - This is the safety net for everything after it.
- A1 (ES + G): collapse the `KPUArchitecture` / `KPUBlock` duplication.
  `KPUBlock` inherits the architecture fields, and the 6 hand-copy sites are
  replaced by one converter.
- A2 (G): `kpu_block_of(cp)` (lookup by kind, error if ambiguous) replaces the
  `blocks[0]` / `_kpu_block` sites. The SoC plan's multi-block dies need the
  same change.
- A3 (G): refresh stale artifacts: the contract snapshot doc, `kpu-modeling.md`,
  `test_phase4_accelerator_energy.py`, and the stale mapper registry description
  strings.

**Phase B: schema, additive** (ES, release 0.7.0)
- B1: `NumberFormat`, `OpKind`, `FunctionalUnit`, `EnergyRef`, `PEDatapath`.
  `KPUTileSpec` gains `tile_kind`, `tile_class_id`, `label` alias, optional
  `datapath`, `footprint`, `local_memory`, `power_domain_id` and `placement`,
  plus the before-validator.
- B2: `FabricInterconnect` / `FabricOverlay`, and `NoCOverlay` on `KPUNoCSpec`.
- B3: `SystolicTile`, `FunctionCore` (new module), `FixedFunctionTile`, and the
  `AnyKPUTile` union.
- B4: `CheckerboardSpec`, generic `PowerDomain` (compute_block_common),
  `domain_operating_points` and `tdp_scenario` on the thermal profile.
- B5: performance roll-up by tile kind, plus the fixed-function throughput map.
  Legacy fields derived.
- B6: tile-class library directory and loader. First entries:
  - pe_fabric: `pe_int8_mac_i32` (the legacy-equivalent), `pe_bf16_fma`,
    `pe_lns16_mac`, `pe_fp16_lerp`, `pe_minplus_i16`;
  - systolic: `systolic_int8_ws`;
  - fixed_function: `ff_isp_raw2yuv`, `ff_vio_stereo_inertial`, `ff_stereo_sgm`;
  - plus the `tsmc_n65` anchor node.

  All THEORETICAL, with citations.
- **Accept:** all 12 YAMLs load unchanged. ES tests are extended for every new
  type. `load_kpus` reports heterogeneous SKUs it cannot express as a legacy
  `KPUEntry` explicitly, instead of skipping them silently.

**Phase C: graphs consumers** (legacy zero-diff on every PR)
- C1: `silicon_math`: per-tile silicon, new transistor sources, per-class
  memory, kind dispatch, no-double-count. *As built:* tile-carried silicon
  in place of the new transistor sources (see 5.5), plus the shared
  synthetic fixture `graphs.hardware.kpu_hetero_fixture`.
- C2: `kpu_power_model`: per-kind energy, per-domain V/f, gating, `tdp_scenario`.
  *As built:*
  - Legacy-shaped profiles keep the original formula. The new
    `compute_heterogeneous_tdp_breakdown` reproduces it exactly for all
    12 SKUs, within 1e-12.
  - It returns a `HeterogeneousTDPBreakdown`: `fixed_function_w`, compute by
    tile class, and gated classes.
  - A RelativeEnergy anchor invocation counts as 2 ops, which keeps
    `pe_int8_mac_i32` legacy-equivalent in this per-op model.
  - Fixed-function energy scales by its logic / SRAM fractions, using the
    logic and SRAM energy ratios between the reference node and the target
    node.
  - PE-fabric overlay-traffic energy is deferred: it needs a workload
    overlay-traffic model, and no library overlay declares energy yet.
- C3: generator and input spec: library `use:`, `--tile-mix`, scoped
  `--pe-array`, roll-ups.
  *As built:*
  - The by-kind performance roll-up is emitted only for heterogeneous
    architectures (or on request), so uniform legacy SKUs regenerate
    byte-identically.
  - A `use:` entry takes its tile fields under `overrides`.
  - `--tile-mix` refuses an explicit placement map.
- C4: validators: the envelope split and the new validators (section 6).
  *As built:*
  - The envelope split compares programmable TOPS with TDP minus the
    fixed-function power.
  - `fixed_function_energy_plausibility` anchors on the tile-class library
    entry for the same function, with a 4x band.
  - `datapath_ops_consistency` became `datapath_energy_resolution`: ops
    consistency is already enforced by the schema.
  - Of the DVFS design's 5 cluster validators, `cluster_rail_and_clock` and
    `cluster_geometry_consistent` are implemented. Harvesting,
    variation-bin and the rail-registry reference need schema fields and
    are deferred.
- C5: resource-model loader, per-class energy model, the
  `fixed_function_units` attribute, capability-aware mapper pools, and the
  reporting `[0]`-representative fixes.
  *As built:* split in two.
  - **C5a** (loader, per-class energy models, `fixed_function_units`,
    reporting fixes). `fixed_function_units` and `tile_energy_models` are
    class attributes rather than dataclass fields, so the golden snapshot
    of every uniform SKU is unchanged. Classes whose ops are not in the
    `Precision` enum (LNS, min-plus) get an energy model but no compute
    fabric.
  - **C5b** (capability-aware mapper pools). Kept separate because pools
    change legacy mapper results wherever a precision is not supported by
    every tile class (FP32 on T64 runs on 13 of 64 tiles), so they are
    enabled only for heterogeneous architectures.
    *As built:* `HardwareResourceModel.is_heterogeneous_kpu` gates the new
    path (more than one programmable tile kind, or any fixed-function
    tile); every catalog SKU is uniform `pe_fabric` and keeps the flat
    pool. A pool holds the classes whose `ops_per_tile_per_clock` lists the
    precision, ordered systolic-first for a dense matrix product
    (`CONV2D`, `CONV2D_POINTWISE`, `LINEAR`, `MATMUL`,
    `MULTIHEAD_ATTENTION`; depthwise convolution deliberately stays on the
    PE fabric). Allocation fills each class before spilling to the next and
    meets both the thread demand and the scratchpad tiling floor. Compute
    time comes from the throughput of the allocated tiles rather than an
    `allocated / compute_units` fraction of the chip -- on a mixed fabric
    those differ by more than the tile ratio. Utilization is reported
    against the tiles with precision-typed ops, not `compute_units` (which
    counts the fixed-function tiles). When no class runs the precision the
    pool declines and the flat path applies, unsupported-precision error
    included. Per-class MAC energy in the mapper is deferred to Phase F;
    the pool path still uses the chip-level per-op energy.
- C6: kind-aware CLIs (`show_kpu`, `list_kpus`, `validate_sku`).
  *As built:* a shared `graphs.hardware.kpu_tile_display` is the one place
  that knows what each kind looks like -- geometry, library and PE count
  return None / 0 where the kind has none, rather than reading a systolic
  or fixed-function tile as a PE fabric (which raised `AttributeError`).
  `show_kpu` gains a kind column, a per-class detail line, a tile census,
  and checkerboard / power-domain / by-tile-kind sections that a uniform
  SKU omits entirely; `list_kpus` gains a `tile kinds` column (shown only
  when some SKU is heterogeneous) and a `--kind` filter. Both CLIs, plus
  `validate_sku`, take `--from-file PATH` so a generated SKU is inspected
  and validated before it reaches the catalog -- the E1 iteration loop.
  Fixed-function tiles contribute no PEs to any total, and site accounting
  counts a 2x2 footprint as four sites.
- **Accept:** the golden snapshot is unchanged. A synthetic heterogeneous
  fixture (one class of each kind, including a 2x2 fixed-function footprint)
  flows through generator -> validators -> loader -> mapper without special
  cases.

**Phase D: heterogeneous checkerboard floorplan** (G)
- D1: placer (explicit plus greedy), footprints, absorbed memory cells,
  affinities, stream-link adjacency.
  *As built:* `graphs.hardware.kpu_checkerboard_placer` answers only the
  combinatorial question -- which class owns which site -- so it can be
  tested on a grid without building a die; `silicon_floorplan` turns sites
  into millimetre boxes. Auto placement runs the four passes in the order
  above; a class that declared an affinity optimises for it first and uses
  adjacency as the tiebreak, so an IO-edge ISP is not dragged inboard by
  its stream partner. Stream-link adjacency reads the NoC overlay's
  endpoints as well as `placement.adjacent_to`, so declaring the overlay
  is enough. An explicit `placement_map` is read back rather than
  re-placed, recovering footprints as rectangles and reporting a tile-count
  disagreement as a note for the C4 site-accounting validator. Each site
  is one compute cell plus its paired memory cell (two physical cells
  wide); a multi-site tile spans the memory halves between its compute
  halves, so it *covers* all of its cells -- geometry -- and *absorbs*
  them only when it declares `absorbs_memory_cells`, which is accounting.
  `TileRole` gains SYSTOLIC and FIXED_FUNCTION. Blocks with no
  `checkerboard` (every catalog SKU) keep the legacy row-major walk, so the
  golden floorplans are byte-identical.
- D2: GEOMETRY validators (site accounting, footprint pitch fit, whitespace per
  class), `show_floorplan` glyphs, and the power-domain overlay.
  *As built:*
  - The **per-class area model** (carried from D1) is fixed. A `count_ref`
    naming a class by `tile_class_id` now resolves through
    `silicon_math.resolve_tile_ref` instead of being matched against
    `tile_type`, and tile-carried silicon feeds the per-class area
    alongside the silicon_bin `per_pe` blocks, split into compute
    (datapath, systolic cells, fabric overlays, function core) and memory
    (`local_memory`). A class carrying its own SRAM uses that in place of a
    share of the chip-wide L2 pool it does not draw on. All 7 fixture
    classes now resolve to real area; before, 6 of them were zero.
  - The memory roll-up counts the cells that survive placement, not one per
    tile, and the what-if estimates fill the grid rather than assuming the
    SKU's tile count. Both are zero-diff for legacy, where
    `mesh_rows * mesh_cols == total_tiles` on every catalog SKU.
  - `show_floorplan` gains `S` / `F` glyphs (the legend is built from the
    roles actually placed, so it picked them up for free), `--from-file`
    like the C6 CLIs, and `--overlay {tile-class,power-domain}`: a
    compute-site grid in site coordinates, which is what a placement
    question is actually about. A `tile_class` power domain covers whichever
    sites its classes landed on -- knowable only after placement -- while a
    `cluster` domain covers its declared site ranges.
  - **No new validators were needed.** C4's `checkerboard_site_accounting`
    and `tile_footprint_pitch_fit` cover site accounting and footprint fit,
    and `floorplan_whitespace_fraction` already names the worst per-class
    contributor.
  - Still open: the fixture does not validate cleanly. Its SGM class needs
    about 6 sites but declares a 1x1 footprint, so the unified pitch is set
    by one tile and the die reads 81% whitespace. The validators say so
    correctly; re-tuning the fixture's footprints belongs with the
    reference SKU in E1.
- **Accept:** legacy floorplans are byte-identical. For the fixture, the
  placement is deterministic and every validator is clean.

**Phase E: reference heterogeneous SKU and energy ladder** (ES data + G study)
- E1: input specs for `kpu_h64_auto1` at `tsmc_n16` and `tsmc_n7`. Sites:
  - PE fabric: INT8, LNS16 and min-plus classes;
  - 4 systolic sites;
  - 1 ISP at the IO edge;
  - 1 VIO at 2x2 absorbed;
  - 1 SGM, stream-linked ISP -> SGM -> VIO.

  Generate, validate, and render the floorplan.
  *As built:* `data/sku_specs/kpu_h64_auto1_lp5x4_{16nm_tsmc_ffp,7nm_tsmc_hpc}.yaml`.
  Every tile class is a library `use:` reference, so the SKU states only
  how many of each and where they sit. Site budget 24+8+6 PE fabric,
  4 systolic at 1x2, ISP 1x1, SGM 2x4, VIO 2x2, 5 spare = 64. The two
  variants differ in exactly three places -- process node, envelope, and
  the systolic accumulator's SRAM library (n7 has `sram_hp`, n16 does not,
  so that one drops to `sram_hc`; embodied-schemas#96) -- and place
  identically, which is the point of expressing a design in sites.
  Both generate and validate with no ERROR findings. At n16: 73.9 mm^2,
  29% whitespace, 38.9 INT8 TOPS.
  - **Three defects this flushed out, all fixed here:**
    1. A class's pitch was computed as if it were 1x1, so declaring a
       footprint changed nothing and one big core set the pitch of all 64
       sites. Pitch is now area *per site*.
    2. The v1 circuit floorplan never got D2's carried-silicon fix, so in
       that view the systolic and fixed-function classes still collapsed
       to the SRAM term and the pitch validators fired on the largest
       classes on the die.
    3. The placer ordered multi-site footprints first regardless of
       constraint, so the unconstrained systolic pairs took the edge sites
       next to the stereo core that the IO-edge ISP needed, and its stream
       link crossed the mesh. Ordering is now most-constrained-first
       (an affinity or a stream partner), largest footprint within that.
       `stream_link_adjacency` also now checks where tiles actually land
       under auto placement instead of trusting that a hint exists.
  - **Not done here:** the SKUs are not in the embodied-schemas catalog.
    Landing them needs an ES data PR, a release and a pin bump, and would
    put them in the golden snapshot and every parametrized catalog test.
    Worth doing, but as its own change.
- E2: `cli/analyze_kpu_tile_ladder.py`: one function (SGM, VIO front-end, GEMM)
  across the fixed_function / systolic / pe_fabric variants, plus the Orin
  GPU/CPU mappers.
  - Reports pJ/op, pJ per function unit, area, and the data movement avoided by
    segment encapsulation.
  - Every rung shows its provenance and confidence.
  *As built:* `graphs.hardware.kpu_tile_ladder` + the CLI. Four functions:
  `stereo.sgm`, `isp.raw_to_yuv`, `vio.stereo_inertial`, `gemm.int8`.
  - **The ops model is explicit and cited.** Comparing a core published in
    pJ/pixel with a tile published in pJ/op needs a figure for the
    arithmetic in a pixel, and that is a workload claim, not a silicon
    one. SGM's comes from the paper's own two figures (2.3 TOPS/W x 13440
    pJ/px, so INTERPOLATED); the others are structural op counts
    (THEORETICAL) with the arithmetic shown.
  - **A class is only priced on work it can run**: it must declare one of
    the required ops *and* an accepted operand format, and a systolic
    class must list the kernel in `supported_kernels`. Without that the
    ladder priced stereo path aggregation on a MAC array and an ISP on a
    weight-stationary array.
  - **A back-check guards the ops model.** A core's published energy
    divided by the ops model gives its implied pJ/op; when that is far
    above the node's arithmetic op the core is not arithmetic-bound, and
    the report says the programmable rungs are a lower bound rather than a
    like-for-like comparison. It fires for the ISP (4x) and VIO (15x).
  - **The monotonicity check is a finding, not an invariant.** Rungs are
    sorted by energy, so asking whether that sort ascends proves nothing;
    the check compares the resulting *kind* order against the expected
    one, over the KPU's own kinds. On `kpu_h64_auto1` it holds for GEMM
    and the ISP and fails for SGM and VIO -- and the SGM result is the
    interesting one: a purpose-built min-plus fabric at n16 (4173 pJ/px)
    beats the n40 stereo ASIC retargeted to n16 (6720 pJ/px), which is the
    argument for having the min-plus class at all.
  - Acceptance note: the plan expected a monotonic ladder. It is not
    monotonic, for the reasons above, and the tooling reports that rather
    than being tuned until it agrees.
- E3: register the tile classes as engines in the SoC study analyzer. Add
  workload `segments` and precision-floor findings. Compare against the uniform
  T64 on the 5 regimes.
  *As built: the half that does not need graphs#269.* That analyzer has no
  code yet -- `PipelineWorkload`, `StageDemand`, `engine_kind` and
  `SoCSpec` appear nowhere, `workloads/pipelines/` does not exist, and its
  item 0.2 (the Autonomy Workload Data Annex) is marked a blocker for four
  of the five regimes. So E3 delivers the export that analyzer will
  consume, and the regime comparison is deferred with it.
  - `graphs.hardware.kpu_engines` + `cli/list_kpu_engines.py`. One
    `EngineDescriptor` per tile class, in declaration order: what it can
    run (formats, systolic kernels, or a function id), what it costs
    (ops/s and pJ/op per format at the default clock, silicon per tile,
    and for a core its own work unit), and its **precision floor**.
  - **Range decides, not speed.** `supports_at_least` ranks formats by the
    range they carry, so a stage needing fp32 does not become runnable
    because an engine is fast at int8. LNS formats rank with the float
    they replace, which is the point of the class.
  - **Segments** are read back off the stream-link overlays, with absorbed
    traffic keyed by the *producer's* work unit -- a chain can mix
    per-pixel and per-frame stages and summing those would be meaningless.
  - **The T64 comparison, in the part reachable today:** the heterogeneous
    die carries no fp32 at all, so a QP solver that runs on the uniform
    T64's `bf16_primary` class has nowhere to go on `kpu_h64_auto1`. That
    is the trade the refactor makes, stated as a finding rather than a
    regime table.
  - A class whose ops are not precision-keyed (min-plus, abs-diff) reports
    no formats *with a note saying why*, rather than a zero a mapper would
    read as "no throughput".
- **Accept:**
  - The ladder is monotonic, with the ratios in bands backed by citations.
  - The ISP/SGM/VIO tiles pass the split envelope validator without waivers.
  - The Navion-anchored VIO tile, scaled to `tsmc_n65`, reproduces its
    published power to within 20%.

**Phase F: deliberate model corrections** (each one a `model-change` PR with an ES Vdd re-tune)
- Add the missing L1 area block (C11) to all 12 SKUs.
  *As built: the block was added, but C11's premise needed correcting
  first.* L1 was declared per PE. At the catalog's own 0.052 Mtx/KiB that
  is ~208,000 transistors per PE, against the 6,000 its own `per_pe` block
  budgets for "MAC + reg + sequencing" -- and binning it would have added
  835-851% of the die on the uniform SKUs, 148% on the T768. No Vdd
  re-tune absorbs that.
  - **The hierarchy, from the architect:** L3 is the block linear-algebra
    reuse scratchpad with its own memory tile; L2 reformats L3 blocks so
    streaming preparation is easy at fabric clock rates, and may sit in
    either the L3 memory tile or the compute tile depending on the bank
    layout needed to feed L1; **L1 is per compute tile**, tightly coupled
    to the fabric edges, turning row and column fetches out of L2 into
    streams of operands pushed into the edges of the fabric. A PE is an
    ALU with datapath registers and a 16-32 entry token CAM, and holds no
    SRAM.
  - embodied-schemas 0.11.0 renames `l1_kib_per_pe` to `l1_kib_per_tile`
    and adds the `l1_sram` block to every KPU SKU: 0.17-0.45% of die area.
  - **No Vdd re-tune was needed, and that was measured rather than
    assumed.** C11 expected one because "the L1 area feeds leakage and
    therefore TDP" -- it does, at 0.003 W/mm^2 for `sram_hd` on n16, but
    the added leakage is under 1 mW, two orders of magnitude below the
    0.1 W the thermal profiles declare their TDP at. Every SKU's rounded
    TDP is unchanged at every profile.
  - **Separately found:** the `lp` and `default` profiles of every
    `7nm_tsmc_hpc` SKU already declare more TDP than the model computes
    (T64 `lp`: declared 1.3 W, computed 1.1 W). Verified against the
    baseline as pre-existing, and left alone rather than folded into an
    unrelated change.
- **F4 (found while measuring F1): the 7 nm TDP drift.** The `lp` and
  `default` profiles of every `7nm_tsmc_hpc` SKU declared more TDP than
  the power model computes -- the T512's `lp` claimed 10.3 W against a
  computed 8.9 W. `tsmc_n7` carries `leakage_vdd_exponent: 4.5`, so below
  nominal Vdd the leakage term falls steeply; these four kept the round
  0.500 / 0.650 / 0.750 V they were authored with, while the 16 nm family
  and the T768 were re-tuned when that scaling landed.
  - **The tell was which profile looked healthy.** `boost` sits exactly at
    the node's nominal 0.75 V, where the scaling is a no-op, so the one
    profile that could not reveal the problem was the one that passed.
  - The declared envelope is the target and Vdd is the knob, so the Vdds
    moved: `lp` 0.500 -> ~0.537, `default` 0.650 -> ~0.663, all four
    converging on the same band and all below nominal. Every KPU profile
    in the catalog now computes the TDP it declares.
  - **The guard that was missing:** nothing compared declared against
    computed TDP, which is why it sat unnoticed. A
    `declared_tdp_matches_model` validator now does (WARNING at 0.05 W,
    ERROR at 10%), with a test that restores the old Vdds and asserts it
    fires, so the validator cannot be silently inert.
- Optionally migrate t768 Matrix to `systolic` (D8).
- Adopt the per-cluster DVFS default on the legacy SKUs.
  *As built (F2):* embodied-schemas 0.13.0 gives each uniform SKU one
  `cluster` domain per k x k block of compute sites, each with its own
  `rail_id` and `clock_domain_id`, plus one `uncore` domain for the memory
  PHYs, IO and control logic. Cluster size follows the DVFS design's table
  and its "recommended k=4": 2x2 for the T64, 4x4 elsewhere, so 16 / 8 /
  16 / 32 clusters, and 48 for the T768 the table omits. Every cluster
  clears the design's ~1 mm floor, measured from the floorplan pitch (the
  tightest is the T64 at 7 nm, 1.06 mm).
  - **No model change, by construction.** No profile declares
    `domain_operating_points`, so every cluster runs at its profile's Vdd
    and clock and `is_legacy_shaped` still holds. The golden snapshot
    confirms it independently: exactly one difference per legacy SKU, the
    input's `power_domains` -- nothing in silicon, floorplan, resource
    model, mapper or validator findings moved.
  - **Schema:** a `cluster` domain's site ranges now resolve against the
    implicit mesh when there is no checkerboard, which is what the
    `checkerboard` field already documented `None` to mean. Requiring an
    explicit checkerboard would have pushed the uniform SKUs onto the
    heterogeneous floorplan path, which lays each memory cell beside its
    compute cell instead of in the 2D interleave.
  - **The design contradicts itself for the T128**: its prose rule of thumb
    ("the smallest cluster such that ... 8-64 clusters") picks 2x2 (32
    clusters), its table says 4x4. The table wins.
  - `generate_kpu_sku --default-power-domains` reproduces the catalog data;
    `show_floorplan --overlay power-domain` now renders a uniform SKU's
    partition on the implicit row-major site plan; `show_kpu` collapses a
    regular partition to one line.
  - **Caught along the way:** `build_heterogeneous_kpu` starts from the T64
    catalog entry, so the fixture silently inherited the 16-cluster
    partition, which happened to validate on its 8x8 checkerboard. Only one
    test, which narrows the mesh, exposed it. Fixed and guarded.
  - **Not modeled**, each needing a schema field or data the catalog lacks:
    the quadrant level, regulator and PLL silicon and power, per-cluster
    harvest, and process-variation bins. Clusters are not `gateable`: the
    design gates tiles and quadrants, and makes the cluster the DVFS and
    floorsweeping unit.

## 9. Risks and mitigations

| Risk | Mitigation |
|---|---|
| The TDP roundtrip (0.5 W) breaks from a well-meant model tweak | Golden snapshot gate. Model changes are isolated in Phase F with the Vdd re-tune protocol. |
| Fixed-function tiles trip the INT8 TOPS/W ceiling (false positives) | Split the envelope (C4) **before** the first heterogeneous SKU lands (E1) |
| "Equivalent TOPS" inflates headline numbers | R13: separate fields, and the legacy fields exclude fixed function. Reports label equivalents explicitly. |
| Invented control structures in PE-fabric accounting | R18. Overlays are statically configured links. No routers, ROMs or sequencers inside the tile. Per-PE structures stay as enumerated (FMA, regs, accumulator, forward, token match, clock). |
| Double-counted silicon during the transition (per-tile and chip-level `per_pe`) | The `silicon_no_double_count` validator is an ERROR |
| Placer scope creep | Explicit map plus a greedy heuristic only. Optimization-based placement is out of scope. |
| Fixed-function numbers from literature at old nodes scale poorly (analog, SRAM) | Split energy into logic and SRAM fractions and scale each by the node ratio for its library. Keep them THEORETICAL and show the reference node in every report. |
| A fixed-function VIO violates the stage precision floor (PDF Class C) | The precision-floor check makes it an explicit finding, with a waiver field and a justification citation. It is never silently accepted. |
| Hidden consumers of the tile shape (hard-coded 64x16x16 models in `operand_fetch.py`, `cycle_energy/kpu.py`, the slide generator) | Out of scope, because they are not schema-fed. List them in the tracking issue so nobody mistakes them for schema-driven results. |

## 10. Reference YAML sketches (values illustrative)

```yaml
# pe_fabric: LNS MAC datapath with row broadcast + express overlay
- tile_kind: pe_fabric
  tile_class_id: pe_lns16_mac
  label: LNS16-MAC
  num_tiles: 24
  pe_array_rows: 32
  pe_array_cols: 32
  datapath:
    datapath_id: lns16_mac
    circuit_class: balanced_logic
    functional_units:
      - {op: mac, operand_format: lns16, accumulate_format: lns16, lanes: 1,
         ops_per_invocation: 2, energy: {anchor: "balanced_logic:int8", ratio: 0.9}, mtx: 0.005}
    token_match: true
  interconnect:
    base: nearest_neighbor_4
    link_bits: 32
    overlays:
      - {kind: row_broadcast, width_bits: 32, instances_per: row, configuration: static_per_schedule}
      - {kind: express, span: 4, width_bits: 32, instances_per: row, configuration: static_per_schedule}
  local_memory:
    - {level: l1, kib: 4, per: pe, circuit_class: sram_hd}
    - {level: l2, kib: 32, per: tile, circuit_class: sram_hd}
  power_domain_id: pe_cluster_default

# systolic: pure weight-stationary INT8 GEMM array
- tile_kind: systolic
  tile_class_id: systolic_int8_ws
  num_tiles: 4
  array_rows: 64
  array_cols: 64
  dataflow: weight_stationary
  mac: {op: mac, operand_format: int8, accumulate_format: int32, lanes: 1, ops_per_invocation: 2,
        energy: {anchor: "balanced_logic:int8", ratio: 0.6}, mtx: 0.004}
  supported_kernels: [gemm, conv2d]
  local_memory:
    - {level: weight_buffer, kib: 64, per: tile, circuit_class: sram_hd}
    - {level: accumulator, kib: 32, per: tile, circuit_class: sram_hp}

# fixed_function: Navion-class VIO segment, 2x2 sites, absorbs its memory cells
- tile_kind: fixed_function
  tile_class_id: ff_vio_stereo_inertial
  num_tiles: 1
  footprint: {rows: 2, cols: 2, absorbs_memory_cells: true}
  placement: {affinity: any, adjacent_to: [ff_stereo_sgm]}
  power_domain_id: ff_vio
  core:
    function_id: vio.stereo_inertial
    contract: {inputs: [stereo_gray_frame, imu_sample], outputs: [pose_6dof],
               config_limits: {max_width: 1440, max_height: 1080, max_fps: 90}}
    numeric_formats: [fixed16, fp32]
    throughput: {unit: frame, cycles_per_unit: TBD, fmax_mhz_ref: TBD}
    energy: {pj_per_unit: TBD, ref_node: tsmc_n65, logic_fraction: 0.5, sram_fraction: 0.5}
    ops_equivalent_per_unit: {fp32: TBD, int8: TBD}
    io_bytes_per_unit: {in: TBD, out: 64}
    provenance: {source: "Suleiman et al., Navion, IEEE JSSC 2019", ref_node: tsmc_n65, confidence: theoretical}

# chip level
checkerboard:
  compute_sites: {rows: 8, cols: 8}
  memory_cell: {l3_kib: 256, circuit_class: sram_hd}
  placement: auto
noc:
  overlays:
    - {kind: stream_link, endpoints: [ff_isp_raw2yuv, ff_stereo_sgm, ff_vio_stereo_inertial], width_bytes: 32}
```

The `TBD` values come from the cited source when the library entry is authored
(B6). The plan deliberately does not fix them here.
