# KPU Cluster Organization for DVFS, Process Variability, Aging, and Floorsweeping

Status: design proposal, not yet implemented (2026-05-12)
Owner: Theo Omtzigt (Stillwater) + Claude Code

## Problem statement

The current KPU schema (`KPUArchitecture` -- tiles + NoC + memory) treats
the chip as a homogeneous mesh: one chip-wide voltage rail, one chip-wide
clock, no first-class concept of floorsweeping. That works for a small
edge SKU run at one operating point. It will not scale to:

- **Aggressive DVFS policies** -- modern accelerators run multiple
  voltage / clock domains independently. Apple M-series products
  document per-cluster rails for P-CPU / E-CPU / GPU / ANE / fabric;
  NVIDIA's Hopper / Blackwell whitepapers describe a per-GPC clock
  hierarchy plus separate memory / fabric clock domains, though the
  per-rail Vdd partitioning is industry inference rather than
  documented in the public whitepapers. Without per-region
  domains, the KPU cannot exploit local idleness or local thermal
  headroom.
- **Process variability** -- per-tile post-test binning informs how much
  Vdd headroom each region needs. Without a domain structure that
  matches the spatial scale of systematic variation (~1-3 mm
  correlation length at advanced nodes), every region runs at the
  worst-tile Vdd.
- **Aging differential** -- the centermost tiles run hottest and age
  fastest. Without per-region DVFS, aging is uneven and worst-region
  reliability gates the whole chip.
- **Floorsweeping (yield recovery)** -- defects cluster spatially
  (Stapper / negative-binomial). Without a floorsweeping unit matched
  to typical defect cluster sizes, harvested SKUs either lose too
  much silicon per defect (granularity too coarse) or require
  software-routable NoC (granularity too fine, only Cerebras-class
  designs achieve this).

This document evaluates three organizational options (by row, by column,
2D clusters of N x N tiles) against the spatial physics of variability,
aging, and defects, and recommends a **3-level hierarchy of tile ->
cluster -> quadrant** with concrete cluster sizing per existing SKU.

This sits alongside the broader assessment at
[`docs/assessments/kpu-as-generic-compute-product.md`](../assessments/kpu-as-generic-compute-product.md)
which calls out voltage / clock domains and floorsweeping as
load-bearing axes for the unified ComputeProduct schema. This document
narrows to the KPU-mesh-specific question: how to group tiles into
domain units.

## Background

### What the literature says about spatial structure on a die

**Process variability has two components, and they behave differently:**

1. *Random (matching) variation* -- per-transistor sigma scales as
   `1/sqrt(W*L)` (Pelgrom et al., "Matching properties of MOS
   transistors", IEEE JSSC 1989). Uncorrelated between transistors.
   Treated as noise floor for domain organization -- doesn't motivate
   any spatial choice.

2. *Systematic across-die variation* -- the picture is more nuanced
   than commonly stated, and the parameter matters:
   - **Gate length (Leff)** shows systematic spatial variation with
     measurable correlation across millimeter scales (Friedberg, Cao,
     Cain, Wang, Rabaey, Spanos, "Modeling within-die spatial correlation
     effects for process-design co-optimization", ISQED 2005). The
     Friedberg work characterizes Leff over a full-field range of
     horizontal and vertical separation in 0.18 micron CMOS and shows
     measurable systematic spatial structure. Earlier work
     (Orshansky et al., 2002, on intra-chip gate length variability)
     reports up to 17% critical-path delay variation from systematic
     Leff spatial structure if untreated.
   - **Threshold voltage (Vt)**, by contrast, is **dominated by random
     dopant fluctuation (RDF)** at advanced nodes, with little
     significant spatial correlation. Drego, Chandrakasan, Boning,
     "Lack of spatial correlation in MOSFET threshold voltage variation
     and implications for voltage scaling" (IEEE TSM 2009), measured
     50,000+ devices per die in a 180 nm CMOS process and found no
     significant within-die Vt spatial correlation -- across-chip
     variation patterns between different die also do not correlate.
     The implication is that Vt-binning at fine spatial granularity
     mostly chases noise; voltage scaling is the dominant Vt-management
     lever, not spatial Vdd binning.

Practical implication for cluster organization: the systematic-variation
case rests on **Leff** (gate-length systematic structure), not Vt.
Cluster sizes that span the Leff systematic-variation correlation
length see useful per-cluster Vdd binning headroom (the cluster's
nominal Leff differs from neighbors' nominal Leff, so it can run at a
slightly different Vdd / clock target). Cluster sizes well above the
correlation length lose this benefit -- you bin to the worst sub-region
within the cluster, wasting Vdd headroom on better sub-regions.

I have not been able to cite a single canonical correlation-length
number for advanced nodes; the literature varies from "uncorrelated at
within-die scales" (Drego 2009 for Vt) to "measurable systematic
structure with millimeter-scale features" (Friedberg 2005 for Leff).
**This proposal's cluster sizing of 2-3 mm on a side is a defensible
default consistent with the Friedberg-class literature, but the exact
correlation length should be characterized for the specific KPU
process node before sizing is finalized.**

**Defect clustering follows Stapper / negative-binomial models**
(Stapper, "Modeling of integrated circuit defect sensitivities", 1983;
Cunningham, Reed, and others have updated for modern nodes). Defects
are NOT Poisson-distributed -- they cluster. Cluster sizes typically
1-100 transistors, larger from particle defects.

Practical implication: adjacent tiles are more likely to fail together
than distant tiles. Floorsweeping at a granularity matched to typical
defect cluster size is the yield-economic sweet spot. Too small wastes
good tiles (a single bad tile harvests one tile out of, say, 4 in a
2x2 cluster -- 25% loss); too large wastes too much silicon per
harvest (one bad tile harvests 16 in a 4x4 -- 6% loss but the loss
becomes 100% of one cluster's contribution).

**Aging (NBTI, HCI, EM) is thermal-gradient-driven** (Borkar,
"Designing reliable systems from unreliable components: the challenges
of transistor variability and degradation", 2005; Bowman et al.,
"Impact of die-to-die and within-die parameter variations on max FREQ
measurements", 2002). Hottest cells age fastest. On a uniform mesh,
the centermost tiles run hottest (worst thermal conductance to package
edges); on a perimeter-IO design, the corner tiles run hottest (PHY
hotspots). **Aging gradient ~= thermal gradient.**

Practical implication: DVFS domains aligned with thermal gradient
regions (center / edge / corner) let firmware equalize aging across
the die by under-clocking the hot-region domains slightly relative to
the cool-region domains for the same workload.

**Memory uses row + column redundancy** (classic DRAM design) because
DRAM defects are bit-line / word-line failures (linear structures).
Logic does NOT get the same payoff -- logic defects are more spatially
scattered than memory defects.

Practical implication: row + column organization is a memory-side
optimization (use it within SRAM macros), not a logic-side one.

### What real mesh accelerators do

| Architecture | Floorsweeping granularity | DVFS granularity | Notes |
|---|---|---|---|
| Cerebras WSE-2/3 | Per-core (850K cores) | Per-tile-region | Fault-tolerant routing maps around dead sites at the *finest* granularity ever shipped. Possible only because the NoC is software-routable. Outlier in economics (one wafer per chip) |
| Tesla Dojo D1 | Per-node (354 nodes/chip) | Per-node power-gate, per-tile DVFS | Tile-level redundancy in the 5x5 D1 stack; per-node power-gating; not aware of formal per-node DVFS |
| Google TPU v4/v5 | Per-MXU (multiple per chip) | Coarse (per-chip) | MXU = 256-element matrix multiply; redundant MXUs allow harvesting at coarse granularity |
| NVIDIA Hopper (GH100) | Per-SM (132 of 144 enabled), per-HBM-stack (5 of 6) | Per-GPC clock hierarchy (8 GPCs x 18 SMs) per the Hopper whitepaper; separate memory / fabric clock domains; per-rail Vdd partitioning is industry inference, not documented in the public whitepaper | Floorsweeping at SM (~10 mm^2), clock-domain at GPC (~80 mm^2) |
| AMD MI300X | Per-XCD (compute chiplet) | Per-XCD + per-IOD | 8 XCDs of 38 CUs; chiplet boundary IS the harvest + DVFS boundary |
| Apple M3 family | Per-GPU-cluster | Per-cluster (P-CPU / E-CPU / GPU / ANE / fabric) | M3-Max -> M3-Pro harvests by disabling GPU clusters |

The pattern: production mesh accelerators harvest at **roughly 5-50 mm^2
per harvest unit** at advanced nodes, matching typical defect cluster
size and amortizable DVFS overhead. Cerebras is the outlier (per-core
fine-grained) because their economics support it; everyone else picks
a granularity in the middle.

For DVFS specifically, cluster sizes in the millimeter-scale range are
consistent with the documented Leff systematic-variation literature
(Friedberg 2005). Sizing below the systematic-correlation length means
the cluster is binning against random noise; sizing well above it means
binning to the worst sub-region. The exact sweet spot needs measurement
on the specific KPU process node.

## Design options

The KPU is a 2D mesh (e.g., 16 x 16 = 256 tiles in T256, 32 x 16 = 512
in T512, 16 x 8 in t128, 8 x 8 in t64). Memory and compute tiles
alternate in a checkerboard. The question: how to group these tiles
into voltage / clock / floorsweeping domain units.

A single tile (32 x 32 PE array, ~0.5-1 mm^2 at N16) is **too small for
its own voltage rail** -- the LDO / PLL overhead is uneconomical, and
the tile is well below the systematic-variation correlation length.
A single tile is **too small to be a useful floorsweeping unit alone**
either -- a typical defect cluster spans multiple tiles, and harvesting
one bad tile from a 2 x 2 group of dependents is expensive.

### Option A: by-row organization

Tiles in the same row share a voltage rail and clock; entire rows can
be disabled for floorsweeping.

- **Pros**:
  - Simple horizontal power-bus distribution
  - Matches the row-addressing convention familiar from DRAM
  - Easy to disable an entire row by gating its supply
- **Cons**:
  - Hot rows accumulate all their heat (poor thermal isolation between
    same-row tiles); thermal coupling between rows is weak by design
  - Systematic variation runs ACROSS rows perpendicular to wafer-stepper
    direction in many process flows -- a row sees varied transistors,
    not uniform ones
  - Floorsweeping per-row loses too much silicon per harvest (16+ tiles
    in a t256 row; one defect kills the row)
  - The wide power-bus has to span the whole die width
- **Best fit**: not for compute. Useful for the SRAM macros INSIDE a
  tile (row + column redundancy classic).

### Option B: by-column organization

Same as by-row, rotated. Same trade-offs. Often paired with by-row in
memory designs (row + column redundancy together) but not standalone
useful for compute.

### Option C: 2D clusters of N x N tiles

Tiles in the same N x N square share a voltage rail and clock; entire
clusters are the floorsweeping unit.

- **Pros**:
  - Matches systematic-variation correlation length when sized to ~1-3 mm
  - Matches defect-cluster typical size (defects cluster spatially;
    so does the harvest unit)
  - Thermal coupling within cluster but better isolated between clusters
  - Bus / clock / Vdd routing crosses cluster boundaries with
    hierarchical structure (one extra level in the PDN, not one extra
    level per tile)
  - Floorsweeping at N^2 x tile granularity is the right size at
    advanced nodes
- **Cons**:
  - Routing complexity: NoC must respect cluster boundaries for
    placement
  - Compiler must respect cluster boundaries for placement to honor
    domain affinity
- **Best fit**: compute + structural metadata; this is the production
  mesh-accelerator pattern.

## Recommendation: 3-level hierarchy (tile -> cluster -> quadrant -> chip)

```
Level 0: Tile (1 tile, ~0.5-1 mm^2 at N16)
  - Base computation unit
  - No DVFS, no Vdd rail of its own
  - Can be individually power-gated (clock-gate + leakage power-gate)
  - Can be individually disabled in floorsweeping at fine grain
    (within an enabled cluster) -- supports partial-cluster harvest

Level 1: Cluster (k x k tiles, recommended k=4)
  - DVFS domain: own Vdd rail + own PLL/clock
  - Floorsweeping unit: a defect anywhere in the cluster harvests
    the cluster by default; matches defect cluster size
  - Thermal coupling: within-cluster tiles share thermal neighborhood;
    isolated thermally from adjacent clusters
  - Sized 2-3 mm on a side at N16 (4 x 4 tiles * ~0.5-1 mm/tile)
  - Sits in the millimeter-scale range where Leff systematic
    variation is documented (Friedberg 2005); for advanced nodes the
    target should be re-validated against per-node correlation-length
    measurements

Level 2: Quadrant (j x j clusters, recommended j=2)
  - Coarse DVFS for chip-wide thermal management
  - Region-level power gating (drop a quadrant for sleep states)
  - Aligns with thermal-gradient regions (center / edge / corner)
  - Matches the natural hot-region / cold-region division of the die

Level 3: Chip
  - TDP envelope, package thermal, cooling solution
  - Already present in the current schema
```

The 3-level hierarchy is the production sweet spot:

- **Tile is the unit-cell**: enables fine-grained partial-cluster
  harvest (one bad tile inside an otherwise-good cluster doesn't have
  to disable the whole cluster, IF the NoC routing can tolerate it).
- **Cluster is the DVFS / floorsweeping unit**: economical because
  cluster size matches both physics scales (correlation length, defect
  cluster size) and engineering scales (LDO / PLL overhead amortized
  over N^2 tiles).
- **Quadrant is the thermal-management unit**: enables firmware to
  apply per-region DVFS for thermal balancing and aging equalization.
- **Chip is the package envelope**: existing schema layer.

### Cluster sizing per SKU

Recommended cluster size = 4 x 4 tiles, organized into 2 x 2 quadrants:

| SKU | Mesh | Cluster size | Cluster count | Quadrants | Cluster size (mm) at N16 |
|---|---|---|---|---|---|
| t64  | 8 x 8   | 2 x 2 | 16 | 2 x 2 = 4 | ~1 mm  (small; below correlation length but limited by total tile count) |
| t128 | 16 x 8  | 4 x 4 | 8  | 2 x 2 = 4 | ~2 mm  (sits in the millimeter range matched to documented Leff variation) |
| t256 | 16 x 16 | 4 x 4 | 16 | 2 x 2 = 4 | ~2 mm  (same range) |
| t512 | 32 x 16 | 4 x 4 | 32 | 4 x 2 = 8 | ~2 mm  (same range) |

For t64 specifically, 4 x 4 clusters would leave only 4 clusters total
(2 x 2 mesh of 4 x 4 tiles) which is too coarse for DVFS; 2 x 2
clusters is the right fit at the small scale.

For very large future SKUs (t1024, t2048), cluster size could grow to
6 x 6 or 8 x 8 to keep cluster count manageable (target: 16-64 clusters
per chip).

**Rule of thumb**: pick the smallest cluster size such that the cluster
is at least 1 mm on a side AND the per-chip cluster count is in the
range 8-64 (enough for meaningful DVFS / harvest decisions, not so many
that VR / PLL overhead dominates).

### Where row + column does fit

**Memory subsystem within each tile**: the L3 scratchpad and L2 cache
*internal* to a tile or cluster should use **row + column redundancy**
(Stapper-style) since DRAM-like defects in SRAM arrays are mostly
bit-line / word-line. This is independent of the cluster-level
floorsweeping for compute -- happens at a finer granularity inside
the SRAM macro, transparent to the cluster boundary. The existing
silicon_bin entries for `l2_sram` / `l3_sram` would gain
`redundancy_rows` / `redundancy_cols` fields per macro.

**HBM stacks (when added to a future datacenter KPU)**: HBM dies have
their own internal redundancy (vendor-handled). Modeled at the
`MemoryBlock` level per the chiplet caveat in the assessment doc;
no impact on the KPU compute mesh's cluster structure.

## Schema additions (concrete)

This builds on the per-die `voltage_rails` / `clock_domains` proposed
in the assessment doc:

```
KPUArchitecture (extended)
  - cluster_geometry: ClusterGeometry          # NEW
      ClusterGeometry:
        - cluster_rows, cluster_cols           # tiles per cluster
        - quadrant_rows, quadrant_cols         # clusters per quadrant
        - tile_to_cluster_map: derived         # auto-computed by row/col
        - cluster_to_quadrant_map: derived

  - clusters: list[Cluster]                    # NEW
      Cluster:
        - cluster_id (e.g., "c_0_0", "c_0_1", ...)
        - tile_range: {row_min, row_max, col_min, col_max}
        - voltage_rail_id: str                 # references voltage_rails
        - clock_domain_id: str                 # references clock_domains
        - quadrant_id: str
        - is_disabled: bool = False            # cluster-level harvest
        - disabled_tiles: list[tile_id] = []   # partial-cluster harvest
                                               # (tile-level disable
                                               # within an enabled cluster)
        - process_variation_bin: enum | None
            slow_corner | typical | fast_corner
            # post-test binning informs per-cluster Vdd headroom

  - quadrants: list[Quadrant]                  # NEW
      Quadrant:
        - quadrant_id
        - cluster_ids: list[str]
        - voltage_rail_id: str | None          # quadrant-level rail
                                               # (separate physical rail
                                               # OR derived as parent of
                                               # cluster rails -- vendor
                                               # choice)
        - is_disabled: bool = False            # quadrant-level harvest
                                               # for major-defect cases
```

Existing `KPUNoCSpec` gains awareness of cluster boundaries:

```
KPUNoCSpec (extended)
  - cluster_boundary_routing: BoundaryRouting  # NEW
      BoundaryRouting:
        - kind: free | preferred_within_cluster | restricted_to_cluster
        - cluster_egress_bandwidth_gbps        # BW available for traffic
                                               # leaving a cluster
        - cluster_egress_latency_ns            # extra latency for
                                               # cross-cluster traffic
```

Relationship to the harvested-SKU pattern (assessment doc):

- Parent product: every cluster has `is_disabled: false` and
  `disabled_tiles: []`
- Harvested SKU: lists which clusters or tiles are disabled;
  `harvested_from` points to parent silicon
- Per-SKU process_variation_bin reflects post-test binning of THIS
  particular SKU's silicon

## Operational implications

### For the kpu_power_model (existing)

The 5-term roll-up (PE compute + L2 + L3 + NoC + DRAM PHY + leakage)
becomes per-cluster:

```
For each cluster c:
  cluster_vdd = Vdd of cluster c's voltage_rail at the active profile
  cluster_clock = Hz of cluster c's clock_domain at the active profile
  cluster_voltage_scale = (cluster_vdd / nominal_vdd)^2

  cluster_compute_w = peak_compute_w_for_cluster(c) *
                      effective_utilization * compute_duty_cycle *
                      cluster_voltage_scale
  ... (similar for L2, L3, NoC, DRAM contributions allocated to c)

chip_dynamic_w = sum over enabled clusters c of cluster_dynamic_w
chip_leakage_w = sum over enabled clusters c of cluster_leakage_w
                  (disabled clusters contribute 0; power-gated)
```

This naturally handles harvested SKUs (disabled clusters drop out of
the sum) and per-cluster DVFS (each cluster scales independently).

### For the floorplanner (Stage 8, existing)

The architectural-role floorplan view (`derive_kpu_architectural_floorplan`)
gains a cluster-overlay rendering: clusters are drawn as bounding boxes
around their N x N tile groups, with voltage-rail color-coding. The
cross-check validators add `floorplan_cluster_geometry_valid` (every
tile is in exactly one cluster; every cluster is in exactly one
quadrant; no overlap; no gaps).

### For the validator framework (existing)

New validators:

- `cluster_voltage_rail_valid`: every cluster's `voltage_rail_id`
  references an entry in `Die.voltage_rails`
- `cluster_clock_domain_valid`: same for `clock_domain_id`
- `cluster_geometry_consistent`: `cluster_rows * cluster_cols *
  cluster_count == total_tiles`; quadrants partition clusters
- `harvested_cluster_disable_consistent`: when `is_disabled: true` on a
  cluster, every tile in that cluster is also marked unreachable;
  silicon_bin area for the disabled cluster's PE area is excluded
  from the as-shipped die area (NOT from the parent silicon's die area)
- `process_variation_bin_explains_vdd`: if a cluster has
  `process_variation_bin: slow_corner`, its Vdd should be at the
  higher end of the rail's range (sanity check the binning)

### For the generator (existing)

`generate_kpu_sku()` gains:

- A pass that auto-builds `clusters: list[Cluster]` from
  `cluster_geometry` (row / col mapping is mechanical)
- A pass that allocates default voltage_rail_id / clock_domain_id per
  cluster (one rail per cluster + one shared chip-wide rail for
  uncore by default; architects override if needed)
- A pass that respects `disabled_tiles` and `is_disabled` when
  computing as-shipped die area / transistor count / TDP / performance

## Future considerations

### Aging-aware DVFS (firmware policy, not schema)

Once per-cluster DVFS is in place, firmware can equalize aging by
under-clocking the centermost (hottest) clusters slightly relative to
the edge clusters for the same workload. The schema gains nothing for
this -- the existing per-cluster `voltage_rail_id` and the telemetry
framework expose what's needed. The Intel "thermal velocity boost"
feature is the same idea at the core level.

### Process-variation-aware DVFS (post-binning policy)

Per-cluster `process_variation_bin` informs runtime: slow-corner clusters
need higher Vdd headroom for the same clock target. The catalog records
the bin from post-silicon testing; firmware's DVFS table is per-bin
specific.

### Cluster size sweeps (a programmable knob)

The same way `--pe-array ROWSxCOLS` lets architects sweep PE-array size
for roadmap studies, a future `--cluster-size NxN` flag would let them
study DVFS / floorsweeping economics across cluster sizes. Mechanical
to implement once the schema is in place.

### Relationship to the unified ComputeProduct schema

This proposal lives within `KPUArchitecture`, but the underlying
concepts (cluster as DVFS / floorsweeping unit, hierarchy of tile ->
cluster -> quadrant) are not KPU-specific:

- GPU SMs aggregate into GPCs aggregate into the chip (NVIDIA's
  3-level hierarchy)
- AMD CDNA CUs aggregate into XCDs (chiplet) aggregate into the
  package (2-level hierarchy with a chiplet break)
- TPU MXUs aggregate into systolic-array clusters aggregate into the
  chip

The same `Cluster` / `Quadrant` concepts will recur in `GPUBlock`,
`CDNABlock`, `TPUBlock` once the unified ComputeProduct schema lands.
This proposal's schema additions should be designed with future
generalization in mind -- the `Cluster` struct should NOT be
KPU-tile-specific; it should describe a generic 2D-mesh cluster of
unit-cells that any spatial accelerator block type can use.

## References

All references below have been verified against publisher / IEEE Xplore
records. Every cited claim in the body should trace to one of these;
where a cited claim is industry inference rather than directly
documented, the body says so.

- Pelgrom, M. J. M., Duinmaijer, A. C. J., Welbers, A. P. G.,
  "Matching properties of MOS transistors", IEEE Journal of
  Solid-State Circuits, vol. 24 no. 5, October 1989, pp. 1433-1440.
  DOI: 10.1109/JSSC.1989.572629.
- Stapper, C. H., "Modeling of integrated circuit defect
  sensitivities", IBM Journal of Research and Development, vol. 27
  no. 6, November 1983, pp. 549-557.
- Borkar, S., "Designing reliable systems from unreliable components:
  the challenges of transistor variability and degradation",
  IEEE Micro, vol. 25 no. 6, November-December 2005, pp. 10-16.
- Bowman, K. A., Duvall, S. G., Meindl, J. D., "Impact of die-to-die
  and within-die parameter fluctuations on the maximum clock frequency
  distribution for gigascale integration", IEEE Journal of Solid-State
  Circuits, vol. 37 no. 2, February 2002, pp. 183-190. (Note: title
  uses "fluctuations", not "variations".)
- Friedberg, P., Cao, Y., Cain, J., Wang, R., Rabaey, J., Spanos, C. J.,
  "Modeling within-die spatial correlation effects for process-design
  co-optimization", International Symposium on Quality Electronic
  Design (ISQED), 2005. (Characterizes Leff systematic spatial
  variation in 0.18 micron CMOS.)
- Drego, N., Chandrakasan, A., Boning, D., "Lack of spatial
  correlation in MOSFET threshold voltage variation and implications
  for voltage scaling", IEEE Transactions on Semiconductor
  Manufacturing, 2009. (Counterpoint: Vt at advanced nodes is
  dominated by RDF and shows no significant within-die spatial
  correlation in the measured 180 nm CMOS process. The cluster
  argument here rests on Leff systematic variation per Friedberg,
  not on Vt; this reference is included to flag the contrasting
  empirical finding.)
- Cunningham, J. A., "The use and evaluation of yield models in
  integrated circuit manufacturing", IEEE Transactions on Semiconductor
  Manufacturing, vol. 3 no. 2, May 1990, pp. 60-71. (Updates Stapper;
  compares Poisson and negative binomial models against actual yield
  data from seven IC companies.)
- Cerebras Hot Chips presentations:
  - HC 31 (2019): original Wafer-Scale Engine (WSE).
  - HC 34 (2022): WSE-2 / CS-2.
  - HC 36 (2024): "Cerebras Wafer-Scale AI" (WSE-3 era).
  All cover the fault-tolerant fabric and per-core redundancy.
  PDFs available via the Hot Chips program archives.
- Tesla Dojo Hot Chips presentations:
  - HC 34 (2022): "DOJO: The Microarchitecture of Tesla's Exa-Scale
    Computer" (also published in IEEE Micro proceedings).
  - HC 36 (2024): Dojo update including the Mojo Dojo Compute Hall
    rack-scale system. Confirms tile-level redundancy + per-node
    power-gating; per-node DVFS specifics not formally documented.
- NVIDIA, "NVIDIA H100 Tensor Core GPU Architecture" whitepaper
  (GTC 2022). Confirms 8-GPC, 18-SMs-per-GPC, 144-SM full-silicon
  topology and the 700 W SXM5 / 350 W PCIe TDP envelopes.
  Per-rail Vdd partitioning is NOT broken out in the whitepaper --
  the per-GPC voltage-rail claim made in this design doc is
  industry inference, not citation-supported.
- NVIDIA, "Blackwell Architecture Technical Overview" (March 2024).
  Confirms GB100 dual-die via NV-HBI on B100; B200 single-die;
  208 billion transistors; HBM3e configurations.
- ITRS roadmaps -- 2009, 2013 main editions plus 2015 final chapters
  (after 2015 the effort was rebranded to IRDS, the International
  Roadmap for Devices and Systems). The Process Integration, Devices,
  and Structures (PIDS) working-group reports cover variability
  scaling. Cited in this doc as the standard industry reference for
  variability trends across nodes; specific quantitative numbers
  should be drawn from the per-edition PIDS chapters.

### What was REMOVED from an earlier draft of this references list

The earlier draft cited "Sengupta, A., Sapatnekar, S. S., 'Statistical
methodology for spatial correlations in MOSFET threshold voltage'."
That citation was incorrect -- no such paper appears to exist with
those authors and that topic. The Sengupta-Sapatnekar work that does
exist is on aging (BTI / HCI ring-oscillator-based sensors), not
spatial correlation. The closest relevant paper on Vt spatial
correlation is the Drego/Chandrakasan/Boning paper above, which is
included as a counterpoint reference.

The earlier draft also implied a specific "0.5-3 mm correlation
length" for systematic variation that combined Vt, Leff, and metal
width. The literature does not support a single canonical number range
across all three parameters at all nodes; the cluster sizing in this
design doc is a defensible default consistent with the Leff literature
(Friedberg 2005), not a citation-derived hard requirement.

## Status / next steps

1. Review and ratify this design with the KPU architecture team
2. Update `embodied-schemas/src/embodied_schemas/kpu.py` with the
   `Cluster` / `Quadrant` / `ClusterGeometry` Pydantic classes (and
   the `BoundaryRouting` extension to `KPUNoCSpec`)
3. Pick a default cluster geometry (4 x 4 tiles, 2 x 2 quadrants) and
   apply to the existing 12 catalog SKUs via the generator
4. Extend `kpu_power_model.py` to roll up per-cluster (already trivial
   given V^2*f scaling per cluster)
5. Extend `silicon_floorplan.py` to render cluster + quadrant overlays
6. Add the 5 cluster validators to the validator registry
7. Update the assessment doc cross-reference (this design is the
   productive answer to the voltage / clock / floorsweeping caveats)
