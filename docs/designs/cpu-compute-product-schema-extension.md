# CPU ComputeProduct schema extension -- paper exercise

Status: draft
Date: 2026-05-15
Tracking issue: graphs#182
Predecessor: GPU sprint #171 (closed); design doc
              `docs/designs/gpu-compute-product-schema-extension.md`

## Purpose

Issue #182 ("Sprint: extend ComputeProduct schema + catalog to CPUs")
proposes a 5-PR sprint to add `CPUBlock` to the `ComputeProduct`
discriminated union, mirroring sprint #171 for GPUs. This document is
**PR 1 of that sprint** -- the paper exercise: enumerate Intel Core
i7-12700K's full feature set against the v2 (KPU + GPU) ComputeProduct
schema and classify every feature as **covered**, **extends an existing
field**, or **needs a new field/type**.

The exercise uses **Intel Core i7-12700K** as the reference SKU because
its mapper is the most thoroughly-modeled CPU in the `graphs` catalog
(see `src/graphs/hardware/models/edge/intel_core_i7_12700k.py`,
425 LOC), the user has measured calibration data on local hardware
(per CLAUDE.md memory), and Alder Lake's **hybrid P-core + E-core**
architecture stresses every part of the schema -- if it fits Alder
Lake it'll fit anything.

The doc deliberately copies the structure of the GPU paper exercise
(`gpu-compute-product-schema-extension.md`) so the diff between the
two designs is easy to spot.

## Method

For each feature in the i7-12700K resource model, this doc records:

- The field/object as it lives in graphs today (with line ref).
- Classification: COVERED / EXTEND / NEW.
- For EXTEND: the existing schema field that grows.
- For NEW: a proposed field name + brief rationale.

Out of scope for this exercise (deferred to later in the sprint, or
to v4):

- iGPU (Intel UHD 770 on i7-12700K) -- separate `GPUBlock` (already
  exists from v2) or a future `IntegratedGraphicsBlock` if we want
  to model the iGPU's tighter LLC sharing.
- Intel ME (Management Engine), display engine, codecs -- separate
  block kinds; not modeled by the existing graphs factory either.
- Multi-socket NUMA, AMD chiplet topology, AMX (Sapphire Rapids
  server-only) -- none apply to the i7-12700K reference SKU. Each
  is called out under "open questions" as a v4+ extension.
- Per-field provenance (`EstimationConfidence`) -- graphs-side
  feature; orthogonal to schema design (same call as the GPU sprint
  made).

## Reference: current v2 schema state

`embodied-schemas/src/embodied_schemas/compute_product.py`:

```python
class BlockKind(str, Enum):
    KPU = "kpu"
    GPU = "gpu"

AnyBlock = Annotated[Union[KPUBlock, GPUBlock], Field(discriminator="kind")]
```

GPU sprint sub-types (v2): `GPUComputeFabric`, `GPUMemorySubsystem`,
`GPUOnDieFabric`, `GPUThermalProfile`, `GPUTheoreticalPerformance`,
`ClockDomain`, plus 4 enums (`GPUFabricKind`, `GPUL1Kind`,
`GPUL2Topology`, `GPUNoCTopology`).

KPU sprint sub-types (v1): `KPUTileSpec`, `KPUNoCSpec`,
`KPUMemorySubsystem`, `KPUSiliconBin`, `KPUClocks`,
`KPUTheoreticalPerformance`, `KPUThermalProfile`.

The v2 design choice (made deliberately, not by oversight) was to keep
KPU and GPU sub-types separate rather than generalize -- premature
unification with only 2 architectures would force tile-shaped fields
onto SM-shaped objects and vice versa. The CPU sprint should follow
the same rule: **ship CPU-specific sub-types in v3; defer rename +
unify to v4 when at least 4 architectures exist** and the right shape
becomes obvious.

## Feature-by-feature audit

### 1. Identity / packaging

| Field (graphs) | Verdict | Mapping |
|---|---|---|
| `name="Intel-Core-i7-12700K"` | COVERED | `ComputeProduct.name` |
| `hardware_type=HardwareType.CPU` | EXTEND | `Die.blocks[0].kind = BlockKind.CPU` (new enum value) |
| (no vendor today) | COVERED | `ComputeProduct.vendor = "intel"` |
| (no product family) | COVERED | `Market.product_family = "Core i7"` |
| (monolithic) | COVERED | `Packaging(kind=MONOLITHIC, num_dies=1)`. Note: AMD chiplet CPUs need `num_dies > 1` -- v4 scope. |

### 2. Process node

| Field (graphs) | Verdict | Mapping |
|---|---|---|
| `process_node_nm=10` (Intel 7 / 10nm ESF) | COVERED via reference | `Die.process_node_id` references `ProcessNodeEntry`. **NEW catalog YAML needed: `data/process-nodes/intel/intel_7.yaml`** -- the catalog has no Intel nodes today (only TSMC + GF + Samsung). |
| Intel 7 (10nm Enhanced SuperFin) | NEW | New foundry directory + node YAML. Same shape as the existing TSMC/GF/Samsung nodes. |

### 3. Hybrid core hierarchy (P-cores + E-cores)

This is the GPU sprint's "SM hierarchy" analog, but **fundamentally
different** because Alder Lake mixes two distinct core types in one
package. The GPU schema's flat `num_sms / cuda_cores_per_sm` doesn't
generalize. Need a new abstraction.

| Field (graphs) | Verdict | Mapping |
|---|---|---|
| `compute_units=10` (effective cores) | NEW | `CPUBlock.total_effective_cores: int` -- legacy proxy for downstream consumers. The TRUE source of cores is `CPUBlock.core_clusters[*].num_cores`. |
| `p_cores=8` | NEW | `CPUBlock.core_clusters[].num_cores` (with `cluster_kind=PERFORMANCE`) |
| `e_cores=4` | NEW | `CPUBlock.core_clusters[].num_cores` (with `cluster_kind=EFFICIENT`) |
| `e_core_efficiency=0.6` (weight on aggregate) | NEW | `CoreClusterSpec.aggregate_weight: float` -- multiplier used when rolling effective_cores |
| `threads_per_unit=2` (P-cores SMT) | NEW | `CoreClusterSpec.smt_threads: int` (P=2, E=1) |
| `warp_size=8` (256-bit AVX2 lanes) | NEW | `CPUBlock.simd_width_lanes: int` -- 8 for AVX2 FP32 |

**Recommendation: introduce `CoreClusterSpec`** (new helper type) with
`cluster_kind: CoreClusterKind` (PERFORMANCE / EFFICIENT / BIG / LITTLE / HOMOGENEOUS),
`num_cores: int`, `smt_threads: int`, `aggregate_weight: float`,
`compute_fabrics: list[CPUComputeFabric]`. AmpereOne / EPYC are
homogeneous (`cluster_kind=HOMOGENEOUS`, single cluster); Alder Lake
has two clusters; ARM big.LITTLE has the same shape as Alder Lake.

### 4. Per-cluster compute fabrics (AVX2 + AVX-VNNI today; AMX / NEON / SVE later)

The graphs model defines two `ComputeFabric` objects -- one per cluster
(P-core AVX2, E-core AVX2). Both have per-precision ops/clock and
per-fabric energy. The GPUComputeFabric type is close but not a
perfect fit (CPU fabrics have ISA extensions, not "fabric_kind" in
the GPU sense).

| Field (graphs) | Verdict | Mapping |
|---|---|---|
| `fabric_type="alder_lake_p_core_avx2"` | NEW | `CPUComputeFabric.isa_extension: CPUISAExtension` enum (AVX2, AVX_VNNI, AVX512, AVX512_VNNI, AVX512_BF16, AVX512_FP16, AMX_TILE, NEON, SVE, SVE2, SVE_BF16, etc.) |
| `circuit_type="simd_packed"` | EXTEND | Reuse `CircuitClass` from process_node module |
| `num_units=8` (8 P-cores) | NEW | This is the *cluster* core count, lives on `CoreClusterSpec.num_cores`. Not on the fabric directly. |
| `ops_per_unit_per_clock` (per Precision) | NEW | `CPUComputeFabric.ops_per_core_per_clock: dict[str, int]` -- same shape as GPU fabric's `ops_per_unit_per_clock` |
| `core_frequency_hz` | EXTEND | Carried by per-thermal-profile clock (DVFS), not the fabric |
| `energy_per_flop_fp32` | NEW | `CPUComputeFabric.energy_per_flop_fp32_pj` (mirror GPU pattern) |
| `energy_scaling: dict[Precision, float]` | NEW | `CPUComputeFabric.energy_scaling: dict[str, float]` |

**Note**: a single CPU core can host multiple SIMD fabrics (a Sapphire
Rapids core has AVX2 + AVX-512 + AMX simultaneously). The schema
should allow `core_clusters[i].compute_fabrics: list[CPUComputeFabric]`
so a core's full ISA portfolio is enumerated, not just the dominant
one. The graphs i7-12700K model only carries one fabric per cluster
because it picks the "best" (AVX-VNNI for INT8, AVX2 for FP). Keep
the list shape so future SKUs can carry the full set.

### 5. Memory hierarchy (richer than GPU/KPU because L1/L2 are per-cluster)

This is where CPU departs the most from KPU/GPU. CPUs have:
- **Per-core L1** that may differ between P and E (i7: 48 KB P, 32 KB E)
- **Per-core or per-cluster L2** (i7: 1.25 MB private per P-core, 2 MB per 4-E-cluster)
- **Shared L3 LLC** across all clusters (i7: 25 MB)
- **Real cache coherence** (snoopy MESI / MOESI / MESIF)
- **DRAM**

| Field (graphs) | Verdict | Mapping |
|---|---|---|
| `peak_bandwidth=76.8e9` (DDR5-4800 dual-channel) | EXTEND | `CPUMemorySubsystem.memory_bandwidth_gbps` |
| `main_memory=64 GB` | EXTEND | `CPUMemorySubsystem.memory_size_gb` |
| `l1_cache_per_unit=48 KB` (P-core proxy) | NEW | Move to `CoreClusterSpec.l1_kib_per_core` so heterogeneous L1 is modeled directly. The chip-level `CPUMemorySubsystem` carries no L1 field. |
| `l2_cache_total=25 MB` (LLC convention!) | EXTEND | This is the LLC = L3 by graphs convention. Map to `CPUMemorySubsystem.l3_total_kib`. The TRUE L2 (12 MB physical = 8x1.25MB P + 1x2MB E) needs a new field: `CoreClusterSpec.l2_kib_per_core` for P-cores and `CoreClusterSpec.l2_kib_shared` for E-cores' shared L2. |
| `l2_cache_per_unit=1.2MB` (= 12MB / 10) | NEW | Replaced by per-cluster fields above |
| `l2_topology="per-unit"` | NEW | Per-cluster: `CoreClusterSpec.l2_layout: L2Layout` (PRIVATE_PER_CORE / SHARED_PER_CLUSTER) |
| `l3_present=True` | EXTEND | Same shape as `GPUMemorySubsystem.l3_present`; the CPU type just has it set to True more often |
| `l3_cache_total=25*1024**2` | EXTEND | `CPUMemorySubsystem.l3_total_kib` |
| `coherence_protocol="snoopy_mesi"` | EXTEND | Same as `GPUMemorySubsystem.coherence_protocol`. Free-form string is fine; values: `none` (GPU SIMT default), `snoopy_mesi`, `snoopy_moesi`, `mesif`, `nvlink-c2c`, `cxl`, `directory_mesi`. |
| `memory_technology="DDR5"` | EXTEND | `CPUMemorySubsystem.memory_type: MemoryType` (existing enum from `embodied_schemas.gpu`) |
| `memory_read_energy_per_byte_pj=25.0` | EXTEND | `CPUMemorySubsystem.read_energy_pj_per_byte` |
| `memory_write_energy_per_byte_pj=30.0` | EXTEND | `CPUMemorySubsystem.write_energy_pj_per_byte` |

**Recommendation**: split memory between `CoreClusterSpec` (carries
per-core L1 and L2) and `CPUMemorySubsystem` (carries chip-shared L3 +
DRAM). Same separation pattern that the GPU schema uses for "per-SM
L1 inside `GPUBlock.memory.l1_kib_per_sm`" but more pronounced because
L2 is per-core on CPUs.

### 6. On-die fabric (ring bus / mesh / IO die)

| Field (graphs) | Verdict | Mapping |
|---|---|---|
| `topology=Topology.RING` | NEW | `CPUOnDieFabric.topology: CPUNoCTopology` enum (RING, MESH_2D, IO_DIE_PLUS_CCD, INFINITY_FABRIC) |
| `bisection_bandwidth_gbps=512.0` | EXTEND | `CPUOnDieFabric.bisection_bandwidth_gbps` |
| `controller_count=12` (= 8 P + 4 E ring stops) | EXTEND | `CPUOnDieFabric.stop_count: int` (renamed from `controller_count` because RING terminology is "stop") |
| `flit_size_bytes=32` | EXTEND | Same |
| `hop_latency_ns=1.5` | EXTEND | Same |
| `pj_per_flit_per_hop=5.0` | EXTEND | Same |
| `routing_distance_factor=1.0` | EXTEND | Same |

**Recommendation**: define `CPUOnDieFabric` separately from
`GPUOnDieFabric` despite the field overlap -- the topology *enum*
differs (CPUs need IO_DIE_PLUS_CCD, GPUs need CROSSBAR; neither makes
sense in the other context). Field names align so future v4
unification is mechanical.

### 7. Thermal profiles (CPU power-limit windowing: PL1 / PL2 / PL3)

CPU thermal profiles are simpler than GPU's per-precision DVFS:
typically two power limits (PL1 long-term, PL2 short-term boost) and
optionally PL3 / PL4 instantaneous. The i7-12700K model carries only
the 125 W PL1 entry today; missing PL2 (190 W default) is a graphs-side
data gap, not a schema issue.

| Field (graphs) | Verdict | Mapping |
|---|---|---|
| `name="125W-PL1"` | COVERED | `CPUThermalProfile.name` (or reuse `KPUThermalProfile`) |
| `tdp_watts=125.0` | COVERED | `CPUThermalProfile.tdp_watts` (or reuse) |
| `cooling_solution="tower-cooler"` | EXTEND | `CPUThermalProfile.cooling_solution_id` references `data/cooling-solutions/`. Need to verify "tower-cooler" / "passive-heatsink" are in the catalog -- looks like only `active_fan` / `passive_heatsink_large` are; minor data PR. |
| (no `clock_mhz` per profile) | NEW | CPUs run different P-core / E-core clocks at different power profiles. Need `CPUThermalProfile.p_core_clock_mhz` and `e_core_clock_mhz` (or generalized `per_cluster_clock_mhz: dict[str, float]`). The GPU schema's `ClockDomain` (base/boost/sustained) generalizes cleanly. |

**Recommendation**: shape `CPUThermalProfile` with **two ClockDomains**
(one for P-cluster, one for E-cluster) keyed by cluster name, plus the
existing `tdp_watts / cooling_solution_id`. Reuse `ClockDomain` from
GPU sprint -- this is the v3 expansion the GPU sprint deferred.

### 8. SIMD efficiency (CPU-specific concept absent in KPU/GPU)

| Field (graphs) | Verdict | Mapping |
|---|---|---|
| `simd_efficiency={"elementwise": 0.95, "matrix": 0.80, "default": 0.70}` | NEW | `CPUBlock.simd_efficiency_by_op_kind: dict[str, float]`. Captures vectorization-friendliness of different op classes. Doesn't apply to KPU (every op vectorizes by construction) or GPU (SIMT handles it implicitly). |

### 9. Performance roll-up

| Field (graphs) | Verdict | Mapping |
|---|---|---|
| `peak_ops_per_sec` per Precision | EXTEND | `CPUTheoreticalPerformance.peak_ops_per_sec_by_precision: dict[str, float]` -- same shape as `GPUTheoreticalPerformance` |

### 10. Silicon bin (per-block transistor decomposition)

`KPUSiliconBin` / its `count_ref` mechanism generalizes cleanly. New
`count_ref` strings for CPUs:

- `cluster.p_core` -> per-P-core Mtx
- `cluster.e_core` -> per-E-core Mtx
- `l1_total_kib` -> per-KiB L1 SRAM
- `l2_total_kib` -> per-KiB L2 SRAM
- `l3_total_kib` -> per-KiB L3 SRAM
- `ring_stops` -> per-stop ring bus logic
- `ddr5_phy.controller` -> per-DDR5-controller PHY
- `igpu` (catch-all for the integrated GPU when not modeled as a separate block)
- `me` (Management Engine)

**Verdict**: NO schema change needed -- just document the convention.
Same approach as GPU sprint took.

### 11. CPU-specific scheduler / mapper attributes

| Field (graphs) | Verdict | Mapping |
|---|---|---|
| `min_occupancy=0.4` | NEW | `CPUBlock.min_occupancy: float` (matches GPU pattern) |
| `max_concurrent_kernels=20` | NEW | `CPUBlock.max_concurrent_threads: int` (CPU equivalent) -- legacy compat field |
| `wave_quantization=1` | NEW | `CPUBlock.wave_quantization: int = 1` -- CPUs don't really wave-quantize; default 1 keeps the field shape consistent with GPU and lets shared mappers read the field uniformly |

### 12. BOM cost profile

Same story as GPU: not yet in the v2 schema; factory overlay until v3
adds `Market.bom`. NEW (optional). Defer.

## Summary table

| Category | Total fields | COVERED | EXTEND | NEW |
|---|---:|---:|---:|---:|
| Identity / packaging | 5 | 4 | 1 | 0 |
| Process node | 1 | 0 | 0 | 1 (new node YAML) |
| Hybrid core hierarchy | 6 | 0 | 0 | 6 |
| Per-cluster compute fabrics | 7 | 0 | 2 | 5 |
| Memory hierarchy | 12 | 0 | 7 | 5 |
| On-die fabric | 7 | 0 | 6 | 1 |
| Thermal profiles | 4 | 2 | 1 | 1 |
| SIMD efficiency | 1 | 0 | 0 | 1 |
| Performance roll-up | 1 | 0 | 1 | 0 |
| Silicon bin | 0 | 0 | 0 | 0 (just new count_ref) |
| Scheduler attrs | 3 | 0 | 0 | 3 |
| BoM cost (deferred) | 1 | 0 | 0 | 1 |
| **Totals** | **48** | **6** | **18** | **24** |

Of 48 fields, ~12% (6) are already covered, ~38% (18) extend an
existing schema field (more than GPU's 25%, because CPU sub-types
inherit a lot of pattern from the GPU sprint), and ~50% (24) are
net-new. The high EXTEND ratio is the payoff from the GPU sprint --
many CPU fields map cleanly onto GPU sprint types.

## Recommended schema diff for the next PR

The Schema PR (PR 2 of the sprint) should land roughly the following
in embodied-schemas:

1. **`compute_product.py`**:
   - Add `BlockKind.CPU = "cpu"` to the discriminator enum.
   - Update `AnyBlock = Union[KPUBlock, GPUBlock, CPUBlock]` with discriminator.

2. **New file `embodied_schemas/cpu_block.py`** (~350 LOC, similar to `gpu_block.py`):
   - `CPUBlock` with: `core_clusters: list[CoreClusterSpec]`,
     `total_effective_cores`, `simd_width_lanes`, `memory`, `noc`,
     `min_occupancy`, `max_concurrent_threads`, `wave_quantization`,
     `simd_efficiency_by_op_kind`.
   - `CoreClusterSpec` with: `cluster_kind: CoreClusterKind`,
     `num_cores`, `smt_threads`, `aggregate_weight`,
     `compute_fabrics: list[CPUComputeFabric]`, `l1_kib_per_core`,
     `l2_kib_per_core` (or `l2_kib_shared` for shared-cluster L2),
     `l2_layout: L2Layout`.
   - `CPUComputeFabric` with: `isa_extension: CPUISAExtension`,
     `circuit_class`, `ops_per_core_per_clock`, `energy_per_flop_fp32_pj`,
     `energy_scaling`.
   - `CPUMemorySubsystem` with: `memory_type`, `memory_size_gb`,
     `memory_bus_bits`, `memory_bandwidth_gbps`, `memory_controllers`,
     `l3_present`, `l3_total_kib`, `coherence_protocol`,
     `read_energy_pj_per_byte`, `write_energy_pj_per_byte`. (Per-core
     L1/L2 lives on `CoreClusterSpec`, not here.)
   - `CPUOnDieFabric` with: `topology: CPUNoCTopology`,
     `bisection_bandwidth_gbps`, `stop_count`, `flit_size_bytes`,
     `hop_latency_ns`, `pj_per_flit_per_hop`, `routing_distance_factor`.
   - `CPUThermalProfile` with: `name`, `tdp_watts`, `cooling_solution_id`,
     `per_cluster_clock_domain: dict[str, ClockDomain]` (cluster name
     -> ClockDomain). **Reuses `ClockDomain` from gpu_block** -- this
     is the first cross-block-kind type sharing in the schema.
   - `CPUTheoreticalPerformance` with `peak_ops_per_sec_by_precision`.
   - Enums: `CoreClusterKind` (PERFORMANCE / EFFICIENT / HOMOGENEOUS /
     BIG / LITTLE), `CPUISAExtension` (AVX2 / AVX_VNNI / AVX512 /
     AVX512_VNNI / AVX512_BF16 / AVX512_FP16 / AMX_TILE / NEON / SVE /
     SVE2 / SVE_BF16 / VPU), `L2Layout` (PRIVATE_PER_CORE /
     SHARED_PER_CLUSTER / SHARED_GLOBAL), `CPUNoCTopology` (RING /
     MESH_2D / IO_DIE_PLUS_CCD / INFINITY_FABRIC / DOUBLE_RING).

3. **New process node YAML**:
   - `data/process-nodes/intel/intel_7.yaml` -- Intel 7 (10nm
     Enhanced SuperFin). First entry in a new `intel/` foundry
     directory. Densities can be sourced from Wikichip + Anandtech
     Alder Lake die-shot analysis.

4. **Cooling solution catalog**:
   - The CPU model uses ``"tower-cooler"`` as a cooling string. The
     catalog has `active_fan`, `passive_heatsink_large`, `liquid_cooled`,
     `vapor_chamber`, `datacenter_dtc`, `passive_fanless`,
     `passive_heatsink_small`. None match. PR 3 (data) needs to either
     add `tower_cooler.yaml` or remap to `active_fan` and let the
     factory carry the descriptive string.

## Risks called out by this exercise

1. **Hybrid clusters are a new abstraction.** `CoreClusterSpec` is the
   first sub-type that maps cleanly to ARM big.LITTLE *and* Apple
   M-series performance/efficiency clusters AND Intel Alder/Raptor/
   Meteor Lake. Get it right the first time. Risk: medium. Mitigation:
   the existing graphs `intel_core_i7_12700k.py` model has battle-
   tested the shape; copy it.

2. **The `compute_units` legacy field is misleading on hybrid CPUs.**
   The graphs i7 model reports 10 effective cores (8 P + 0.6 * 4 E).
   Downstream consumers that read `model.compute_units` get a fictional
   number. Ship `total_effective_cores` on `CPUBlock` AND keep the
   roll-up via `compute_units` for backward compat.

3. **Per-cluster L2 split.** P-cores have 1.25 MB private L2 each;
   E-cores share 2 MB across the 4-E cluster. Schema needs to express
   both shapes. `CoreClusterSpec.l2_layout: L2Layout` enum +
   `l2_kib_per_core` (private) OR `l2_kib_shared` (cluster-shared)
   handles it. AmpereOne (homogeneous) only uses `l2_kib_per_core`;
   future Apple M-series uses both shapes per cluster. Risk: low if
   the L2Layout enum gets the right values.

4. **AMX, AVX-512, NEON, SVE all ship in real CPUs.** The
   `CPUComputeFabric.isa_extension` enum needs to be extensible.
   Pydantic's `str, Enum` pattern works -- add values as catalog
   demands. Don't try to enumerate everything up front. Risk: low
   (this is exactly the GPU sprint's `GPUFabricKind` pattern).

5. **Cache coherence as a free-form string vs enum.** GPU used a
   free-form string (`"none"`, `"nvlink-c2c"`, etc.) which works; CPU
   could too. Recommend KEEP it free-form for CPU as well -- the
   protocol space is rich (MESI / MOESI / MESIF / Directory-MESI /
   snoopy / source-snoop) and pinning to an enum forces a schema
   bump for every new variant. Risk: low.

6. **Multi-socket NUMA is out of scope.** Single-socket i7-12700K
   doesn't exercise it, but Xeon / EPYC do. Defer to v4 once at least
   one server CPU YAML lands -- at which point `CPUOnDieFabric.numa_domains`
   or a separate `Interconnect` at the `DIE_TO_DIE` level becomes
   appropriate. Same call as the GPU sprint made for multi-GPU NVLink.

7. **AMD chiplet topology (IOD + CCDs) is multi-die.** Falls under
   the existing `Packaging.kind=CHIPLET` deferred from v1. v4 unlocks
   it. For sprint #182 we only need single-die monolithic CPUs.

## Next step

Land this doc as PR 1 of the sprint (graphs-side). The Schema PR
(PR 2, embodied-schemas-side) implements the "Recommended schema
diff" above. Schema PR can be reviewed and merged independently of
PR 3 (data PR for first CPU YAML) since the new types are additive.

Estimated total sprint cost: ~3 days (matches GPU sprint) for the
i7-12700K reference. Each subsequent CPU SKU is then a YAML-only
addition, same as Thor was for GPUs.
