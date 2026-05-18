# v10 OnDieFabric unification -- paper exercise

Status: draft
Date: 2026-05-17
Tracking issue: `graphs#217`
Predecessors:
  - v8 unification: `compute_block_common` + ``TheoreticalPerformance`` (`graphs#208`, `graphs#210`)
  - v9 unification: ``ThermalProfile`` collapses 5 byte-identical classes (`graphs#215`)
  - DSP sprint: 8th + final block kind closes the category schema gap (`graphs#211`)

## Purpose

Issue `#217` proposes a v10 follow-up to v8/v9: collapse the **6
``*OnDieFabric`` classes** (CPU/GPU/NPU/CGRA/DPU/TPU) via an
**inheritance hierarchy** with a shared base in ``compute_block_common``.
Reconciles the endpoint-count naming inconsistency at the same time
(CPU's ``stop_count`` and GPU's ``controller_count`` → ``unit_count``,
matching NPU/CGRA/DPU/TPU).

This doc is **PR 1 of the v10 sprint** -- the paper exercise.
Mirrors the structure of the v8 and v9 paper exercises so the diff
between the three designs is easy to spot.

## Why inheritance (vs alias like v8/v9)

v8 unified ``TheoreticalPerformance`` via alias (5 classes were
byte-identical: ``GPUTheoreticalPerformance = TheoreticalPerformance``).
v9 unified ``ThermalProfile`` via alias (5 classes were byte-identical).

v10 ``OnDieFabric`` cannot use alias because **each block kind carries
its own ``topology`` enum** that is genuinely architecture-specific:

  - CPU: ``ring`` / ``double_ring`` / ``io_die_plus_ccd`` / ``infinity_fabric``
  - GPU: ``crossbar`` / ``ring`` / ``hierarchical``
  - NPU: ``dataflow_ring`` / ``systolic`` / ``crossbar`` / ``shared`` / ``partitioned``
  - CGRA: ``crossbar``
  - DPU: ``aie_mesh`` / ``crossbar``
  - TPU: ``crossbar`` / ``multi_crossbar``

Only ``crossbar`` appears in multiple enums. Unifying these into a
single ``NoCTopology`` enum would lose architectural specificity
(``infinity_fabric`` is meaningless to a TPU; ``aie_mesh`` is
meaningless to a CPU). **Topology enums stay per-block-kind.**

The inheritance approach gives the best of both worlds:

  - Shared fields defined once (in base)
  - Per-block-kind ``topology`` enum validation preserved
  - ``isinstance(x, NPUOnDieFabric)`` AND ``isinstance(x, OnDieFabric)``
    both work (the subclass relationship is real)
  - Each per-kind subclass shrinks from ~30 LOC to ~5 LOC

Pydantic supports model inheritance directly; this is a standard
pattern.

## Method

For each of the 6 ``*OnDieFabric`` classes in
``src/embodied_schemas/<kind>_block.py``, this doc records the
exact field set + endpoint-count field name, then identifies the
shared base (7 fields all 6 have) + optional extensions (mesh dims,
confidence) + per-kind unique fields (topology).

Out of scope:

- **Topology enum unification** -- intentionally per-block-kind (different architectural primitives)
- **``has_external_dram`` vs ``has_host_dram``** naming reconciliation -- touches more SKU YAMLs; semantic difference matters (chip-attached HBM vs host-bus PCIe DRAM); defer to v11
- **``KPUNoCSpec``** -- KPU NoC type is doubly-purposed (also chip-level fabric description); v11+
- ``MemorySubsystem`` / ``ComputeFabric`` unification -- KEEP_SEPARATE per the v8 paper exercise

## Reference: 6 ``*OnDieFabric`` classes today

```
embodied-schemas/src/embodied_schemas/
  cpu_block.py     -> CPUOnDieFabric      (line 383)
  gpu_block.py     -> GPUOnDieFabric      (line 255)
  npu_block.py     -> NPUOnDieFabric      (line 282)
  cgra_block.py    -> CGRAOnDieFabric     (line 271)
  dpu_block.py     -> DPUOnDieFabric      (line 294)
  tpu_block.py     -> TPUOnDieFabric      (line 358)
```

## Per-class audit

### NPUOnDieFabric -- 10 fields (canonical mesh-capable shape)

```python
topology: NPUNoCTopology      # per-kind enum
bisection_bandwidth_gbps: float
unit_count: int                # endpoint-count name
flit_size_bytes: int
mesh_rows: int | None          # optional
mesh_cols: int | None          # optional
hop_latency_ns: float
pj_per_flit_per_hop: float
routing_distance_factor: float
confidence: DataConfidence
```

### CGRAOnDieFabric -- **byte-identical** to NPU shape (just different topology enum)

Same 10 fields; ``topology: CGRANoCTopology`` is the only difference.

### DPUOnDieFabric -- **byte-identical** to NPU shape (just different topology enum)

Same 10 fields; ``topology: DPUNoCTopology`` is the only difference.

### TPUOnDieFabric -- 8 fields (no mesh)

```python
topology: TPUNoCTopology
bisection_bandwidth_gbps: float
unit_count: int
flit_size_bytes: int
# (no mesh_rows / mesh_cols -- TPU has no mesh)
hop_latency_ns: float
pj_per_flit_per_hop: float
routing_distance_factor: float
confidence: DataConfidence
```

Drops ``mesh_rows`` / ``mesh_cols`` because TPU MXUs share the UB via crossbar (no mesh routing). Otherwise identical to NPU/CGRA/DPU shape.

### CPUOnDieFabric -- 7 fields (``stop_count`` instead of ``unit_count``; no confidence)

```python
topology: CPUNoCTopology       # ring / double_ring / etc.
bisection_bandwidth_gbps: float
stop_count: int                # << different name!
flit_size_bytes: int
hop_latency_ns: float
pj_per_flit_per_hop: float
routing_distance_factor: float
```

**Differences from NPU shape**:
1. ``stop_count`` (CPU ring-bus jargon: ""stops"" are routing endpoints on the ring)
2. No ``confidence`` field
3. No mesh dims

Rename ``stop_count`` -> ``unit_count`` is straightforward; the per-arch field description will document that ""unit"" means ""ring stop"" for CPUs.

### GPUOnDieFabric -- 7 fields (``controller_count`` instead of ``unit_count``; no confidence)

```python
topology: GPUNoCTopology       # crossbar / ring / hierarchical
bisection_bandwidth_gbps: float
controller_count: int          # << different name!
flit_size_bytes: int
hop_latency_ns: float
pj_per_flit_per_hop: float
routing_distance_factor: float
```

**Differences from NPU shape**:
1. ``controller_count`` (GPU jargon: ""memory controllers"" are the partition unit for GPU fabric)
2. No ``confidence`` field
3. No mesh dims

Rename ``controller_count`` -> ``unit_count`` is straightforward; the per-arch field description will document that ""unit"" means ""memory controller"" for GPUs.

## Summary table

| Class | Endpoint name | mesh? | confidence? | Verdict |
|---|---|---|---|---|
| NPUOnDieFabric  | unit_count       | yes | yes | inherit + topology |
| CGRAOnDieFabric | unit_count       | yes | yes | inherit + topology |
| DPUOnDieFabric  | unit_count       | yes | yes | inherit + topology |
| TPUOnDieFabric  | unit_count       | no  | yes | inherit + topology |
| CPUOnDieFabric  | **stop_count**       | no  | no  | inherit + topology + rename + add confidence |
| GPUOnDieFabric  | **controller_count** | no  | no  | inherit + topology + rename + add confidence |

**Net LOC saved**: ~150 (6 × ~25 LOC of shared-field definitions removed; per-kind subclasses shrink to ~5 LOC each).

## Recommended ``compute_block_common.py`` diff (PR 2 scope)

Add the base class to ``compute_block_common.py``:

```python
class OnDieFabric(BaseModel):
    """Base for on-die fabric descriptions. Per-block-kind subclasses
    contribute the architecture-specific ``topology`` enum; this base
    holds the 7 fields all 6 block-kind NoCs share + 2 optional
    mesh dims + optional confidence.

    Inheritance (not alias) because the topology field is genuinely
    architecture-specific:
      - CPU: ring / infinity_fabric (Intel/AMD)
      - GPU: crossbar / ring / hierarchical
      - NPU/DPU/CGRA/TPU: each has its own enum

    Per-block-kind subclasses look like:

        class NPUOnDieFabric(OnDieFabric):
            topology: NPUNoCTopology = Field(...)

    ``isinstance(x, NPUOnDieFabric)`` AND ``isinstance(x, OnDieFabric)``
    both work (proper subclass relationship).
    """

    bisection_bandwidth_gbps: float = Field(..., gt=0)
    unit_count: int = Field(
        ..., gt=0,
        description=(
            "Number of fabric endpoints. Meaning varies by block kind: "
            "compute units (NPU/DPU/CGRA/TPU); ring stops (CPU); "
            "memory controllers (GPU)."
        ),
    )
    flit_size_bytes: int = Field(..., gt=0)
    hop_latency_ns: float = Field(..., ge=0)
    pj_per_flit_per_hop: float = Field(..., ge=0)
    routing_distance_factor: float = Field(1.0, gt=0)

    # Mesh-specific (optional; only populated when topology is a
    # mesh-like one). NPU/CGRA/DPU populate these for 2D meshes;
    # TPU/CPU/GPU leave them None.
    mesh_rows: int | None = Field(default=None, gt=0)
    mesh_cols: int | None = Field(default=None, gt=0)

    # NoC provenance. NPU/CGRA/DPU/TPU populate; CPU/GPU default to
    # THEORETICAL when not set.
    confidence: DataConfidence = Field(DataConfidence.THEORETICAL)

    model_config = {"extra": "forbid"}
```

Update ``__all__`` to include ``OnDieFabric``. Update the module docstring to note that ``OnDieFabric`` joins ``TheoreticalPerformance`` + ``ThermalProfile`` as a v8/v9/v10 unified type (the first **inheritance-based** one).

## Migration strategy (PR 3 proof of concept)

PR 3 of v10 sprint migrates **all 6 per-kind classes in one PR** + 3 YAML field renames:

### Schema changes

For each ``*_block.py``:

```python
# Before (e.g., NPU)
class NPUOnDieFabric(BaseModel):
    topology: NPUNoCTopology = Field(...)
    bisection_bandwidth_gbps: float = Field(..., gt=0)
    unit_count: int = Field(..., gt=0)
    # ... 6 more fields ...

# After
from embodied_schemas.compute_block_common import OnDieFabric

class NPUOnDieFabric(OnDieFabric):
    topology: NPUNoCTopology = Field(...)
```

For CPU and GPU, the subclass is identical but the old ``stop_count`` / ``controller_count`` field is removed (inherited as ``unit_count``).

### YAML changes (3 files)

  - ``intel/intel_core_i7_12700k.yaml``: ``stop_count: N`` -> ``unit_count: N``
  - ``nvidia/jetson_agx_orin_64gb.yaml``: ``controller_count: N`` -> ``unit_count: N``
  - ``nvidia/jetson_agx_thor_128gb.yaml``: ``controller_count: N`` -> ``unit_count: N``

Same commit as the schema changes so the catalog never validates against an inconsistent pair.

### Test impact

- Tests that import ``NPUOnDieFabric`` continue to work.
- ``isinstance(x, NPUOnDieFabric)`` still works.
- ``isinstance(x, OnDieFabric)`` ALSO works (proper subclass).
- Tests that pin field names (``stop_count`` / ``controller_count``) need updating in PR 3.
- graphs-side loaders that read ``stop_count`` or ``controller_count`` need updating in their own follow-up PRs.

## Backward-compat guarantees

1. Every existing YAML in ``data/compute_products/`` validates unchanged after the rename (PR 3 atomic update).
2. Every existing ``*OnDieFabric`` name remains importable.
3. ``isinstance(x, NPUOnDieFabric)`` continues to work.
4. ``isinstance(x, OnDieFabric)`` ALSO works (new affordance).
5. graphs-side loaders for NPU/DPU/TPU/CGRA work unchanged.
6. graphs-side loaders for CPU and GPU need to update one field name; tracked as a follow-up after PR 3.

## Risks called out by this exercise

1. **Inheritance changes Pydantic discriminator behavior.** The unified ``OnDieFabric`` base is NOT part of the ``AnyBlock`` discriminated union; it's used as a field type on ``*Block`` classes. No discriminator impact. Verified mentally; PR 2 should add a smoke test.

2. **Field discovery on subclass.** Pydantic 2 supports model inheritance well, but ``model_fields`` on a subclass shows base + own fields together. The PR 2 test should assert that ``NPUOnDieFabric().model_fields`` includes all 10 fields (7 from base + topology + mesh + confidence).

3. **CPU and GPU gain a confidence field (with default).** Previously CPU/GPU ``*OnDieFabric`` had no ``confidence`` field at all. v10 base adds one with ``DataConfidence.THEORETICAL`` default; existing YAMLs continue to validate (no field required). PR 3 verifies via load-then-dump round-trip.

4. **The rename is a breaking change for callers** that pin ``stop_count`` or ``controller_count`` by name. Survey of caller code (3 YAMLs identified; loader updates tracked separately) suggests scope is small but non-zero. Documented as a v10 release note.

5. **mesh_rows / mesh_cols semantics depend on topology.** The base class makes them optional unconditionally; per-kind subclasses don't enforce ``mesh_rows + mesh_cols required when topology=MESH_2D``. That validation should be added as a per-kind validator in PR 3 OR left as a v11 polish item. Current behavior matches today's (no validator), so PR 3 ships as-is.

## Next step

Sprint sequencing:

- **PR 1 (this document)** -- docs/design, graphs-side. Lands the field-by-field audit + inheritance design + recommended schema diff. Pure docs PR.
- **PR 2 -- schema, embodied-schemas-side.** Adds ``OnDieFabric`` base class to ``compute_block_common.py`` (additive). Existing per-kind classes unchanged. Tests verify base constructs + field-set invariant.
- **PR 3 -- migrations, embodied-schemas-side.** All 6 per-kind classes inherit from the base. CPU and GPU also get the ``stop_count`` / ``controller_count`` -> ``unit_count`` rename + 3 YAML updates atomically.

After this sprint closes:
- ``compute_block_common`` hosts 4 re-exported primitives + 2 unified types (alias-based: ``TheoreticalPerformance``, ``ThermalProfile``) + 1 base class (inheritance-based: ``OnDieFabric``).
- Endpoint-count naming is consistent across all 6 block kinds.
- 6 ``*OnDieFabric`` class bodies collapse from ~30 LOC each to ~5 LOC each.
- **v11 candidate**: ``has_external_dram`` vs ``has_host_dram`` naming reconciliation (touches more SKU YAMLs; preserves the chip-attached-vs-host-bus distinction as a separate field).
- **v12+ candidate**: KPU schema unification (oldest module; 12 SKUs; ``KPUNoCSpec`` doubly-purposed).
