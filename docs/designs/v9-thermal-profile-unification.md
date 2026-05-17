# v9 ThermalProfile unification -- paper exercise

Status: draft
Date: 2026-05-17
Tracking issue: `graphs#215`
Predecessors:
  - v8 unification: `compute_block_common` + ``TheoreticalPerformance`` (`graphs#208`, `graphs#210`)
  - DSP sprint: 8th + final block kind closes the category schema gap (`graphs#211`)

## Purpose

Issue `#215` proposes a v9 follow-up to the v8 unification sprint:
collapse the **byte-identical** ``*ThermalProfile`` classes across
NPU / CGRA / DPU / TPU / DSP into a single shared ``ThermalProfile``
definition in ``embodied_schemas.compute_block_common``.

This is the natural next refactoring after v8. The v8 sprint
landed `compute_block_common` + unified ``TheoreticalPerformance``
and flagged ``ThermalProfile`` / ``OnDieFabric`` as v9-candidate
NEAR_UNIFIABLE patterns. The DSP sprint then exercised the v8 module
end-to-end (first block kind using `compute_block_common` from day 1).
With every block kind now landed, the NEAR_UNIFIABLE deferrals can
be addressed cleanly.

This doc is **PR 1 of the v9 sprint** -- the paper exercise:
audit all 7 ``*ThermalProfile`` classes side-by-side, justify the
5-vs-2 unification split (5 unify, 2 stay separate), document the
migration plan + risk analysis.

The doc deliberately mirrors the structure of the v8 paper exercise
(``v8-compute-block-common-unification.md``) so the diff between
the two designs is easy to spot.

## Sprint scope (narrower than v8)

v8 unified 4 primitives + 1 unified type across 7 block kinds.
v9 is more focused:

| What | Scope | Risk |
|---|---|---|
| ``ThermalProfile`` (5 byte-identical classes) | **THIS SPRINT** | LOW: alias migration; byte-identical bodies |
| ``OnDieFabric`` (3 byte-identical + 3 variants) | **defer to v10** | MEDIUM: needs naming reconciliation (``stop_count`` vs ``controller_count`` vs ``unit_count``) |
| ``has_external_dram`` vs ``has_host_dram`` | defer to v10 | MEDIUM: touches SKU YAMLs |
| ``KPUThermalProfile`` | defer to v11+ | HIGH: also used for chip-level ``Power.thermal_profiles`` across whole catalog |
| CPU / GPU thermal profile shape | **stays separate** | N/A: legitimate architectural divergence |

The v9 sprint deliberately picks the **highest-confidence**, **lowest-risk** unification target: 5 byte-identical classes → 1 aliased type. Same mechanical pattern v8 used for ``TheoreticalPerformance``.

## Method

For each of the 7 ``*ThermalProfile`` classes in
``src/embodied_schemas/<kind>_block.py``, this doc records the
exact field set + validator behavior, then classifies each class
as **IDENTICAL** (collapses to alias), **NEAR_IDENTICAL** (close
but different shape; v10 candidate), or **KEEP_SEPARATE** (different
shape for legitimate architectural reasons).

Out of scope:

- **``OnDieFabric`` unification** -- the 3 byte-identical NPU/CGRA/DPU classes have ``mesh_cols`` + ``mesh_rows`` that TPU lacks (no mesh). CPU + GPU use ``stop_count`` / ``controller_count`` for the same conceptual ""endpoint count"" field. Naming reconciliation is its own sprint; defer to v10.
- **``KPUThermalProfile``** -- oldest module; pre-dates the pattern. Also used for chip-level ``Power.thermal_profiles`` across the whole catalog (not just KPU SKUs). Touching it is a larger change; defer to v11+ KPU unification sprint.
- **``CPUThermalProfile`` / ``GPUThermalProfile``** -- different shapes for legitimate architectural reasons (see classifications below). Keep separate.

## Reference: 7 ``*ThermalProfile`` classes today

```
embodied-schemas/src/embodied_schemas/
  cpu_block.py     -> CPUThermalProfile      (line 414)
  gpu_block.py     -> GPUThermalProfile      (line 292)
  npu_block.py     -> NPUThermalProfile      (line 354)
  cgra_block.py    -> CGRAThermalProfile     (line 346)
  dpu_block.py     -> DPUThermalProfile      (line 363)
  tpu_block.py     -> TPUThermalProfile      (line 398)
  dsp_block.py     -> DSPThermalProfile      (line 371)
```

Plus ``KPUThermalProfile`` from ``kpu.py`` (out of scope; v11+).

## Per-class audit

### NPUThermalProfile -- **IDENTICAL** (canonical shape)

Fields (9):
```python
name: str
tdp_watts: float                              # > 0
cooling_solution_id: str
clock_mhz: float                              # > 0
dvfs_enabled: bool = False
efficiency_factor_by_precision: dict[str, float] = {}
instruction_efficiency_by_precision: dict[str, float] = {}
memory_bottleneck_factor_by_precision: dict[str, float] = {}
vdd_v: float | None = None                    # > 0 when set
```

Validators:
- ``_validate_efficiency_ranges``: rejects values outside [0, 1] for the 3 ``*_by_precision`` dicts.

``model_config = {"extra": "forbid"}``.

### CGRAThermalProfile -- **IDENTICAL** to NPU

Same 9 fields, same validator, same ``model_config``. Byte-for-byte identical body to ``NPUThermalProfile``.

### DPUThermalProfile -- **IDENTICAL** to NPU

Same 9 fields, same validator, same ``model_config``.

### TPUThermalProfile -- **IDENTICAL** to NPU

Same 9 fields, same validator, same ``model_config``.

### DSPThermalProfile -- **IDENTICAL** to NPU

Same 9 fields, same validator, same ``model_config``.

(DSP was authored after v8 unification with the explicit expectation that v9 would collapse this class. See the DSP paper exercise risk #6.)

### CPUThermalProfile -- **KEEP_SEPARATE**

Fields (6):
```python
name: str
tdp_watts: float
cooling_solution_id: str
per_cluster_clock_domain: dict[str, ClockDomain] | None   # << different!
efficiency_factor_by_precision: dict[str, float] = {}
vdd_v: float | None = None
```

**Differences from NPU shape**:
1. Uses ``per_cluster_clock_domain`` (per-cluster ClockDomain) instead of scalar ``clock_mhz`` + ``dvfs_enabled``. CPUs have per-core or per-cluster clock domains (P-cores vs E-cores; different DVFS curves per cluster) and a single scalar clock can't capture this.
2. No ``instruction_efficiency_by_precision`` or ``memory_bottleneck_factor_by_precision`` (CPUs use IPC + roofline modeling differently).

Verdict: **legitimate architectural divergence; keep separate**.

### GPUThermalProfile -- **KEEP_SEPARATE**

Fields (10):
```python
name: str
tdp_watts: float
cooling_solution_id: str
clock_domain: ClockDomain                                 # << different!
memory_clock_mhz: float | None                            # << GPU-specific
native_acceleration_by_precision: dict[str, bool] = {}    # << GPU-specific
efficiency_factor_by_precision: dict[str, float] = {}
instruction_efficiency_by_precision: dict[str, float] = {}
memory_bottleneck_factor_by_precision: dict[str, float] = {}
vdd_v: float | None = None
```

**Differences from NPU shape**:
1. ``clock_domain: ClockDomain`` instead of ``clock_mhz: float`` + ``dvfs_enabled: bool``. GPUs ship richer DVFS (base / boost / sustained) that scalar ``clock_mhz`` can't capture.
2. ``memory_clock_mhz``: GPU HBM/GDDR has its own clock that throttles independently of core clock.
3. ``native_acceleration_by_precision: dict[str, bool]``: GPUs report which precisions have native acceleration (e.g., BF16 on Hopper but not on Volta) -- not just efficiency.

Verdict: **legitimate architectural divergence; keep separate**.

## Summary table

| Class | Shape | Verdict | LOC saved |
|---|---|---|---|
| NPUThermalProfile  | 9 fields | **IDENTICAL** | ~50 |
| CGRAThermalProfile | 9 fields | **IDENTICAL** | ~50 |
| DPUThermalProfile  | 9 fields | **IDENTICAL** | ~50 |
| TPUThermalProfile  | 9 fields | **IDENTICAL** | ~50 |
| DSPThermalProfile  | 9 fields | **IDENTICAL** | ~50 |
| CPUThermalProfile  | 6 fields, ClockDomain-per-cluster | **KEEP_SEPARATE** | 0 |
| GPUThermalProfile  | 10 fields, ClockDomain + memory_clock + native_accel | **KEEP_SEPARATE** | 0 |

**Net LOC saved**: ~250 (5 × ~50 LOC class bodies + validators); precise number depends on whether tests need updating.

## Recommended ``compute_block_common.py`` diff (PR 2 scope)

Add the unified type to ``compute_block_common.py``:

```python
class ThermalProfile(BaseModel):
    """Per-precision thermal operating point. Shared shape across
    5 modern inference-accelerator block kinds (NPU/CGRA/DPU/TPU/
    DSP); collapsed into one definition here.

    CPU and GPU use different shapes (per-cluster ClockDomain on
    CPU; ClockDomain + memory_clock + native_acceleration on GPU)
    and stay separate. KPUThermalProfile (oldest module) is also
    excluded; v11+ KPU unification may revisit.

    Per-block-kind aliases (NPUThermalProfile = ThermalProfile etc.)
    preserve backward-compat for existing callers. The aliases land
    in PR 3 of the v9 sprint.
    """

    name: str = Field(...)
    tdp_watts: float = Field(..., gt=0)
    cooling_solution_id: str = Field(...)
    clock_mhz: float = Field(..., gt=0, description="Operating frequency")
    dvfs_enabled: bool = Field(
        False,
        description=(
            "False is the IP-core / single-profile default; True for "
            "SKUs with multiple thermal profiles (multi-mode DVFS)."
        ),
    )
    efficiency_factor_by_precision: dict[str, float] = Field(default_factory=dict)
    instruction_efficiency_by_precision: dict[str, float] = Field(default_factory=dict)
    memory_bottleneck_factor_by_precision: dict[str, float] = Field(default_factory=dict)
    vdd_v: float | None = Field(default=None, gt=0)

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _validate_efficiency_ranges(self) -> "ThermalProfile":
        for attr_name, label in (
            ("efficiency_factor_by_precision", "efficiency_factor"),
            ("instruction_efficiency_by_precision", "instruction_efficiency"),
            ("memory_bottleneck_factor_by_precision", "memory_bottleneck_factor"),
        ):
            mapping = getattr(self, attr_name)
            for precision, value in mapping.items():
                if not 0.0 <= value <= 1.0:
                    raise ValueError(
                        f"{attr_name}[{precision!r}] = {value} is outside "
                        f"[0, 1]; {label} is a unit fraction."
                    )
        return self
```

Update ``__all__`` to include ``ThermalProfile``. Update the module docstring to note that ``ThermalProfile`` joins ``TheoreticalPerformance`` as a v8/v9 unified type.

## Migration strategy (PR 3 proof of concept)

PR 3 of v9 sprint migrates **all 5 byte-identical classes in one PR**, matching the v8 follow-up batch (`graphs#210` / embodied-schemas#42) precedent:

For each of NPU / CGRA / DPU / TPU / DSP ``*_block.py``:

1. Remove the class body + validator
2. Add at the original class location:
   ```python
   from embodied_schemas.compute_block_common import ThermalProfile
   NPUThermalProfile = ThermalProfile   # alias preserves the class name
   ```
3. ``__init__.py`` re-exports unchanged (the alias IS the same class).

**Pydantic class-identity behavior**:
- ``isinstance(profile, NPUThermalProfile)`` continues to work because the alias IS the same class object.
- ``isinstance(profile, ThermalProfile)`` ALSO works (since they're the same class).
- This is the exact pattern v8 follow-up (#210) used for ``TheoreticalPerformance``; verified end-to-end on the v8 sprint.

**Test impact**:
- Tests that import ``NPUThermalProfile`` continue to work.
- Tests that pin class identity ASSERTIONS like ``NPUThermalProfile is not GPUThermalProfile`` continue to work (they're still distinct from CPU + GPU shapes).
- Tests that pin ``NPUThermalProfile is DPUThermalProfile`` would now PASS where before they'd FAIL (the alias makes them the same class). Unlikely to exist; check during PR 3.
- Existing YAMLs validate unchanged (alias accepts the same data shape).

## Backward-compat guarantees

1. Every existing YAML in ``data/compute_products/`` validates unchanged.
2. Every existing ``*ThermalProfile`` name remains importable.
3. Every ``isinstance(x, NPUThermalProfile)`` continues to work.
4. Every serialized JSON round-trips unchanged.
5. graphs-side loaders work unchanged (they construct via field names, not type identity).

## Risks called out by this exercise

1. **None of the 5 unification targets diverge.** Unlike v8 where GPU's ``sparse_peak_ops_per_sec_by_precision`` field had to be added to the unified type, the 5 ``*ThermalProfile`` classes are truly byte-identical. The unified type can be byte-equivalent to ``NPUThermalProfile`` today; no schema relaxation needed.

2. **CPU and GPU stay separate by design.** This is the expected outcome of a NEAR_UNIFIABLE pattern that turned out to be IDENTICAL-for-5 / DIFFERENT-for-2. The paper exercise documents *why* they differ; future readers can decide whether to revisit if CPU / GPU shapes converge naturally.

3. **KPUThermalProfile is still doubly-purposed.** It serves both KPU-specific block thermal data AND chip-level ``Power.thermal_profiles`` for the whole catalog. v9 doesn't touch it because the second purpose has a wider blast radius. v11+ KPU unification sprint will need to address this.

4. **``OnDieFabric`` is genuinely harder.** The byte-identical 3 (NPU/CGRA/DPU) have ``mesh_rows`` + ``mesh_cols`` that TPU lacks. CPU has ``stop_count`` and GPU has ``controller_count`` for what NPU/CGRA/DPU/TPU call ``unit_count``. Three name reconciliations needed in one sprint. Defer to v10 with its own paper exercise.

5. **DSP was authored expecting this.** The DSP paper exercise (`graphs#212`) explicitly noted ``ThermalProfile`` unification as a v9 deferral. PR 3 of this sprint closes that expectation; failure to land v9 cleanly would leave DSP carrying a class that should have been a one-line alias.

## Next step

Sprint sequencing:

- **PR 1 (this document)** -- docs/design, graphs-side. Lands the field-by-field audit + classifications + recommended schema diff. Pure docs PR.
- **PR 2 -- schema, embodied-schemas-side.** Adds ``ThermalProfile`` to ``compute_block_common.py`` (additive). Existing ``*ThermalProfile`` classes unchanged. Test that ``ThermalProfile`` constructs cleanly with the canonical NPU shape data.
- **PR 3 -- migrations, embodied-schemas-side.** Aliases NPU/CGRA/DPU/TPU/DSP ``*ThermalProfile`` to the unified type in one batched commit (matches v8 follow-up #210 precedent). Removes ~250 LOC of duplicated definitions.

The scope is **smaller than v8** (only 1 unified type vs 4 primitives + 1 type) because the v8 sprint did the heavier lifting; v9 just sweeps up the next layer.

After this sprint closes:
- The ``compute_block_common`` module hosts 2 unified types (``TheoreticalPerformance`` + ``ThermalProfile``) and re-exports 4 shared primitives.
- 5 byte-identical class duplications removed; the catalog is cleaner.
- v10 candidate sprint: ``OnDieFabric`` unification (more complex due to endpoint-count naming reconciliation) and/or ``has_external_dram`` vs ``has_host_dram`` naming reconciliation.
- v11+ candidate sprint: KPU schema unification (oldest module; 12 SKUs to migrate).
