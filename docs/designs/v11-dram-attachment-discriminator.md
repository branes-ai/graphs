# v11 DRAM-attachment naming reconciliation -- paper exercise

Status: draft
Date: 2026-05-17
Tracking issue: `graphs#219`
Predecessors:
  - v8 unification: `compute_block_common` + ``TheoreticalPerformance`` (`graphs#208`, `graphs#210`)
  - v9 unification: ``ThermalProfile`` collapses 5 byte-identical classes (`graphs#215`)
  - v10 unification: ``OnDieFabric`` inheritance base + endpoint-count rename (`graphs#217`)

## Purpose

Issue `#219` proposes a v11 follow-up: reconcile the ``has_host_dram``
(CGRA) vs ``has_external_dram`` (NPU/DPU/TPU/DSP) naming inconsistency
across the catalog. Rename CGRA's fields to match the 5-of-6 convention,
AND add an explicit ``dram_attachment`` discriminator that preserves
the architectural distinction the original naming captured implicitly.

This is the v11 NEAR_UNIFIABLE pattern flagged repeatedly across v8 /
v9 / v10:

  - v8: "``has_external_dram`` vs ``has_host_dram`` naming reconciliation (touches SKU YAMLs; defer to v9)"
  - v9: same, deferred to v10
  - v10: same, deferred to v11

v11 picks it up.

This doc is **PR 1 of the v11 sprint** -- the paper exercise.
Mirrors the structure of v9 / v10 so the diff between the four
sprints is easy to spot.

## Why a discriminator (not pure rename)

Pure rename would consolidate naming but **lose the architectural
distinction** the original CGRA author encoded. The two field names
captured genuinely different DRAM attachment models:

| Attachment | Owner | Bandwidth gated by | Energy/byte includes | Block kinds today |
|---|---|---|---|---|
| **chip-attached** | chip's own DRAM controller | chip's HBM/DDR PHY | chip's DRAM access | NPU/DPU/TPU/DSP |
| **host-bus** | host CPU's memory subsystem | PCIe lane count + host RAM | PCIe transit + host DRAM access | CGRA (Plasticine) |

These different bandwidth + energy models matter for roofline analysis. The v11 fix consolidates the field naming AND adds an explicit ``dram_attachment`` discriminator (``CHIP_ATTACHED`` / ``HOST_BUS``) so consumers can branch on attachment type without parsing block-kind names.

## Method

Audit the 5 ``*MemorySubsystem`` classes that carry the external/host
DRAM fields:

  - ``NPUMemorySubsystem`` (npu_block.py)
  - ``DPUMemorySubsystem`` (dpu_block.py)
  - ``TPUMemorySubsystem`` (tpu_block.py)
  - ``DSPMemorySubsystem`` (dsp_block.py)
  - ``CGRAMemorySubsystem`` (cgra_block.py)

Plus the 7 YAMLs that actually populate these fields:
  - 4 with chip-attached DRAM: hailo_10h (NPU), vitis_ai_b4096 (DPU), tpu_v4 (TPU), cadence_vision_q8 (DSP)
  - 1 with host-bus DRAM: plasticine_v2 (CGRA)
  - 2 explicit "no external DRAM": coral_edge_tpu (NPU), hailo_8 (NPU)

Out of scope:

- **``KPUMemorySubsystem``** -- different shape (KPU has its own tile-mesh memory model that doesn't fit the *MemorySubsystem pattern). v12+ KPU unification.
- **Full ``*MemorySubsystem`` unification** -- v8 marked KEEP_SEPARATE; TPU's Unified Buffer vs DSP's L1/L2 are architecturally meaningful differences.
- **Updating chip-attached YAMLs to populate ``dram_attachment`` explicitly** -- a follow-up "explicit discriminator" PR after v11 closes (low priority; defaulting to None is fine).

## Reference: current state

### CGRAMemorySubsystem (the outlier)

```python
# CGRA-specific naming
has_host_dram: bool = Field(False)
host_dram_type: MemoryType | None = Field(default=None)
host_dram_size_gb: float | None = Field(default=None, ge=0)
host_dram_bandwidth_gbps: float | None = Field(default=None, ge=0)
host_dram_access_energy_pj_per_byte: float = Field(0.0, ge=0)

@model_validator(mode="after")
def _validate_host_dram_consistency(self) -> "CGRAMemorySubsystem":
    # ... checks all host_dram_* fields when has_host_dram=True ...
```

### NPU/DPU/TPU/DSP (the 5-of-6 convention)

```python
# Shared naming (NPUMemorySubsystem shown)
has_external_dram: bool = Field(False)
external_dram_type: MemoryType | None = Field(default=None)
external_dram_size_gb: float | None = Field(default=None, ge=0)
external_dram_bandwidth_gbps: float | None = Field(default=None, ge=0)
external_dram_access_energy_pj_per_byte: float = Field(0.0, ge=0)

@model_validator(mode="after")
def _validate_external_dram_consistency(self) -> "NPUMemorySubsystem":
    # ... checks all external_dram_* fields when has_external_dram=True ...
```

(DSP also has ``external_dram_bandwidth_kind: Literal["typical", "measured"]`` added in v9 -- IP-core vs SoC distinction.)

## Audit summary

| Class | Field naming | Has discriminator? | Action |
|---|---|---|---|
| NPUMemorySubsystem  | has_external_dram | no | + dram_attachment field |
| DPUMemorySubsystem  | has_external_dram | no | + dram_attachment field |
| TPUMemorySubsystem  | has_external_dram | no | + dram_attachment field |
| DSPMemorySubsystem  | has_external_dram | no | + dram_attachment field |
| CGRAMemorySubsystem | **has_host_dram**     | no | **rename + dram_attachment field** |

## Recommended schema changes (PR 2 scope: additive)

Add the discriminator to ``compute_block_common``:

```python
class DramAttachment(str, Enum):
    """How external DRAM is attached to the compute chip.

    CHIP_ATTACHED: the chip has its own DRAM controller; bandwidth
      is gated by the chip's HBM/DDR/LPDDR PHY. Energy/byte is the
      chip's own DRAM access cost. Typical: TPU + HBM2e; DSP + LPDDR;
      DPU + DDR4; NPU + LPDDR.
    HOST_BUS: the chip reaches DRAM via the host CPU's memory
      subsystem (PCIe + host DRAM controllers). Bandwidth is gated
      by PCIe lane count * link speed. Energy/byte includes PCIe
      transit + host DRAM access. Typical: CGRA accelerator cards
      (Plasticine).
    """

    CHIP_ATTACHED = "chip_attached"
    HOST_BUS = "host_bus"
```

Add ``dram_attachment: DramAttachment | None`` field to each of the 5 ``*MemorySubsystem`` classes. Default ``None`` preserves backward compat -- existing YAMLs validate unchanged. The field is informational today; v12+ may make it required when external DRAM is present.

## Recommended migration (PR 3 scope)

### CGRA schema rename

Rename ``CGRAMemorySubsystem`` fields:

```python
# Before                              After
has_host_dram                      -> has_external_dram
host_dram_type                     -> external_dram_type
host_dram_size_gb                  -> external_dram_size_gb
host_dram_bandwidth_gbps           -> external_dram_bandwidth_gbps
host_dram_access_energy_pj_per_byte -> external_dram_access_energy_pj_per_byte
_validate_host_dram_consistency    -> _validate_external_dram_consistency
```

The renamed fields match the NPU/DPU/TPU/DSP convention exactly.

### Plasticine v2 YAML migration

```yaml
# Before
has_host_dram: true
host_dram_type: ddr4
host_dram_size_gb: 4.0
host_dram_bandwidth_gbps: 25.6
host_dram_access_energy_pj_per_byte: 20.0

# After
has_external_dram: true
external_dram_type: ddr4
external_dram_size_gb: 4.0
external_dram_bandwidth_gbps: 25.6
external_dram_access_energy_pj_per_byte: 20.0
dram_attachment: host_bus  # << explicit; preserves the "via PCIe" semantics
```

The added ``dram_attachment: host_bus`` is the load-bearing addition: without it, the rename would silently merge CGRA's PCIe-DRAM into the chip-attached convention used by NPU/DPU/TPU/DSP and consumers reading bandwidth + energy would mis-attribute the cost model.

### Test updates

  - ``test_compute_product_v5_cgra_block.py``: rename ``has_host_dram`` -> ``has_external_dram`` in fixtures
  - ``test_compute_product_v5_plasticine_v2_yaml.py``: same in assertions + assert ``dram_attachment == "host_bus"``
  - graphs-side ``cgra_yaml_loader``: rename field reads + propagate ``dram_attachment`` to ``HardwareResourceModel`` if applicable

### Optional: chip-attached YAMLs

The 4 chip-attached YAMLs (hailo_10h, vitis_ai_b4096, tpu_v4, cadence_vision_q8) can OPTIONALLY populate ``dram_attachment: chip_attached`` for explicitness. Recommended to defer to a separate "fill in the discriminator" PR after v11 closes -- keeps the v11 commit focused on CGRA reconciliation.

## Sprint shape (3 PRs)

- **PR 1 (this document)** -- paper exercise (graphs docs)
- **PR 2 -- additive schema (embodied-schemas).** Add ``DramAttachment`` enum + ``dram_attachment`` field to 5 ``*MemorySubsystem`` classes. Default ``None``; existing YAMLs validate unchanged.
- **PR 3 -- migration (embodied-schemas).** Rename CGRA fields + update Plasticine YAML + set ``dram_attachment: host_bus`` + update tests + graphs-side loader follow-up (separate PR).

## Backward-compat guarantees

1. **NPU/DPU/TPU/DSP YAMLs** validate unchanged after PR 2 (additive ``dram_attachment`` field defaults to None).
2. **CGRA Plasticine YAML** changes atomically with the schema rename in PR 3 -- no intermediate state where the catalog is broken.
3. Existing import-by-name continues to work for ``CGRAMemorySubsystem`` (only field names change, not class name).
4. graphs-side ``cgra_yaml_loader`` needs a small update to read the new field names; tracked as a separate PR after the embodied-schemas pin bumps.

## Risks

1. **The rename is a hard schema break.** Unlike v8 / v9 / v10 (additive type unification with backward-compat aliases), v11 PR 3 is a hard rename that requires atomic YAML migration. Risk is small (only Plasticine v2 affected) but documented.

2. **graphs-side loader needs an update.** ``cgra_yaml_loader.load_cgra_resource_model_from_yaml`` reads the host_dram fields by name. After PR 3 merges in embodied-schemas, graphs CI will fail until the loader is updated. Plan: ship the loader update as a separate graphs PR + CI pin bump in the same window.

3. **Default ``dram_attachment=None`` doesn't enforce explicit population.** A SKU could declare ``has_external_dram=True`` without setting ``dram_attachment``. This is intentional for backward compat in v11; v12+ may add a validator requiring ``dram_attachment`` when ``has_external_dram=True``.

4. **The 4 chip-attached YAMLs are NOT updated in v11.** They keep ``dram_attachment=None``. Consumers that need the discriminator must either infer ``CHIP_ATTACHED`` as the default OR wait for the follow-up PR. Documented as a known limitation.

5. **``DSPMemorySubsystem`` has its own ``external_dram_bandwidth_kind`` discriminator** (typical | measured) added in v9. That's orthogonal to ``dram_attachment`` and stays on DSP only.

## Next step

Sprint sequencing:

- **PR 1 (this document)** -- docs/design, graphs-side. Lands the audit + recommended schema diff + migration plan. Pure docs PR.
- **PR 2 -- additive schema, embodied-schemas-side.** ``DramAttachment`` enum + ``dram_attachment`` field on 5 ``*MemorySubsystem`` classes.
- **PR 3 -- CGRA rename + Plasticine YAML migration, embodied-schemas-side.** Atomic. Closes the v11 sprint.
- **Follow-up (graphs)** -- update ``cgra_yaml_loader`` to read the new field names + bump embodied-schemas pin. Separate PR; not part of the v11 sprint.

After this sprint closes:
- Field naming is consistent across all 5 block kinds with external DRAM
- ``dram_attachment`` discriminator is available (optional today) for consumers that need the chip-attached-vs-host-bus distinction
- Plasticine's host-bus semantic is preserved explicitly via the new discriminator
- **v12+ candidates**: KPU schema unification (oldest module; 12 SKUs; ``KPUMemorySubsystem`` doesn't fit the *MemorySubsystem pattern); making ``dram_attachment`` mandatory when has_external_dram=True (after all YAMLs have been backfilled)
