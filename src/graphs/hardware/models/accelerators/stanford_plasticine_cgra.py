"""Stanford Plasticine v2 resource model.

Final cleanup PR for issue #196 (CGRA mini-sprint). As of this PR,
the chip's architectural and thermal-profile data lives in the
canonical YAML at
``embodied-schemas:data/compute_products/stanford/plasticine_v2.yaml``
(landed in embodied-schemas#35) and is loaded via ``cgra_yaml_loader``.
The previous ~170-LOC hand-coded ``HardwareResourceModel``
constructor was retired here; its parity with the YAML-loaded model
is pinned in ``tests/hardware/test_cgra_yaml_loader_plasticine_parity.py``.

Schema precursors that had to land first:
  - embodied-schemas#34 -- CGRABlock added to the discriminated union
  - embodied-schemas#35 -- the Plasticine v2 YAML itself

The public function name and signature are preserved so existing
callers (``create_stanford_plasticine_cgra_mapper``, validation
scripts, ``cli/compare_*.py``) continue to work unchanged.

**Zero overlays remain.** This is the simplest cleanup of any of the
v2-v5 sprints because:

  - ``HardwareType.CGRA`` was already in the graphs enum (no
    transitional period like NPU's #191).
  - ``peak_bandwidth`` convention (on-chip PCU mesh; the loader
    correctly picks this for CGRA) matches the legacy hand-coded
    value (40 GB/s).
  - Plasticine v2 is a research SKU with no BoM data, so no
    BOMCostProfile overlay is needed (unlike Hailo/Coral).
  - ``memory_technology`` was not set in the legacy at all; the YAML
    loader populates it correctly ("DDR4").

Four documented drifts captured by the parity test where the YAML
**CORRECTS** legacy bugs:

  - INT8 peak: 10.24 TOPS (correct chip-level) vs 0.307 TOPS (legacy
    used a buggy ``num_pcus * macs_per_pcu * ops_per_mac`` formula
    that ignored the per-PCU dual-issue / pipeline multiplier).
  - FP16 peak: 2.56 TOPS vs 0.077 TOPS (same root cause).
  - ``memory_technology``: "DDR4" (correct) vs None (legacy didn't set).
  - ``soc_fabric``: 4x8 mesh populated (correct) vs None (legacy
    didn't model -- now unblocks downstream Layer 6 reporting for CGRA).

One v6 reconciliation item:
  - ``reconfig_overhead_cycles`` (the defining CGRA Achilles heel,
    Plasticine = 1000) lives on ``CGRABlock`` in the v5 schema but
    has no equivalent field on ``HardwareResourceModel``. Downstream
    consumers that need it read from the underlying ComputeProduct
    directly. v6 will add the field properly.

The legacy resource-model ``name`` ("CGRA-Plasticine-v2") is preserved.
"""

from ...resource_model import HardwareResourceModel
from .cgra_yaml_loader import load_cgra_resource_model_from_yaml


_LEGACY_NAME = "CGRA-Plasticine-v2"
_YAML_BASE_ID = "stanford_plasticine_v2"


def stanford_plasticine_cgra_resource_model() -> HardwareResourceModel:
    """Stanford Plasticine v2 -- coarse-grained reconfigurable
    architecture research prototype.

    See the YAML for the canonical chip description (32 PCUs * 8 MACs
    at 1 GHz, 64 KiB PMU per PCU, 2 MiB shared L2, 4 GiB host DDR4,
    4x8 PCU mesh NoC, 1000-cycle reconfig overhead, GF 28nm SLP
    process, single 15W operating point with passive air cooling).
    """
    return load_cgra_resource_model_from_yaml(
        _YAML_BASE_ID,
        name_override=_LEGACY_NAME,
    )
