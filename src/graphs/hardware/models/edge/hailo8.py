"""Hailo-8 resource model.

PR 5 of the NPU sprint scoped at #187. As of this PR, the chip's
architectural and thermal-profile data lives in the canonical YAML
at ``embodied-schemas:data/compute_products/hailo/hailo_8.yaml``
(landed in embodied-schemas#26) and is loaded via ``npu_yaml_loader``.
The previous ~360-LOC hand-coded ``HardwareResourceModel``
constructor was retired here; its parity with the YAML-loaded model
was established in PR #189's
``test_npu_yaml_loader_hailo8_parity.py``.

The public function name and signature are preserved so existing
callers (``create_hailo8_mapper``, layer1 reporting, validation
scripts, ``cli/compare_*.py``) continue to work unchanged.

One overlay remains in the factory after loading -- BoM cost --
which is data the v4 ComputeProduct schema doesn't yet carry:

  - ``BOMCostProfile`` -- die / package / memory / PCB / thermal /
    margin / retail; v5 ``Market.bom`` will absorb it.

No memory_clock overlay needed (unlike Jetson AGX Orin / Thor):
NPUs have no external DRAM in the common case (Hailo-8 is SRAM-only)
and run at a single fixed frequency, so per-profile memory_clock_mhz
isn't an NPU concern.

Two known shape quirks documented in PR #189's loader docstring,
both v5 reconciliation items:
  - ``HardwareType.KPU`` (sic!) -- the graphs ``HardwareType`` enum
    has no NPU value; loader preserves the hand-coded choice for
    parity. Adding ``HardwareType.NPU`` is a separate graphs-side
    followup.
  - ``energy_per_flop_fp32`` synthesis -- NPUs don't ship FP32; the
    loader synthesizes it as ``energy_per_op_int8 * 8`` per the
    standard-cell rule of thumb. Within tolerance of the prior
    hand-coded ``get_base_alu_energy(16, 'standard_cell')`` value.

The legacy resource-model ``name`` ("Hailo-8") is preserved.
"""

from ...resource_model import BOMCostProfile, HardwareResourceModel
from .npu_yaml_loader import load_npu_resource_model_from_yaml


_LEGACY_NAME = "Hailo-8"
_YAML_BASE_ID = "hailo_hailo_8"


def hailo8_resource_model() -> HardwareResourceModel:
    """Hailo-8 Computer Vision AI Accelerator.

    See the YAML for the canonical chip description (32 dataflow
    units, structure-driven dataflow architecture, INT8/INT4 only,
    24 MiB on-chip SRAM, no external DRAM, MESH_2D NoC, single
    2.5W operating point) and the migration notes at
    ``docs/designs/npu-compute-product-schema-extension.md``.
    """
    model = load_npu_resource_model_from_yaml(
        _YAML_BASE_ID,
        name_override=_LEGACY_NAME,
    )

    # BoM cost overlay -- not yet carried by the v4 ComputeProduct schema.
    # Numbers from teardowns + Hailo public pricing (M.2 module retails
    # at ~$160 with $40 BOM).
    model.bom_cost_profile = BOMCostProfile(
        silicon_die_cost=25.0,
        package_cost=8.0,
        memory_cost=0.0,           # all on-chip SRAM
        pcb_assembly_cost=4.0,
        thermal_solution_cost=1.0,
        other_costs=2.0,
        total_bom_cost=0,          # auto-calculated: $40
        margin_multiplier=4.0,
        retail_price=160.0,
        volume_tier="10K+",
        process_node="16nm",
        year=2025,
        notes=(
            "Ultra-efficient edge AI accelerator. Low BOM due to "
            "all-on-chip design. Highest TOPS/$ and TOPS/W in "
            "entry-level segment. M.2 module retails at $160."
        ),
    )

    return model
