"""Jetson AGX Thor 128GB resource model.

As of this PR, the chip's architectural and thermal-profile data
lives in the canonical YAML at
``embodied-schemas:data/compute_products/nvidia/jetson_agx_thor_128gb.yaml``
(landed in embodied-schemas#21) and is loaded via ``gpu_yaml_loader``.
The previous ~330-LOC hand-coded ``HardwareResourceModel``
constructor was retired here; its parity with the YAML-loaded model
is the same shape established for the AGX Orin migration in
graphs#181.

The public function name and signature are preserved so existing
callers (``create_jetson_thor_128gb_mapper``, the roadmap chart,
validation scripts) continue to work unchanged.

Same two overlays as AGX Orin -- both are data the v2 ComputeProduct
schema doesn't yet carry:
  - ``BOMCostProfile`` -- silicon / package / memory / PCB / thermal
    / margin / retail; v3 ``Market.bom`` will absorb it.
  - ``ThermalOperatingPoint.memory_clock_mhz`` -- LPDDR5X-8533 runs
    at 4267 MHz internal across all three nvpmodel profiles on Thor.
    The chip-level ``KPUThermalProfile`` shape doesn't carry
    ``memory_clock_mhz``; v3 ``ThermalProfile`` will absorb it.

The legacy resource-model ``name`` ("Jetson-Thor-128GB", no spaces)
is preserved for backward compatibility with the roadmap chart
(``cli/analyze_jetson_roadmap.py``) and existing test fixtures.
"""

from ...resource_model import BOMCostProfile, HardwareResourceModel
from ..edge.gpu_yaml_loader import load_gpu_resource_model_from_yaml


_LEGACY_NAME = "Jetson-Thor-128GB"
_YAML_BASE_ID = "nvidia_jetson_agx_thor_128gb"


def jetson_thor_128gb_resource_model() -> HardwareResourceModel:
    """NVIDIA Jetson Thor 128GB with realistic DVFS modeling (next-gen
    edge AI, 2025+).

    See the YAML for the canonical chip description (Blackwell SMs,
    Tensor cores, LPDDR5X, three nvpmodel profiles) and the migration
    notes at ``docs/designs/gpu-compute-product-schema-extension.md``.
    """
    model = load_gpu_resource_model_from_yaml(
        _YAML_BASE_ID,
        name_override=_LEGACY_NAME,
    )

    # Memory-clock overlay -- LPDDR5X-8533 internal clock is 4267 MHz on
    # Thor (vs Orin's 3200 MHz at LPDDR5-6400), identical across all
    # three nvpmodel profiles. KPUThermalProfile shape used by chip-
    # level Power.thermal_profiles doesn't carry memory_clock_mhz; a
    # v3 ThermalProfile generalization will absorb this overlay.
    for profile in model.thermal_operating_points.values():
        profile.memory_clock_mhz = 4267.0

    # BoM cost overlay -- not yet carried by the v2 ComputeProduct schema.
    # Numbers from NVIDIA pricing / industry teardowns (2025).
    model.bom_cost_profile = BOMCostProfile(
        silicon_die_cost=850.0,
        package_cost=180.0,
        memory_cost=350.0,
        pcb_assembly_cost=90.0,
        thermal_solution_cost=80.0,
        other_costs=50.0,
        total_bom_cost=1600.0,
        margin_multiplier=1.56,
        retail_price=2500.0,
        volume_tier="10K+",
        process_node="4nm",
        year=2025,
        notes=(
            "Next-gen automotive AI platform. 128GB LPDDR5X, Blackwell "
            "architecture, 4nm process. Advanced flip-chip BGA "
            "packaging. Active cooling required. Target: autonomous "
            "vehicles, humanoid robots, industrial AGVs."
        ),
    )

    return model
