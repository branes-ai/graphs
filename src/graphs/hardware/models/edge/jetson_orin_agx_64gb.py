"""Jetson Orin AGX 64GB resource model.

PR 5 of the GPU sprint scoped at #171. As of this PR, the chip's
architectural and thermal-profile data lives in the canonical YAML
at ``embodied-schemas:data/compute_products/nvidia/jetson_agx_orin_64gb.yaml``
and is loaded via ``gpu_yaml_loader``. The previous ~630-LOC
hand-coded ``HardwareResourceModel`` constructor was retired here;
its parity with the YAML-loaded model was established field-by-field
in PR #180's ``test_gpu_yaml_loader_jetson_agx_orin_parity.py``.

The public function name and signature are preserved so existing
callers (``create_jetson_orin_agx_64gb_mapper``, layer1 reporting,
validation scripts, ``cli/compare_*.py``) continue to work unchanged.

Three pieces of data don't yet live in the YAML and are therefore
attached here after loading:
  - ``BOMCostProfile`` -- die / package / memory / PCB / thermal /
    margin / retail BOM. The v2 ComputeProduct schema doesn't carry
    BoM cost; v3 schema PR will add ``Market.bom`` and this
    overlay can move into the YAML.
  - ``ThermalOperatingPoint.memory_clock_mhz`` -- LPDDR5-6400 runs at
    3200 MHz internal across all four nvpmodel profiles on Orin AGX
    (NVIDIA datasheet). The chip-level ``thermal_profiles`` use the
    ``KPUThermalProfile`` shape which doesn't carry a
    ``memory_clock_mhz`` field; the v3 schema generalization
    (``ThermalProfile`` with optional memory clock) covered in the
    design doc will eliminate this overlay.
  - The legacy resource-model ``name`` ("Jetson-Orin-AGX-64GB", no
    spaces) is preserved for backward compatibility with existing
    test fixtures and chart legends. The YAML's ``name`` field is
    "NVIDIA Jetson AGX Orin 64GB" -- that's the human-readable form
    that future schema-aware consumers should prefer.
"""

from ...resource_model import BOMCostProfile, HardwareResourceModel
from .gpu_yaml_loader import load_gpu_resource_model_from_yaml


_LEGACY_NAME = "Jetson-Orin-AGX-64GB"
_YAML_BASE_ID = "nvidia_jetson_agx_orin_64gb"


def jetson_orin_agx_64gb_resource_model() -> HardwareResourceModel:
    """NVIDIA Jetson Orin AGX 64GB with realistic DVFS-aware multi-power-profile modeling.

    See the YAML for the canonical chip description (Ampere SMs,
    Tensor cores, LPDDR5, four nvpmodel profiles) and the migration
    notes at ``docs/designs/gpu-compute-product-schema-extension.md``.
    """
    model = load_gpu_resource_model_from_yaml(
        _YAML_BASE_ID,
        name_override=_LEGACY_NAME,
    )

    # Memory-clock overlay -- LPDDR5-6400 internal clock is 3200 MHz on
    # Orin AGX, identical across all four nvpmodel profiles. The
    # KPUThermalProfile shape used by chip-level Power.thermal_profiles
    # doesn't carry memory_clock_mhz; a v3 ThermalProfile generalization
    # will absorb this overlay.
    for profile in model.thermal_operating_points.values():
        profile.memory_clock_mhz = 3200.0

    # BoM cost overlay -- not yet carried by the v2 ComputeProduct schema.
    # Numbers from NVIDIA pricing / industry teardowns (2025).
    model.bom_cost_profile = BOMCostProfile(
        silicon_die_cost=380.0,
        package_cost=60.0,
        memory_cost=160.0,
        pcb_assembly_cost=35.0,
        thermal_solution_cost=15.0,
        other_costs=20.0,
        total_bom_cost=670.0,
        margin_multiplier=1.34,
        retail_price=899.0,
        volume_tier="10K+",
        process_node="8nm",
        year=2025,
        notes=(
            "High-end edge AI module. 64GB LPDDR5. NVIDIA pricing: "
            "competitive margin for robotics/automotive. Competes with "
            "datacenter inference cards."
        ),
    )

    return model
