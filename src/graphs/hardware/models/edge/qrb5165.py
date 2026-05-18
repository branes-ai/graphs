"""Qualcomm QRB5165 (Dragonwing) resource model.

Cleanup PR of DSP batch #223 SKU 8. Retires the 328-LOC hand-coded
body of qrb5165_resource_model() in favor of a thin loader wrapper
with a BOM overlay (same pattern as SA8775P #230).

Schema precursors:
  - DSP sprint #211 (closed): DSPBlock + dsp_yaml_loader
  - embodied-schemas#60: the Qualcomm QRB5165 YAML

**4 documented drifts collapse to identity post-cleanup**:

  - compute_units: hand=32 (abstracted HVX+HTA) -> 1 (HTA's single
    num_units; loader picks fabric[0])
  - warp_size: hand=32 (SIMD estimate) -> 1 (=compute_units)
  - energy_per_flop_fp32: hand=1.62 pJ (HVX 7nm simd_packed) ->
    1.53 pJ (HTA 7nm tensor_core; HTA is fabric[0])
  - bom_cost_profile: hand-coded has it; YAML loader doesn't;
    wrapper attaches BOM overlay post-load

threads_per_unit and precision_profiles already matched pre-cleanup
(no drift on those).

The public function name + signature are preserved.

The legacy resource-model ``name`` ("Qualcomm-QRB5165-Hexagon698") is preserved.
"""

from embodied_schemas.dsp_block import DSPFabricKind

from ...resource_model import BOMCostProfile, HardwareResourceModel
from ..ip_cores.dsp_yaml_loader import load_dsp_resource_model_from_yaml


_LEGACY_NAME = "Qualcomm-QRB5165-Hexagon698"
_YAML_BASE_ID = "qualcomm_qrb5165"
_LEGACY_FABRIC_TYPE_OVERRIDES = {
    DSPFabricKind.TENSOR_MATRIX: "hta_tensor",
    DSPFabricKind.VECTOR_SIMD: "hvx_vector",
}


# QRB5165 BOM overlay (carried by the wrapper; YAMLs don't ship BOM).
# Estimated @ 10K units, consumer/robotics grade (vs SA8775P automotive).
_QRB5165_BOM_COST_PROFILE = BOMCostProfile(
    silicon_die_cost=55.0,        # 7nm die (larger than 6nm)
    package_cost=14.0,             # Advanced flip-chip package
    memory_cost=18.0,              # 4GB LPDDR5 on-package
    pcb_assembly_cost=7.0,         # SMT assembly
    thermal_solution_cost=3.0,     # Heatsink for 7W
    other_costs=6.0,               # Testing, connectors, robotics certification
    total_bom_cost=0,              # Auto-calculated
    margin_multiplier=2.7,         # Qualcomm robotics platform margin
    retail_price=0,                # Auto-calculated
    volume_tier="10K+",
    process_node="7nm",
    year=2021,
    notes=(
        "Robotics platform based on Snapdragon 865. Higher BOM than "
        "QCS6490 due to 7nm vs 6nm process."
    ),
)


def qrb5165_resource_model() -> HardwareResourceModel:
    """Qualcomm QRB5165 (Dragonwing) -- robotics platform SoC with
    Hexagon 698 DSP on 7nm TSMC.

    See the YAML for the canonical chip description (HTA tensor + HVX
    vector multi-fabric, 15 TOPS INT8, single 7W passive thermal,
    LPDDR5 quad-channel @ 44 GB/s, consumer-grade lifecycle). BOM
    overlay attached post-load.
    """
    model = load_dsp_resource_model_from_yaml(
        _YAML_BASE_ID,
        name_override=_LEGACY_NAME,
        fabric_type_overrides=_LEGACY_FABRIC_TYPE_OVERRIDES,
    )
    # BOM overlay -- YAML loader doesn't produce BOM
    model.bom_cost_profile = _QRB5165_BOM_COST_PROFILE
    return model
