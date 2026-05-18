"""Qualcomm SA8775P (Snapdragon Ride) resource model.

Cleanup PR of DSP batch #223 SKU 7. Retires the 369-LOC hand-coded
body of qualcomm_sa8775p_resource_model() in favor of a thin loader
wrapper **with a BOM overlay** -- SA8775P is the first DSP SKU with
commercial BOM; the YAML loader doesn't produce BOM so this wrapper
attaches it post-load.

Schema precursors:
  - DSP sprint #211 (closed): DSPBlock + dsp_yaml_loader
  - embodied-schemas#59: the Qualcomm SA8775P YAML

**6 documented drifts collapse to identity post-cleanup**:

  - compute_units: hand=32 (abstracted HVX+HMX) -> 2 (HMX's num_units;
    loader picks fabric[0])
  - threads_per_unit: hand=128 (HMX accelerator threads) -> 4 (wave_quantization)
  - warp_size: hand=32 (SIMD width estimate) -> 2 (=compute_units)
  - energy_per_flop_fp32: hand=1.35 pJ (HVX baseline, 5nm simd_packed)
    -> 1.27 pJ (HMX baseline, 5nm tensor_core, fabric[0])
  - precision_profiles: hand has INT8/INT4 only -> full
    INT8/INT16/FP16/INT4
  - bom_cost_profile: hand-coded has it; YAML loader doesn't; this
    wrapper attaches the BOM overlay post-load. Both fixtures have
    BOM post-cleanup.

The public function name + signature are preserved.

The legacy resource-model ``name`` ("SA8775P-Snapdragon-Ride") is preserved.
"""

from embodied_schemas.dsp_block import DSPFabricKind

from ...resource_model import BOMCostProfile, HardwareResourceModel
from ..ip_cores.dsp_yaml_loader import load_dsp_resource_model_from_yaml


_LEGACY_NAME = "SA8775P-Snapdragon-Ride"
_YAML_BASE_ID = "qualcomm_sa8775p"
_LEGACY_FABRIC_TYPE_OVERRIDES = {
    DSPFabricKind.TENSOR_MATRIX: "hmx_tensor",
    DSPFabricKind.VECTOR_SIMD: "hvx_vector",
}


# SA8775P BOM overlay (carried by the wrapper; YAMLs don't ship BOM).
# Estimated @ 10K units, automotive-grade ASIL-D.
_SA8775P_BOM_COST_PROFILE = BOMCostProfile(
    silicon_die_cost=180.0,      # 5nm automotive (ASIL D) -- expensive
    package_cost=35.0,            # Advanced automotive package
    memory_cost=80.0,             # 16GB LPDDR5 (automotive-grade)
    pcb_assembly_cost=25.0,       # Automotive PCB with safety features
    thermal_solution_cost=15.0,   # Enhanced thermal for automotive
    other_costs=15.0,             # Testing, certification, safety
    total_bom_cost=0,             # Auto-calculated
    margin_multiplier=2.2,        # Lower automotive margin (B2B)
    retail_price=0,               # Auto-calculated
    volume_tier="10K+",
    process_node="5nm",
    year=2025,
    notes=(
        "Automotive-grade SoC with ASIL D certification. Higher BOM due "
        "to safety features and testing."
    ),
)


def qualcomm_sa8775p_resource_model() -> HardwareResourceModel:
    """Qualcomm SA8775P (Snapdragon Ride) -- mid-range automotive ADAS
    SoC with Hexagon DSP (HMX tensor + HVX vector) on 5nm TSMC.

    See the YAML for the canonical chip description (HMX tensor + HVX
    vector multi-fabric on TSMC N5, 32 TOPS INT8 + 64 TOPS INT4,
    ASIL-D safety, 3-profile DVFS: 20W passive cockpit + 30W active
    ADAS + 45W max L3-autonomous). BOM overlay attached post-load.
    """
    model = load_dsp_resource_model_from_yaml(
        _YAML_BASE_ID,
        name_override=_LEGACY_NAME,
        fabric_type_overrides=_LEGACY_FABRIC_TYPE_OVERRIDES,
    )
    # BOM overlay -- YAML loader doesn't produce BOM (no *Block YAML
    # carries BOM today); this is a graphs-side enrichment.
    model.bom_cost_profile = _SA8775P_BOM_COST_PROFILE
    return model
