"""Qualcomm QCS6490 resource model.

**Closes DSP batch #223 sprint** (10/10 SKUs YAML-backed). Retires
the 365-LOC hand-coded body of qualcomm_qcs6490_resource_model() in
favor of a thin loader wrapper with a BOM overlay (same pattern as
SA8775P #230 and QRB5165 #231).

Schema precursors:
  - DSP sprint #211 (closed): DSPBlock + dsp_yaml_loader
  - embodied-schemas#61: the Qualcomm QCS6490 YAML + tsmc_n6 process
    node

**6 documented drifts collapse to identity post-cleanup** (same shape
as SA8775P #230):

  - compute_units: hand=16 (HVX num_units) -> 1 (HTA num_units;
    loader picks fabric[0])
  - threads_per_unit: hand=64 (DSP vector threads) -> 2 (wave_quantization)
  - warp_size: hand=32 (SIMD estimate) -> 1 (=compute_units)
  - energy_per_flop_fp32: hand=1.485 pJ (HVX 6nm simd_packed) ->
    1.40 pJ (HTA 6nm tensor_core; HTA is fabric[0])
  - precision_profiles: hand has INT8/INT4 only -> full
    INT8/INT16/FP16/INT4
  - bom_cost_profile: hand-coded has it; YAML loader doesn't;
    wrapper attaches BOM overlay post-load

The public function name + signature are preserved.

The legacy resource-model ``name`` ("Qualcomm-QCS6490-Hexagon-V79")
is preserved.
"""

from embodied_schemas.dsp_block import DSPFabricKind

from ...resource_model import BOMCostProfile, HardwareResourceModel
from ..ip_cores.dsp_yaml_loader import load_dsp_resource_model_from_yaml


_LEGACY_NAME = "Qualcomm-QCS6490-Hexagon-V79"
_YAML_BASE_ID = "qualcomm_qcs6490"
_LEGACY_FABRIC_TYPE_OVERRIDES = {
    DSPFabricKind.TENSOR_MATRIX: "hta_tensor",
    DSPFabricKind.VECTOR_SIMD: "hvx_vector",
}


# QCS6490 BOM overlay (carried by the wrapper; YAMLs don't ship BOM).
# Estimated @ 10K units, entry-level edge AI consumer grade.
_QCS6490_BOM_COST_PROFILE = BOMCostProfile(
    silicon_die_cost=45.0,        # 6nm die (smaller than 16nm; cheaper than QRB5165's 7nm)
    package_cost=12.0,             # Advanced flip-chip package
    memory_cost=15.0,              # 2GB LPDDR4X on-package
    pcb_assembly_cost=6.0,         # SMT assembly
    thermal_solution_cost=2.0,     # Small heatsink
    other_costs=5.0,               # Testing, connectors
    total_bom_cost=0,              # Auto-calculated
    margin_multiplier=2.8,         # Qualcomm typical margin
    retail_price=0,                # Auto-calculated
    volume_tier="10K+",
    process_node="6nm",
    year=2025,
    notes=(
        "Entry-level edge AI SoC. Competitive with Hailo-8 ($40) but "
        "higher BOM due to CPU/GPU integration."
    ),
)


def qualcomm_qcs6490_resource_model() -> HardwareResourceModel:
    """Qualcomm QCS6490 -- entry-level edge AI SoC with Hexagon NPU V79
    on 6nm TSMC.

    See the YAML for the canonical chip description (HTA tensor + 16x
    HVX vector multi-fabric, 12 TOPS INT8 + 24 TOPS INT4, 3-profile
    DVFS: 5W battery + 10W standard + 15W max, LPDDR4X @ 40 GB/s,
    consumer-grade lifecycle). BOM overlay attached post-load.
    """
    model = load_dsp_resource_model_from_yaml(
        _YAML_BASE_ID,
        name_override=_LEGACY_NAME,
        fabric_type_overrides=_LEGACY_FABRIC_TYPE_OVERRIDES,
    )
    # BOM overlay -- YAML loader doesn't produce BOM
    model.bom_cost_profile = _QCS6490_BOM_COST_PROFILE
    return model
