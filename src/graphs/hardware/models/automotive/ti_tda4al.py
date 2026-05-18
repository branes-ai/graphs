"""Texas Instruments TDA4AL (Jacinto 7 Advanced Low-Power) resource model.

Cleanup PR of DSP batch #223 SKU 4. Retires the 188-LOC hand-coded
body of ti_tda4al_resource_model() in favor of a thin wrapper. Parity
pinned in tests/hardware/test_dsp_yaml_loader_ti_tda4al_parity.py.

Schema precursors:
  - DSP sprint #211 (closed): DSPBlock + dsp_yaml_loader
  - embodied-schemas#56: the TI TDA4AL YAML

**5 documented drifts collapse to identity post-cleanup** (more than
TDA4VM's 3 because the TDA4AL hand-coded factory had additional bugs):

  - compute_units: hand=32 (abstracted) -> 8 (real C7x num_units)
  - threads_per_unit: hand=250 (BUG: ops/clock value pasted in) -> 4
  - warp_size: hand=1 (BUG: forgotten) -> 8
  - energy_per_flop_fp32: hand=3.6 pJ (28nm bug; same as TDA4VM) ->
    2.43 pJ (16nm per TI's official "16nm FinFET" docs)
  - precision_profiles: hand has only INT8/FP32 (sparse) -> full
    INT8/INT16/FP16/FP32 (matches the multi-fabric capability)

The public function name + signature are preserved.

The legacy resource-model ``name`` ("TI-TDA4AL-C7x-MMAv2") is preserved.
"""

from embodied_schemas.dsp_block import DSPFabricKind

from ...resource_model import HardwareResourceModel
from ..ip_cores.dsp_yaml_loader import load_dsp_resource_model_from_yaml


_LEGACY_NAME = "TI-TDA4AL-C7x-MMAv2"
_YAML_BASE_ID = "ti_tda4al"
_LEGACY_FABRIC_TYPE_OVERRIDES = {
    DSPFabricKind.VLIW_SCALAR: "c7x_dsp",
    DSPFabricKind.TENSOR_MATRIX: "mma_v2",
}


def ti_tda4al_resource_model() -> HardwareResourceModel:
    """Texas Instruments TDA4AL (Jacinto 7 Advanced Low-Power) --
    automotive ADAS SoC with MMAv2 tensor accelerator.

    Same C7x + MMA architecture as TDA4VM but with MMAv2 (more efficient
    than MMAv1) and a lower thermal ceiling (18W vs TDA4VM's 20W).
    See the YAML for the canonical chip description.
    """
    return load_dsp_resource_model_from_yaml(
        _YAML_BASE_ID,
        name_override=_LEGACY_NAME,
        fabric_type_overrides=_LEGACY_FABRIC_TYPE_OVERRIDES,
    )
