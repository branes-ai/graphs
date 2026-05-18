"""Texas Instruments TDA4VL (Jacinto 7 Entry-Level) resource model.

Cleanup PR of DSP batch #223 SKU 6. **Closes TI TDA4 family** (batch 2
complete). Retires the 225-LOC hand-coded body of
ti_tda4vl_resource_model() in favor of a thin loader wrapper. Parity
pinned in tests/hardware/test_dsp_yaml_loader_ti_tda4vl_parity.py.

Schema precursors:
  - DSP sprint #211 (closed): DSPBlock + dsp_yaml_loader
  - embodied-schemas#58: the TI TDA4VL YAML

**5 documented drifts collapse to identity post-cleanup** -- identical
shape to TDA4AL (#227) and TDA4VH (#228):

  - compute_units: hand=16 (abstracted) -> 4 (real C7x num_units)
  - threads_per_unit: hand=250 (BUG) -> 4 (wave_quantization)
  - warp_size: hand=1 (BUG) -> 4 (=compute_units)
  - energy_per_flop_fp32: hand=3.6 pJ (28nm bug) -> 2.43 pJ (16nm correct)
  - precision_profiles: hand has INT8/FP32 only -> full
    INT8/INT16/FP16/FP32

The public function name + signature are preserved.

The legacy resource-model ``name`` ("TI-TDA4VL-C7x-MMAv2") is preserved.
"""

from embodied_schemas.dsp_block import DSPFabricKind

from ...resource_model import HardwareResourceModel
from ..ip_cores.dsp_yaml_loader import load_dsp_resource_model_from_yaml


_LEGACY_NAME = "TI-TDA4VL-C7x-MMAv2"
_YAML_BASE_ID = "ti_tda4vl"
_LEGACY_FABRIC_TYPE_OVERRIDES = {
    DSPFabricKind.VLIW_SCALAR: "c7x_dsp",
    DSPFabricKind.TENSOR_MATRIX: "mma_v2",
}


def ti_tda4vl_resource_model() -> HardwareResourceModel:
    """Texas Instruments TDA4VL (Jacinto 7 Entry-Level) -- entry-level
    automotive ADAS SoC for cost-sensitive deployments.

    Half the compute of TDA4VM: 4 C7x DSP cores + 1 MMAv2 at 4000
    ops/cycle. 4 TOPS INT8 + 40 GFLOPS FP32. 7W + 12W thermal
    profiles (both passive cooling). ASIL-B/C safety. See the YAML
    for the canonical chip description.
    """
    return load_dsp_resource_model_from_yaml(
        _YAML_BASE_ID,
        name_override=_LEGACY_NAME,
        fabric_type_overrides=_LEGACY_FABRIC_TYPE_OVERRIDES,
    )
