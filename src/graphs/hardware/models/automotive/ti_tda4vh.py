"""Texas Instruments TDA4VH (Jacinto 7 Very High Performance) resource model.

Cleanup PR of DSP batch #223 SKU 5. Retires the 193-LOC hand-coded
body of ti_tda4vh_resource_model() in favor of a thin loader wrapper.
Parity pinned in tests/hardware/test_dsp_yaml_loader_ti_tda4vh_parity.py.

Schema precursors:
  - DSP sprint #211 (closed): DSPBlock + dsp_yaml_loader
  - embodied-schemas#57: the TI TDA4VH YAML

**5 documented drifts collapse to identity post-cleanup** (same
pattern as TDA4AL #227):

  - compute_units: hand=128 (abstracted) -> 32 (real C7x num_units)
  - threads_per_unit: hand=250 (BUG) -> 8 (wave_quantization)
  - warp_size: hand=1 (BUG) -> 32 (=compute_units)
  - energy_per_flop_fp32: hand=3.6 pJ (28nm bug) -> 2.43 pJ (16nm correct)
  - precision_profiles: hand has INT8/FP32 only -> full
    INT8/INT16/FP16/FP32 (matches fabric capability)

The public function name + signature are preserved.

The legacy resource-model ``name`` ("TI-TDA4VH-4xC7x-4xMMAv2") is preserved.
"""

from embodied_schemas.dsp_block import DSPFabricKind

from ...resource_model import HardwareResourceModel
from ..ip_cores.dsp_yaml_loader import load_dsp_resource_model_from_yaml


_LEGACY_NAME = "TI-TDA4VH-4xC7x-4xMMAv2"
_YAML_BASE_ID = "ti_tda4vh"
_LEGACY_FABRIC_TYPE_OVERRIDES = {
    DSPFabricKind.VLIW_SCALAR: "c7x_dsp",
    DSPFabricKind.TENSOR_MATRIX: "mma_v2",
}


def ti_tda4vh_resource_model() -> HardwareResourceModel:
    """Texas Instruments TDA4VH (Jacinto 7 Very High Performance) --
    high-end automotive ADAS SoC for Level 3-4 autonomy.

    4x TDA4VM compute: 32 C7x DSP cores + 4 MMAv2 tensor accelerators,
    16 MiB MSMC SRAM, 16 GiB LPDDR5 @ 100 GB/s. 20W + 35W thermal
    profiles. See the YAML for the canonical chip description.
    """
    return load_dsp_resource_model_from_yaml(
        _YAML_BASE_ID,
        name_override=_LEGACY_NAME,
        fabric_type_overrides=_LEGACY_FABRIC_TYPE_OVERRIDES,
    )
