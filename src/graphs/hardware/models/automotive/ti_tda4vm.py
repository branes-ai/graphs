"""Texas Instruments TDA4VM (Jacinto 7) resource model.

Cleanup PR of DSP batch #223 SKU 3 (graphs#223). **First SoC-integrated
DSP** retirement: the chip's architectural and thermal-profile data
lives in the canonical YAML at
``embodied-schemas:data/compute_products/ti/ti_tda4vm.yaml`` (landed
in embodied-schemas#55) and is loaded via ``dsp_yaml_loader``. The
previous ~368-LOC hand-coded ``HardwareResourceModel`` constructor
was retired here; its parity with the YAML-loaded model is pinned in
``tests/hardware/test_dsp_yaml_loader_ti_tda4vm_parity.py``.

Schema precursors that had to land first:
  - DSP sprint #211 (closed): DSPBlock + dsp_yaml_loader
  - embodied-schemas#55: the TI TDA4VM YAML

The public function name + signature are preserved so all callers
continue to work unchanged.

**3 documented drifts collapse to identity post-cleanup**:

  - ``compute_units``: pre-cleanup hand=32 (abstracted via
    num_dsp_units = MMA's 8000 ops/cycle / 250 ops/cycle/unit);
    post-cleanup both fixtures = 8 (C7x's real num_units; loader
    picks fabric[0]).
  - ``energy_per_flop_fp32``: pre-cleanup hand=3.6 pJ (28nm
    simd_packed, factory bug); post-cleanup both = 2.43 pJ (16nm
    simd_packed, per TI's official "16nm FinFET" docs).
  - ``precision_profiles``: pre-cleanup hand omits FP16 from chip-
    level profiles despite the C7x fabric supporting it; post-cleanup
    both include FP16 (YAML completer).

**First multi-profile DVFS DSP**: 2 thermal profiles (10W front-camera,
20W full-ADAS-system) preserved through the loader; the
DSPBlock.thermal_profiles list with len > 1 works end-to-end on DSP.

The ``fabric_type_overrides`` argument preserves the legacy fabric
strings ``"c7x_dsp"`` / ``"mma_v1"``.

The legacy resource-model ``name`` ("TI-TDA4VM-C7x-DSP") is preserved.
"""

from embodied_schemas.dsp_block import DSPFabricKind

from ...resource_model import HardwareResourceModel
from ..ip_cores.dsp_yaml_loader import load_dsp_resource_model_from_yaml


_LEGACY_NAME = "TI-TDA4VM-C7x-DSP"
_YAML_BASE_ID = "ti_tda4vm"
_LEGACY_FABRIC_TYPE_OVERRIDES = {
    DSPFabricKind.VLIW_SCALAR: "c7x_dsp",
    DSPFabricKind.TENSOR_MATRIX: "mma_v1",
}


def ti_tda4vm_resource_model() -> HardwareResourceModel:
    """Texas Instruments TDA4VM (Jacinto 7) -- automotive ADAS SoC.

    See the YAML for the canonical chip description (8x C7x VLIW DSP
    cores + MMAv1 tensor accelerator on 16nm FinFET, 8 TOPS INT8 +
    80 GFLOPS FP32, ASIL-D safety certified, 10W front-camera +
    20W full-ADAS-system thermal profiles, LPDDR4x measured 60 GB/s).
    """
    return load_dsp_resource_model_from_yaml(
        _YAML_BASE_ID,
        name_override=_LEGACY_NAME,
        fabric_type_overrides=_LEGACY_FABRIC_TYPE_OVERRIDES,
    )
