"""CEVA NeuPro-M NPM11 resource model.

Cleanup PR of the DSP follow-up batch (graphs#223). Closes batch 1
of 3: both IP-core SKUs (Synopsys EV7x + CEVA NPM11) are now YAML-
backed. The chip's architectural and thermal-profile data lives in
the canonical YAML at
``embodied-schemas:data/compute_products/ceva/ceva_neupro_npm11.yaml``
(landed in embodied-schemas#54) and is loaded via ``dsp_yaml_loader``.
The previous ~254-LOC hand-coded ``HardwareResourceModel``
constructor was retired here; its parity with the YAML-loaded model
is pinned in
``tests/hardware/test_dsp_yaml_loader_ceva_npm11_parity.py``.

Schema precursors that had to land first:
  - DSP sprint #211 (closed): DSPBlock + dsp_yaml_loader
  - embodied-schemas#54: the CEVA NeuPro-M NPM11 YAML

The public function name and signature are preserved so existing
callers (mapper factory, validation, CLI) continue to work unchanged.

**Tensor-first ordering = zero drift**: unlike Synopsys EV7x (#224)
which had 3 documented drifts pre-cleanup from VPU-first fabric
ordering, NPM11's YAML places the tensor fabric first so the loader's
pick-first convention maps it to chip-level surfaces (compute_units
= 64 = tensor's num_units; energy_per_flop_fp32 = 2.295 pJ from
tensor_core baseline). Only INT4 differs (YAML completer than
hand-coded; the hand-coded factory omitted INT4 from precision_profiles
despite the tensor fabric supporting it). Post-cleanup both fixtures
resolve through the loader so even that drift collapses to identity.

The ``fabric_type_overrides`` argument preserves the legacy fabric
strings ``"neupro_tensor"`` / ``"neupro_vector"``.

The legacy resource-model ``name`` ("CEVA-NeuPro-M-NPM11") is
preserved.
"""

from embodied_schemas.dsp_block import DSPFabricKind

from ...resource_model import HardwareResourceModel
from .dsp_yaml_loader import load_dsp_resource_model_from_yaml


_LEGACY_NAME = "CEVA-NeuPro-M-NPM11"
_YAML_BASE_ID = "ceva_neupro_m_npm11"
_LEGACY_FABRIC_TYPE_OVERRIDES = {
    DSPFabricKind.TENSOR_MATRIX: "neupro_tensor",
    DSPFabricKind.VECTOR_SIMD: "neupro_vector",
}


def ceva_neupro_npm11_resource_model() -> HardwareResourceModel:
    """CEVA NeuPro-M NPM11 -- single-engine neural processing IP core
    for edge AI acceleration.

    See the YAML for the canonical chip description (64-unit tensor +
    64-unit vector fabrics at 1.0 GHz on TSMC N16, 20 TOPS INT8 +
    40 TOPS INT4 + 10 TFLOPS FP16, 2W passive, multi-precision
    INT4 / INT8 / INT16 / FP16, edge-optimized).
    """
    return load_dsp_resource_model_from_yaml(
        _YAML_BASE_ID,
        name_override=_LEGACY_NAME,
        fabric_type_overrides=_LEGACY_FABRIC_TYPE_OVERRIDES,
    )
