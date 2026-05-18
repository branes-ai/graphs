"""Synopsys ARC EV7x resource model.

Cleanup PR of the DSP follow-up batch (graphs#223). As of this PR,
the chip's architectural and thermal-profile data lives in the
canonical YAML at
``embodied-schemas:data/compute_products/synopsys/synopsys_arc_ev7x.yaml``
(landed in embodied-schemas#53) and is loaded via ``dsp_yaml_loader``.
The previous ~244-LOC hand-coded ``HardwareResourceModel``
constructor was retired here; its parity with the YAML-loaded model
is pinned in
``tests/hardware/test_dsp_yaml_loader_synopsys_ev7x_parity.py``.

Schema precursors that had to land first:
  - DSP sprint #211 (closed): DSPBlock + dsp_yaml_loader
  - embodied-schemas#53: the Synopsys ARC EV7x YAML itself

The public function name and signature are preserved so existing
callers (mapper factory, validation, CLI) continue to work unchanged.

**Multi-fabric clean-cut migration**: like Cadence Vision Q8 (#214),
no overlays remain except the ``fabric_type_overrides`` argument
that preserves the legacy ``"ev7x_vpu"`` / ``"ev7x_dnn_accelerator"``
fabric_type strings.

3 documented drifts where the YAML loader follows the dsp_yaml_loader's
"pick fabric[0]" convention while the hand-coded factory picked
the DNN fabric as the primary:

  - ``compute_units``: yaml=4 (VPU), hand=128 (DNN). The DNN fabric's
    num_units is 128 in both; the drift is which fabric the chip-
    level surface mirrors.
  - ``energy_per_flop_fp32``: yaml=2.43 pJ (simd_packed baseline,
    fabric[0]), hand=2.295 pJ (tensor_core baseline, DNN).
  - ``precision_profiles``: yaml additionally includes INT32 (from
    VPU); hand-coded omits.

All three drifts could be eliminated by reordering YAML fabrics
(DNN first) in a follow-up; punted to keep the v9 DSP follow-up
batch landing fast.

The legacy resource-model ``name`` ("Synopsys-ARC-EV7x-4core") is
preserved.
"""

from embodied_schemas.dsp_block import DSPFabricKind

from ...resource_model import HardwareResourceModel
from .dsp_yaml_loader import load_dsp_resource_model_from_yaml


_LEGACY_NAME = "Synopsys-ARC-EV7x-4core"
_YAML_BASE_ID = "synopsys_arc_ev7x"
_LEGACY_FABRIC_TYPE_OVERRIDES = {
    DSPFabricKind.VECTOR_SIMD: "ev7x_vpu",
    DSPFabricKind.TENSOR_MATRIX: "ev7x_dnn_accelerator",
}


def synopsys_arc_ev7x_resource_model() -> HardwareResourceModel:
    """Synopsys ARC EV7x (4-core) -- 7th-generation DesignWare ARC
    embedded vision IP core.

    See the YAML for the canonical chip description (4 VPU vector
    cores + 128-unit DNN tensor accelerator at 1.0 GHz on TSMC N16,
    35 TOPS INT8 + 8.8 GFLOPS FP32, 5W passive, multi-precision
    INT8 / INT16 / INT32 / FP32, vision-optimized).
    """
    return load_dsp_resource_model_from_yaml(
        _YAML_BASE_ID,
        name_override=_LEGACY_NAME,
        fabric_type_overrides=_LEGACY_FABRIC_TYPE_OVERRIDES,
    )
