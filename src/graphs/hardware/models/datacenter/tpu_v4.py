"""Google TPU v4 resource model.

Final cleanup PR for issue #204 (TPU mini-sprint). As of this PR,
the chip's architectural and thermal-profile data lives in the
canonical YAML at
``embodied-schemas:data/compute_products/google/tpu_v4.yaml``
(landed in embodied-schemas#39) and is loaded via ``tpu_yaml_loader``.
The previous ~176-LOC hand-coded ``HardwareResourceModel``
constructor was retired here; its parity with the YAML-loaded model
is pinned in ``tests/hardware/test_tpu_yaml_loader_v4_parity.py``.

Schema precursors that had to land first:
  - embodied-schemas#38 -- TPUBlock added to the discriminated union
  - embodied-schemas#39 -- the TPU v4 YAML itself

The public function name and signature are preserved so existing
callers (``create_tpu_v4_mapper``, validation scripts, energy / BERT
benchmarks, ``cli/compare_*.py``) continue to work unchanged.

**Zero overlays remain.** Like Plasticine (#199) and Vitis AI (#203),
this is a clean-cut migration because:

  - ``HardwareType.TPU`` was already in the graphs enum (no
    transitional period like NPU's #191).
  - Loader's ``peak_bandwidth`` convention (external DRAM = HBM2e at
    1.2 TB/s) matches the legacy hand-coded value exactly.
  - Loader reconstructs the ``TPUTileEnergyModel`` from the schema's
    ``TPUTileEnergyCoefficients`` sub-type with the same 9 canonical
    coefficients the hand-coded factory used.
  - No BOMCostProfile overlay needed (TPU v4 is Google Cloud rental
    only; no commercial BoM).

Four documented drifts where the YAML loader follows v7 schema
conventions that differ from the legacy:

  - ``energy_per_flop_fp32``: ~4x drift (0.45 pJ vs 1.8 pJ). YAML
    computes from BF16 baseline * FP32 scaling = 0.225 * 2.0 = 0.45.
    Hand-coded used ``get_base_alu_energy(7, 'standard_cell')`` =
    1.8 pJ static 7nm baseline. Both valid 7nm estimates; v8 may add
    a direct ``energy_per_flop_fp32_override`` field on
    ``TPUComputeFabric``.
  - ``memory_technology``: "HBM2E" (correct) vs None (legacy didn't set).
  - ``soc_fabric``: populated 2-endpoint crossbar (vs legacy None).
    Unblocks downstream Layer 6 reporting for TPU.
  - **MXU count modeling**: both YAML and legacy model TPU v4 as
    2 MXUs * 128x128 = 32,768 MACs (= 68.8 TOPS BF16). Google's actual
    TPU v4 has 2 TensorCores * 4 MXUs each = 8 MXUs (= 275 TFLOPS
    marketed). The schema's ``chip.performance.bf16_tflops = 275.0``
    captures the marketed number, but the precision_profiles[BF16]
    derived from the compute_fabric uses the 2-MXU model. v8
    reconciliation: model 2 TC * 4 MXUs properly (might require
    multi-block-per-die support).

Two v8 reconciliation items:
  - ``is_statically_reconfigurable=False`` (TPUs are fixed-function)
    -- not in TPUBlock schema yet.
  - ``ici_port_count`` / ``ici_bandwidth_per_port_gbps`` /
    ``ici_topology_hint`` -- live on TPUBlock in the v7 schema but
    have no equivalent fields on ``HardwareResourceModel``.
    Downstream consumers read from ComputeProduct directly meanwhile.

The legacy resource-model ``name`` ("TPU-v4") is preserved.
"""

from ...resource_model import HardwareResourceModel
from .tpu_yaml_loader import load_tpu_resource_model_from_yaml


_LEGACY_NAME = "TPU-v4"
_YAML_BASE_ID = "google_tpu_v4"


def tpu_v4_resource_model() -> HardwareResourceModel:
    """Google TPU v4 -- 4th-generation datacenter training TPU.

    See the YAML for the canonical chip description (2 MXUs * 128x128
    at 1.05 GHz on TSMC N7, 32 MiB on-chip Unified Buffer, 32 GiB
    chip-attached HBM2e at 1.2 TB/s, 350W active liquid cooling,
    6-port ICI for 3D-torus pod, multi-precision BF16 + INT8 + emulated
    FP32, full 9-coefficient tile energy model).
    """
    return load_tpu_resource_model_from_yaml(
        _YAML_BASE_ID,
        name_override=_LEGACY_NAME,
    )
