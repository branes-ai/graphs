"""Cadence Tensilica Vision Q8 resource model.

Final cleanup PR for issue #211 (DSP mini-sprint). As of this PR,
the chip's architectural and thermal-profile data lives in the
canonical YAML at
``embodied-schemas:data/compute_products/cadence/cadence_vision_q8.yaml``
(landed in embodied-schemas#44) and is loaded via ``dsp_yaml_loader``.
The previous ~222-LOC hand-coded ``HardwareResourceModel``
constructor was retired here; its parity with the YAML-loaded model
is pinned in ``tests/hardware/test_dsp_yaml_loader_cadence_q8_parity.py``.

Schema precursors that had to land first:
  - embodied-schemas#43 -- DSPBlock added to the discriminated union
    (closes the final category schema gap; eighth and last block kind)
  - embodied-schemas#44 -- the Cadence Vision Q8 YAML itself

The public function name and signature are preserved so existing
callers (mappers, validation scripts, benchmarks, ``cli/compare_*.py``)
continue to work unchanged.

**Zero overlays remain.** Like Plasticine (#199), Vitis AI (#203),
and TPU v4 (sprint #204 PR 5), this is a clean-cut migration because:

  - ``HardwareType.DSP`` was already in the graphs enum (no
    transitional period like NPU's #191).
  - Loader's ``peak_bandwidth`` convention (external DRAM = LPDDR4
    typical-integration at 40 GB/s) matches the legacy hand-coded
    value exactly.
  - Loader's ``compute_units`` / ``threads_per_unit`` / ``warp_size``
    conventions match the hand-coded factory's SIMD-fabric model.
  - No BOMCostProfile overlay needed (Cadence Vision Q8 is licensable
    IP; no per-unit BOM).

The ``fabric_type_overrides`` argument preserves the legacy
``"vision_q8_simd"`` fabric_type string (the loader's generic mapping
would emit ``"vector_simd"`` for ``DSPFabricKind.VECTOR_SIMD``).

The legacy resource-model ``name`` ("Cadence-Tensilica-Vision-Q8") is
preserved.

DSP sprint closes here:
  - All 10 hand-coded DSP factories will be migrated as pure data PRs
    after this sprint closes:
      ip_cores/ceva_neupro_npm11.py, synopsys_arc_ev7x.py
      automotive/qualcomm_sa8775p.py, ti_tda4{al,vh,vl,vm}.py
      edge/qrb5165.py, qualcomm_qcs6490.py
  - DSP was the final category schema gap. Every architecture in the
    catalog now has a YAML-backed thin-loader path.
"""

from embodied_schemas.dsp_block import DSPFabricKind

from ...resource_model import HardwareResourceModel
from .dsp_yaml_loader import load_dsp_resource_model_from_yaml


_LEGACY_NAME = "Cadence-Tensilica-Vision-Q8"
_YAML_BASE_ID = "cadence_tensilica_vision_q8"
_LEGACY_FABRIC_TYPE = "vision_q8_simd"


def cadence_vision_q8_resource_model() -> HardwareResourceModel:
    """Cadence Tensilica Vision Q8 -- 7th-generation vision DSP IP core.

    See the YAML for the canonical chip description (32 SIMD units
    @ 1.0 GHz on TSMC N16, 1024-bit SIMD engine, 32 KiB L1 per unit,
    1 MiB shared L2, 4 GB typical LPDDR4 pairing at 40 GB/s, 1W
    passive cooling, multi-precision INT8 / INT16 / FP16 / FP32,
    vision-optimized).
    """
    return load_dsp_resource_model_from_yaml(
        _YAML_BASE_ID,
        name_override=_LEGACY_NAME,
        fabric_type_overrides={DSPFabricKind.VECTOR_SIMD: _LEGACY_FABRIC_TYPE},
    )
