"""Xilinx Vitis AI DPU (Versal VE2302, B4096 config) resource model.

Final cleanup PR for issue #200 (DPU mini-sprint). As of this PR,
the chip's architectural and thermal-profile data lives in the
canonical YAML at
``embodied-schemas:data/compute_products/xilinx/vitis_ai_b4096.yaml``
(landed in embodied-schemas#37) and is loaded via ``dpu_yaml_loader``.
The previous ~160-LOC hand-coded ``HardwareResourceModel``
constructor was retired here; its parity with the YAML-loaded model
is pinned in ``tests/hardware/test_dpu_yaml_loader_vitis_ai_parity.py``.

Schema precursors that had to land first:
  - embodied-schemas#36 -- DPUBlock added to the discriminated union
  - embodied-schemas#37 -- the Vitis AI B4096 YAML itself

The public function name and signature are preserved so existing
callers (``create_xilinx_vitis_ai_dpu_mapper``, validation scripts,
``cli/compare_*.py``) continue to work unchanged.

**Zero overlays remain.** Like Plasticine (issue #196 cleanup), this
is a clean-cut migration because:

  - ``HardwareType.DPU`` was already in the graphs enum (no
    transitional period like NPU's #191).
  - ``peak_bandwidth`` convention (external DRAM tier when present;
    50 GB/s DDR4 for Vitis AI) matches the legacy hand-coded value
    exactly.
  - ``memory_technology`` was None in the legacy; the YAML loader
    populates it correctly ("DDR4").
  - No BOMCostProfile overlay needed (the legacy factory didn't
    include one either; Versal SoM/SoC pricing not yet modeled).

Six documented drifts captured by the parity test where the YAML
follows schema conventions that differ from the legacy:

  - INT8 peak: 10.24 TOPS (theoretical chip-level, schema convention)
    vs 7.68 TOPS (legacy realistic at 75% efficiency, pre-multiplied).
    Downstream consumers that want realistic peak should multiply by
    ``thermal_operating_points[default].efficiency_factor_by_precision[int8]``.
  - FP16 peak: 2.56 TFLOPS vs 1.92 TFLOPS (same 75% efficiency factor).
  - FP32 peak: not in YAML precision_profiles vs 0.96 TFLOPS legacy.
    YAML's compute_fabrics declare INT8 + FP16; FP32 is captured only
    as energy_scaling since it's emulated.
  - ``memory_technology``: "DDR4" (correct) vs None (legacy didn't set).
  - ``soc_fabric``: 8x8 AIE mesh populated vs None (legacy didn't
    model -- now unblocks downstream Layer 6 reporting for DPU).
  - ``energy_per_flop_fp32``: ~8% drift (2.5 pJ vs 2.7 pJ); both valid.

One v7 reconciliation item:
  - ``is_statically_reconfigurable`` / ``bitstream_load_time_ms`` /
    ``fpga_fabric_overhead_factor`` (the defining DPU characteristics)
    live on ``DPUBlock`` in the v6 schema but have no equivalent
    fields on ``HardwareResourceModel``. Downstream consumers that
    need them read from the underlying ComputeProduct directly. v7
    will add the fields properly.

The legacy resource-model ``name`` ("DPU-Vitis-AI-B4096") is preserved.
"""

from ...resource_model import HardwareResourceModel
from .dpu_yaml_loader import load_dpu_resource_model_from_yaml


_LEGACY_NAME = "DPU-Vitis-AI-B4096"
_YAML_BASE_ID = "xilinx_vitis_ai_b4096"


def xilinx_vitis_ai_dpu_resource_model() -> HardwareResourceModel:
    """Xilinx Vitis AI B4096 DPU on Versal VE2302 -- FPGA-based AI
    accelerator with AIE-ML v1 tile array.

    See the YAML for the canonical chip description (64 AIE-ML tiles *
    64 MACs at 1.25 GHz, 64 KiB scratchpad per tile, 4 MiB shared L2,
    8 GiB chip-attached DDR4, 8x8 AIE_MESH NoC, 20W active-fan,
    TSMC N16 process, multi-precision INT8 + native FP16 + emulated
    FP32, static FPGA reconfiguration with ~2 second bitstream load).
    """
    return load_dpu_resource_model_from_yaml(
        _YAML_BASE_ID,
        name_override=_LEGACY_NAME,
    )
