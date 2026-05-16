"""Hailo-10H resource model.

Graphs cleanup PR for the Hailo-10H half of issue #192 (the NPU
sprint follow-up). As of this PR, the chip's architectural and
thermal-profile data lives in the canonical YAML at
``embodied-schemas:data/compute_products/hailo/hailo_10h.yaml``
(landed in embodied-schemas#31) and is loaded via ``npu_yaml_loader``.
The previous ~360-LOC hand-coded ``HardwareResourceModel``
constructor was retired here; its parity with the YAML-loaded model
is pinned in ``tests/hardware/test_npu_yaml_loader_hailo10h_parity.py``.

Schema precursors that had to land first:
  - embodied-schemas#30 -- ``KVCacheSpec`` extension to ``NPUBlock``
    (Hailo-10H is the first SKU to populate it: 8K context, K=INT8 /
    V=INT4 asymmetric quantization, ring-buffer streaming, DRAM
    offload).
  - embodied-schemas#31 -- the Hailo-10H YAML itself.

The public function name and signature are preserved so existing
callers (``create_hailo10h_mapper``, layer1/4/5/7 reporting,
validation scripts, ``cli/compare_*.py``) continue to work unchanged.

One overlay remains in the factory after loading -- BoM cost --
which is data the v4 ComputeProduct schema doesn't yet carry:

  - ``BOMCostProfile`` -- die / package / memory / PCB / thermal /
    margin / retail; v5 ``Market.bom`` will absorb it.

Two documented shape drifts remain (v5 reconciliation items):

  - ``energy_per_flop_fp32`` synthesis -- NPUs don't ship FP32; the
    loader synthesizes it as ``energy_per_op_int8 * 8`` per the
    standard-cell rule of thumb. Within ~1% of the prior hand-coded
    ``get_base_alu_energy(16, 'standard_cell')`` value.
  - ``default_precision`` -- the loader picks INT8 by convention
    (the NPU default); the hand-coded Hailo-10H factory used INT4
    (the primary GenAI use case). The thin factory does NOT override
    this; downstream consumers that want INT4 must select it
    explicitly. v5 cleanup: add ``default_precision`` hint to
    ``NPUBlock`` so per-SKU preference is captured in the YAML.

The legacy resource-model ``name`` ("Hailo-10H") is preserved.
"""

from ...resource_model import BOMCostProfile, HardwareResourceModel
from .npu_yaml_loader import load_npu_resource_model_from_yaml


_LEGACY_NAME = "Hailo-10H"
_YAML_BASE_ID = "hailo_hailo_10h"


def hailo10h_resource_model() -> HardwareResourceModel:
    """Hailo-10H Generative AI Edge Accelerator.

    See the YAML for the canonical chip description (40 dataflow
    units, structure-driven dataflow architecture, INT8/INT4 only,
    32 MiB total on-chip SRAM with 12 MiB KV-cache-sized shared L2,
    8 GiB LPDDR4X external DRAM, MESH_2D 8x5 NoC, single 2.5W
    operating point, transformer KV cache surface).
    """
    model = load_npu_resource_model_from_yaml(
        _YAML_BASE_ID,
        name_override=_LEGACY_NAME,
    )

    # BoM cost overlay -- not yet carried by the v4 ComputeProduct schema.
    # Numbers from teardowns + Hailo public pricing (M.2 module estimated
    # retail $240 with $70 BOM, vs $40 for Hailo-8 due to LPDDR4X).
    model.bom_cost_profile = BOMCostProfile(
        silicon_die_cost=30.0,         # 16nm die (slightly larger than Hailo-8)
        package_cost=10.0,             # package with LPDDR4X interface
        memory_cost=20.0,              # 4 GB LPDDR4X on-module (KV cache + weights)
        pcb_assembly_cost=5.0,         # more complex than Hailo-8
        thermal_solution_cost=1.0,     # passive heatsink (2.5W envelope)
        other_costs=4.0,               # testing, connectors, certification
        total_bom_cost=0,              # auto-calculated: $70
        margin_multiplier=3.5,         # high margin for cutting-edge edge GenAI
        retail_price=240.0,            # estimated retail (Hailo-8 is $160, this adds GenAI)
        volume_tier="10K+",
        process_node="16nm",
        year=2025,
        notes=(
            "First edge Gen AI accelerator. KV cache support for LLMs. "
            "Higher BOM than Hailo-8 due to external LPDDR4X. Production "
            "2026 for automotive (AEC-Q100 Grade 2)."
        ),
    )

    return model
