"""Coral Edge TPU resource model.

Final cleanup PR for issue #192. As of this PR, the chip's
architectural and thermal-profile data lives in the canonical YAML
at ``embodied-schemas:data/compute_products/google/coral_edge_tpu.yaml``
(landed in embodied-schemas#33) and is loaded via ``npu_yaml_loader``.
The previous ~360-LOC hand-coded ``HardwareResourceModel``
constructor was retired here; its parity with the YAML-loaded model
is pinned in ``tests/hardware/test_npu_yaml_loader_coral_parity.py``.

Schema precursors that had to land first:
  - embodied-schemas#32 -- GF 28nm SLP process node YAML
  - embodied-schemas#33 -- the Coral Edge TPU YAML itself

The public function name and signature are preserved so existing
callers (``create_coral_edge_tpu_mapper``, layer 1-7 reporting,
validation scripts, ``cli/compare_*.py``) continue to work unchanged.

SIX overlays remain in the factory after loading -- the v4 ComputeProduct
schema doesn't carry these yet, and one is a taxonomy mismatch:

  1. ``hardware_type=TPU`` -- the v4 schema classifies Coral as an
     NPU (it has an NPUBlock); the graphs ``HardwareType`` taxonomy
     uses TPU as a separate value for systolic accelerators with
     TPU-specific mapping logic in ``TPUMapper``. Hand-coded keeps
     ``HardwareType.TPU`` so the existing TPUMapper guard
     (``hardware_type.value == "tpu"``) accepts Coral unchanged.
     v5 reconciliation: unify the schema NPU + graphs TPU/NPU split
     under a vendor-neutral common module, or add a per-SKU
     ``mapper_class`` hint to NPUBlock.

  2. ``peak_bandwidth=4 GB/s`` -- the YAML loader picks
     on_chip_bandwidth_gbps (128 GB/s for the UB-to-systolic direct
     connect) because Coral has has_external_dram=False. But Coral's
     real roofline bottleneck is the host bus (USB 3.0 / PCIe Gen2
     at ~4 GB/s), since the chip has no external DRAM and weights
     must stream from host memory. v5 reconciliation: add a
     ``host_memory_bandwidth`` concept to NPUMemorySubsystem.

  3. ``memory_technology="LPDDR4 (host)"`` -- consistent with the
     peak_bandwidth overlay; the YAML loader sets "on-chip SRAM (no
     external DRAM)" by default, which doesn't reflect Coral's
     host-memory-via-bus architecture.

  4. ``pipeline_fill_overhead=0.15`` -- TPU-systolic-specific field
     derived from ``TPUMapper._analyze_systolic_utilization`` averaged
     over typical edge-AI matrix shapes. Not carried by NPUBlock;
     v5 reconciliation: add to schema or move to NPU-side default.

  5. ``tile_energy_model=TPUTileEnergyModel(...)`` -- the
     architectural tile-energy decomposition (weight FIFO,
     accumulator, UB read/write, MAC) is TPU-specific. v5 reconciliation:
     a generalized ``TileEnergyModel`` field on NPUBlock.

  6. ``BOMCostProfile`` -- die / package / memory / PCB / thermal /
     margin / retail; v5 ``Market.bom`` will absorb this (same pattern
     as Hailo-8 / Hailo-10H factories).

Two documented shape drifts captured by the parity test:

  - ``energy_per_flop_fp32`` synthesis -- NPUs don't ship FP32; the
    loader synthesizes it as ``energy_per_op_int8 * 8`` per the
    standard-cell rule of thumb. ~38% drift for Coral (3.6 pJ vs
    2.6 pJ hand-coded) because Coral's fabric int8 energy
    (0.45 pJ) reflects systolic reuse already. v5 reconciliation item.
  - ``threads_per_unit`` -- the YAML reports 4096 (= lanes_per_unit
    for the 64x64 systolic array); the hand-coded value was 256
    (an estimate). The YAML value is structurally correct; the
    parity test pins the YAML value as the new contract.

The legacy resource-model ``name`` ("Coral-Edge-TPU") is preserved.
"""

from graphs.core.confidence import EstimationConfidence

from ...architectural_energy import TPUTileEnergyModel
from ...resource_model import BOMCostProfile, HardwareResourceModel, HardwareType
from .npu_yaml_loader import load_npu_resource_model_from_yaml


_LEGACY_NAME = "Coral-Edge-TPU"
_YAML_BASE_ID = "google_coral_edge_tpu"


def coral_edge_tpu_resource_model() -> HardwareResourceModel:
    """Google Coral Edge TPU (single-tile systolic NPU; INT8-only).

    See the YAML for the canonical chip description (64x64 INT8
    systolic array as 1 dataflow unit with 4096 lanes, 512 KiB
    on-chip unified buffer, no external DRAM, crossbar NoC, single
    2W operating point on GF 28nm SLP).
    """
    model = load_npu_resource_model_from_yaml(
        _YAML_BASE_ID,
        name_override=_LEGACY_NAME,
    )

    # Overlay 1: hardware_type. The v4 schema's NPUBlock + the graphs
    # HardwareType taxonomy split is a v5 reconciliation item; until
    # then, Coral keeps TPU so TPUMapper's guard accepts it unchanged.
    model.hardware_type = HardwareType.TPU

    # Overlay 2-3: memory bandwidth + technology. Coral's real
    # bottleneck is the host bus (USB 3.0 / PCIe Gen2), not the
    # UB-to-systolic direct connect that the YAML's on-chip
    # bandwidth captures.
    model.peak_bandwidth = 4e9                # ~4 GB/s host bus
    model.memory_technology = "LPDDR4 (host)"
    model.memory_read_energy_per_byte_pj = 20.0
    model.memory_write_energy_per_byte_pj = 24.0

    # Overlay 4: TPU-systolic-specific pipeline-fill overhead. Derived
    # from TPUMapper._analyze_systolic_utilization averaged over
    # typical edge-AI matrix shapes (~mobilenet / detection backbones).
    model.pipeline_fill_overhead = 0.15
    model.set_provenance(
        "pipeline_fill_overhead",
        EstimationConfidence.theoretical(
            score=0.50,
            source=("Coral Edge TPU 64x64 systolic array, derived from "
                    "TPUMapper._analyze_systolic_utilization formula "
                    "averaged over typical edge-AI matrix shapes"),
        ),
    )

    # Overlay 5: TPU-systolic-specific tile energy model. Scaled-down
    # from the datacenter TPU equivalents; ultra-low-power coefficients.
    model.tile_energy_model = TPUTileEnergyModel(
        array_width=64,
        array_height=64,
        num_arrays=1,
        weight_tile_size=4 * 1024,         # 4 KiB tiny tiles (edge)
        weight_fifo_depth=1,                # minimal buffering
        pipeline_fill_cycles=64,
        clock_frequency_hz=500e6,
        accumulator_size=512 * 1024,
        accumulator_width=64,
        unified_buffer_size=512 * 1024,
        weight_memory_energy_per_byte=20.0e-12,        # host bus
        weight_fifo_energy_per_byte=0.5e-12,           # on-chip SRAM
        unified_buffer_read_energy_per_byte=0.5e-12,
        unified_buffer_write_energy_per_byte=0.5e-12,
        accumulator_write_energy_per_element=0.4e-12,
        accumulator_read_energy_per_element=0.3e-12,
        weight_shift_in_energy_per_element=0.3e-12,
        activation_stream_energy_per_element=0.2e-12,
        mac_energy=0.15e-12,                # 0.15 pJ per INT8 MAC (efficient)
    )

    # Overlay 6: BoM cost -- not yet carried by the v4 ComputeProduct
    # schema. Numbers from Coral teardowns + Google retail pricing
    # ($75 M.2 module; $59 USB Type-A; $34 PCIe).
    model.bom_cost_profile = BOMCostProfile(
        silicon_die_cost=12.0,             # 28nm small die (~25 mm^2)
        package_cost=5.0,
        memory_cost=0.0,                   # uses host memory
        pcb_assembly_cost=3.0,
        thermal_solution_cost=1.0,
        other_costs=4.0,
        total_bom_cost=25.0,                # auto-calculated: $25
        margin_multiplier=3.0,
        retail_price=75.0,                  # M.2 module retail
        volume_tier="10K+",
        process_node="28nm",
        year=2025,
        notes=(
            "Ultra-low-cost edge TPU. Minimal on-chip memory, uses "
            "host CPU memory via USB / PCIe. USB Type-A, M.2, and "
            "PCIe form factors. Target: IoT cameras, embedded vision, "
            "battery devices."
        ),
    )

    return model
