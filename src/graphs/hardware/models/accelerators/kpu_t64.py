"""Stillwater KPU-T64 resource model.

Thin wrapper around ``load_kpu_resource_model_from_yaml`` (Phase 4b PR 5).
Architecture, performance, thermal profiles, energy, fabric, and M3-M7
layer attributes all come from
``embodied-schemas:kpus/stillwater/kpu_t64_32x32_lp5x4_16nm_tsmc_ffp.yaml`` via the
loader. This factory only adds the BOM cost profile, which is
economic / market data not tracked in the silicon-architecture YAML.

Pre-collapse this file was ~655 LOC of hand-coded
``HardwareResourceModel`` construction with values that drifted from
the YAML; the contract snapshot at
``docs/designs/kpu-test-contract-snapshot.md`` documented every delta
and Phase 4b PRs 1-4 reconciled them.
"""

from ...resource_model import BOMCostProfile, HardwareResourceModel
from .kpu_yaml_loader import load_kpu_resource_model_from_yaml


def kpu_t64_resource_model() -> HardwareResourceModel:
    """Stillwater KPU-T64 -- entry-level edge AI accelerator.

    64 heterogeneous tiles (44 INT8, 13 BF16, 7 Matrix), 32x32 PE
    arrays, TSMC N16. Targets battery-powered drones, robots, edge AI.
    Power profiles: 3W (battery), 6W (standard, default), 10W (peak).
    """
    model = load_kpu_resource_model_from_yaml("kpu_t64_32x32_lp5x4_16nm_tsmc_ffp")
    # M0.5 domain-flow MAC energies are more aggressive than the
    # generic N16 BALANCED_LOGIC numbers in process-nodes/tsmc/n16.yaml.
    # Preserved here so the per-MAC energy ranges asserted by
    # tests/hardware/test_kpu_tile_energy.py continue to hold. Future
    # work: extend the YAML schema with a per-SKU mac_energy override
    # so this can move into embodied-schemas (Phase 5+ scope).
    model.tile_energy_model.mac_energy_int8 = 0.10e-12
    model.tile_energy_model.mac_energy_bf16 = 0.16e-12
    model.tile_energy_model.mac_energy_fp32 = 0.30e-12
    # Per-tile L1 scratchpad bandwidth: each PE delivers ~1.5 GB/s of
    # steady-state demand. Aggregate L1 BW ~96 TB/s across 64 tiles.
    # Shared L2 BW: 4 MB shared L2 feeds the tile mesh at NoC bisection
    # BW; vendor spec is 200 GB/s aggregate. Used by the V4 classifier;
    # locked in by tests/validation_model_v4/test_classify_with_bw_peaks.py.
    model.l1_bandwidth_per_unit_bps = 1.5e12
    model.l2_bandwidth_bps = 200e9
    model.bom_cost_profile = BOMCostProfile(
        silicon_die_cost=75.0,        # 16nm TSMC (small die)
        package_cost=15.0,             # Flip-chip BGA
        memory_cost=20.0,              # 2GB LPDDR4X on-package
        pcb_assembly_cost=8.0,
        thermal_solution_cost=2.0,     # Small heatsink (3-10W)
        other_costs=5.0,               # Testing, connectors
        total_bom_cost=0,              # Auto-calculated: $125
        margin_multiplier=2.4,
        retail_price=0,                # Auto-calculated: $300
        volume_tier="10K+",
        process_node="16nm",
        year=2025,
        notes="Entry-level KPU for edge AI. Competitive with Hailo-8 "
              "($40-60 BOM) but higher BOM due to integrated memory. "
              "Superior to Jetson Orin Nano ($200 BOM).",
    )
    return model
