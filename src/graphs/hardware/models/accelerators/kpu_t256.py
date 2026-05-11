"""Stillwater KPU-T256 resource model -- thin wrapper around the YAML loader.

See ``kpu_t64.py`` for the Phase 4b PR 5 collapse rationale; this file
follows the same pattern.
"""

from ...resource_model import BOMCostProfile, HardwareResourceModel
from .kpu_yaml_loader import load_kpu_resource_model_from_yaml


def kpu_t256_resource_model() -> HardwareResourceModel:
    """Stillwater KPU-T256 -- high-performance edge / embodied AI.

    256 heterogeneous tiles (179 INT8, 51 BF16, 26 Matrix), 20x20 PE
    arrays, TSMC N16. The M0.5 sweet spot: big enough for serious
    peak compute, dense enough that per-tile utilization stays high.
    Targets edge servers, autonomous vehicles, advanced drones.
    Power profiles: 15W, 30W (default), 50W.
    """
    model = load_kpu_resource_model_from_yaml("kpu_t256_32x32_lp5x16_16nm_tsmc_ffp")
    # M0.5 domain-flow MAC energies (see kpu_t64.py for rationale).
    model.tile_energy_model.mac_energy_int8 = 0.10e-12
    model.tile_energy_model.mac_energy_bf16 = 0.16e-12
    model.tile_energy_model.mac_energy_fp32 = 0.30e-12
    model.bom_cost_profile = BOMCostProfile(
        silicon_die_cost=280.0,
        package_cost=45.0,
        memory_cost=90.0,
        pcb_assembly_cost=20.0,
        thermal_solution_cost=8.0,
        other_costs=12.0,
        total_bom_cost=455.0,
        margin_multiplier=2.4,
        retail_price=1092.0,
        volume_tier="10K+",
        process_node="16nm",
        year=2025,
        notes="Mid-range edge AI accelerator. 256 tiles, advanced "
              "flip-chip packaging. Competitive with high-end edge GPUs "
              "but better efficiency and predictability.",
    )
    return model
