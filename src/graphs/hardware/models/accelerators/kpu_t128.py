"""Stillwater KPU-T128 resource model -- thin wrapper around the YAML loader.

See ``kpu_t64.py`` for the Phase 4b PR 5 collapse rationale; this file
follows the same pattern.
"""

from ...resource_model import BOMCostProfile, HardwareResourceModel
from .kpu_yaml_loader import load_kpu_resource_model_from_yaml


def kpu_t128_resource_model() -> HardwareResourceModel:
    """Stillwater KPU-T128 -- mid-range embodied AI.

    128 heterogeneous tiles (89 INT8, 26 BF16, 13 Matrix), 32x32 PE
    arrays (131K PEs), TSMC N16 -- the same canonical tile as T64 and
    T256. Targets autonomous robots, advanced edge AI. Power
    profiles: 6W, 12W (default), 18W.
    """
    model = load_kpu_resource_model_from_yaml("kpu_t128_32x32_lp5x8_16nm_tsmc_ffp")
    # M0.5 domain-flow MAC energies (see kpu_t64.py for rationale).
    # Applied to the chip-level model and every per-class model, so a
    # report that selects one class still sees this SKU's numbers.
    model.apply_mac_energy_override(int8=0.10e-12, bf16=0.16e-12, fp32=0.30e-12)
    model.bom_cost_profile = BOMCostProfile(
        silicon_die_cost=100.0,       # 16nm TSMC, between T64 and T256
        package_cost=18.0,
        memory_cost=30.0,              # 4GB LPDDR5 on-package
        pcb_assembly_cost=10.0,
        thermal_solution_cost=3.0,
        other_costs=6.0,
        total_bom_cost=0,              # auto-calculated
        margin_multiplier=2.4,
        retail_price=0,                # auto-calculated
        volume_tier="10K+",
        process_node="16nm",
        year=2026,
        notes="Mid-range KPU for embodied AI. 32x32 PE array per tile "
              "balances pipeline utilization against tile count.",
    )
    return model
