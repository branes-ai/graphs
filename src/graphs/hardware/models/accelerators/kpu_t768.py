"""Stillwater KPU-T768 resource model -- thin wrapper around the YAML loader.

See ``kpu_t64.py`` for the Phase 4b PR 5 collapse rationale; this file
follows the same pattern.
"""

from ...resource_model import BOMCostProfile, HardwareResourceModel
from .kpu_yaml_loader import load_kpu_resource_model_from_yaml


def kpu_t768_resource_model() -> HardwareResourceModel:
    """Stillwater KPU-T768 -- datacenter AI inference accelerator.

    768 heterogeneous tiles (537 INT8, 154 BF16, 77 Matrix). INT8/BF16
    tiles use 16x8 PE arrays; Matrix tiles use 8x8 systolic blocks
    (tensor-core-style) for LLM inference. TSMC N7. 192 MiB
    distributed L3 scratchpad. HBM3 64 GB / 1.6 TB/s. Targets
    datacenter LLM inference, batch-processing AI workloads.
    Power profiles: 30W, 60W (default), 100W (liquid-cooled).
    """
    model = load_kpu_resource_model_from_yaml("stillwater_kpu_t768")
    # M0.5 domain-flow MAC energies, datacenter-tuned. See kpu_t64.py
    # for the override rationale; T768 uses lower per-MAC numbers
    # than the T64-T256 family (more aggressive datacenter binning).
    model.tile_energy_model.mac_energy_int8 = 0.08e-12
    model.tile_energy_model.mac_energy_bf16 = 0.13e-12
    model.tile_energy_model.mac_energy_fp32 = 0.24e-12
    model.bom_cost_profile = BOMCostProfile(
        silicon_die_cost=680.0,
        package_cost=120.0,
        memory_cost=280.0,
        pcb_assembly_cost=65.0,
        thermal_solution_cost=45.0,
        other_costs=35.0,
        total_bom_cost=1225.0,
        margin_multiplier=2.4,
        retail_price=2940.0,
        volume_tier="10K+",
        process_node="7nm",
        year=2025,
        notes="Datacenter AI inference accelerator. 768 tiles, "
              "7nm process, multi-chip or interposer packaging. "
              "HBM3 memory (64 GB / 1.6 TB/s). Liquid cooling at the "
              "100 W profile.",
    )
    return model
