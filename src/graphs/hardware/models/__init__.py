"""
Hardware resource models organized by deployment category.

This module re-exports all resource model functions for backward compatibility.
Models are now organized into subdirectories by category:
- datacenter/: High-end GPUs, TPUs, and server CPUs
- edge/: Edge AI accelerators and SBCs
- automotive/: Automotive-grade SoCs
- mobile/: Mobile GPUs and SoCs
- accelerators/: Fixed-function and reconfigurable accelerators
- ip_cores/: Licensable IP cores for SoC integration

Usage:
    from graphs.hardware.models import jetson_thor_resource_model
    model = jetson_thor_resource_model()
"""

from .datacenter.h100_pcie import h100_pcie_resource_model
from .datacenter.v100_sxm2 import v100_sxm2_resource_model
from .datacenter.a100_sxm4_80gb import a100_sxm4_80gb_resource_model
from .datacenter.t4 import t4_resource_model
from .datacenter.tpu_v4 import tpu_v4_resource_model
from .datacenter.cpu_x86 import cpu_x86_resource_model
from .datacenter.ampere_ampereone_192 import ampere_ampereone_192_resource_model
from .datacenter.ampere_ampereone_128 import ampere_ampereone_128_resource_model
from .datacenter.intel_xeon_platinum_8490h import intel_xeon_platinum_8490h_resource_model
from .datacenter.intel_xeon_platinum_8592plus import intel_xeon_platinum_8592plus_resource_model
from .datacenter.intel_granite_rapids import intel_granite_rapids_resource_model
from .datacenter.amd_epyc_9654 import amd_epyc_9654_resource_model
from .datacenter.amd_epyc_9754 import amd_epyc_9754_resource_model
from .datacenter.amd_epyc_turin import amd_epyc_turin_resource_model
from .edge.jetson_orin_agx import jetson_orin_agx_resource_model
from .edge.jetson_orin_nano import jetson_orin_nano_resource_model
from .edge.coral_edge_tpu import coral_edge_tpu_resource_model
from .edge.qrb5165 import qrb5165_resource_model
from .automotive.jetson_thor import jetson_thor_resource_model
from .automotive.ti_tda4vm import ti_tda4vm_resource_model
from .automotive.ti_tda4vl import ti_tda4vl_resource_model
from .automotive.ti_tda4al import ti_tda4al_resource_model
from .automotive.ti_tda4vh import ti_tda4vh_resource_model
from .mobile.arm_mali_g78_mp20 import arm_mali_g78_mp20_resource_model
from .accelerators.kpu_t64 import kpu_t64_resource_model
from .accelerators.kpu_t256 import kpu_t256_resource_model
from .accelerators.kpu_t768 import kpu_t768_resource_model
from .accelerators.xilinx_vitis_ai_dpu import xilinx_vitis_ai_dpu_resource_model
from .accelerators.stanford_plasticine_cgra import stanford_plasticine_cgra_resource_model
from .ip_cores.ceva_neupro_npm11 import ceva_neupro_npm11_resource_model
from .ip_cores.cadence_vision_q8 import cadence_vision_q8_resource_model
from .ip_cores.synopsys_arc_ev7x import synopsys_arc_ev7x_resource_model

__all__ = [
    'h100_pcie_resource_model',
    'v100_sxm2_resource_model',
    'a100_sxm4_80gb_resource_model',
    't4_resource_model',
    'tpu_v4_resource_model',
    'cpu_x86_resource_model',
    'ampere_ampereone_192_resource_model',
    'ampere_ampereone_128_resource_model',
    'intel_xeon_platinum_8490h_resource_model',
    'intel_xeon_platinum_8592plus_resource_model',
    'intel_granite_rapids_resource_model',
    'amd_epyc_9654_resource_model',
    'amd_epyc_9754_resource_model',
    'amd_epyc_turin_resource_model',
    'jetson_orin_agx_resource_model',
    'jetson_orin_nano_resource_model',
    'coral_edge_tpu_resource_model',
    'qrb5165_resource_model',
    'jetson_thor_resource_model',
    'ti_tda4vm_resource_model',
    'ti_tda4vl_resource_model',
    'ti_tda4al_resource_model',
    'ti_tda4vh_resource_model',
    'arm_mali_g78_mp20_resource_model',
    'kpu_t64_resource_model',
    'kpu_t256_resource_model',
    'kpu_t768_resource_model',
    'xilinx_vitis_ai_dpu_resource_model',
    'stanford_plasticine_cgra_resource_model',
    'ceva_neupro_npm11_resource_model',
    'cadence_vision_q8_resource_model',
    'synopsys_arc_ev7x_resource_model',
]
