"""
Accelerators hardware resource models.
"""

from .kpu_t64 import kpu_t64_resource_model
from .kpu_t256 import kpu_t256_resource_model
from .kpu_t768 import kpu_t768_resource_model
from .xilinx_vitis_ai_dpu import xilinx_vitis_ai_dpu_resource_model
from .stanford_plasticine_cgra import stanford_plasticine_cgra_resource_model
from .ceva_neupro_npm11 import ceva_neupro_npm11_resource_model
from .cadence_vision_q8 import cadence_vision_q8_resource_model
from .synopsys_arc_ev7x import synopsys_arc_ev7x_resource_model

__all__ = [
    'kpu_t64_resource_model',
    'kpu_t256_resource_model',
    'kpu_t768_resource_model',
    'xilinx_vitis_ai_dpu_resource_model',
    'stanford_plasticine_cgra_resource_model',
    'ceva_neupro_npm11_resource_model',
    'cadence_vision_q8_resource_model',
    'synopsys_arc_ev7x_resource_model',
]
