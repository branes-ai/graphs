"""
Accelerators hardware resource models.

This module contains standalone fixed-function and reconfigurable accelerators.
IP cores have been moved to models/ip_cores/.
"""

from .kpu_t64 import kpu_t64_resource_model
from .kpu_t256 import kpu_t256_resource_model
from .kpu_t768 import kpu_t768_resource_model
from .xilinx_vitis_ai_dpu import xilinx_vitis_ai_dpu_resource_model
from .stanford_plasticine_cgra import stanford_plasticine_cgra_resource_model

__all__ = [
    'kpu_t64_resource_model',
    'kpu_t256_resource_model',
    'kpu_t768_resource_model',
    'xilinx_vitis_ai_dpu_resource_model',
    'stanford_plasticine_cgra_resource_model',
]
