"""
Automotive hardware resource models.
"""

from .jetson_thor_128gb import jetson_thor_128gb_resource_model
from .ti_tda4vm import ti_tda4vm_resource_model
from .ti_tda4vl import ti_tda4vl_resource_model
from .ti_tda4al import ti_tda4al_resource_model
from .ti_tda4vh import ti_tda4vh_resource_model

__all__ = [
    'jetson_thor_128gb_resource_model',
    'ti_tda4vm_resource_model',
    'ti_tda4vl_resource_model',
    'ti_tda4al_resource_model',
    'ti_tda4vh_resource_model',
]
