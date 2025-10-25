"""
Edge hardware resource models.
"""

from .jetson_orin_agx import jetson_orin_agx_resource_model
from .jetson_orin_nano import jetson_orin_nano_resource_model
from .coral_edge_tpu import coral_edge_tpu_resource_model
from .qrb5165 import qrb5165_resource_model

__all__ = [
    'jetson_orin_agx_resource_model',
    'jetson_orin_nano_resource_model',
    'coral_edge_tpu_resource_model',
    'qrb5165_resource_model',
]
